"""
FastAPI backend for the RAG HR App — Week 5.

Endpoints:
  GET  /health
  POST /upload
  GET  /collections
  GET  /documents?collection=...
  GET  /chunks?source_file=...&collection=...
  POST /query
"""
from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

import sys, os, time
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests as http_lib
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder

from api.models import (
    UploadResponse, DocumentInfo, ChunkDetail, CollectionInfo,
    QueryRequest, QueryResponse, RetrievedChunk,
)
from config import (
    EMBED_MODEL_NAME, CROSS_ENCODER_MODEL, QUERY_PREFIX,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_URL,
    DEFAULT_COLLECTION, MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS,
)
from vectorstore import get_store
from ingestion.pipeline import ingest_file
from tracing import lf_span, lf_generation, lf_set_output, lf_get_ids, lf_flush

app = FastAPI(title="RAG HR App", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_embed_model: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        get_store().list_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    return {"status": "ok", "version": "2.0.0", "qdrant": qdrant_ok}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(default=DEFAULT_COLLECTION),
    chunking_strategy: str = Form(default="auto"),
):
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    file_bytes = await file.read()
    if len(file_bytes) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")

    try:
        result = ingest_file(file_bytes, filename, collection=collection, chunking_strategy=chunking_strategy)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Ingestion failed: {exc}")

    return UploadResponse(**result)


# ── Collections ───────────────────────────────────────────────────────────────

@app.get("/collections", response_model=list[CollectionInfo])
def list_collections():
    store = get_store()
    result = []
    for name in store.list_collections():
        try:
            result.append(CollectionInfo(**store.get_collection_info(name)))
        except Exception:
            result.append(CollectionInfo(name=name))
    return result


# ── Documents ────────────────────────────────────────────────────────────────

@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(collection: str = Query(default=DEFAULT_COLLECTION)):
    try:
        return [DocumentInfo(**d) for d in get_store().get_documents(collection)]
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/chunks", response_model=list[ChunkDetail])
def get_chunks(
    source_file: str = Query(...),
    collection: str = Query(default=DEFAULT_COLLECTION),
):
    try:
        raw = get_store().get_chunks_for_document(collection, source_file)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return [
        ChunkDetail(
            id=pt.get("id", 0),
            text=pt.get("text", ""),
            heading=pt.get("heading", ""),
            source_file=pt.get("source_file") or pt.get("source", ""),
            source_type=pt.get("source_type", ""),
            chunk_index=pt.get("chunk_index"),
            upload_timestamp=pt.get("upload_timestamp"),
            has_table=pt.get("has_table"),
        )
        for pt in raw
    ]


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    store = get_store()
    model = _get_embed_model()

    with lf_span("rag_query", input={"question": req.question, "collection": req.collection, "filters": req.filters}) as root_obs:

        # ── Retrieval ─────────────────────────────────────────────────────────
        with lf_span("retrieval", span_type="retriever",
                     input={"question": req.question, "top_k": req.top_k, "use_hybrid": req.use_hybrid}) as ret_obs:
            candidate_k = req.top_k * 3
            try:
                if req.use_hybrid:
                    from hybrid import HybridRetriever
                    hr = HybridRetriever(store, req.collection, model, QUERY_PREFIX)
                    hits = hr.retrieve(req.question, top_k=candidate_k)
                    if req.filters:
                        hits = [h for h in hits if _matches_filter(h, req.filters)]
                else:
                    qvec = model.encode([QUERY_PREFIX + req.question], normalize_embeddings=True).tolist()[0]
                    hits = store.query(req.collection, qvec, candidate_k, metadata_filter=req.filters)
            except Exception as exc:
                ret_obs.update(output={"error": str(exc)}, level="ERROR")
                raise HTTPException(500, f"Retrieval failed: {exc}")
            ret_obs.update(output={"hits_count": len(hits), "top_scores": [round(h.get("similarity", 0), 4) for h in hits[:5]]})

        if not hits:
            root_obs.update(output={"answer": None, "reason": "no_hits"})
            lf_set_output({"answer": None, "reason": "no_hits"})
            lf_flush()
            tid, url = lf_get_ids()
            return QueryResponse(
                question=req.question, answer=None, reason="no_results",
                message="No documents in this collection. Upload documents first.",
                confidence=0.0, chunks=[], collection=req.collection,
                model=OPENROUTER_MODEL, trace_id=tid, trace_url=url,
            )

        # ── Reranking ─────────────────────────────────────────────────────────
        with lf_span("reranking", input={"candidate_count": len(hits)}) as rerank_obs:
            ce = _get_cross_encoder()
            scores = ce.predict([(req.question, h["text"]) for h in hits])
            ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
            hits = [h for _, h in ranked[:req.top_k]]
            top_ce = float(np.max(scores))
            ce_confidence = float(1.0 / (1.0 + np.exp(-top_ce)))  # sigmoid → [0, 1]
            rerank_obs.update(output={"top_ce_score": round(top_ce, 4), "ce_confidence": round(ce_confidence, 4), "kept": len(hits)})

        # ── Confidence gate ───────────────────────────────────────────────────
        best_sim = max((h.get("similarity", 0.0) for h in hits), default=0.0)
        # Use the higher of dense similarity and CE-based confidence as the final confidence.
        # CE (cross-encoder) is a better relevance signal; dense sim catches off-topic queries early.
        confidence = round(max(best_sim, ce_confidence), 4)
        with lf_span("confidence_gate",
                     input={"best_similarity": round(best_sim, 4), "ce_confidence": round(ce_confidence, 4), "threshold": req.confidence_threshold}) as gate_obs:
            if best_sim < req.confidence_threshold:
                gate_obs.update(output={"passed": False})
                root_obs.update(output={"answer": None, "reason": "below_threshold", "confidence": confidence})
                lf_set_output({"answer": None, "reason": "below_threshold", "confidence": confidence})
                lf_flush()
                tid, url = lf_get_ids()
                return QueryResponse(
                    question=req.question, answer=None, reason="no_relevant_context",
                    message=(
                        f"Documents don't contain relevant information for your question. "
                        f"Best similarity: {best_sim:.2f} (threshold: {req.confidence_threshold:.2f}). "
                        "Try rephrasing or upload more relevant documents."
                    ),
                    confidence=confidence, chunks=_hits_to_chunks(hits),
                    collection=req.collection, model=OPENROUTER_MODEL, trace_id=None, trace_url=None,
                )
            gate_obs.update(output={"passed": True, "best_similarity": round(best_sim, 4), "ce_confidence": round(ce_confidence, 4)})

        # ── Generation ────────────────────────────────────────────────────────
        from query import build_prompt
        prompt = build_prompt(req.question, hits)

        with lf_generation("llm_generation", model=OPENROUTER_MODEL, input=prompt) as gen_obs:
            try:
                answer = _call_llm(prompt)
            except Exception as exc:
                gen_obs.update(output={"error": str(exc)}, level="ERROR")
                raise HTTPException(502, f"LLM call failed: {exc}")
            gen_obs.update(output=answer)

        root_obs.update(output={"answer": answer, "confidence": confidence})
        lf_set_output({"answer": answer, "confidence": confidence})
        lf_flush()
        tid, url = lf_get_ids()

    return QueryResponse(
        question=req.question, answer=answer,
        confidence=confidence, chunks=_hits_to_chunks(hits),
        collection=req.collection, model=OPENROUTER_MODEL,
        trace_id=tid, trace_url=url,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hits_to_chunks(hits: list) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=h.get("text", ""),
            heading=h.get("heading", ""),
            source_file=h.get("source_file") or h.get("source", ""),
            source_type=h.get("source_type", ""),
            similarity=round(h.get("similarity", 0.0), 4),
            rrf_score=h.get("rrf_score"),
            chunk_index=h.get("chunk_index"),
        )
        for h in hits
    ]


def _matches_filter(hit: dict, filters: dict) -> bool:
    return all(hit.get(k) == v for k, v in filters.items() if v is not None)


FALLBACK_MODELS = [
    OPENROUTER_MODEL,
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "minimax/minimax-m3:free",
]


def _call_llm(prompt: str, max_retries: int = 3) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")

    models_to_try = list(dict.fromkeys(FALLBACK_MODELS))
    last_error = None

    for model_id in models_to_try:
        for attempt in range(max_retries):
            try:
                resp = http_lib.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                    },
                    timeout=120,
                )
                if resp.status_code == 429:
                    wait = 5 * (2 ** attempt)
                    print(f"  [rate limited on {model_id} — waiting {wait}s]")
                    time.sleep(wait)
                    continue
                if resp.status_code in (500, 502, 503):
                    print(f"  [model {model_id} returned {resp.status_code} — trying next]")
                    last_error = f"{model_id} HTTP {resp.status_code}"
                    break
                resp.raise_for_status()
                answer = resp.json()["choices"][0]["message"]["content"].strip()
                if model_id != OPENROUTER_MODEL:
                    print(f"  [used fallback model: {model_id}]")
                return answer
            except http_lib.exceptions.Timeout:
                last_error = f"{model_id} timed out"
                print(f"  [timeout on {model_id} — trying next]")
                break
            except Exception as exc:
                last_error = str(exc)
                break

    raise RuntimeError(f"All models failed. Last error: {last_error}")
