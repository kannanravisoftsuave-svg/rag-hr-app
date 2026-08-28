"""
Ingestion pipeline: file bytes → Qdrant chunks.

Stages:
  1. extract   — pull text/tables from PDF, DOCX, or MD
  2. chunk     — split into retrievable pieces
  3. embed     — encode with BGE-small
  4. store     — upsert into Qdrant with full metadata
"""
from __future__ import annotations
import os
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL_NAME, UPLOAD_DIR, DEFAULT_COLLECTION, EMBED_DIM
from ingestion.extractor import extract
from ingestion.chunker import chunk_document
from tracing import lf_span, lf_set_output, lf_get_ids, lf_flush

_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def ingest_file(
    file_bytes: bytes,
    filename: str,
    collection: str = DEFAULT_COLLECTION,
    chunking_strategy: str = "auto",
) -> dict:
    ext = Path(filename).suffix.lower()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    upload_ts = datetime.now(timezone.utc).isoformat()

    with lf_span("document_ingestion",
                 input={"filename": filename, "collection": collection, "size_bytes": len(file_bytes)},
                 metadata={"filename": filename, "source_type": ext.lstrip("."), "collection": collection}) as root_obs:

        # ── 1. Extract ────────────────────────────────────────────────────────
        with lf_span("text_extraction", input={"filename": filename, "format": ext}) as ext_obs:
            try:
                extracted = extract(filename, file_bytes)
            except Exception as exc:
                ext_obs.update(output={"error": str(exc)}, level="ERROR")
                raise

            if not extracted["text"].strip():
                msg = (
                    f"No text extracted from '{filename}'. "
                    "The file may be empty or image-only (scanned PDFs need OCR)."
                )
                ext_obs.update(output={"error": msg}, level="ERROR")
                raise ValueError(msg)

            ext_obs.update(output={
                "char_count": len(extracted["text"]),
                "total_pages": extracted["metadata"].get("total_pages"),
                "has_tables": extracted["metadata"].get("has_tables", False),
                "warnings": extracted["warnings"],
            })

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        source_metadata = {
            "source_file": filename,
            "source_type": ext.lstrip("."),
            "file_hash": file_hash,
            "upload_timestamp": upload_ts,
            "collection": collection,
            "has_table": extracted["metadata"].get("has_tables", False),
            "total_pages": extracted["metadata"].get("total_pages"),
        }

        with lf_span("chunking",
                     input={"strategy": chunking_strategy, "source_type": ext.lstrip(".")}) as chunk_obs:
            chunks = chunk_document(extracted, source_metadata, strategy=chunking_strategy)
            chunks = [c for c in chunks if c.get("text", "").strip()]
            chunk_obs.update(output={
                "chunk_count": len(chunks),
                "strategy": chunking_strategy,
                "avg_chars": int(sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)),
            })

        if not chunks:
            raise ValueError(f"No chunks produced from '{filename}'. Document may be too short.")

        # ── 3. Embed ──────────────────────────────────────────────────────────
        with lf_span("embedding", input={"model": EMBED_MODEL_NAME, "chunk_count": len(chunks)}) as emb_obs:
            model = get_embed_model()
            texts = [c["text"] for c in chunks]
            BATCH_SIZE = 32
            all_embeddings = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                all_embeddings.extend(model.encode(batch, normalize_embeddings=True).tolist())
            embeddings = all_embeddings
            emb_obs.update(output={"embedding_dim": EMBED_DIM, "chunk_count": len(chunks)})

        # ── 4. Store ──────────────────────────────────────────────────────────
        from vectorstore import get_store
        from qdrant_client.models import PointStruct, VectorParams, Distance

        with lf_span("vector_store", input={"collection": collection, "chunk_count": len(chunks)}) as store_obs:
            store = get_store()
            _ensure_collection(store, collection)

            # Delete existing chunks for this file before re-inserting (dedup)
            deleted = _delete_by_source(store, collection, filename)

            metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload={**metadatas[i], "text": texts[i]},
                )
                for i in range(len(chunks))
            ]
            store.client.upsert(collection_name=collection, points=points)
            store_obs.update(output={
                "stored": len(chunks),
                "deleted_old": deleted,
                "collection": collection,
            })

        # ── Save file locally ─────────────────────────────────────────────────
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        dest = Path(UPLOAD_DIR) / filename
        dest.write_bytes(file_bytes)

        lf_flush()
        tid, _ = lf_get_ids()

        result = {
            "filename": filename,
            "collection": collection,
            "chunk_count": len(chunks),
            "char_count": len(extracted["text"]),
            "total_pages": extracted["metadata"].get("total_pages"),
            "has_tables": extracted["metadata"].get("has_tables", False),
            "file_hash": file_hash,
            "upload_timestamp": upload_ts,
            "warnings": extracted["warnings"],
            "trace_id": tid,
        }
        root_obs.update(output=result)
        lf_set_output(result)
        return result


def _ensure_collection(store, name: str) -> None:
    from qdrant_client.models import Distance, VectorParams
    if not store.client.collection_exists(name):
        store.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def _delete_by_source(store, collection: str, filename: str) -> int:
    """Delete all points whose source_file or source matches filename. Returns count deleted."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    deleted = 0
    for field in ("source_file", "source"):
        try:
            result = store.client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[FieldCondition(key=field, match=MatchValue(value=filename))]
                ),
            )
            if hasattr(result, "deleted"):
                deleted += result.deleted or 0
        except Exception:
            pass
    return deleted
