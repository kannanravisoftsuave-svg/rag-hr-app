"""
Streamlit UI for the RAG HR App - Week 5.

Pages (sidebar radio):
  Upload     -> upload PDF / DOCX / MD and trigger ingestion
  Documents  -> browse all ingested documents and their chunks
  Query      -> ask questions, see retrieved chunks + answer
  Traces     -> link to Langfuse dashboard for error analysis

Run:  streamlit run ui/app.py
Requires the FastAPI backend running at API_URL (default http://localhost:8000).
"""
from __future__ import annotations

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="RAG HR App",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Helpers -------------------------------------------------------------------

def _api(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=300, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach the FastAPI backend at **{API_URL}**. "
            "Make sure it is running:\n\n"
            "```\nuvicorn api.main:app --reload --port 8000\n```"
        )
        return None
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"API error {exc.response.status_code}: {detail or exc}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


def _get_collections():
    data = _api("GET", "/collections")
    if not data:
        return ["documents"]
    return [c["name"] for c in data] or ["documents"]


# -- Sidebar navigation --------------------------------------------------------
with st.sidebar:
    st.title(":books: RAG HR App")
    st.caption("Week 5 - Dynamic RAG")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Upload Documents", "Query", "View Documents", "Traces"],
        label_visibility="collapsed",
    )
    st.divider()
    health = _api("GET", "/health")
    if health:
        qdrant = ":green_circle: Online" if health.get("qdrant") else ":red_circle: Offline"
        st.caption(f"API v{health.get('version','?')}  -  Qdrant {qdrant}")


# ==============================================================================
# PAGE: Upload Documents
# ==============================================================================
if page == "Upload Documents":
    st.header(":outbox_tray: Upload Documents")
    st.markdown(
        "Upload a **PDF**, **DOCX**, or **Markdown** file. The system will extract text, "
        "chunk it, generate embeddings, and store everything in Qdrant automatically."
    )

    with st.form("upload_form"):
        uploaded = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "md"],
            help="Max 10 MB. PDF: text + tables extracted. DOCX: paragraphs + tables. MD: full text.",
        )
        col1, col2 = st.columns(2)
        with col1:
            collections = _get_collections()
            collection = st.selectbox(
                "Target collection",
                options=["documents"] + [c for c in collections if c != "documents"],
                help="The Qdrant collection to store chunks in.",
            )
        with col2:
            strategy = st.selectbox(
                "Chunking strategy",
                options=["auto", "heading", "page", "sliding_window"],
                help=(
                    "auto = smart default per file type.  "
                    "heading = split at ## / ###.  "
                    "page = one chunk per PDF page.  "
                    "sliding_window = fixed-size overlapping windows."
                ),
            )
        submitted = st.form_submit_button("Upload & Ingest", type="primary")

    if submitted:
        if not uploaded:
            st.warning("Please select a file before uploading.")
        else:
            with st.spinner(f"Ingesting **{uploaded.name}**..."):
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    data={"collection": collection, "chunking_strategy": strategy},
                    timeout=300,
                )
            if resp.status_code == 200:
                r = resp.json()
                st.success(f"Ingested **{r['filename']}** into collection **{r['collection']}**")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Chunks created", r["chunk_count"])
                col_b.metric("Characters extracted", f"{r['char_count']:,}")
                col_c.metric("Pages", r.get("total_pages") or "-")

                with st.expander("Details"):
                    st.json({
                        "filename": r["filename"],
                        "collection": r["collection"],
                        "has_tables": r["has_tables"],
                        "file_hash": r["file_hash"],
                        "upload_timestamp": r["upload_timestamp"],
                        "chunking_strategy": strategy,
                        "trace_id": r.get("trace_id"),
                    })

                if r.get("warnings"):
                    for w in r["warnings"]:
                        st.warning(w)

                if r.get("trace_id"):
                    st.info(f"[View ingestion trace in Langfuse]({LANGFUSE_HOST}/trace/{r['trace_id']})")
            else:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                st.error(f"Upload failed ({resp.status_code}): {detail}")


# ==============================================================================
# PAGE: Query
# ==============================================================================
elif page == "Query":
    st.header(":speech_balloon: Ask a Question")

    with st.sidebar:
        st.subheader("Query settings")
        collections = _get_collections()
        q_collection = st.selectbox("Collection", collections, key="q_col")

        docs_data = _api("GET", f"/documents?collection={q_collection}") or []
        doc_names = [d["filename"] for d in docs_data]
        filter_file = st.selectbox(
            "Filter by document (optional)",
            options=["-- all documents --"] + doc_names,
            key="q_filter_file",
        )
        top_k = st.slider("Top chunks (top_k)", min_value=1, max_value=20, value=8)
        threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
        use_hybrid = st.checkbox("Hybrid search (BM25 + dense)", value=True)

    question = st.text_area("Your question", height=80, placeholder="e.g. How many vacation days do I get after 2 years?")
    ask_btn = st.button("Ask", type="primary")

    if ask_btn:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            filters = None
            if filter_file != "-- all documents --":
                filters = {"source_file": filter_file}

            payload = {
                "question": question,
                "collection": q_collection,
                "top_k": top_k,
                "filters": filters,
                "use_hybrid": use_hybrid,
                "confidence_threshold": threshold,
            }

            with st.spinner("Retrieving and generating..."):
                resp = requests.post(f"{API_URL}/query", json=payload, timeout=300)

            if resp.status_code == 200:
                r = resp.json()
                st.session_state["last_result"] = r

                conf = r["confidence"]
                if conf >= 0.80:
                    badge = ":green_circle:"
                elif conf >= threshold:
                    badge = ":yellow_circle:"
                else:
                    badge = ":red_circle:"
                st.caption(f"{badge} Confidence: **{conf:.2f}** | Collection: `{r['collection']}` | Model: `{r['model']}`")

                if r.get("answer"):
                    st.subheader("Answer")
                    st.markdown(r["answer"])
                else:
                    st.warning(r.get("message", "No answer generated."))
                    st.caption(f"Reason: `{r.get('reason', 'unknown')}`")

                if r.get("chunks"):
                    st.subheader(f"Retrieved Chunks ({len(r['chunks'])})")
                    for i, chunk in enumerate(r["chunks"], 1):
                        label = (
                            f"#{i} - `{chunk['source_file']}` > **{chunk['heading']}** "
                            f"| similarity: {chunk['similarity']:.3f}"
                        )
                        with st.expander(label):
                            st.markdown(chunk["text"])
                            meta_cols = st.columns(3)
                            meta_cols[0].caption(f"Type: `{chunk['source_type']}`")
                            meta_cols[1].caption(f"Similarity: `{chunk['similarity']:.4f}`")
                            if chunk.get("rrf_score"):
                                meta_cols[2].caption(f"RRF: `{chunk['rrf_score']:.4f}`")

                if r.get("trace_url"):
                    st.info(f":mag: [View full trace in Langfuse]({r['trace_url']})")
                elif r.get("trace_id"):
                    st.info(f"Trace ID: `{r['trace_id']}` - configure Langfuse keys in `.env` to see the dashboard link.")
            else:
                try:
                    st.error(resp.json().get("detail", resp.text))
                except Exception:
                    st.error(resp.text)


# ==============================================================================
# PAGE: View Documents
# ==============================================================================
elif page == "View Documents":
    st.header(":open_file_folder: Uploaded Documents")

    collections = _get_collections()
    selected_col = st.selectbox("Collection", collections)

    docs = _api("GET", f"/documents?collection={selected_col}")
    if docs is None:
        st.stop()

    if not docs:
        st.info(f"No documents in collection **{selected_col}**. Upload some on the Upload page.")
    else:
        st.caption(f"{len(docs)} document(s) in **{selected_col}**")

        import pandas as pd
        df = pd.DataFrame([{
            "Filename": d["filename"],
            "Type": d.get("source_type", "").upper(),
            "Chunks": d["chunk_count"],
            "Has Tables": "Yes" if d.get("has_table") else "-",
            "Uploaded": d.get("upload_timestamp", "")[:19].replace("T", " "),
        } for d in docs])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Chunk Explorer")
        doc_choice = st.selectbox("Select a document to inspect", [d["filename"] for d in docs])
        if doc_choice:
            chunks = _api("GET", f"/chunks?source_file={requests.utils.quote(doc_choice)}&collection={selected_col}")
            if chunks:
                st.caption(f"{len(chunks)} chunks from **{doc_choice}**")
                for chunk in chunks:
                    heading = chunk.get("heading", "-")
                    idx = chunk.get("chunk_index", "?")
                    with st.expander(f"Chunk {idx} - {heading}"):
                        st.text(chunk["text"])
                        c1, c2, c3 = st.columns(3)
                        c1.caption(f"ID: `{chunk['id']}`")
                        c2.caption(f"Type: `{chunk.get('source_type','')}`")
                        c3.caption(f"Has table: `{chunk.get('has_table', False)}`")
            elif chunks == []:
                st.info("No chunks found for this document.")


# ==============================================================================
# PAGE: Traces
# ==============================================================================
elif page == "Traces":
    st.header(":mag: Langfuse Traces - Error Analysis")
    st.markdown(
        """
This page links you to your Langfuse dashboard where all RAG traces are stored.
Each query creates one trace with spans for:

| Span | What it records |
|---|---|
| `retrieval` | Query text, hit count, top similarity scores |
| `reranking` | Candidate count, top cross-encoder score |
| `confidence_gate` | Best similarity vs threshold, pass/fail |
| `llm_generation` | Full prompt, model response |
| `text_extraction` | File format, char count, table detection, warnings |
| `chunking` | Strategy used, chunk count, avg size |
| `embedding` | Model name, chunk count |
| `vector_store` | Collection, number stored |

### Week 5 Module 3 - Error Analysis task
1. Run **20 real queries** from the Query page (mix of on-topic, off-topic, ambiguous, multi-hop)
2. Open each trace in Langfuse and read the full span tree
3. Write one honest sentence per failure noting **what went wrong**
4. Group the notes into ~5 named problem types
5. Rank by frequency x severity -> your ranked taxonomy deliverable
        """
    )

    lf_configured = (
        os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-your") not in ("pk-lf-your", "pk-lf-your-public-key-here", "")
        and os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-your") not in ("sk-lf-your", "sk-lf-your-secret-key-here", "")
    )

    if lf_configured:
        st.success("Langfuse is configured - traces are being captured.")
        st.link_button("Open Langfuse Dashboard", LANGFUSE_HOST, type="primary")
    else:
        st.warning(
            "Langfuse is **not yet configured**. "
            "Add your keys to `.env` to start capturing traces:\n\n"
            "```\nLANGFUSE_PUBLIC_KEY=pk-lf-...\nLANGFUSE_SECRET_KEY=sk-lf-...\n```\n\n"
            f"Sign up free at [{LANGFUSE_HOST}]({LANGFUSE_HOST})"
        )

    last = st.session_state.get("last_result")
    if last and last.get("trace_id"):
        st.subheader("Last Query Trace")
        st.json({
            "question": last.get("question"),
            "answer_preview": (last.get("answer") or "")[:200] + "..." if last.get("answer") else None,
            "confidence": last.get("confidence"),
            "reason": last.get("reason"),
            "trace_id": last.get("trace_id"),
            "trace_url": last.get("trace_url"),
        })
        if last.get("trace_url"):
            st.link_button("View this trace", last["trace_url"])
