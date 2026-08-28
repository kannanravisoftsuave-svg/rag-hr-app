"""
Langfuse v4 tracing wrapper.

v4 uses OpenTelemetry context propagation — the API is context-manager-based,
not the old lf.trace() / trace.span() pattern from v2/v3.

Usage:
    from tracing import lf_span, lf_generation, lf_set_output, lf_get_ids, lf_flush

    with lf_span("rag_query", input={"question": q}) as root:
        with lf_span("retrieval", span_type="retriever") as s:
            hits = do_retrieval()
            s.update(output={"count": len(hits)})

        with lf_generation("llm", model="gpt-4o") as g:
            answer = call_llm(prompt)
            g.update(output=answer)

        lf_set_output({"answer": answer})
        trace_id, trace_url = lf_get_ids()

All functions degrade gracefully to no-ops when Langfuse is not configured.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

_langfuse = None
_init_attempted = False


def get_langfuse():
    global _langfuse, _init_attempted
    if _init_attempted:
        return _langfuse
    _init_attempted = True

    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return None
    if "your" in LANGFUSE_PUBLIC_KEY or "your" in LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        print(f"[Langfuse] tracing enabled → {LANGFUSE_HOST}")
    except Exception as exc:
        print(f"[Langfuse] init failed: {exc}")
        _langfuse = None
    return _langfuse


def langfuse_enabled() -> bool:
    return get_langfuse() is not None


# ── Span context managers ─────────────────────────────────────────────────────

def lf_span(name: str, span_type: str = "span", input=None, metadata=None):
    """
    Context manager that creates a Langfuse span (or trace root when at top level).
    Returns a no-op context when Langfuse is not configured.
    """
    lf = get_langfuse()
    if lf is None:
        return nullcontext(_NoopObs())
    try:
        return lf.start_as_current_observation(
            name=name,
            as_type=span_type,
            input=input,
            metadata=metadata,
        )
    except Exception:
        return nullcontext(_NoopObs())


def lf_generation(name: str, model: str = "", input=None):
    """Context manager for an LLM generation span."""
    lf = get_langfuse()
    if lf is None:
        return nullcontext(_NoopObs())
    try:
        return lf.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input,
        )
    except Exception:
        return nullcontext(_NoopObs())


# ── Trace-level helpers ───────────────────────────────────────────────────────

def lf_set_input(input) -> None:
    lf = get_langfuse()
    if lf:
        try:
            lf.set_current_trace_io(input=input)
        except Exception:
            pass


def lf_set_output(output) -> None:
    lf = get_langfuse()
    if lf:
        try:
            lf.set_current_trace_io(output=output)
        except Exception:
            pass


def lf_get_ids() -> tuple[str | None, str | None]:
    """Returns (trace_id, trace_url) for the current active trace."""
    lf = get_langfuse()
    if lf is None:
        return None, None
    try:
        tid = lf.get_current_trace_id()
        url = lf.get_trace_url(trace_id=tid) if tid else None
        return tid, url
    except Exception:
        return None, None


def lf_flush() -> None:
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass


# ── No-op stub ────────────────────────────────────────────────────────────────

class _NoopObs:
    """Returned inside nullcontext when Langfuse is disabled."""
    def update(self, *a, **kw): pass
    def score(self, *a, **kw): pass
