"""
Chunking strategies for ingested documents.

- heading:        split at ## / ### markdown headings (best for MD; fallback to
                  sliding window if no headings found)
- page:           one chunk per page (best for PDFs)
- sliding_window: fixed-size overlapping windows (fallback / DOCX default)
- auto:           picks the best strategy based on source_type
"""
from __future__ import annotations
import re
import hashlib

SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
SUBSECTION_RE = re.compile(r"^###\s+(.*)$", re.MULTILINE)

SLIDE_CHUNK_CHARS = 400
SLIDE_OVERLAP_CHARS = 80


def chunk_document(extracted: dict, source_metadata: dict, strategy: str = "auto") -> list[dict]:
    """
    Chunk an extracted document.

    Args:
        extracted:       output of ingestion.extractor.extract()
        source_metadata: base metadata dict merged into every chunk
        strategy:        "auto" | "heading" | "page" | "sliding_window"

    Returns:
        List of chunk dicts each containing at minimum:
          text, heading, source, source_file, source_type, chunk_index
    """
    src_type = source_metadata.get("source_type", "")

    if strategy == "auto":
        if src_type == "md":
            strategy = "heading"
        else:
            # sliding_window for both PDF and DOCX — page-level chunks are too
            # coarse when a page mixes unrelated content (org charts, bar charts,
            # tables) which dilutes embedding similarity for focused queries.
            strategy = "sliding_window"

    if strategy == "heading":
        chunks = _chunk_by_heading(extracted["text"], source_metadata)
        if not chunks:
            chunks = _chunk_sliding_window(extracted["text"], source_metadata)
        return chunks

    if strategy == "page":
        return _chunk_by_page(extracted["pages"], source_metadata)

    if strategy == "sliding_window":
        return _chunk_sliding_window(extracted["text"], source_metadata)

    raise ValueError(f"Unknown chunking strategy: '{strategy}'")


# ── Strategies ────────────────────────────────────────────────────────────────

def _chunk_by_heading(text: str, base_meta: dict) -> list[dict]:
    """Split at ### headings; fall back to ## if no ### exist."""
    pairs = _split_by_pattern(text, SUBSECTION_RE)
    if not pairs:
        pairs = _split_by_pattern(text, SECTION_RE)
    return [_make_chunk(body, heading, base_meta, i) for i, (heading, body) in enumerate(pairs) if body.strip()]


def _chunk_by_page(pages: list[dict], base_meta: dict) -> list[dict]:
    """One chunk per page; apply sliding window if a page is very long."""
    chunks = []
    idx = 0
    for page in pages:
        page_text = page.get("text", "")
        for tbl in page.get("tables", []):
            page_text = page_text + "\n\n" + tbl if page_text else tbl
        page_text = page_text.strip()
        if not page_text:
            continue
        heading = f"Page {page['page_num']}"
        if len(page_text) <= SLIDE_CHUNK_CHARS * 2:
            chunks.append(_make_chunk(page_text, heading, base_meta, idx))
            idx += 1
        else:
            # Long page → sliding window sub-chunks
            sub = _sliding_window_text(page_text)
            for j, piece in enumerate(sub):
                chunks.append(_make_chunk(piece, f"{heading} (part {j+1})", base_meta, idx))
                idx += 1
    return chunks


def _chunk_sliding_window(text: str, base_meta: dict) -> list[dict]:
    parts = _sliding_window_text(text)
    return [_make_chunk(p, f"Chunk {i+1}", base_meta, i) for i, p in enumerate(parts) if p.strip()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_by_pattern(text: str, pattern) -> list[tuple[str, str]]:
    matches = list(pattern.finditer(text))
    result = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(1).strip()
        body = text[start:end].strip()
        result.append((heading, body))
    return result


def _sliding_window_text(text: str, chunk_size: int = SLIDE_CHUNK_CHARS, overlap: int = SLIDE_OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping windows.

    Prefers sentence/line boundaries but always hard-splits at chunk_size so
    punctuation-free text (org charts, extracted tables) still gets chunked.
    """
    # Split on sentence endings OR newlines so org-chart / table text splits too
    units = re.split(r"(?<=[.!?\n])\s*|\n", text)
    units = [u.strip() for u in units if u.strip()]

    chunks, current, current_len = [], [], 0
    for unit in units:
        # Hard-split units that are longer than chunk_size themselves
        while len(unit) > chunk_size:
            remaining_space = chunk_size - current_len
            if remaining_space > 0:
                current.append(unit[:remaining_space])
                current_len += remaining_space
                unit = unit[remaining_space:]
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            tail = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
            current = [tail] if tail else []
            current_len = len(tail)

        if current_len + len(unit) > chunk_size and current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            tail = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
            current = [tail] if tail else []
            current_len = len(tail)

        current.append(unit)
        current_len += len(unit)

    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def _make_chunk(text: str, heading: str, base_meta: dict, chunk_index: int) -> dict:
    uid = hashlib.md5(
        f"{base_meta.get('source_file','')}{heading}{text[:80]}".encode()
    ).hexdigest()[:12]
    return {
        "text": text,
        "heading": heading,
        "chunk_index": chunk_index,
        "chunk_id": uid,
        # alias so existing query.py build_prompt keeps working
        "source": base_meta.get("source_file", ""),
        **base_meta,
    }
