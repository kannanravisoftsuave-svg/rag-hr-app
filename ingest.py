"""
Ingest HR policy documents into two Chroma collections using two different
chunking granularities, so we can compare retrieval quality between them:

  - "subsection": one chunk per ### heading (small, precise)
  - "section":    one chunk per ## heading, merging all its ### children (larger, more context)

Usage: python ingest.py
"""
import re
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DB_DIR = "chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
SUBSECTION_RE = re.compile(r"^###\s+(.*)$", re.MULTILINE)


def split_by_heading(text, pattern):
    """Split text into (heading, body) pairs at each heading matched by pattern."""
    matches = list(pattern.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(1).strip()
        body = text[start:end].strip()
        chunks.append((heading, body))
    return chunks


def chunk_subsection(text, source):
    """One chunk per ### heading. Falls back to ## headings if a document has none
    (e.g. the addendum only uses ## headings) so no content is silently dropped."""
    chunks = []
    subsections = split_by_heading(text, SUBSECTION_RE)
    if not subsections:
        subsections = split_by_heading(text, SECTION_RE)
    for heading, body in subsections:
        chunks.append({"text": body, "heading": heading, "source": source})
    return chunks


def chunk_section(text, source):
    """One chunk per ## heading, keeping all its ### children merged together."""
    chunks = []
    for heading, body in split_by_heading(text, SECTION_RE):
        chunks.append({"text": body, "heading": heading, "source": source})
    return chunks


def build_collection(client, name, chunk_fn):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name)

    print(f"\n=== Building collection '{name}' ===")
    model = get_model()
    all_chunks = []
    for path in sorted(glob.glob(f"{DATA_DIR}/*.md")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = path.replace("\\", "/").split("/")[-1]
        chunks = chunk_fn(text, source)
        all_chunks.extend(chunks)
        print(f"  {source}: {len(chunks)} chunks")

    if not all_chunks:
        print("  No chunks produced — check your source files.")
        return

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    ids = [f"{name}-{i}" for i in range(len(all_chunks))]
    metadatas = [{"source": c["source"], "heading": c["heading"]} for c in all_chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"  Total chunks stored: {len(all_chunks)}")
    sizes = [len(t) for t in texts]
    print(f"  Chunk size (chars): min={min(sizes)} max={max(sizes)} avg={sum(sizes)//len(sizes)}")


_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model '{EMBED_MODEL_NAME}' (first run downloads it)...")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


if __name__ == "__main__":
    client = chromadb.PersistentClient(path=DB_DIR)
    build_collection(client, "hr_policy_subsection", chunk_subsection)
    build_collection(client, "hr_policy_section", chunk_section)
    print("\nIngestion complete.")
