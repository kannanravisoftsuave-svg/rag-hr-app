"""
Thin wrapper over Qdrant.  Extended in Week 5 to support:
  - metadata filtering on queries (passed as a simple dict)
  - listing collections with stats
  - listing unique documents inside a collection
  - fetching all chunks for a specific document
"""
from __future__ import annotations
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, VectorParams,
    Filter, FieldCondition, MatchValue,
)

EMBED_DIM = 384
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


class QdrantStore:
    def __init__(self, url: str = QDRANT_URL):
        self.client = QdrantClient(url=url)

    # ── Write ─────────────────────────────────────────────────────────────────

    def build_collection(self, name: str, texts: list, embeddings: list, metadatas: list) -> None:
        """Rebuild a collection from scratch (used by legacy ingest.py)."""
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=embeddings[i], payload={**metadatas[i], "text": texts[i]})
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=name, points=points)

    # ── Read / Query ──────────────────────────────────────────────────────────

    def query(self, name: str, query_embedding: list, top_k: int, metadata_filter: dict | None = None) -> list[dict]:
        """Vector search with optional metadata pre-filter."""
        qdrant_filter = _build_filter(metadata_filter) if metadata_filter else None
        result = self.client.query_points(
            collection_name=name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [{**p.payload, "similarity": p.score, "id": p.id} for p in result.points]

    def count(self, name: str) -> int:
        return self.client.count(collection_name=name).count

    def get_all(self, name: str) -> list[dict]:
        """Fetch every point (for BM25 index building). Returns list sorted by id."""
        all_records, offset = [], None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=name, limit=256, offset=offset,
                with_payload=True, with_vectors=False,
            )
            all_records.extend(records)
            if next_offset is None:
                break
            offset = next_offset
        all_records.sort(key=lambda p: p.id)
        return [{**p.payload, "id": p.id} for p in all_records]

    # ── Collection-level helpers ───────────────────────────────────────────────

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def get_collection_info(self, name: str) -> dict:
        info = self.client.get_collection(name)
        return {
            "name": name,
            "vector_count": info.vectors_count,
            "points_count": info.points_count,
        }

    # ── Document-level helpers ────────────────────────────────────────────────

    def get_documents(self, collection: str) -> list[dict]:
        """Return one record per unique source_file in a collection, with chunk_count."""
        all_points = self.get_all(collection)
        docs: dict[str, dict] = {}
        for pt in all_points:
            src = pt.get("source_file") or pt.get("source", "unknown")
            if src not in docs:
                docs[src] = {
                    "filename": src,
                    "source_type": pt.get("source_type", ""),
                    "upload_timestamp": pt.get("upload_timestamp", ""),
                    "collection": pt.get("collection", collection),
                    "has_table": pt.get("has_table", False),
                    "chunk_count": 0,
                }
            docs[src]["chunk_count"] += 1
        return list(docs.values())

    def get_chunks_for_document(self, collection: str, source_file: str) -> list[dict]:
        """Return all chunks that belong to a specific source file."""
        all_records, offset = [], None
        filt = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))])
        while True:
            records, next_offset = self.client.scroll(
                collection_name=collection,
                limit=256, offset=offset,
                with_payload=True, with_vectors=False,
                scroll_filter=filt,
            )
            all_records.extend(records)
            if next_offset is None:
                break
            offset = next_offset
        return [{**p.payload, "id": p.id} for p in all_records]


# ── Filter builder ────────────────────────────────────────────────────────────

def _build_filter(metadata_filter: dict) -> Filter | None:
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in metadata_filter.items()
        if v is not None
    ]
    return Filter(must=conditions) if conditions else None


def get_store(url: str = QDRANT_URL) -> QdrantStore:
    return QdrantStore(url=url)
