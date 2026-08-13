"""
Thin wrapper over Qdrant so ingest.py/query.py/eval.py share one interface
for storing and querying chunks. Talks to a Qdrant server (e.g. running in
Docker via `docker run ... qdrant/qdrant`) over HTTP, at QDRANT_URL.
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBED_DIM = 384  # BAAI/bge-small-en-v1.5 output size
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


class QdrantStore:
    def __init__(self, url=QDRANT_URL):
        self.client = QdrantClient(url=url)

    def build_collection(self, name, texts, embeddings, metadatas):
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

    def query(self, name, query_embedding, top_k):
        result = self.client.query_points(collection_name=name, query=query_embedding, limit=top_k)
        hits = []
        for point in result.points:
            payload = point.payload
            # Qdrant reports cosine similarity directly (higher = more similar).
            hits.append({"text": payload["text"], "source": payload["source"], "heading": payload["heading"], "similarity": point.score})
        return hits

    def count(self, name):
        return self.client.count(collection_name=name).count


def get_store(url=QDRANT_URL):
    return QdrantStore(url=url)
