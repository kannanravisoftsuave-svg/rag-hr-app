"""
Inspect what's actually stored in a Qdrant collection: the chunk text,
its metadata, and a peek at its embedding vector.

Usage: python inspect_db.py [collection_name] [n]
"""
import sys
from vectorstore import get_store

collection_name = sys.argv[1] if len(sys.argv) > 1 else "hr_policy_subsection"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

store = get_store()
client = store.client

print(f"Collection: {collection_name}")
print(f"Total chunks stored: {store.count(collection_name)}\n")

points = client.scroll(collection_name=collection_name, limit=n, with_vectors=True)[0]

for i, point in enumerate(points):
    payload = point.payload
    print(f"--- Chunk {i+1} ---")
    print(f"id: {point.id}")
    print(f"metadata: {{'source': {payload['source']!r}, 'heading': {payload['heading']!r}}}")
    print(f"text: {payload['text'][:150]}...")
    vec = point.vector
    print(f"embedding: length={len(vec)}, first 8 values={[round(v, 4) for v in vec[:8]]}")
    print()
