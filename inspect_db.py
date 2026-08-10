"""
Inspect what's actually stored in a Chroma collection: the chunk text,
its metadata, and a peek at its embedding vector.

Usage: python inspect_db.py [collection_name] [n]
"""
import sys
import chromadb

DB_DIR = "chroma_db"

collection_name = sys.argv[1] if len(sys.argv) > 1 else "hr_policy_subsection"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(collection_name)

print(f"Collection: {collection_name}")
print(f"Total chunks stored: {collection.count()}\n")

# include=... controls what comes back; embeddings are included so we can peek at one
result = collection.get(limit=n, include=["documents", "metadatas", "embeddings"])

for i in range(len(result["ids"])):
    print(f"--- Chunk {i+1} ---")
    print(f"id: {result['ids'][i]}")
    print(f"metadata: {result['metadatas'][i]}")
    print(f"text: {result['documents'][i][:150]}...")
    vec = result["embeddings"][i]
    print(f"embedding: length={len(vec)}, first 8 values={[round(v, 4) for v in vec[:8]]}")
    print()
