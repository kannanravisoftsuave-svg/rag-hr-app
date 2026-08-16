"""
Hybrid retriever: BM25 keyword search + dense vector search, fused with
Reciprocal Rank Fusion (RRF). Catches exact keyword matches (policy terms,
codes, names) that pure semantic search misses when query and document
don't share the same vocabulary.
"""
import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60  # lower = stronger boost to top-ranked results from each retriever


class HybridRetriever:
    def __init__(self, store, collection_name, embed_model, query_prefix=""):
        self.store = store
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.query_prefix = query_prefix

        print(f"Building BM25 index for '{collection_name}'...")
        self.all_points = store.get_all(collection_name)  # sorted by id, id == list index
        self.id_to_point = {p["id"]: p for p in self.all_points}
        tokenized = [p["text"].lower().split() for p in self.all_points]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, question, top_k=8):
        # Dense retrieval
        query_vec = self.embed_model.encode(
            [self.query_prefix + question], normalize_embeddings=True
        ).tolist()[0]
        dense_hits = self.store.query(self.collection_name, query_vec, top_k)

        # BM25 retrieval
        bm25_scores = self.bm25.get_scores(question.lower().split())
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()

        # RRF fusion — accumulate 1/(k+rank) from each retriever
        rrf: dict[int, float] = {}
        for rank, hit in enumerate(dense_hits):
            pid = hit["id"]
            rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_top_indices):
            pid = self.all_points[idx]["id"]
            rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)

        dense_by_id = {h["id"]: h for h in dense_hits}
        top_ids = sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k]

        results = []
        for pid in top_ids:
            if pid in dense_by_id:
                hit = {**dense_by_id[pid], "rrf_score": rrf[pid]}
            else:
                # BM25-only hit — no dense similarity score available
                hit = {**self.id_to_point[pid], "similarity": 0.0, "rrf_score": rrf[pid]}
            results.append(hit)
        return results
