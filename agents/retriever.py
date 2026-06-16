"""
Hybrid retriever — combines BM25 keyword search + ChromaDB vector search,
then reranks results with Cohere for maximum relevance.

This is the key differentiator from naive RAG:
  - BM25 catches exact keyword matches (equipment IDs, part numbers)
  - Vector search catches semantic similarity
  - Reranking re-scores the combined pool by actual relevance to the query
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from agents.models import RetrievedChunk

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")


class HybridRetriever:
    def __init__(self):
        self._collection = None
        self._bm25 = None
        self._all_docs = None
        self._all_ids = None
        self._all_metadata = None
        self._cohere_client = None

    def _ensure_loaded(self):
        if self._collection is not None:
            return

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef = embedding_functions.DefaultEmbeddingFunction()

        self._collection = client.get_collection(
            name="industrial_knowledge_base",
            embedding_function=ef,
        )

        # Load all docs for BM25 index
        all_data = self._collection.get(include=["documents", "metadatas"])
        self._all_docs = all_data["documents"]
        self._all_ids = all_data["ids"]
        self._all_metadata = all_data["metadatas"]

        # Build BM25 index
        tokenized = [doc.lower().split() for doc in self._all_docs]
        self._bm25 = BM25Okapi(tokenized)

    def _get_cohere_client(self):
        if self._cohere_client is None:
            api_key = os.getenv("COHERE_API_KEY", "")
            if api_key:
                import cohere
                self._cohere_client = cohere.Client(api_key)
        return self._cohere_client

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """
        1. BM25 keyword search → top 5 candidates
        2. Vector semantic search → top 5 candidates
        3. Merge and deduplicate
        4. Rerank with Cohere (if API key available) else score-based merge
        5. Return top_k results
        """
        self._ensure_loaded()

        # ── BM25 search ──────────────────────────────────────────────────
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:5]
        bm25_candidates = {
            self._all_ids[i]: {
                "doc": self._all_docs[i],
                "meta": self._all_metadata[i],
                "bm25_score": float(bm25_scores[i]),
                "vector_score": 0.0,
            }
            for i in bm25_top_indices
        }

        # ── Vector search ────────────────────────────────────────────────
        vector_results = self._collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "distances", "metadatas"],
        )
        for doc, dist, meta, doc_id in zip(
            vector_results["documents"][0],
            vector_results["distances"][0],
            vector_results["metadatas"][0],
            vector_results["ids"][0],
        ):
            score = max(0.0, 1.0 - dist / 2.0)
            if doc_id in bm25_candidates:
                bm25_candidates[doc_id]["vector_score"] = score
            else:
                bm25_candidates[doc_id] = {
                    "doc": doc,
                    "meta": meta,
                    "bm25_score": 0.0,
                    "vector_score": score,
                }

        # ── Merge: combined score ────────────────────────────────────────
        candidates = list(bm25_candidates.items())

        # Normalise BM25 scores to 0-1
        max_bm25 = max((v["bm25_score"] for _, v in candidates), default=1.0) or 1.0
        for _, v in candidates:
            v["bm25_score_norm"] = v["bm25_score"] / max_bm25

        for _, v in candidates:
            v["combined_score"] = 0.5 * v["bm25_score_norm"] + 0.5 * v["vector_score"]

        candidates.sort(key=lambda x: x[1]["combined_score"], reverse=True)
        top_candidates = candidates[:min(6, len(candidates))]

        # ── Cohere reranking ─────────────────────────────────────────────
        cohere_client = self._get_cohere_client()
        if cohere_client and len(top_candidates) > 1:
            try:
                docs_to_rerank = [v["doc"] for _, v in top_candidates]
                rerank_response = cohere_client.rerank(
                    query=query,
                    documents=docs_to_rerank,
                    top_n=top_k,
                    model="rerank-english-v3.0",
                )
                print(f"[Retriever] Cohere reranking applied — top result score: {rerank_response.results[0].relevance_score:.3f}")
                reranked = []
                for result in rerank_response.results:
                    doc_id, v = top_candidates[result.index]
                    reranked.append(RetrievedChunk(
                        doc_id=doc_id,
                        content=v["doc"],
                        score=round(result.relevance_score, 3),
                        category=v["meta"].get("category", ""),
                        equipment=v["meta"].get("equipment", ""),
                        location=v["meta"].get("location", ""),
                    ))
                return reranked[:top_k]
            except Exception as e:
                print(f"[Retriever] Cohere reranking failed ({e}), using combined score")

        # Fallback: return by combined score
        return [
            RetrievedChunk(
                doc_id=doc_id,
                content=v["doc"],
                score=round(v["combined_score"], 3),
                category=v["meta"].get("category", ""),
                equipment=v["meta"].get("equipment", ""),
                location=v["meta"].get("location", ""),
            )
            for doc_id, v in top_candidates[:top_k]
        ]


# Singleton
retriever = HybridRetriever()
