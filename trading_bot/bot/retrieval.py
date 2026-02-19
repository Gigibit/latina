from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextChunk:
    text: str
    score: float
    index: int | None = None


class EmbeddingRetriever:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install dependencies from requirements.txt"
            ) from exc

        self.model = SentenceTransformer(model_name)

    def top_k(self, query: str, corpus: list[str], k: int = 3) -> list[ContextChunk]:
        import numpy as np

        if not corpus:
            return []

        embeddings = self.model.encode(corpus, normalize_embeddings=True)
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(embeddings, query_vec)

        ranked_indices = np.argsort(scores)[::-1][:k]
        return [
            ContextChunk(text=corpus[i], score=float(scores[i]), index=int(i))
            for i in ranked_indices
        ]


class FaissEmbeddingRetriever(EmbeddingRetriever):
    def top_k(self, query: str, corpus: list[str], k: int = 3) -> list[ContextChunk]:
        import numpy as np

        if not corpus:
            return []

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is not installed. Install dependencies from requirements.txt"
            ) from exc

        embeddings = self.model.encode(corpus, normalize_embeddings=True)
        query_vec = self.model.encode([query], normalize_embeddings=True)

        embeddings = np.asarray(embeddings, dtype="float32")
        query_vec = np.asarray(query_vec, dtype="float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        top_n = min(k, len(corpus))
        similarities, indices = index.search(query_vec, top_n)

        return [
            ContextChunk(text=corpus[int(doc_idx)], score=float(score), index=int(doc_idx))
            for score, doc_idx in zip(similarities[0], indices[0], strict=False)
        ]
