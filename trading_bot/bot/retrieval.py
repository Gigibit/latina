from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextChunk:
    text: str
    score: float


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
        return [ContextChunk(text=corpus[i], score=float(scores[i])) for i in ranked_indices]
