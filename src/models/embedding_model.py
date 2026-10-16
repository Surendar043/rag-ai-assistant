"""Embedding model wrapper.

Uses sentence-transformers by default. Falls back to a deterministic
hash-based embedding if the model cannot be loaded (offline / no internet).
"""

from typing import List
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dim = 384
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            # Determine actual dimension from a test encoding
            test = self.model.encode(["test"], convert_to_numpy=True)
            self.dim = test.shape[1]
        except Exception:
            self.model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts into a 2D numpy array of shape (n, dim)."""
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True)
        # Deterministic fallback: hashed bag-of-words embedding
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                idx = abs(hash(word)) % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.vstack(embeddings)
