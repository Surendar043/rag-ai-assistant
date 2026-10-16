"""FAISS-based vector store for chunk embeddings."""

import os
import json
from typing import List, Dict, Optional
import numpy as np
import faiss
from src.config.settings import settings


class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []

    def add_chunks(self, chunks: List[Dict]):
        """Embed chunks and add them to the index."""
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.embed(texts).astype("float32")
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Return top-k most similar chunks with similarity scores."""
        if self.index is None or self.index.ntotal == 0:
            return []
        top_k = top_k or settings.TOP_K
        q_emb = self.embedding_model.embed([query]).astype("float32")
        distances, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            score = 1.0 / (1.0 + float(dist))
            if score < settings.SIMILARITY_THRESHOLD:
                continue
            result = dict(self.metadata[idx])
            result["score"] = score
            results.append(result)
        return results

    def save(self, path: Optional[str] = None):
        path = path or settings.VECTOR_STORE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, path)
        with open(path + ".meta.json", "w") as f:
            json.dump(self.metadata, f)

    def load(self, path: Optional[str] = None):
        path = path or settings.VECTOR_STORE_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")
        self.index = faiss.read_index(path)
        with open(path + ".meta.json") as f:
            self.metadata = json.load(f)
