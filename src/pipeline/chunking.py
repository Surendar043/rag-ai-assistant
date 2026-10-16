"""Document chunking with overlap."""

from typing import List, Dict
from src.config.settings import settings


def chunk_text(text: str, source: str = "") -> List[Dict]:
    """Split text into overlapping chunks with metadata."""
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i : i + settings.CHUNK_SIZE]
        chunk = " ".join(chunk_words)
        chunks.append(
            {
                "text": chunk,
                "source": source,
                "chunk_id": idx,
                "start_word": i,
                "end_word": min(i + settings.CHUNK_SIZE, len(words)),
            }
        )
        idx += 1
        i += settings.CHUNK_SIZE - settings.CHUNK_OVERLAP
    return chunks


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """Chunk a list of {text, source} documents."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(
            chunk_text(doc.get("text", ""), doc.get("source", ""))
        )
    return all_chunks
