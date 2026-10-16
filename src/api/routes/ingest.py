"""Document ingestion endpoint."""

from fastapi import APIRouter, HTTPException
from src.api.schemas.models import IngestRequest, IngestResponse
from src.pipeline.chunking import chunk_documents
from src.pipeline.vector_store import VectorStore
from src.models.embedding_model import EmbeddingModel
from config.settings import settings

router = APIRouter()

# Shared components (wire via dependency injection in production)
embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL)
vs = VectorStore(embedding_model)


@router.post("", response_model=IngestResponse)
def ingest_documents(request: IngestRequest):
    """Upload and index documents for retrieval."""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    chunks = chunk_documents(request.documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from documents")

    vs.add_chunks(chunks)
    vs.save()

    return IngestResponse(chunks_indexed=len(chunks))
