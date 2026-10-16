"""Question-answering endpoint powered by the LangGraph agent."""

from fastapi import APIRouter, HTTPException
from src.api.schemas.models import QueryRequest, QueryResponse
from src.agent.graph import run_agent
from src.pipeline.retrieval import Retriever
from src.pipeline.vector_store import VectorStore
from src.models.embedding_model import EmbeddingModel
from src.models.llm import LLM
from src.pipeline.generation import Generator
from config.settings import settings

router = APIRouter()

embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL)
vs = VectorStore(embedding_model)
try:
    vs.load(settings.VECTOR_STORE_PATH)
except Exception:
    vs = None

llm = LLM(settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
retriever = Retriever(vs) if vs else None
generator = Generator(llm)


@router.post("", response_model=QueryResponse)
def ask(request: QueryRequest):
    """Ask a question; the agent retrieves context and generates an answer."""
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Use POST /ingest first.",
        )
    answer = run_agent(request.question, retriever, generator)
    return QueryResponse(answer=answer)
