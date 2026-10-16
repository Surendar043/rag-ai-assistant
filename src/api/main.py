"""FastAPI application entrypoint."""

from fastapi import FastAPI
from src.api.routes import ingest, query

app = FastAPI(title="RAG AI Assistant", version="1.0.0")

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/ask", tags=["query"])


@app.get("/health")
def health():
    return {"status": "ok"}
