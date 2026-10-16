"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Dict


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class Document(BaseModel):
    text: str
    source: str = ""


class IngestRequest(BaseModel):
    documents: List[Document]


class IngestResponse(BaseModel):
    chunks_indexed: int
