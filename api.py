"""
FastAPI wrapper for the Albanian Legal RAG System
Exposes REST endpoints reusing the existing RAG engine in app.py
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Import the RAG system
from app import CloudEnhancedAlbanianLegalRAG

app = FastAPI(title="Albanian Legal RAG API", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class Document(BaseModel):
    title: str
    content: str
    similarity: Optional[float] = None
    source: Optional[str] = None
    url: Optional[str] = None
    document_type: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    documents: List[Document]
    response: str


# Single global instance (simple for now)
rag = CloudEnhancedAlbanianLegalRAG()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embeddings_ready": rag.document_embeddings is not None,
        "documents": len(rag.legal_documents),
        "provider": "google" if getattr(rag, 'use_google_embeddings', False) else "local",
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    top_k = req.top_k or 3
    docs = rag.search_documents(req.query, top_k=top_k)

    # Build documents list
    api_docs: List[Document] = []
    for d in docs:
        api_docs.append(Document(
            title=d.get('title', ''),
            content=d.get('content', ''),
            similarity=float(d.get('similarity', 0.0)),
            source=d.get('source', None),
            url=d.get('url', None),
            document_type=d.get('document_type', None)
        ))

    # Generate RAG response
    answer = rag.generate_response(req.query, docs)

    return SearchResponse(
        query=req.query,
        top_k=top_k,
        documents=api_docs,
        response=answer
    )


# Simple root
@app.get("/")
def root():
    return {"message": "Albanian Legal RAG API. Use POST /search with { query }"}
