"""
FastAPI-based REST API for Albanian Legal RAG System

This module provides a RESTful API interface extracted and refactored 
from the original api.py, with proper modular structure.
"""

import os
import sys
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to Python path if not already there
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import core components
try:
    from juristi.core.rag_engine import AlbanianLegalRAG
    from juristi.core.llm_client import CloudLLMClient
except ImportError:
    # Fallback for relative imports
    from ..core.rag_engine import AlbanianLegalRAG
    from ..core.llm_client import CloudLLMClient


# Pydantic models for API requests/responses
class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="The legal question or query", min_length=1)
    top_k: int = Field(3, description="Number of documents to return", ge=1, le=20)
    mode: str = Field("hybrid", description="Search mode: hybrid, embedding, or sparse")
    rag_mode: str = Field("traditional", description="RAG mode: traditional, hierarchical, or sentence_window")
    multi_query: bool = Field(True, description="Whether to use query expansion")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")


class SearchResponse(BaseModel):
    """Response model for document search."""
    success: bool = Field(..., description="Whether the search was successful")
    query: str = Field(..., description="The original query")
    results: List[Dict[str, Any]] = Field(..., description="List of search results")
    total_results: int = Field(..., description="Total number of results")
    search_time: float = Field(..., description="Search time in seconds")
    timestamp: str = Field(..., description="Response timestamp")
    session_id: Optional[str] = Field(None, description="Session ID if provided")
    error: Optional[str] = Field(None, description="Error message if any")


class AIRequest(BaseModel):
    """Request model for AI-generated responses."""
    query: str = Field(..., description="The legal question", min_length=1)
    context: Optional[str] = Field(None, description="Optional context for the AI")
    provider: str = Field("gemini", description="AI provider: gemini, groq, or huggingface")
    session_id: Optional[str] = Field(None, description="Optional session ID")


class AIResponse(BaseModel):
    """Response model for AI-generated responses."""
    success: bool = Field(..., description="Whether the request was successful")
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="AI-generated response")
    provider: str = Field(..., description="AI provider used")
    response_time: float = Field(..., description="Response generation time")
    timestamp: str = Field(..., description="Response timestamp")
    session_id: Optional[str] = Field(None, description="Session ID if provided")
    error: Optional[str] = Field(None, description="Error message if any")


class SystemStatus(BaseModel):
    """System status response model."""
    status: str = Field(..., description="System status")
    documents_loaded: int = Field(..., description="Number of documents loaded")
    embeddings_ready: bool = Field(..., description="Whether embeddings are ready")
    rag_mode: str = Field(..., description="Current RAG mode")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")


class AlbanianLegalAPI:
    """FastAPI-based REST API for Albanian Legal RAG System."""
    
    def __init__(self):
        """Initialize the API with core systems."""
        self.app = FastAPI(
            title="Albanian Legal RAG API",
            description="REST API for Albanian Legal Research Assistant",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize core systems
        self.rag_system = None
        self.llm_client = None
        self.startup_time = time.time()
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize systems on startup."""
            await self._initialize_systems()
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """API root endpoint."""
            return {
                "message": "Albanian Legal RAG API",
                "version": "2.0.0",
                "status": "active",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/status", response_model=SystemStatus)
        async def get_status():
            """Get system status and statistics."""
            try:
                documents_count = 0
                embeddings_ready = False
                rag_mode = "unknown"
                
                if self.rag_system:
                    if hasattr(self.rag_system, 'legal_documents') and self.rag_system.legal_documents:
                        documents_count = len(self.rag_system.legal_documents)
                    
                    embeddings_ready = (
                        hasattr(self.rag_system, 'document_embeddings') and
                        self.rag_system.document_embeddings is not None
                    )
                    
                    rag_mode = getattr(self.rag_system, 'rag_mode', 'traditional')
                
                uptime = time.time() - self.startup_time
                
                return SystemStatus(
                    status="operational",
                    documents_loaded=documents_count,
                    embeddings_ready=embeddings_ready,
                    rag_mode=rag_mode,
                    uptime=uptime,
                    version="2.0.0"
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")
        
        @self.app.post("/search", response_model=SearchResponse)
        async def search_documents(request: SearchRequest):
            """Search legal documents."""
            start_time = time.time()
            
            try:
                # Ensure systems are initialized
                if not self.rag_system:
                    await self._initialize_systems()
                
                # Configure RAG system
                if self.rag_system.rag_mode != request.rag_mode:
                    self.rag_system.rag_mode = request.rag_mode
                
                # Perform search
                results = self.rag_system.search_documents(
                    query=request.query,
                    top_k=request.top_k,
                    mode=request.mode,
                    session_id=request.session_id,
                    multi_query=request.multi_query
                )
                
                search_time = time.time() - start_time
                
                return SearchResponse(
                    success=True,
                    query=request.query,
                    results=results,
                    total_results=len(results),
                    search_time=search_time,
                    timestamp=datetime.now().isoformat(),
                    session_id=request.session_id
                )
                
            except Exception as e:
                search_time = time.time() - start_time
                return SearchResponse(
                    success=False,
                    query=request.query,
                    results=[],
                    total_results=0,
                    search_time=search_time,
                    timestamp=datetime.now().isoformat(),
                    session_id=request.session_id,
                    error=str(e)
                )
        
        @self.app.post("/ai-response", response_model=AIResponse)
        async def generate_ai_response(request: AIRequest):
            """Generate AI response for legal questions."""
            start_time = time.time()
            
            try:
                # Ensure systems are initialized
                if not self.llm_client:
                    await self._initialize_systems()
                
                # Configure LLM provider if different
                if self.llm_client.current_provider != request.provider:
                    self.llm_client.current_provider = request.provider
                
                # Generate response
                response = self.llm_client.get_legal_response(
                    user_query=request.query,
                    context=request.context or ""
                )
                
                response_time = time.time() - start_time
                
                return AIResponse(
                    success=True,
                    query=request.query,
                    response=response,
                    provider=request.provider,
                    response_time=response_time,
                    timestamp=datetime.now().isoformat(),
                    session_id=request.session_id
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                return AIResponse(
                    success=False,
                    query=request.query,
                    response="",
                    provider=request.provider,
                    response_time=response_time,
                    timestamp=datetime.now().isoformat(),
                    session_id=request.session_id,
                    error=str(e)
                )
        
        @self.app.post("/combined-search", response_model=Dict[str, Any])
        async def combined_search(request: SearchRequest, include_ai: bool = Query(False)):
            """Combined search with optional AI response."""
            try:
                # Perform document search
                search_response = await search_documents(request)
                
                result = {
                    "search": search_response.dict(),
                    "ai_response": None
                }
                
                # Generate AI response if requested and search was successful
                if include_ai and search_response.success and search_response.results:
                    # Prepare context from search results
                    context_parts = []
                    for doc in search_response.results[:3]:  # Top 3 results
                        context_parts.append(f"""
Dokumenti: {doc.get('title', 'Pa titull')}
Burimi: {doc.get('source', 'Pa burim')}  
Përmbajtja: {doc.get('content', '')[:1000]}...
                        """)
                    
                    context = "\n".join(context_parts)
                    
                    ai_request = AIRequest(
                        query=request.query,
                        context=context,
                        provider="gemini",  # Default provider
                        session_id=request.session_id
                    )
                    
                    ai_response = await generate_ai_response(ai_request)
                    result["ai_response"] = ai_response.dict()
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error in combined search: {str(e)}")
        
        @self.app.get("/documents/count")
        async def get_document_count():
            """Get total number of loaded documents."""
            try:
                if not self.rag_system:
                    await self._initialize_systems()
                
                count = 0
                if hasattr(self.rag_system, 'legal_documents') and self.rag_system.legal_documents:
                    count = len(self.rag_system.legal_documents)
                
                return {"total_documents": count}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error getting document count: {str(e)}")
        
        @self.app.post("/reload-documents")
        async def reload_documents():
            """Reload all documents (admin endpoint)."""
            try:
                if self.rag_system:
                    # Force reload documents
                    self.rag_system._documents_loaded = False
                    self.rag_system.legal_documents = []
                    self.rag_system.document_embeddings = None
                    
                    # Trigger reload on next search
                    return {"message": "Documents marked for reload", "status": "success"}
                else:
                    return {"message": "RAG system not initialized", "status": "warning"}
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reloading documents: {str(e)}")
    
    async def _initialize_systems(self):
        """Initialize RAG and LLM systems asynchronously."""
        try:
            if not self.rag_system:
                # Initialize with quick start for better API response times
                self.rag_system = AlbanianLegalRAG(
                    quick_start=True,
                    rag_mode='traditional'
                )
            
            if not self.llm_client:
                self.llm_client = CloudLLMClient()
                
        except Exception as e:
            print(f"Error initializing systems: {e}")
            raise
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


# Create global API instance
api_instance = AlbanianLegalAPI()
app = api_instance.get_app()


def create_app() -> FastAPI:
    """Factory function to create FastAPI app."""
    return api_instance.get_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
