"""
FastAPI application to serve the Albanian Legal RAG system.

This API provides endpoints to:
- Query the RAG system.
- Get the current status of the RAG system.
- Reset the conversation memory.
"""

import sys
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.juristi.core.rag_engine import AlbanianLegalRAG

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 15
    mode: str = "hybrid" 
    rag_mode: str = "traditional"

class DocumentResult(BaseModel):
    id: str
    title: str
    content: str
    source: str
    similarity_score: float

class SearchResponse(BaseModel):
    answer: str
    sources: list[DocumentResult]
    total_sources: int
    processing_time: float
    query: str

class AnalyseRequest(BaseModel):
    query: str

class AnalyseResponse(BaseModel):
    answer: str
    sources: list
    total_sources: int
    processing_time: float
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    error: str | None = None

class StatusResponse(BaseModel):
    documents_loaded: bool
    total_documents: int
    vectorstore_initialized: bool
    chain_ready: bool
    embedding_provider: str
    embedding_model: str
    llm_model: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Juristi AI API",
    description="API for the Albanian Legal RAG System",
    version="1.0.0"
)

# --- CORS Middleware ---
# Allow all origins for development purposes.
# For production, you should restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- RAG Engine Initialization ---
# The RAG engine is initialized once when the API starts.
# It runs in UI-only mode, as it relies on a pre-built vector database.
try:
    rag_engine = AlbanianLegalRAG(ui_only=True, verbose=True)
except Exception as e:
    rag_engine = None
    print(f"FATAL: Could not initialize RAG Engine: {e}")

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the Juristi AI API"}

@app.post("/search", response_model=SearchResponse, tags=["RAG"])
async def search_documents(request: SearchRequest):
    """
    Search for legal documents with precise mode (like Streamlit precise mode).
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        import time
        start_time = time.time()
        
        # Use the same query function as Streamlit with precise mode
        response = rag_engine.query(
            question=request.query,
            session_state={},  # Empty session state for API
            query_mode="precise"
        )
        
        # Debug: Print response structure
        print(f"DEBUG: Response type: {type(response)}")
        print(f"DEBUG: Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
        
        # Check for errors in response
        if isinstance(response, dict) and response.get('error'):
            raise HTTPException(status_code=500, detail=f"RAG Engine Error: {response['error']}")
        
        # Format response to match the expected structure
        sources = []
        if isinstance(response, dict) and response.get('sources'):
            print(f"DEBUG: Sources count: {len(response['sources'])}")
            print(f"DEBUG: First source type: {type(response['sources'][0]) if response['sources'] else 'No sources'}")
            
            try:
                for i, source in enumerate(response['sources']):
                    print(f"DEBUG: Processing source {i}: {type(source)}")
                    
                    # Handle different source formats
                    if hasattr(source, 'metadata') and hasattr(source, 'page_content'):
                        # LangChain Document format
                        sources.append(DocumentResult(
                            id=f"source_{i}",
                            title=source.metadata.get('source', 'Unknown Document'),
                            content=source.page_content[:500],  # Limit content length
                            score=0.0,
                            metadata=source.metadata
                        ))
                    elif isinstance(source, dict):
                        # Dictionary format
                        sources.append(DocumentResult(
                            id=f"source_{i}",
                            title=source.get('metadata', {}).get('source', 'Unknown Document'),
                            content=source.get('page_content', source.get('content', ''))[:500],
                            score=source.get('score', 0.0),
                            metadata=source.get('metadata', {})
                        ))
                    else:
                        print(f"DEBUG: Unknown source format: {type(source)}")
                        
            except Exception as source_error:
                print(f"DEBUG: Error processing sources: {source_error}")
                sources = []  # Continue without sources if there's an error
        
        processing_time = time.time() - start_time
        
        # Get answer from response
        answer = "No answer generated"
        if isinstance(response, dict):
            answer = response.get('answer', response.get('result', 'No answer generated'))
        elif isinstance(response, str):
            answer = response
            
        print(f"DEBUG: Final answer length: {len(answer) if answer else 0}")
        print(f"DEBUG: Final sources count: {len(sources)}")
        
        return SearchResponse(
            answer=answer,
            sources=sources,
            total_sources=len(sources),
            processing_time=processing_time,
            query=request.query
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

@app.post("/analyse", response_model=AnalyseResponse, tags=["RAG"])
async def analyse_query(request: AnalyseRequest):
    """
    Analyze a legal query with AI-generated comprehensive response (like Streamlit analyzed mode).
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        import time
        start_time = time.time()
        
        # Use the same query function as Streamlit with analyzed mode
        response = rag_engine.query(
            question=request.query,
            session_state={},  # Empty session state for API
            query_mode="analyzed"
        )
        
        # Debug: Print response structure
        print(f"DEBUG (Analyse): Response type: {type(response)}")
        print(f"DEBUG (Analyse): Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
        
        # Check for errors in response
        if isinstance(response, dict) and response.get('error'):
            raise HTTPException(status_code=500, detail=f"RAG Engine Error: {response['error']}")
        
        # Format response to match the expected structure
        sources = []
        if isinstance(response, dict) and response.get('sources'):
            print(f"DEBUG (Analyse): Sources count: {len(response['sources'])}")
            
            try:
                for i, source in enumerate(response['sources']):
                    # Handle different source formats
                    if hasattr(source, 'metadata') and hasattr(source, 'page_content'):
                        # LangChain Document format
                        sources.append(DocumentResult(
                            id=f"source_{i}",
                            title=source.metadata.get('source', 'Unknown Document'),
                            content=source.page_content[:500],  # Limit content length
                            score=0.0,
                            metadata=source.metadata
                        ))
                    elif isinstance(source, dict):
                        # Dictionary format
                        sources.append(DocumentResult(
                            id=f"source_{i}",
                            title=source.get('metadata', {}).get('source', 'Unknown Document'),
                            content=source.get('page_content', source.get('content', ''))[:500],
                            score=source.get('score', 0.0),
                            metadata=source.get('metadata', {})
                        ))
                    else:
                        print(f"DEBUG (Analyse): Unknown source format: {type(source)}")
                        
            except Exception as source_error:
                print(f"DEBUG (Analyse): Error processing sources: {source_error}")
                sources = []  # Continue without sources if there's an error
        
        analysis_time = time.time() - start_time
        
        # Get answer from response
        answer = "No answer generated"
        if isinstance(response, dict):
            answer = response.get('answer', response.get('result', 'No answer generated'))
        elif isinstance(response, str):
            answer = response
            
        print(f"DEBUG (Analyse): Final answer length: {len(answer) if answer else 0}")
        print(f"DEBUG (Analyse): Final sources count: {len(sources)}")
        
        return AnalyseResponse(
            answer=answer,
            sources=sources,
            total_sources=len(sources),
            processing_time=analysis_time,
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Receives a question and returns the answer from the RAG system.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")
    
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_engine.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")

@app.get("/status", response_model=StatusResponse, tags=["RAG"])
async def get_status():
    """
    Returns the current status of the RAG system.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")
    
    return rag_engine.get_system_status()

@app.post("/reset-memory", tags=["RAG"])
async def reset_memory():
    """
    Resets the conversation memory of the RAG engine.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")
    
    rag_engine.reset_memory()
    return {"message": "Conversation memory has been reset."}

# To run this API, use the command:
# uvicorn src.juristi.api.main:app --reload
