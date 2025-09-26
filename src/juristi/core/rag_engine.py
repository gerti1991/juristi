"""
Modern Albanian Legal RAG Engine using LangChain with Multiple Embedding Providers

This module provides a modern RAG implementation supporting:
- Google Generative AI embeddings (embedding-001) - Priority 1
- BGE Small English embeddings for better Albanian language understanding - Priority 2  
- Sentence Transformers embeddings (all-MiniLM-L6-v2) - Priority 3 (fallback)
- Google Gemini 2.5 Flash for efficient LLM responses
- LangChain for advanced retrieval with MMR
- ChromaDB for vector storage
- Optimized for Albanian/English legal documents
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

# LangChain imports with updated community imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Multiple embedding providers with error handling
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HUGGINGFACE_AVAILABLE = True
    except ImportError:
        HUGGINGFACE_AVAILABLE = False
        HuggingFaceEmbeddings = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Environment and utilities
from dotenv import load_dotenv
import streamlit as st

# Load environment variables (force reload to get latest .env changes)
load_dotenv(override=True)

# Setup logging
logger = logging.getLogger(__name__)


class DimensionNormalizedEmbeddings:
    """
    Wrapper class to normalize all embeddings to 384 dimensions for ChromaDB compatibility.
    Uses PCA for dimensionality reduction when needed.
    """
    
    def __init__(self, base_embeddings, target_dimension: int = 384):
        self.base_embeddings = base_embeddings
        self.target_dimension = target_dimension
        self._pca_reducer = None
        self._needs_reduction = None
        
    def _check_dimension_and_setup_reducer(self, embeddings):
        """Check if we need dimension reduction and set up PCA if needed."""
        if self._needs_reduction is None:
            original_dim = len(embeddings[0]) if isinstance(embeddings[0], list) else embeddings.shape[1]
            self._needs_reduction = original_dim > self.target_dimension
            
            if self._needs_reduction:
                try:
                    from sklearn.decomposition import PCA
                    import numpy as np
                    
                    # Initialize PCA reducer
                    self._pca_reducer = PCA(n_components=self.target_dimension, random_state=42)
                    
                    # Fit on current embeddings
                    embeddings_array = np.array(embeddings)
                    self._pca_reducer.fit(embeddings_array)
                    
                    logger.info(f"🔧 PCA reducer initialized: {original_dim} → {self.target_dimension} dimensions")
                    
                except ImportError:
                    logger.warning("⚠️ scikit-learn not available for PCA - using truncation instead")
                    self._pca_reducer = None
                    
        return self._needs_reduction
    
    def _normalize_embeddings(self, embeddings):
        """Normalize embeddings to target dimension."""
        import numpy as np
        embeddings_array = np.array(embeddings)
        
        if not self._check_dimension_and_setup_reducer(embeddings_array):
            # No reduction needed
            return embeddings_array.tolist()
        
        # Apply dimension reduction
        if self._pca_reducer is not None:
            # Use PCA for intelligent dimension reduction
            reduced = self._pca_reducer.transform(embeddings_array)
            return reduced.tolist()
        else:
            # Fallback: simple truncation
            return embeddings_array[:, :self.target_dimension].tolist()
    
    def embed_documents(self, texts):
        """Embed documents with dimension normalization."""
        embeddings = self.base_embeddings.embed_documents(texts)
        return self._normalize_embeddings(embeddings)
    
    def embed_query(self, text):
        """Embed query with dimension normalization."""
        embedding = self.base_embeddings.embed_query(text)
        if isinstance(embedding, list) and len(embedding) > self.target_dimension:
            if self._pca_reducer is not None:
                import numpy as np
                # For single queries, we need to reshape for PCA
                embedding_array = np.array([embedding])
                reduced = self._pca_reducer.transform(embedding_array)
                return reduced[0].tolist()
            else:
                # Simple truncation
                return embedding[:self.target_dimension]
        return embedding


class AlbanianLegalRAG:
    """
    Modern Albanian Legal RAG System using LangChain with Multiple Embedding Providers.
    
    Features:
    - Priority-based embedding providers: Google -> BGE -> SentenceTransformers
    - Google Gemini 2.5 Flash LLM for efficient responses
    - MMR retrieval for better chunk diversity
    - ChromaDB for vector storage
    - Session state management
    - Memory for conversation history
    """
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 verbose: bool = True,
                 ui_only: bool = False):
        """
        Initialize the modern RAG system with multiple embedding providers.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            verbose: Enable verbose logging
            ui_only: If True, only load existing ChromaDB without embedding initialization
        """
        self.persist_directory = persist_directory
        self.verbose = verbose
        self.ui_only = ui_only
        
        # Document tracking for incremental loading
        self.document_index_file = os.path.join(persist_directory, "document_index.json")
        self.processed_documents = self._load_document_index()
        
        # Configuration from environment
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()
        self.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.google_embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        self.bge_embedding_model = os.getenv("BGE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.fallback_embedding_model = os.getenv("FALLBACK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.llm_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.device = os.getenv("DEVICE", "auto")
        
        # Initialize components based on mode
        if self.ui_only:
            # UI-only mode: Initialize embeddings for search compatibility, load existing vectorstore and LLM
            self._initialize_embeddings()  # Need embeddings for search to work properly
            self._initialize_llm()
            self._initialize_memory() 
            self._try_load_existing_vectorstore_ui_only()
        else:
            # Full mode: Initialize everything including embeddings
            self._initialize_embeddings()
            self._initialize_llm()
            self._initialize_text_splitter()
            self._initialize_vectorstore()
            self._initialize_memory()
            self._initialize_chain()
        
        # Document tracking
        self.documents_loaded = False
        self.total_documents = 0
        
        if self.verbose:
            mode = "UI-only" if self.ui_only else "Full"
            logger.info(f"✅ Modern Albanian Legal RAG System initialized ({mode} mode)")
    
    def _try_load_existing_vectorstore_ui_only(self):
        """Load existing ChromaDB for UI-only mode, using the correctly initialized embeddings."""
        try:
            if not os.path.exists(self.persist_directory):
                if self.verbose:
                    logger.warning("⚠️ No existing ChromaDB found - UI-only mode requires pre-built embeddings")
                self.vectorstore = None
                self.qa_chain = None
                return False
            
            # The main embeddings should already be initialized.
            if not self.embeddings:
                logger.error("❌ Embeddings not initialized in UI-only mode. Cannot load vectorstore.")
                return False

            # Load with the correct embedding function
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Check if it has data
            try:
                count = self.vectorstore._collection.count()
                if count > 0:
                    self.total_documents = count
                    self.documents_loaded = True
                    
                    # Initialize QA chain for UI-only mode
                    self._initialize_chain_ui_only()
                    
                    if self.verbose:
                        logger.info(f"✅ Loaded existing vectorstore with {count} documents (UI-only mode)")
                    return True
                else:
                    if self.verbose:
                        logger.warning("⚠️ ChromaDB exists but is empty")
                    return False
            except Exception as count_error:
                if self.verbose:
                    logger.warning(f"⚠️ Could not get document count: {count_error}")
                # Try to initialize anyway, might still work
                self._initialize_chain_ui_only()
                return True
            
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Failed to load existing vectorstore in UI-only mode: {e}")
            self.vectorstore = None
            self.qa_chain = None
            return False
            
        return False
    
    def _initialize_chain_ui_only(self):
        """Initialize QA chain for UI-only mode with existing embeddings."""
        if self.vectorstore is not None and hasattr(self, 'llm'):
            # Create retriever with MMR for better diversity - improved parameters for Albanian legal search
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 15,          # Increased to get more diverse results
                    "fetch_k": 100,   # Fetch many more candidates for MMR selection  
                    "lambda_mult": 0.3  # Even more diversity to avoid similar articles
                }
            )
            
            # Create custom prompt template for Albanian legal questions (UI-only mode)
            custom_prompt_template = """Ti jeni një ekspert i lartë juridik për legjislacionin shqiptar. Jepni përgjigje të sakta dhe profesionale.

**PYETJA:** {question}

**DOKUMENTET LIGJORE:**
{context}

**PËRGJIGJA:** Jepni një përgjigje të drejtpërdrejtë duke cituar nenin dhe ligjin specifik."""

            custom_prompt = PromptTemplate(
                template=custom_prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain with custom prompt
            from langchain.chains import RetrievalQA
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": custom_prompt},
                memory=self.memory,
                return_source_documents=True,
                verbose=self.verbose
            )
            
            if self.verbose:
                logger.info("✅ QA chain initialized (UI-only mode)")
        else:
            self.qa_chain = None
            if self.verbose:
                logger.warning("⚠️ Cannot initialize QA chain - missing vectorstore or LLM")
    
    def _initialize_embeddings(self):
        """Initialize embeddings with configurable priority based on EMBEDDING_PROVIDER env var."""
        self.embeddings = None
        
        # Check if specific provider is requested via environment variable
        preferred_provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()
        if self.verbose:
            logger.info(f"🔧 EMBEDDING_PROVIDER env var: '{os.getenv('EMBEDDING_PROVIDER', 'NOT_SET')}' -> preferred: '{preferred_provider}'")
        
        # Build the priority list dynamically
        all_providers = ["google", "bge", "gemma", "sentence-transformers"]
        
        # Handle gemma as an alias for bge (Google Gemma uses BGE embeddings)
        if preferred_provider == "gemma":
            preferred_provider = "bge"
        
        if preferred_provider in all_providers:
            # Start with the preferred provider
            embedding_providers = [preferred_provider]
            # Add the rest as fallbacks (exclude gemma alias)
            for p in ["google", "bge", "sentence-transformers"]:
                if p != preferred_provider:
                    embedding_providers.append(p)
        else:
            # Default priority order if the preferred one is invalid
            embedding_providers = ["google", "bge", "sentence-transformers"]
        
        for provider in embedding_providers:
            try:
                if provider == "google" and GOOGLE_GENAI_AVAILABLE and self.google_api_key:
                    if self.verbose:
                        logger.info("🔄 Trying Google embeddings (Priority 1)...")
                    
                    # Handle Streamlit event loop issues
                    try:
                        import asyncio
                        # Check if we're in Streamlit environment
                        try:
                            asyncio.get_running_loop()
                            # We're in an async context (likely Streamlit), skip Google for now
                            if self.verbose:
                                logger.warning("⚠️ Skipping Google embeddings in async environment (Streamlit)")
                            raise Exception("Event loop conflict in Streamlit")
                        except RuntimeError:
                            # No event loop running, safe to use Google embeddings
                            pass
                    except:
                        # If any issue with event loop detection, skip Google
                        if self.verbose:
                            logger.warning("⚠️ Skipping Google embeddings due to async environment")
                        continue
                    
                    base_embeddings = GoogleGenerativeAIEmbeddings(
                        model=self.google_embedding_model,
                        google_api_key=self.google_api_key
                    )
                    # Wrap with dimension normalizer to ensure 384 dimensions
                    self.embeddings = DimensionNormalizedEmbeddings(base_embeddings, target_dimension=384)
                    self.active_embedding_provider = "google"
                    if self.verbose:
                        logger.info(f"✅ Google embeddings initialized: {self.google_embedding_model} (normalized to 384 dims)")
                    break
                    
                elif provider == "bge" and HUGGINGFACE_AVAILABLE:
                    if self.verbose:
                        logger.info("🔄 Google failed, trying BGE embeddings (Priority 2)...")
                    
                    base_embeddings = HuggingFaceEmbeddings(
                        model_name=self.bge_embedding_model,
                        model_kwargs={'device': self.device}
                    )
                    # BGE is already 384 dims, but wrap for consistency
                    self.embeddings = DimensionNormalizedEmbeddings(base_embeddings, target_dimension=384)
                    self.active_embedding_provider = "bge"
                    if self.verbose:
                        logger.info(f"✅ BGE embeddings initialized: {self.bge_embedding_model} (384 dims)")
                    break
                    
                elif provider == "sentence-transformers":
                    if self.verbose:
                        logger.info("🔄 Previous providers failed, using fallback embeddings (Priority 3)...")
                    
                    # Use SentenceTransformerEmbeddings from langchain_community
                    from langchain_community.embeddings import SentenceTransformerEmbeddings
                    
                    base_embeddings = SentenceTransformerEmbeddings(
                        model_name=self.fallback_embedding_model
                    )
                    # all-MiniLM-L6-v2 is already 384 dims, but wrap for consistency
                    self.embeddings = DimensionNormalizedEmbeddings(base_embeddings, target_dimension=384)
                    self.active_embedding_provider = "sentence-transformers"
                    if self.verbose:
                        logger.info(f"✅ Fallback embeddings initialized: {self.fallback_embedding_model} (384 dims)")
                    break
                    
            except Exception as e:
                if self.verbose:
                    logger.warning(f"⚠️ Failed to initialize {provider} embeddings: {e}")
                continue
        
        if self.embeddings is None:
            raise ValueError("❌ All embedding providers failed. Please check your configuration.")
        
        if self.verbose:
            logger.info(f"🎯 Active embedding provider: {self.active_embedding_provider}")
    
    def _initialize_llm(self):
        """Initialize Google Gemini LLM with fallback."""
        try:
            if GOOGLE_GENAI_AVAILABLE and self.google_api_key:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.llm_model,
                    google_api_key=self.google_api_key,
                    temperature=0.2,
                    verbose=self.verbose,
                    convert_system_message_to_human=True  # For better Albanian language handling
                )
                if self.verbose:
                    logger.info(f"✅ Google Gemini LLM initialized: {self.llm_model}")
            else:
                # Fallback to a different LLM if needed
                raise ValueError("Google Gemini LLM not available")
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️ Failed to initialize Google Gemini LLM: {e}")
            # Could add fallback LLM here (e.g., local model)
            raise ValueError("❌ No LLM provider available. Please check GOOGLE_API_KEY.")
    
    def _initialize_text_splitter(self):
        """Initialize text splitter optimized for Albanian legal documents."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            # Prioritize splitting on legal articles ("Neni"), then sections, then paragraphs
            separators=[
                "\nNeni ", "\nKREU ", "\nPJESA ", "\nSEKSIONI ",  # Albanian legal structure
                "\n\n", "\n", ". ", "; ", ", ", " ", ""
            ]
        )
        if self.verbose:
            logger.info("✅ Text splitter initialized for legal documents")
    
    def _initialize_vectorstore(self):
        """Initialize vectorstore - will be created when documents are loaded."""
        self.vectorstore = None
        if self.verbose:
            logger.info("📝 ChromaDB will be created when documents are loaded")
    
    def _initialize_memory(self):
        """Initialize conversation memory."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="result"
        )
        if self.verbose:
            logger.info("✅ Conversation memory initialized")
    
    def _load_document_index(self) -> Dict[str, Dict]:
        """Load document index for tracking processed documents."""
        if os.path.exists(self.document_index_file):
            try:
                with open(self.document_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose:
                    logger.warning(f"⚠️ Could not load document index: {e}")
                return {}
        return {}
    
    def rebuild_document_index(self, documents_path: str = "legal_documents/pdfs"):
        """
        Rebuild document index from existing PDF files.
        This is useful when the index was lost but documents were already processed.
        """
        try:
            pdf_dir = Path(documents_path)
            if not pdf_dir.exists():
                if self.verbose:
                    logger.error(f"❌ Document directory not found: {pdf_dir}")
                return False
            
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                if self.verbose:
                    logger.warning(f"⚠️ No PDF files found in: {pdf_dir}")
                return False
                
            if self.verbose:
                logger.info(f"🔧 Rebuilding document index from {len(pdf_files)} PDF files")
                
            # Clear existing index
            self.processed_documents = {}
            
            # Process each PDF and add to index
            for pdf_file in pdf_files:
                doc_id = self._get_document_id(pdf_file)
                
                # Get page count from PDF (quick check without full processing)
                try:
                    import PyPDF2
                    with open(pdf_file, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        page_count = len(pdf_reader.pages)
                except:
                    try:
                        import fitz  # PyMuPDF
                        with fitz.open(pdf_file) as pdf_doc:
                            page_count = pdf_doc.page_count
                    except:
                        page_count = 1  # Default if can't read
                
                # Add to processed documents
                self.processed_documents[doc_id] = {
                    'file_path': str(pdf_file),
                    'processed_at': datetime.now().isoformat(),
                    'chunk_count': page_count,  # Approximate
                    'embedding_provider': self.active_embedding_provider or 'bge'
                }
                
                if self.verbose:
                    logger.info(f"✅ Added to index: {pdf_file.name} ({page_count} chunks)")
            
            # Save the rebuilt index
            self._save_document_index()
            
            if self.verbose:
                logger.info(f"🎉 Successfully rebuilt document index with {len(self.processed_documents)} documents")
            return True
            
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Failed to rebuild document index: {e}")
            return False

    def _save_document_index(self):
        """Save document index to disk."""
        try:
            os.makedirs(os.path.dirname(self.document_index_file), exist_ok=True)
            with open(self.document_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_documents, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️ Could not save document index: {e}")
    
    def _get_document_id(self, file_path: Path) -> str:
        """Generate unique document ID based on file path and modification time."""
        stat = file_path.stat()
        return f"{file_path.name}_{int(stat.st_mtime)}_{stat.st_size}"
    
    def _is_document_processed(self, file_path: Path) -> bool:
        """Check if document has already been processed."""
        doc_id = self._get_document_id(file_path)
        return doc_id in self.processed_documents
    
    def _mark_document_processed(self, file_path: Path, chunk_count: int = 0):
        """Mark document as processed in the index."""
        doc_id = self._get_document_id(file_path)
        self.processed_documents[doc_id] = {
            'file_path': str(file_path),
            'processed_at': datetime.now().isoformat(),
            'chunk_count': chunk_count,
            'embedding_provider': self.active_embedding_provider
        }
        self._save_document_index()

    def _reset_document_index(self):
        """Reset document index when vectorstore is cleared or corrupted."""
        self.processed_documents = {}
        self._save_document_index()
        if self.verbose:
            logger.info("🗑️ Document index reset - all documents will be re-processed")

    def _initialize_chain(self):
        """Initialize the QA chain."""
        if self.vectorstore is not None:
            # Configure MMR retriever as requested - increased parameters for better Albanian legal search
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 15,          # Number of documents to return (increased)
                    "fetch_k": 100,   # Number of documents to fetch before MMR (increased)
                    "lambda_mult": 0.3  # Even more diversity for better coverage
                }
            )
            
            # Create custom prompt template for Albanian legal questions
            custom_prompt_template = """Ti jeni një ekspert i lartë juridik për legjislacionin shqiptar. Jepni përgjigje të sakta dhe profesionale.

**PYETJA:** {question}

**DOKUMENTET LIGJORE:**
{context}

**PËRGJIGJA:** Jepni një përgjigje të drejtpërdrejtë duke cituar nenin dhe ligjin specifik."""

            custom_prompt = PromptTemplate(
                template=custom_prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": custom_prompt},
                memory=self.memory,
                return_source_documents=True,
                verbose=self.verbose
            )
            
            if self.verbose:
                logger.info("✅ QA chain initialized with MMR retrieval")
        else:
            self.qa_chain = None
            if self.verbose:
                logger.info("📝 QA chain will be initialized when vectorstore is ready")
    
    def _check_dimension_compatibility(self) -> bool:
        """Check if existing ChromaDB dimensions match current embedding provider"""
        try:
            if not os.path.exists(self.persist_directory):
                if self.verbose:
                    logger.info("📁 No existing ChromaDB found")
                # Reset document index for fresh start
                self.processed_documents = {}
                self._save_document_index()
                return True  # No existing DB, so compatible
            
            # Check if there's any collection data first
            collection_dir = Path(self.persist_directory)
            chroma_db_file = collection_dir / "chroma.sqlite3"
            
            if not chroma_db_file.exists():
                if self.verbose:
                    logger.info("📊 ChromaDB directory exists but no database file found")
                # Reset document index for fresh start
                self.processed_documents = {}
                self._save_document_index()
                return True
            
            # Try to load existing vectorstore with a simple compatibility check
            temp_vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Try to get collection info to check dimensions
            collection = temp_vectorstore._collection
            if collection.count() == 0:
                if self.verbose:
                    logger.info("📊 ChromaDB exists but is empty")
                return True
            
            # Get embedding dimension from current provider
            test_embedding = self.embeddings.embed_query("test")
            current_dim = len(test_embedding)
            
            # Try a test query to see if dimensions match
            test_results = temp_vectorstore.similarity_search("test", k=1)
            if self.verbose:
                logger.info(f"✅ ChromaDB compatible with {current_dim}-dim {self.active_embedding_provider} embeddings")
            return True
                        
        except Exception as e:
            error_str = str(e)
            if "dimension" in error_str.lower():
                if self.verbose:
                    logger.warning(f"⚠️ Dimension mismatch detected: {error_str}")
                    logger.warning(f"⚠️ ChromaDB incompatible with {self.active_embedding_provider} embeddings")
                return False
            else:
                if self.verbose:
                    logger.warning(f"⚠️ Could not check ChromaDB compatibility: {e}")
                return True
    
    def _clear_incompatible_vectorstore(self):
        """Clear ChromaDB if it's incompatible with current embedding provider"""
        try:
            import shutil
            import time
            import gc
            
            # Close any existing vectorstore connections
            if hasattr(self, 'vectorstore') and self.vectorstore is not None:
                try:
                    # Try to close the connection if possible
                    if hasattr(self.vectorstore, '_client'):
                        self.vectorstore._client = None
                    if hasattr(self.vectorstore, '_collection'):
                        self.vectorstore._collection = None
                    self.vectorstore = None
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"⚠️ Could not properly close vectorstore: {e}")
            
            # Force garbage collection to release any remaining references
            gc.collect()
            time.sleep(0.5)  # Give time for cleanup
            
            if os.path.exists(self.persist_directory):
                if self.verbose:
                    logger.info(f"🗑️ Clearing incompatible ChromaDB directory: {self.persist_directory}")
                
                # Try multiple times with increasing delays
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        shutil.rmtree(self.persist_directory)
                        break
                    except PermissionError as e:
                        if attempt < max_attempts - 1:
                            if self.verbose:
                                logger.warning(f"⚠️ Attempt {attempt + 1} failed, retrying in {(attempt + 1) * 2}s...")
                            time.sleep((attempt + 1) * 2)
                        else:
                            # Last attempt failed, try alternative method
                            if self.verbose:
                                logger.warning("⚠️ Using alternative deletion method...")
                            self._force_delete_directory(self.persist_directory)
                
                # DON'T reset document index since documents were already processed successfully
                # The issue was vector dimension mismatch, not document processing
                if self.verbose:
                    logger.info("✅ ChromaDB cleared - keeping document index since documents were processed successfully")
            return True
            
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Failed to clear ChromaDB: {e}")
            return False
    
    def _force_delete_directory(self, directory_path: str):
        """Force delete directory using system commands as fallback"""
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                # Use Windows rmdir with force flag
                subprocess.run(['rmdir', '/S', '/Q', directory_path], 
                             shell=True, check=False, capture_output=True)
            else:
                # Use Unix rm command
                subprocess.run(['rm', '-rf', directory_path], 
                             check=False, capture_output=True)
                             
            if self.verbose:
                logger.info(f"🔨 Force deleted directory: {directory_path}")
                
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Force deletion also failed: {e}")
                logger.info("💡 Manual deletion may be required - please delete chroma_db directory manually")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        # Count total documents if vectorstore exists
        total_docs = 0
        if hasattr(self, 'vectorstore') and self.vectorstore is not None:
            try:
                # Get document count from vectorstore
                collection = self.vectorstore._collection
                total_docs = collection.count()
            except:
                total_docs = 0
        
        self.total_documents = total_docs
        self.documents_loaded = total_docs > 0
        
        # Get the current embedding model name based on active provider
        current_embedding_model = "unknown"
        if hasattr(self, 'active_embedding_provider'):
            if self.active_embedding_provider == "google":
                current_embedding_model = self.google_embedding_model
            elif self.active_embedding_provider == "bge":
                current_embedding_model = self.bge_embedding_model
            elif self.active_embedding_provider == "sentence-transformers":
                current_embedding_model = self.fallback_embedding_model
        
        return {
            'documents_loaded': self.documents_loaded,
            'total_documents': self.total_documents,
            'vectorstore_initialized': self.vectorstore is not None,
            'chain_ready': self.qa_chain is not None,
            'embedding_provider': getattr(self, 'active_embedding_provider', 'unknown'),
            'embedding_model': current_embedding_model,
            'llm_model': self.llm_model
        }
    
    def reset_memory(self):
        """Reset conversation memory."""
        self.memory.clear()
        if self.verbose:
            logger.info("🔄 Conversation memory reset")
    
    def load_documents_from_directory(self, directory_path: str) -> bool:
        """
        Load documents from a directory and build the vector database.
        Handles API quota limits with batch processing.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.verbose:
                logger.info(f"📂 Loading documents from: {directory_path}")
            
            # Check if we can load existing vectorstore first
            if self._try_load_existing_vectorstore():
                # Check if there are new documents to add
                new_documents = []
                
                # Load from PDF directory if it exists
                pdf_dir = Path(directory_path) / "pdfs"
                if pdf_dir.exists():
                    new_documents.extend(self._load_pdf_documents(pdf_dir))
                
                # Load from processed JSON files if they exist
                processed_dir = Path(directory_path) / "processed"
                if processed_dir.exists():
                    new_documents.extend(self._load_processed_documents(processed_dir))
                
                # If we have new documents, add them to existing vectorstore
                if new_documents:
                    if self.verbose:
                        logger.info(f"📝 Adding {len(new_documents)} new documents to existing vectorstore...")
                    
                    # Split new documents into chunks
                    new_chunks = self.text_splitter.split_documents(new_documents)
                    
                    if self.verbose:
                        logger.info(f"📊 Created {len(new_chunks)} new document chunks")
                    
                    # Add to existing vectorstore
                    return self._add_chunks_to_vectorstore(new_chunks)
                else:
                    if self.verbose:
                        logger.info("✅ All documents already processed - no new documents to add")
                    return True
            
            # No existing vectorstore - load all documents
            documents = []
            
            # Load from PDF directory if it exists
            pdf_dir = Path(directory_path) / "pdfs"
            if pdf_dir.exists():
                documents.extend(self._load_pdf_documents(pdf_dir))
            
            # Load from processed JSON files if they exist
            processed_dir = Path(directory_path) / "processed"
            if processed_dir.exists():
                documents.extend(self._load_processed_documents(processed_dir))
            
            if not documents:
                logger.warning(f"⚠️ No documents found in {directory_path}")
                return False
            
            # Split documents into chunks
            if self.verbose:
                logger.info(f"📝 Splitting {len(documents)} documents into chunks...")
            
            text_chunks = self.text_splitter.split_documents(documents)
            
            if self.verbose:
                logger.info(f"📊 Created {len(text_chunks)} document chunks")
            
            # Create vectorstore with quota handling
            success = self._create_vectorstore_with_quota_handling(text_chunks)
            
            if success:
                self.documents_loaded = True
                self.total_documents = len(text_chunks)
                if self.verbose:
                    logger.info(f"✅ Successfully loaded {len(text_chunks)} document chunks")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
            if self.verbose:
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _try_load_existing_vectorstore(self) -> bool:
        """Try to load existing ChromaDB vectorstore with dimension compatibility check."""
        try:
            if Path(self.persist_directory).exists():
                if self.verbose:
                    logger.info("🔍 Found existing ChromaDB, checking compatibility...")
                
                # Check if existing vectorstore is compatible with current embedding dimensions
                if not self._check_dimension_compatibility():
                    if self.verbose:
                        logger.warning(f"⚠️ ChromaDB incompatible with {self.active_embedding_provider} embeddings")
                        logger.info("🔄 Clearing incompatible vectorstore...")
                    
                    if not self._clear_incompatible_vectorstore():
                        return False
                    
                    if self.verbose:
                        logger.info("✅ Ready to create fresh vectorstore with correct dimensions")
                    return False  # Need to create new vectorstore
                
                # Compatible - load existing vectorstore
                if self.verbose:
                    logger.info(f"✅ ChromaDB compatible with {self.active_embedding_provider} embeddings")
                
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                
                # Check if it has data
                if hasattr(self.vectorstore._collection, 'count'):
                    count = self.vectorstore._collection.count()
                    if count > 0:
                        if self.verbose:
                            logger.info(f"✅ Loaded existing vectorstore with {count} documents")
                        
                        self._initialize_chain()
                        self.documents_loaded = True
                        self.total_documents = count
                        return True
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️ Could not load existing vectorstore: {e}")
        else:
            # No existing ChromaDB directory found - reset document index
            if self.verbose:
                logger.info("📁 No existing ChromaDB found - starting fresh")
            self._reset_document_index()
        
        return False
    
    def _create_vectorstore_with_quota_handling(self, text_chunks: List[Document]) -> bool:
        """Create vectorstore with API quota limit handling and automatic fallback."""
        
        # Try current embedding provider first
        success = self._try_create_vectorstore(text_chunks)
        
        if not success and self.active_embedding_provider == "google":
            if self.verbose:
                logger.warning("⚠️ Google embeddings failed (likely quota exceeded)")
                logger.info("🔄 Falling back to BGE embeddings (Priority 2)...")
            
            # Clear any existing ChromaDB that might have incompatible dimensions
            self._clear_incompatible_vectorstore()
            
            # Try to switch to BGE embeddings
            if self._switch_to_fallback_embeddings("bge"):
                success = self._try_create_vectorstore(text_chunks)
            
            # If BGE also fails, try sentence-transformers
            if not success:
                if self.verbose:
                    logger.warning("⚠️ BGE embeddings also failed")
                    logger.info("🔄 Falling back to SentenceTransformers (Priority 3)...")
                
                # Clear ChromaDB again for sentence transformers
                self._clear_incompatible_vectorstore()
                
                if self._switch_to_fallback_embeddings("sentence-transformers"):
                    success = self._try_create_vectorstore(text_chunks)
        
        return success
    
    def _switch_to_fallback_embeddings(self, provider: str) -> bool:
        """Switch to a fallback embedding provider."""
        try:
            if provider == "bge" and HUGGINGFACE_AVAILABLE:
                base_embeddings = HuggingFaceEmbeddings(
                    model_name=self.bge_embedding_model,
                    model_kwargs={'device': self.device}
                )
                # Wrap with dimension normalizer
                self.embeddings = DimensionNormalizedEmbeddings(base_embeddings, target_dimension=384)
                self.active_embedding_provider = "bge"
                if self.verbose:
                    logger.info(f"✅ Switched to BGE embeddings: {self.bge_embedding_model} (384 dims)")
                return True
                
            elif provider == "sentence-transformers":
                from langchain_community.embeddings import SentenceTransformerEmbeddings
                base_embeddings = SentenceTransformerEmbeddings(
                    model_name=self.fallback_embedding_model
                )
                # Wrap with dimension normalizer  
                self.embeddings = DimensionNormalizedEmbeddings(base_embeddings, target_dimension=384)
                self.active_embedding_provider = "sentence-transformers"
                if self.verbose:
                    logger.info(f"✅ Switched to SentenceTransformers: {self.fallback_embedding_model} (384 dims)")
                return True
                
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Failed to switch to {provider}: {e}")
        
        return False
    
    def _try_create_vectorstore(self, text_chunks: List[Document]) -> bool:
        """Try to create vectorstore with current embedding provider."""
        try:
            # For Google embeddings, use small batches to avoid quota limits
            if self.active_embedding_provider == "google":
                batch_size = 10  # Small batch size for Google API
                delay_between_batches = 10  # 10 seconds delay
            else:
                # For local models, we can use larger batches
                batch_size = 50
                delay_between_batches = 1
            
            if len(text_chunks) > batch_size:
                if self.verbose:
                    logger.info(f"📊 Processing {len(text_chunks)} chunks in batches of {batch_size}")
                    if self.active_embedding_provider == "google":
                        logger.info(f"⏱️ Using delays due to Google API rate limits...")
                
                # Process first batch to create vectorstore
                first_batch = text_chunks[:batch_size]
                if self.verbose:
                    logger.info(f"📝 Processing batch 1/{(len(text_chunks) + batch_size - 1) // batch_size}")
                
                self.vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                
                # Process remaining batches
                remaining_chunks = text_chunks[batch_size:]
                
                for i in range(0, len(remaining_chunks), batch_size):
                    batch_num = (i // batch_size) + 2
                    total_batches = (len(text_chunks) + batch_size - 1) // batch_size
                    
                    if self.verbose:
                        logger.info(f"⏳ Waiting {delay_between_batches}s before next batch...")
                    
                    time.sleep(delay_between_batches)
                    
                    batch = remaining_chunks[i:i + batch_size]
                    if self.verbose:
                        logger.info(f"📝 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    
                    try:
                        self.vectorstore.add_documents(batch)
                    except Exception as e:
                        if "429" in str(e) or "quota" in str(e).lower():
                            if self.verbose:
                                logger.warning(f"⚠️ Hit rate limit, increasing delay to {delay_between_batches * 2}s")
                            delay_between_batches *= 2
                            time.sleep(delay_between_batches)
                            self.vectorstore.add_documents(batch)
                        else:
                            raise e
            else:
                # Small number of chunks, process all at once
                self.vectorstore = Chroma.from_documents(
                    documents=text_chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            
            # Initialize chain now that we have documents
            self._initialize_chain()
            
            return True
            
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Error creating vectorstore with {self.active_embedding_provider}: {e}")
            return False
    
    def _add_chunks_to_vectorstore(self, text_chunks: List[Document]) -> bool:
        """Add new document chunks to existing vectorstore."""
        if not self.vectorstore:
            if self.verbose:
                logger.error("❌ No existing vectorstore to add documents to")
            return False
        
        try:
            # Use similar batch processing as create method
            batch_size = 50 if self.active_embedding_provider == "bge" else 10
            delay_between_batches = 1 if self.active_embedding_provider != "google" else 3
            
            if len(text_chunks) > batch_size:
                if self.verbose:
                    logger.info(f"📊 Adding {len(text_chunks)} chunks in batches of {batch_size}")
                
                # Process in batches
                for i in range(0, len(text_chunks), batch_size):
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(text_chunks) + batch_size - 1) // batch_size
                    
                    batch = text_chunks[i:i + batch_size]
                    if self.verbose:
                        logger.info(f"📝 Adding batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    
                    try:
                        self.vectorstore.add_documents(batch)
                        
                        if i + batch_size < len(text_chunks):  # Not the last batch
                            if self.verbose:
                                logger.info(f"⏳ Waiting {delay_between_batches}s before next batch...")
                            time.sleep(delay_between_batches)
                            
                    except Exception as e:
                        if "429" in str(e) or "quota" in str(e).lower():
                            if self.verbose:
                                logger.warning(f"⚠️ Hit rate limit, increasing delay to {delay_between_batches * 2}s")
                            delay_between_batches *= 2
                            time.sleep(delay_between_batches)
                            self.vectorstore.add_documents(batch)
                        else:
                            raise e
            else:
                # Small number of chunks, add all at once
                self.vectorstore.add_documents(text_chunks)
            
            # Update document count
            if hasattr(self.vectorstore._collection, 'count'):
                self.total_documents = self.vectorstore._collection.count()
            
            if self.verbose:
                logger.info(f"✅ Successfully added {len(text_chunks)} new chunks to vectorstore")
            
            return True
            
        except Exception as e:
            if self.verbose:
                logger.error(f"❌ Error adding chunks to vectorstore: {e}")
            return False

    def _load_pdf_documents(self, pdf_dir: Path) -> List[Document]:
        """Load documents from PDF directory with incremental processing."""
        documents = []
        new_documents = []
        
        try:
            # Import PDF processing libraries, prioritizing PyMuPDF (fitz)
            try:
                import fitz  # PyMuPDF
                PDF_READER_AVAILABLE = True
                PDF_READER_TYPE = "pymupdf"
            except ImportError:
                try:
                    import PyPDF2
                    PDF_READER_AVAILABLE = True
                    PDF_READER_TYPE = "pypdf2"
                except ImportError:
                    if self.verbose:
                        logger.warning("⚠️ No PDF reader available (PyMuPDF or PyPDF2). Install with: pip install PyMuPDF")
                    return documents
            
            if not PDF_READER_AVAILABLE:
                return documents
            
            # Get all PDF files
            pdf_files = list(pdf_dir.glob("*.pdf"))
            
            if not pdf_files:
                if self.verbose:
                    logger.info(f"📄 No PDF files found in: {pdf_dir}")
                return documents
            
            if self.verbose:
                logger.info(f"📄 Found {len(pdf_files)} PDF files to check")
            
            # Process each PDF
            for pdf_file in pdf_files:
                # Check if this PDF has already been processed
                if self._is_document_processed(pdf_file):
                    if self.verbose:
                        logger.info(f"⏩ Skipping already processed PDF: {pdf_file.name}")
                    continue
                
                if self.verbose:
                    logger.info(f"📖 Processing new PDF: {pdf_file.name}")
                
                # Extract text from PDF
                pdf_documents = self._extract_text_from_pdf(pdf_file, PDF_READER_TYPE)
                
                if pdf_documents:
                    # Mark this PDF as processed
                    self._mark_document_processed(pdf_file, len(pdf_documents))
                    documents.extend(pdf_documents)
                    new_documents.extend(pdf_documents)
                    
                    if self.verbose:
                        logger.info(f"✅ Extracted {len(pdf_documents)} pages from {pdf_file.name}")
                else:
                    if self.verbose:
                        logger.warning(f"⚠️ Could not extract text from {pdf_file.name}")
            
            if new_documents and self.verbose:
                logger.info(f"📝 Found {len(new_documents)} new PDF pages to process")
            elif self.verbose:
                logger.info("✅ All PDF files already processed - using existing data")
                
        except Exception as e:
            logger.error(f"❌ Error loading PDF documents: {e}")
            if self.verbose:
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return documents
    
    def _extract_text_from_pdf(self, pdf_file: Path, reader_type: str) -> List[Document]:
        """Extract text from a PDF file."""
        documents = []
        
        try:
            if reader_type == "pypdf2":
                import PyPDF2
                
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            text = page.extract_text()
                            if text.strip():  # Only add non-empty pages
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        'title': pdf_file.stem,
                                        'source': str(pdf_file),
                                        'document_type': 'pdf',
                                        'page_number': page_num + 1,
                                        'filename': pdf_file.name,
                                        'file_path': str(pdf_file)
                                    }
                                )
                                documents.append(doc)
                        except Exception as e:
                            if self.verbose:
                                logger.warning(f"⚠️ Could not extract text from page {page_num + 1} of {pdf_file.name}: {e}")
                            
            elif reader_type == "pymupdf":
                import fitz
                
                pdf_document = fitz.open(str(pdf_file))
                
                for page_num in range(len(pdf_document)):
                    try:
                        page = pdf_document.load_page(page_num)
                        text = page.get_text()
                        
                        if text.strip():  # Only add non-empty pages
                            doc = Document(
                                page_content=text,
                                metadata={
                                    'title': pdf_file.stem,
                                    'source': str(pdf_file),
                                    'document_type': 'pdf',
                                    'page_number': page_num + 1,
                                    'filename': pdf_file.name,
                                    'file_path': str(pdf_file)
                                }
                            )
                            documents.append(doc)
                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"⚠️ Could not extract text from page {page_num + 1} of {pdf_file.name}: {e}")
                
                pdf_document.close()
                
        except Exception as e:
            logger.error(f"❌ Error extracting text from {pdf_file.name}: {e}")
        
        return documents
    
    def _load_processed_documents(self, processed_dir: Path) -> List[Document]:
        """Load documents from processed JSON files with incremental loading support."""
        documents = []
        new_documents = []
        
        try:
            for json_file in processed_dir.glob("*.json"):
                # Check if this document has already been processed
                if self._is_document_processed(json_file):
                    if self.verbose:
                        logger.info(f"⏩ Skipping already processed: {json_file.name}")
                    continue
                
                if self.verbose and len(documents) % 100 == 0:
                    logger.info(f"📄 Processing new document: {json_file.name}")
                
                file_documents = []
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    # Handle list of documents
                    for item in data:
                        if isinstance(item, dict) and 'content' in item:
                            doc = Document(
                                page_content=item['content'],
                                metadata={
                                    'title': item.get('title', ''),
                                    'source': item.get('source', str(json_file)),
                                    'document_type': item.get('document_type', 'processed')
                                }
                            )
                            file_documents.append(doc)
                elif isinstance(data, dict):
                    # Handle single document with chunks structure
                    if 'chunks' in data:
                        title = data.get('title', '')
                        source = data.get('source_url', str(json_file))
                        doc_type = data.get('document_type', 'processed')
                        
                        for chunk in data['chunks']:
                            if isinstance(chunk, dict) and 'content' in chunk:
                                doc = Document(
                                    page_content=chunk['content'],
                                    metadata={
                                        'title': title,
                                        'source': source,
                                        'document_type': doc_type,
                                        'chunk': chunk.get('metadata', {}).get('chunk', 1),
                                        'filename': data.get('filename', '')
                                    }
                                )
                                file_documents.append(doc)
                    elif 'content' in data:
                        # Handle single document with direct content
                        doc = Document(
                            page_content=data['content'],
                            metadata={
                                'title': data.get('title', ''),
                                'source': data.get('source', str(json_file)),
                                'document_type': data.get('document_type', 'processed')
                            }
                        )
                        file_documents.append(doc)
                
                # Mark document as processed and add to collections
                if file_documents:
                    self._mark_document_processed(json_file, len(file_documents))
                    documents.extend(file_documents)
                    new_documents.extend(file_documents)
                
        except Exception as e:
            logger.error(f"❌ Error loading processed documents: {e}")
            if self.verbose:
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
        
        if new_documents and self.verbose:
            logger.info(f"📝 Found {len(new_documents)} new documents to process")
        elif self.verbose:
            logger.info("✅ All documents already processed - using existing embeddings")
        
        return documents
    
    def query(self, question: str, session_state: Optional[Dict] = None, query_mode: str = "precise") -> Dict[str, Any]:
        """
        Query the legal RAG system with enhanced dual-mode responses.
        
        Args:
            question: User's legal question
            session_state: Streamlit session state for callbacks
            query_mode: "precise" for direct answers or "analyzed" for comprehensive analysis
            
        Returns:
            Dict containing answer, sources, and metadata
        """
        if not self.qa_chain:
            if self.ui_only:
                return {
                    'error': 'No existing embeddings found. Please run embedding process first.',
                    'answer': '',
                    'sources': []
                }
            else:
                return {
                    'error': 'System not ready. Please load documents first.',
                    'answer': '',
                    'sources': []
                }
        
        try:
            # Add callbacks if in Streamlit context
            callbacks = []
            if session_state and 'callback_handler' in session_state:
                callbacks.append(session_state['callback_handler'])
            
            # Run the query based on mode
            if self.verbose:
                mode_info = " (UI-only mode)" if self.ui_only else ""
                logger.info(f"🔍 Querying{mode_info} in '{query_mode}' mode: {question[:100]}...")
            
            if query_mode == "analyzed":
                # For analyzed mode, use a more comprehensive query approach
                result = self._query_analyzed_mode(question, callbacks)
            else:
                # Default precise mode - use existing chain
                result = self.qa_chain(
                    {"query": question},
                    callbacks=callbacks
                )
            
            # Extract sources
            sources = []
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    sources.append({
                        'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        'metadata': doc.metadata
                    })
            
            return {
                'answer': result.get('result', ''),
                'sources': sources,
                'error': None,
                'query_mode': query_mode
            }
            
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return {
                'error': str(e),
                'answer': '',
                'sources': [],
                'query_mode': query_mode
            }
    
    def _query_analyzed_mode(self, question: str, callbacks: List = None) -> Dict[str, Any]:
        """
        Execute a query in analyzed mode - comprehensive analysis with multiple chunk synthesis.
        """
        # First get more documents for comprehensive analysis
        docs = self.vectorstore.similarity_search_with_relevance_scores(
            question, 
            k=20,  # Get more documents for analysis
            score_threshold=0.3  # Lower threshold for broader coverage
        )
        
        if not docs:
            return {
                'result': 'Nuk u gjetën dokumente të përshtatshme për këtë pyetje.',
                'source_documents': []
            }
        
        # Extract documents and scores
        documents = [doc for doc, score in docs]
        
        # Create comprehensive analysis prompt
        analyzed_prompt_template = """Ti jeni një ekspert i lartë juridik shqiptar. Analizoni pyetjen dhe jepni një përgjigje të detajuar.

**KONTEKSTI LIGJOR:**
{context}

**PYETJA:** {question}

**PËRGJIGJA E ANALIZUAR:** Jepni një përgjigje të plotë dhe të detajuar në shqip."""

        # Create analyzed prompt
        analyzed_prompt = PromptTemplate(
            template=analyzed_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Prepare context from all documents
        context = "\n\n---\n\n".join([
            f"BURIMI {i+1}: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        # Generate response using LLM directly for analyzed mode
        try:
            formatted_prompt = analyzed_prompt.format(context=context, question=question)
            response = self.llm.invoke(formatted_prompt)
            
            return {
                'result': response.content if hasattr(response, 'content') else str(response),
                'source_documents': documents
            }
        except Exception as e:
            logger.error(f"❌ Error in analyzed mode: {e}")
            return {
                'result': f'Gabim në përpunimin e pyetjes në mënyrën analitike: {e}',
                'source_documents': documents
            }

    def process_documents_for_embeddings(self, documents_path: str = "legal_documents/pdfs"):
        """
        Process documents and create embeddings - for separate embedding process.
        This method should be called in embedding-only mode, not in UI mode.
        
        Args:
            documents_path: Path to directory containing PDF documents
        """
        if self.ui_only:
            logger.error("❌ Cannot process documents in UI-only mode. Use full mode for embedding processing.")
            return False
            
        if not self.embeddings:
            logger.error("❌ Embeddings not initialized. Cannot process documents.")
            return False
            
        try:
            # Check if documents_path ends with /pdfs - if so, use parent directory
            if documents_path.endswith("/pdfs") or documents_path.endswith("\\pdfs"):
                # Remove /pdfs since load_documents_from_directory adds it automatically
                parent_path = str(Path(documents_path).parent)
                success = self.load_documents_from_directory(parent_path)
            else:
                # Use as is - load_documents_from_directory will look for pdfs subdirectory
                success = self.load_documents_from_directory(documents_path)
            
            if success:
                if self.verbose:
                    logger.info(f"✅ Successfully processed documents for embeddings")
                return True
            else:
                if self.verbose:
                    logger.warning("⚠️ No new documents to process")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to process documents for embeddings: {e}")
            return False
