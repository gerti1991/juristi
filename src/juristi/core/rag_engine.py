"""
Core RAG Engine for Albanian Legal Research System

This module contains the main RAG engine extracted and refactored from the original app.py.
Provides document retrieval, embedding generation, and search functionality.
"""

import os
import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Local imports - adjust based on package structure
from .llm_client import CloudLLMClient
from ..data.processing import AlbanianLegalScraper

# Load environment variables
load_dotenv()

# Set environment variables for PyTorch compatibility
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class AlbanianLegalRAG:
    """
    Advanced Albanian Legal RAG System with multiple retrieval strategies.
    
    Features:
    - Hybrid retrieval (dense + sparse)
    - Hierarchical RAG
    - Sentence-window RAG
    - Lazy loading for quick startup
    - Google embeddings integration
    - Advanced document processing
    """
    
    def __init__(self, quick_start: bool = None, rag_mode: str = None, 
                 use_google_embeddings: bool = None, hybrid_alpha: float = None):
        """
        Initialize the RAG system with configurable startup mode.
        
        Args:
            quick_start: Enable quick startup (lazy loading)
            rag_mode: RAG mode ('traditional', 'hierarchical', 'sentence_window')
            use_google_embeddings: Whether to use Google embeddings
            hybrid_alpha: Balance between dense and sparse retrieval
        """
        
        # Configuration from environment or parameters
        if quick_start is None:
            quick_start = os.getenv("RAG_QUICK_START", "0") == "1"
        self.quick_start = quick_start
        
        # Core configuration
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.15'))
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))
        self.max_chunks_to_return = int(os.getenv('MAX_CHUNKS_TO_RETURN', '5'))
        
        # Embedding configuration
        if use_google_embeddings is None:
            use_google_embeddings = os.getenv('USE_GOOGLE_EMBEDDINGS', '1') == '1'
        self.use_google_embeddings = use_google_embeddings
        
        self.model_name = 'all-MiniLM-L6-v2'  # Fallback model
        self.model = None
        self.gemini_api_key = None
        
        # Document storage and processing
        self.legal_documents = []
        self.chunks = []  # Text chunks for search
        self.document_embeddings = None
        self.superchunk_chars = int(os.getenv('RAG_SUPERCHUNK_CHARS', '4500'))
        self.superchunk_overlap = int(os.getenv('RAG_SUPERCHUNK_OVERLAP', '500'))
        
        # Retrieval configuration
        if hybrid_alpha is None:
            hybrid_alpha = float(os.getenv('RAG_HYBRID_ALPHA', '0.5'))
        self.hybrid_alpha = hybrid_alpha
        
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.15'))
        self.mmr_lambda = float(os.getenv('RAG_MMR_LAMBDA', '0.5'))
        self.neighbor_expansion = int(os.getenv('RAG_NEIGHBOR_EXPANSION', '1'))
        self.context_packing = os.getenv('RAG_CONTEXT_PACKING', '1') == '1'
        
        # Advanced RAG modes
        if rag_mode is None:
            rag_mode = os.getenv('RAG_MODE', 'traditional')
        self.rag_mode = rag_mode
        
        self.sentence_window_size = int(os.getenv('RAG_SENTENCE_WINDOW_SIZE', '5'))
        self.hierarchical_top_docs = int(os.getenv('RAG_HIERARCHICAL_TOP_DOCS', '2'))
        
        # Sparse retrieval components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Advanced data structures
        self.document_summaries = []
        self.document_summary_embeddings = None
        self.sentences = []
        self.sentence_embeddings = None
        self.sentence_to_chunk_map = {}
        
        # Components
        self.scraper = None
        self.cloud_llm = None
        self.google_api = None
        
        # Memory and caching
        self.memory_store = {}
        self.embeddings_cache_file = 'document_embeddings_cache_google.json'
        self.google_cache_file = 'document_embeddings_cache_google.json'  # Alias for compatibility
        
        # State tracking
        self._components_initialized = False
        self._documents_loaded = False
        self._embeddings_ready = False
        
        # Initialize components based on startup mode
        if not self.quick_start:
            # Full initialization
            self._setup_google_api()
            self._load_fallback_model()
            self._load_cloud_llm()
            self._initialize_scraper()
            self._load_all_documents()
            self._generate_embeddings()
            self._documents_loaded = True
            self._google_api_setup = True
            self._cloud_llm_loaded = True
        else:
            # Minimal initialization for quick start
            self.legal_documents = []
            self.use_google_embeddings = False  # Will be determined later
            self.cloud_llm = None
            self._documents_loaded = False
            self._google_api_setup = False
            self._cloud_llm_loaded = False
        
        # Advanced structures flag
        self._advanced_structures_initialized = False
    
    # ================================
    # COMPONENT INITIALIZATION
    # ================================
    
    def _setup_google_api(self):
        """Setup Google API for embeddings."""
        # Skip entirely in quick start mode during initial load
        if self.quick_start and not getattr(self, '_google_api_setup', False):
            return
            
        try:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            print(f"🔑 GEMINI_API_KEY found: {'Yes' if self.gemini_api_key else 'No'}")
            if self.gemini_api_key:
                print(f"   Key starts with: {self.gemini_api_key[:10]}...")
                # Import and initialize Google API
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.gemini_api_key)
                    print("✅ Google Generative AI configured successfully")
                    
                    # Create a simple wrapper for the Google embeddings API
                    class GoogleEmbeddingsAPI:
                        def __init__(self, api_key):
                            self.api_key = api_key
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                        
                        def get_embeddings(self, texts):
                            """Get embeddings for a list of texts."""
                            try:
                                import google.generativeai as genai
                                embeddings = []
                                
                                # Test with first text to validate API
                                print(f"🔍 Testing Google API with {len(texts)} texts...")
                                if len(texts) > 0:
                                    print(f"   First text preview: {texts[0][:100]}...")
                                
                                for i, text in enumerate(texts):
                                    if i == 0 or (i + 1) % 100 == 0 or i == len(texts) - 1:
                                        print(f"   Processing text {i+1}/{len(texts)}")
                                    
                                    response = genai.embed_content(
                                        model="models/text-embedding-004",
                                        content=text
                                    )
                                    embeddings.append(response['embedding'])
                                
                                print(f"✅ Successfully generated {len(embeddings)} Google embeddings")
                                return embeddings
                            except Exception as e:
                                print(f"❌ Error getting Google embeddings: {type(e).__name__}: {e}")
                                import traceback
                                print(f"   Full error: {traceback.format_exc()}")
                                return None
                    
                    self.google_api = GoogleEmbeddingsAPI(self.gemini_api_key)
                    # Only enable Google embeddings if explicitly requested
                    if self.use_google_embeddings:
                        print("✅ Google embeddings API initialized and enabled")
                    else:
                        print("ℹ️  Google embeddings API initialized but disabled (use_google_embeddings=False)")
                        
                except ImportError as e:
                    print(f"❌ Could not import Google Generative AI: {e}")
                    self.google_api = None
                except Exception as e:
                    print(f"❌ Error initializing Google API: {type(e).__name__}: {e}")
                    import traceback
                    print(f"   Full error: {traceback.format_exc()}")
                    self.google_api = None
            else:
                print("❌ No GEMINI_API_KEY found in environment")
                self.google_api = None
        except Exception as e:
            print(f"❌ Error in Google API setup: {type(e).__name__}: {e}")
            self.google_api = None
        
        # Initialize sentence transformer model (fallback and default)
    
    def _load_fallback_model(self):
        """Load fallback sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            if not self.model:
                print(f"🔄 Loading fallback model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                print(f"✅ Fallback model loaded successfully")
        except ImportError:
            print("❌ Warning: sentence_transformers not available")
        except Exception as e:
            print(f"❌ Warning: Could not load fallback model: {e}")
    
    def _load_cloud_llm(self):
        """Initialize cloud LLM client."""
        try:
            self.cloud_llm = CloudLLMClient("gemini")
        except Exception as e:
            print(f"Warning: Cloud LLM initialization failed: {e}")
            self.cloud_llm = None
    
    def _initialize_scraper(self):
        """Initialize the document scraper."""
        if not self.scraper:
            try:
                self.scraper = AlbanianLegalScraper(max_docs=10)
            except Exception as e:
                print(f"Warning: Scraper initialization failed: {e}")
                self.scraper = None
    
    # ================================
    # LAZY LOADING METHODS
    # ================================
    
    def _ensure_all_components_ready(self):
        """Ensure all components are loaded when needed."""
        if self.quick_start:
            if self.scraper is None:
                self._initialize_scraper()
            
            if not self._google_api_setup:
                self._setup_google_api()
                self._google_api_setup = True
            
            # Ensure fallback model is loaded for embeddings
            if self.model is None:
                self._load_fallback_model()
            
            if not self._cloud_llm_loaded:
                self._load_cloud_llm()
                self._cloud_llm_loaded = True
            
            if not self._documents_loaded:
                self._load_all_documents()
                self._documents_loaded = True
    
    def _ensure_documents_loaded(self):
        """Ensure documents are loaded (lazy loading for quick start)."""
        if not self._documents_loaded:
            self._load_all_documents()
            self._documents_loaded = True
    
    def _ensure_embeddings_ready(self):
        """Ensure embeddings are available, generate if needed."""
        # Try to load from cache first
        if self._load_embeddings_cache():
            return
        
        # If no cache, generate now
        self._generate_embeddings_now()
    
    def _ensure_advanced_structures_initialized(self):
        """Lazy initialization of advanced RAG structures."""
        if self._advanced_structures_initialized:
            return
            
        if self.rag_mode in ['hierarchical', 'sentence_window']:
            try:
                if self.rag_mode == 'hierarchical':
                    self._build_hierarchical_structures()
                elif self.rag_mode == 'sentence_window':
                    self._build_sentence_structures()
                    
                self._advanced_structures_initialized = True
            except Exception as e:
                print(f"Warning: Could not initialize advanced structures: {e}")
    
    # ================================
    # DOCUMENT LOADING
    # ================================
    
    def _load_all_documents(self):
        """Load all document sources."""
        hardcoded = self._load_hardcoded_documents()
        scraped = self._load_scraped_documents()
        pdf_docs = self._load_pdf_documents()
        
        # Combine all documents
        self.legal_documents = hardcoded + scraped + pdf_docs
        
        # Ensure IDs exist
        for i, doc in enumerate(self.legal_documents):
            if 'id' not in doc or not doc.get('id'):
                prefix = (doc.get('source') or doc.get('url') or doc.get('title') or 'doc').replace(' ', '_')
                doc['id'] = f"{prefix}_chunk_{i}"
        
        # Extract text chunks for search
        self.chunks = []
        for doc in self.legal_documents:
            content = doc.get('content', '')
            if content:
                self.chunks.append(content)
        
        print(f"📄 Loaded {len(self.legal_documents)} documents with {len(self.chunks)} chunks")
    
    def _load_hardcoded_documents(self) -> List[Dict]:
        """Load hardcoded legal documents."""
        return [
            {
                "id": "family_law_1",
                "title": "Kodi i Familjes - Martesa",
                "content": "Martesa është bashkimi i ligjshëm i burrit dhe gruas me qëllim krijimin e familjes. Martesa lidhet para zyrtarit të gjendjes civile. Për t'u martuar duhet plotësuar kushtet: 1) Mosha minimale 18 vjeç për të dy palët 2) Pëlqimi i lirë i të dy palëve 3) Mungesa e pengesave ligjore",
                "source": "Kodi i Familjes, Neni 7-15",
                "document_type": "hardcoded"
            },
            {
                "id": "criminal_law_1", 
                "title": "Kodi Penal - Vrasja me dashje",
                "content": "Vrasja me dashje dënohet me burgim jo më pak se 15 vjet. Kur vrasja kryhet: a) ndaj dy ose më shumë personave b) me mizori të veçantë c) për motive të ulta dënohet me burgim të përjetshëm ose jo më pak se 20 vjet burgim",
                "source": "Kodi Penal, Neni 76",
                "document_type": "hardcoded"
            },
            {
                "id": "civil_law_1",
                "title": "Kodi Civil - Pronësia",
                "content": "E drejta e pronësisë është e drejta për të shfrytëzuar, administruar dhe disponuar lirisht me gjënë. Pronari mund të kërkojë nga çdo person kthimin e gjësë së tij dhe eliminimin e çdo pengese të paligjshme për ushtrimin e së drejtës së pronësisë",
                "source": "Kodi Civil, Neni 159",
                "document_type": "hardcoded"
            }
        ]
    
    def _load_scraped_documents(self) -> List[Dict]:
        """Load scraped documents."""
        if not self.scraper:
            return []
            
        try:
            scraped_docs = self.scraper.get_processed_documents_for_rag()
            # Merge to super-chunks
            merged = self._to_superchunks(scraped_docs, self.superchunk_chars, self.superchunk_overlap)
            return merged
        except Exception as e:
            print(f"Warning: Could not load scraped documents: {e}")
            return []
    
    def _load_pdf_documents(self) -> List[Dict]:
        """Load processed PDF documents."""
        try:
            # Try enhanced documents first if in advanced mode
            enhanced_file = os.path.join("legal_documents", "pdf_rag_documents_advanced.json")
            traditional_file = os.path.join("legal_documents", "pdf_rag_documents.json")
            
            if self.rag_mode in ['hierarchical', 'sentence_window'] and os.path.exists(enhanced_file):
                return self._load_enhanced_pdf_documents(enhanced_file)
            elif os.path.exists(traditional_file):
                return self._load_traditional_pdf_documents(traditional_file)
            else:
                return []
        except Exception as e:
            print(f"Warning: Could not load PDF documents: {e}")
            return []
    
    def _load_traditional_pdf_documents(self, file_path: str) -> List[Dict]:
        """Load traditional PDF documents."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
            
            # Convert to expected format
            normalized_docs = []
            for i, doc in enumerate(docs):
                normalized_doc = {
                    'id': doc.get('id', f'pdf_{i}'),
                    'title': doc.get('title', 'PDF Document'),
                    'content': doc.get('content', ''),
                    'source': doc.get('source_url', doc.get('source', 'PDF')),
                    'document_type': doc.get('document_type', 'local_pdf'),
                    'filename': doc.get('filename', '')
                }
                normalized_docs.append(normalized_doc)
            
            print(f"📄 Loaded {len(normalized_docs)} traditional PDF documents")
            
            # Don't merge into super-chunks for traditional documents to preserve original structure
            return normalized_docs
            
        except Exception as e:
            print(f"Warning: Could not load traditional PDF documents: {e}")
            return []
    
    def _load_enhanced_pdf_documents(self, file_path: str) -> List[Dict]:
        """Load enhanced PDF documents with hierarchical data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                enhanced_docs = json.load(f)
            
            # Store hierarchical data for advanced RAG modes
            self.hierarchical_data = enhanced_docs
            
            # Extract traditional chunks for main document list
            normalized_docs = []
            for doc_data in enhanced_docs:
                chunks = doc_data.get('chunks', [])
                for chunk in chunks:
                    normalized_doc = {
                        'id': chunk.get('chunk_id', f'enhanced_{len(normalized_docs)}'),
                        'title': f"{doc_data.get('title', 'PDF Document')} - Part {chunk.get('chunk_index', 1)}",
                        'content': chunk.get('content', ''),
                        'source': doc_data.get('source', 'Enhanced PDF'),
                        'document_type': 'enhanced_pdf',
                        'parent_doc': doc_data.get('title', ''),
                        'chunk_index': chunk.get('chunk_index', 0)
                    }
                    normalized_docs.append(normalized_doc)
            
            return normalized_docs
            
        except Exception as e:
            print(f"Warning: Could not load enhanced PDF documents: {e}")
            return []
    
    # ================================
    # UTILITY METHODS
    # ================================
    
    def _to_superchunks(self, documents: List[Dict], chars_per_chunk: int, overlap_chars: int) -> List[Dict]:
        """Merge small documents into larger super-chunks."""
        if not documents:
            return []
        
        superchunks = []
        current_chunk = {
            'id': f'superchunk_{len(superchunks)}',
            'title': '',
            'content': '',
            'source': 'Merged',
            'document_type': 'superchunk',
            'merged_from': []
        }
        current_length = 0
        
        for doc in documents:
            doc_content = doc.get('content', '')
            doc_length = len(doc_content)
            
            # If adding this doc would exceed limit, finalize current chunk
            if current_length + doc_length > chars_per_chunk and current_length > 0:
                superchunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk['content'][-overlap_chars:] if overlap_chars > 0 else ''
                current_chunk = {
                    'id': f'superchunk_{len(superchunks)}',
                    'title': '',
                    'content': overlap_text,
                    'source': 'Merged',
                    'document_type': 'superchunk',
                    'merged_from': []
                }
                current_length = len(overlap_text)
            
            # Add document to current chunk
            if current_chunk['title']:
                current_chunk['title'] += ' + ' + doc.get('title', '')
            else:
                current_chunk['title'] = doc.get('title', '')
            
            current_chunk['content'] += ' ' + doc_content
            current_chunk['merged_from'].append(doc.get('id', ''))
            current_length += doc_length + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_length > 0:
            superchunks.append(current_chunk)
        
        return superchunks
    
    # ================================
    # SEARCH INTERFACE
    # ================================
    
    def search_documents(self, query: str, top_k: int = 3, mode: str = 'hybrid', 
                        session_id: Optional[str] = None, multi_query: bool = True) -> List[Dict]:
        """
        Search for relevant documents using various retrieval strategies.
        
        Args:
            query: User query
            top_k: Number of documents to return
            mode: Retrieval mode ('hybrid', 'embedding', 'sparse', 'hierarchical', 'sentence_window')
            session_id: Optional conversation ID for context
            multi_query: Whether to use query expansion
            
        Returns:
            List of relevant documents with metadata
        """
        # Ensure all components are ready (lazy loading)
        self._ensure_all_components_ready()
        
        # Ensure embeddings are ready
        if self.document_embeddings is None:
            self._ensure_embeddings_ready()
            
        if self.document_embeddings is None or len(self.legal_documents) == 0:
            return []
        
        # Ensure advanced structures are initialized if needed
        self._ensure_advanced_structures_initialized()
        
        # Route to appropriate search method based on RAG mode
        if self.rag_mode == 'hierarchical':
            return self._search_hierarchical(query, top_k, session_id, multi_query)
        elif self.rag_mode == 'sentence_window':
            return self._search_sentence_window(query, top_k, session_id, multi_query)
        else:
            # Traditional hybrid/embedding/sparse search
            return self._search_traditional(query, top_k, mode, session_id, multi_query)
    
    def generate_response(self, query: str, max_results: int = 3) -> str:
        """Generate a comprehensive response to a legal query."""
        # Search for relevant documents
        results = self.search_documents(query, top_k=max_results)
        
        if not self.cloud_llm:
            # Fallback response
            if results:
                response = f"📚 **Dokumentet e gjetur për: {query}**\n\n"
                for i, doc in enumerate(results, 1):
                    response += f"**{i}. {doc['title']}**\n{doc['content'][:300]}...\n\n"
                return response
            else:
                return "Nuk u gjetën dokumente relevante për pyetjen tuaj."
        
        # Use cloud LLM for enhanced response
        try:
            return self.cloud_llm.generate_response(query, results)
        except Exception as e:
            print(f"Warning: LLM response generation failed: {e}")
            # Fallback to simple response
            if results:
                return f"Gjeta {len(results)} dokumente relevante, por nuk mund të gjeneroj përgjigje të detajuar."
            else:
                return "Nuk u gjetën dokumente relevante."


    # ================================
    # EMBEDDING METHODS
    # ================================
    
    def _generate_embeddings(self):
        """Generate embeddings with quick start check."""
        if self.quick_start:
            return
        self._generate_embeddings_now()
    
    def _generate_embeddings_now(self):
        """Force generate embeddings (bypasses quick start check)"""
        if not self.use_google_embeddings and not self.model:
            print("❌ No embedding model available - cannot generate embeddings")
            return
        
        if not self.legal_documents:
            print("❌ No documents loaded - cannot generate embeddings")
            return
        
        # Try to load from cache first
        if self._load_embeddings_cache():
            return
        
        try:
            if self.use_google_embeddings:
                print("🔄 Generating Google embeddings for better Albanian legal search...")
                
                # Prepare document texts with enhanced content
                document_texts = []
                for doc in self.legal_documents:
                    # Combine title and content for better context
                    enhanced_text = f"Dokument ligjor: {doc['title']}. Përmbajtja: {doc['content']}"
                    if doc.get('content_en'):
                        enhanced_text += f" English: {doc['content_en']}"
                    document_texts.append(enhanced_text)
                
                # Generate Google embeddings
                self.document_embeddings = self._get_google_embeddings(document_texts)
                
                if self.document_embeddings is None:
                    print("❌ Failed to generate Google embeddings - falling back to local model")
                    # Fallback to local model
                    if self.model:
                        print(f"🔄 Generating embeddings for {len(document_texts)} documents with local model...")
                        # Process in batches for large datasets to avoid memory issues
                        batch_size = 100
                        all_embeddings = []
                        
                        for i in range(0, len(document_texts), batch_size):
                            batch = document_texts[i:i+batch_size]
                            print(f"   Processing batch {i//batch_size + 1}/{(len(document_texts)-1)//batch_size + 1} ({len(batch)} documents)")
                            batch_embeddings = self.model.encode(batch)
                            all_embeddings.append(batch_embeddings)
                        
                        import numpy as np
                        self.document_embeddings = np.vstack(all_embeddings)
                        print(f"✅ Generated {len(self.document_embeddings)} document embeddings")
                    else:
                        print("❌ No local model available for fallback")
                        return
                else:
                    # Convert Google embeddings list to numpy array
                    import numpy as np
                    self.document_embeddings = np.array(self.document_embeddings)
                    print(f"✅ Successfully generated {len(self.document_embeddings)} Google embeddings (converted to numpy array)")
            else:
                # Use local SentenceTransformer model
                print(f"🔄 Generating embeddings with local model for {len(self.legal_documents)} documents...")
                
                document_texts = [f"{doc['title']}. {doc['content']}" for doc in self.legal_documents]
                
                # Process in batches for large datasets
                batch_size = 100
                all_embeddings = []
                
                for i in range(0, len(document_texts), batch_size):
                    batch = document_texts[i:i+batch_size]
                    print(f"   Processing batch {i//batch_size + 1}/{(len(document_texts)-1)//batch_size + 1} ({len(batch)} documents)")
                    batch_embeddings = self.model.encode(batch)
                    all_embeddings.append(batch_embeddings)
                
                import numpy as np
                self.document_embeddings = np.vstack(all_embeddings)
                print(f"✅ Generated {len(self.document_embeddings)} local embeddings")
            
            # Save to cache for future use
            self._save_embeddings_cache()
            
            # Generate TF-IDF matrix for hybrid search
            self._generate_tfidf_matrix()
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
    
    def _ensure_embeddings_ready(self):
        """Ensure embeddings are available, generate if needed"""
        if self.document_embeddings is None:
            print("🔄 First search - generating embeddings...")
            self._generate_embeddings_now()
    
    def _get_google_embeddings(self, texts: List[str]):
        """Generate embeddings using Google's API."""
        if not self.google_api:
            return None
        
        try:
            return self.google_api.get_embeddings(texts)
        except Exception as e:
            print(f"Error getting Google embeddings: {e}")
            return None
    
    def _get_google_query_embedding(self, query: str):
        """Get embedding for a single query using Google's API."""
        embeddings = self._get_google_embeddings([query])
        return embeddings[0] if embeddings is not None and len(embeddings) > 0 else None
    
    def _generate_tfidf_matrix(self):
        """Generate TF-IDF matrix for sparse retrieval."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            document_texts = [doc['content'] for doc in self.legal_documents]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,  # Keep Albanian stopwords
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(document_texts)
            print("✅ Generated TF-IDF matrix for hybrid search")
            
        except Exception as e:
            print(f"Warning: Could not generate TF-IDF matrix: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def _load_embeddings_cache(self) -> bool:
        """Load embeddings from cache file."""
        cache_file = self.google_cache_file if self.use_google_embeddings else "document_embeddings_cache.json"
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # Check if cache is valid (same number of documents)
            if len(cache.get('embeddings', [])) == len(self.legal_documents):
                self.document_embeddings = np.array(cache['embeddings'])
                print(f"✅ Loaded embeddings from cache: {cache_file}")
                return True
        except Exception as e:
            print(f"Warning: Could not load embeddings cache: {e}")
        
        return False
    
    def _save_embeddings_cache(self):
        """Save embeddings to cache file."""
        if self.document_embeddings is None:
            return
        
        cache_file = self.google_cache_file if self.use_google_embeddings else "document_embeddings_cache.json"
        
        try:
            # Convert numpy array to list for JSON serialization
            import numpy as np
            if isinstance(self.document_embeddings, np.ndarray):
                embeddings_list = self.document_embeddings.tolist()
            else:
                embeddings_list = self.document_embeddings
                
            cache = {
                'embeddings': embeddings_list,
                'timestamp': time.time(),
                'document_count': len(self.legal_documents)
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
            
            print(f"💾 Saved embeddings cache: {cache_file}")
            
        except Exception as e:
            print(f"Warning: Could not save embeddings cache: {e}")
    
    # ================================
    # SEARCH METHODS
    # ================================
    
    def _search_traditional(self, query: str, top_k: int, mode: str, session_id: Optional[str], multi_query: bool) -> List[Dict]:
        """Traditional similarity-based search."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Enhanced query with Albanian legal context
            enhanced_query = self._enhance_albanian_query(query)
            
            # Get query embedding - ensure it matches the document embedding space
            query_embedding = None
            
            # Check if we have Google embeddings available and working
            if self.use_google_embeddings and self.google_api:
                query_embedding = self._get_google_query_embedding(enhanced_query)
                
            # If Google failed or not available, use local model
            if query_embedding is None:
                if not self.model:
                    print("❌ Local model not loaded for query embedding")
                    return []
                query_embedding = self.model.encode([enhanced_query])
                print(f"🔍 Using local model for query: '{query}'")
            
            # Ensure query_embedding is 2D for cosine_similarity
            import numpy as np
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Check dimension compatibility
            query_dim = query_embedding.shape[1]
            doc_dim = self.document_embeddings.shape[1]
            
            if query_dim != doc_dim:
                print(f"⚠️ Dimension mismatch: Query={query_dim}, Documents={doc_dim}")
                print("🔄 Switching to local model for consistency...")
                
                # Force local model for query to match document embeddings
                if not self.model:
                    print("❌ Local model not available for dimension matching")
                    return []
                
                query_embedding = self.model.encode([enhanced_query]).reshape(1, -1)
                print(f"✅ Using local model query embedding: {query_embedding.shape}")
            
            # Calculate dense similarities
            dense_sim = cosine_similarity(query_embedding, self.document_embeddings)[0]

            # Calculate sparse similarities if enabled
            sparse_sim = None
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None and mode in ('hybrid', 'sparse'):
                try:
                    q_vec = self.tfidf_vectorizer.transform([enhanced_query])
                    sparse_sim = (q_vec @ self.tfidf_matrix.T).toarray()[0]
                except Exception:
                    sparse_sim = None

            # Combine similarities
            if mode == 'embedding' or sparse_sim is None:
                combined_sim = dense_sim
            elif mode == 'sparse':
                combined_sim = sparse_sim
            else:
                alpha = self.hybrid_alpha
                combined_sim = alpha * dense_sim + (1 - alpha) * sparse_sim
            
            # Multi-query expansion if enabled
            if multi_query:
                rewrites = self._expand_queries(enhanced_query)
                for rw in rewrites:
                    if rw.strip() == enhanced_query.strip():
                        continue
                    
                    # Get rewrite embedding
                    if self.use_google_embeddings:
                        rw_emb = self._get_google_query_embedding(rw)
                        if rw_emb is None and self.model:
                            rw_emb = self.model.encode([rw])
                    else:
                        rw_emb = self.model.encode([rw]) if self.model else None
                    
                    if rw_emb is not None:
                        # Ensure rw_emb is 2D for cosine_similarity
                        if isinstance(rw_emb, list):
                            rw_emb = np.array(rw_emb)
                        if rw_emb.ndim == 1:
                            rw_emb = rw_emb.reshape(1, -1)
                        
                        # Check dimension compatibility for rewrite embeddings
                        if rw_emb.shape[1] != self.document_embeddings.shape[1]:
                            print(f"⚠️ Rewrite dimension mismatch, using local model...")
                            if self.model:
                                rw_emb = self.model.encode([rw]).reshape(1, -1)
                            else:
                                continue
                            
                        rw_dense_sim = cosine_similarity(rw_emb, self.document_embeddings)[0]
                        
                        # Combine with sparse if available
                        if sparse_sim is not None and mode in ('hybrid', 'sparse'):
                            try:
                                rw_q_vec = self.tfidf_vectorizer.transform([rw])
                                rw_sparse_sim = (rw_q_vec @ self.tfidf_matrix.T).toarray()[0]
                                rw_combined = alpha * rw_dense_sim + (1 - alpha) * rw_sparse_sim
                            except Exception:
                                rw_combined = rw_dense_sim
                        else:
                            rw_combined = rw_dense_sim
                        
                        # Take max of original and rewrite similarities
                        combined_sim = np.maximum(combined_sim, rw_combined)
            
            # MMR selection for diversity (if implemented)
            if hasattr(self, 'mmr_lambda') and hasattr(self, '_mmr_select'):
                selected_indices = self._mmr_select(query_embedding[0], self.document_embeddings, combined_sim, k=top_k, lambda_param=self.mmr_lambda)
            else:
                # Simple top-k selection
                selected_indices = np.argsort(combined_sim)[-top_k:][::-1]
            
            # Expand with neighbors if enabled
            if hasattr(self, 'neighbor_expansion') and hasattr(self, '_expand_neighbors'):
                expanded_indices = self._expand_neighbors(selected_indices, self.neighbor_expansion)
            else:
                expanded_indices = selected_indices
            
            results = []
            max_similarity = 0
            
            # Find the maximum similarity for adaptive thresholding
            for idx in expanded_indices:
                similarity = float(combined_sim[idx])
                if similarity > max_similarity:
                    max_similarity = similarity
                    
            # Use adaptive thresholding like the original
            base_threshold = getattr(self, 'similarity_threshold', 0.15)
            if self.use_google_embeddings:
                # Google embeddings have different scale
                base_threshold = max(0.1, base_threshold * 0.8)
                
            adaptive_threshold = max(base_threshold, max_similarity * 0.7) if max_similarity > 0.2 else base_threshold
            
            # Build results with original logic
            for i, idx in enumerate(expanded_indices):
                similarity = float(combined_sim[idx])
                
                # Always include first result if it meets minimum threshold, or use adaptive threshold
                if similarity > adaptive_threshold or (len(results) == 0 and i == 0 and similarity > 0.1):
                    doc = self.legal_documents[idx].copy()
                    doc['similarity_score'] = similarity
                    doc['similarity'] = similarity  # For compatibility
                    doc['rank'] = len(results) + 1
                    doc['score'] = similarity  # Also add 'score' for compatibility
                    doc['text'] = doc.get('content', '')  # Add 'text' field for compatibility
                    results.append(doc)
            
            print(f"🔍 Search for '{query}': max_sim={max_similarity:.4f}, threshold={adaptive_threshold:.4f}, found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error in traditional search: {e}")
            return []
    
    def _expand_queries(self, query: str) -> List[str]:
        """Expand query with Albanian legal synonyms."""
        expansions = [query]
        
        # Albanian legal term expansions
        legal_expansions = {
            'martesë': ['martesa', 'kurorëzim', 'lidhja martesore'],
            'pronësi': ['pronësia', 'të drejta pronësie', 'zotërim'],
            'punësim': ['punë', 'marrëdhënie pune', 'kontratë pune'],
            'gjykatë': ['gjykata', 'gjykim', 'proces gjyqësor'],
            'ligj': ['ligjin', 'legjislacion', 'normë ligjore'],
            'vrasje': ['vrasja', 'vdekje me dashje', 'homicid'],
            'dënim': ['dënimin', 'sanksion', 'ndëshkim'],
            'kontratë': ['kontrata', 'marrëveshje', 'akt juridik']
        }
        
        query_lower = query.lower()
        for term, synonyms in legal_expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expansions.append(expanded)
                break  # Only expand first match to avoid too many variations
        
        return expansions[:3]  # Limit to avoid too many queries
    
    def _search_hierarchical(self, query: str, top_k: int, session_id: Optional[str], multi_query: bool) -> List[Dict]:
        """Hierarchical search implementation (first summary, then detailed)."""
        if not self.document_summaries or not hasattr(self, 'document_summary_embeddings'):
            print("Warning: Hierarchical structures not available, falling back to traditional search")
            return self._search_traditional(query, top_k, 'hybrid', session_id, multi_query)
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Step 1: Search document summaries first
            enhanced_query = f"Pyetje ligjore shqiptare: {query}"
            
            if self.use_google_embeddings:
                query_embedding = self._get_google_query_embedding(enhanced_query)
            else:
                query_embedding = self.model.encode([enhanced_query]) if self.model else None
            
            if query_embedding is None:
                return self._search_traditional(query, top_k, 'hybrid', session_id, multi_query)
            
            # Ensure query_embedding is 2D for cosine_similarity
            import numpy as np
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Find most relevant document summaries
            summary_similarities = cosine_similarity(query_embedding, self.document_summary_embeddings)[0]
            top_summary_indices = np.argsort(summary_similarities)[-min(top_k*2, len(self.document_summaries)):][::-1]
            
            # Step 2: Search within selected documents' chunks
            relevant_chunks = []
            for summary_idx in top_summary_indices:
                if summary_similarities[summary_idx] > 0.1:  # Threshold for relevance
                    doc_summary = self.document_summaries[summary_idx]
                    chunk_indices = doc_summary.get('chunk_indices', [])
                    
                    # Search within this document's chunks
                    for chunk_idx in chunk_indices:
                        if chunk_idx < len(self.legal_documents):
                            relevant_chunks.append((chunk_idx, summary_similarities[summary_idx]))
            
            # Step 3: Re-rank selected chunks
            if relevant_chunks:
                chunk_indices = [chunk[0] for chunk in relevant_chunks]
                chunk_embeddings = self.document_embeddings[chunk_indices]
                chunk_similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                
                # Combine summary and chunk scores
                combined_scores = []
                for i, (chunk_idx, summary_score) in enumerate(relevant_chunks):
                    combined_score = 0.7 * chunk_similarities[i] + 0.3 * summary_score
                    combined_scores.append((chunk_idx, combined_score))
                
                # Sort by combined score and return top results
                combined_scores.sort(key=lambda x: x[1], reverse=True)
                
                results = []
                for i, (chunk_idx, score) in enumerate(combined_scores[:top_k]):
                    if score > 0.1:
                        doc = self.legal_documents[chunk_idx].copy()
                        doc['similarity_score'] = float(score)
                        doc['rank'] = i + 1
                        doc['search_method'] = 'hierarchical'
                        results.append(doc)
                
                return results
            
            # Fallback to traditional search
            return self._search_traditional(query, top_k, 'hybrid', session_id, multi_query)
            
        except Exception as e:
            print(f"Error in hierarchical search: {e}")
            return self._search_traditional(query, top_k, 'hybrid', session_id, multi_query)
    
    def _search_sentence_window(self, query: str, top_k: int, session_id: Optional[str], multi_query: bool) -> List[Dict]:
        """Sentence window search implementation."""
        if not hasattr(self, 'sentence_windows') or not self.sentence_windows:
            print("Warning: Sentence window structures not available, falling back to traditional search")
            return self._search_traditional(query, top_k, 'hybrid', session_id, multi_query)
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Search sentence windows
            enhanced_query = f"Pyetje ligjore shqiptare: {query}"
            
            if self.use_google_embeddings:
                query_embedding = self._get_google_query_embedding(enhanced_query)
            else:
                query_embedding = self.model.encode([enhanced_query]) if self.model else None
            
            if query_embedding is None:
                return self._search_traditional(query, top_k, 'hybrid', session_id, multi_query)
            
            # Ensure query_embedding is 2D for cosine_similarity
            import numpy as np
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate similarities with sentence windows
            window_embeddings = np.array([window['embedding'] for window in self.sentence_windows])
            similarities = cosine_similarity(query_embedding, window_embeddings)[0]
            
            # Get top sentence windows
            top_indices = np.argsort(similarities)[-top_k*3:][::-1]  # Get more windows initially
            
            # Expand to full context and deduplicate by parent document
            seen_docs = set()
            results = []
            
            for idx in top_indices:
                if len(results) >= top_k:
                    break
                    
                window = self.sentence_windows[idx]
                parent_doc_id = window.get('parent_doc_id')
                
                if parent_doc_id not in seen_docs and similarities[idx] > 0.1:
                    seen_docs.add(parent_doc_id)
                    
                    # Create result with expanded context
                    result = {
                        'id': window.get('parent_doc_id', f'sentence_window_{idx}'),
                        'title': window.get('parent_title', 'Unknown Document'),
                        'content': window.get('expanded_content', window.get('sentence', '')),
                        'source': window.get('source', 'Unknown'),
                        'similarity_score': float(similarities[idx]),
                        'rank': len(results) + 1,
                        'search_method': 'sentence_window',
                        'matched_sentence': window.get('sentence', ''),
                        'document_type': window.get('document_type', 'unknown')
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in sentence window search: {e}")
            return self._search_traditional(query, top_k, 'hybrid', session_id, multi_query)
    
    def _build_hierarchical_structures(self):
        """Build hierarchical search structures (document summaries)."""
        if not self.legal_documents:
            return
        
        try:
            print("🔄 Building hierarchical search structures...")
            
            # Group chunks by document/source
            doc_groups = {}
            for i, doc in enumerate(self.legal_documents):
                # Create a key to group related chunks
                source = doc.get('source', 'unknown')
                title = doc.get('title', 'unknown')
                doc_key = f"{source}_{title}"
                
                if doc_key not in doc_groups:
                    doc_groups[doc_key] = {
                        'title': title,
                        'source': source,
                        'filename': doc.get('filename', ''),
                        'document_type': doc.get('document_type', ''),
                        'chunks': [],
                        'chunk_indices': []
                    }
                
                doc_groups[doc_key]['chunks'].append(doc.get('content', ''))
                doc_groups[doc_key]['chunk_indices'].append(i)
            
            # Create summaries for each document
            self.document_summaries = []
            for doc_key, doc_data in doc_groups.items():
                # Create summary by taking first part of each chunk
                summary_parts = []
                total_chars = 0
                for chunk in doc_data['chunks']:
                    if total_chars >= 3000:  # Limit summary length
                        break
                    chunk_part = chunk[:1000]  # First 1000 chars of chunk
                    summary_parts.append(chunk_part)
                    total_chars += len(chunk_part)
                
                summary = ' '.join(summary_parts)
                enhanced_summary = f"Dokument ligjor: {doc_data['title']}. Përmban: {summary}"
                
                self.document_summaries.append({
                    'id': doc_key,
                    'title': doc_data['title'],
                    'source': doc_data['source'],
                    'filename': doc_data['filename'],
                    'document_type': doc_data['document_type'],
                    'summary': enhanced_summary,
                    'chunk_indices': doc_data['chunk_indices']
                })
            
            # Generate embeddings for summaries
            self._generate_summary_embeddings()
            
            print(f"✅ Built hierarchical structures: {len(self.document_summaries)} document summaries")
            
        except Exception as e:
            print(f"Error building hierarchical structures: {e}")
            self.document_summaries = []
    
    def _generate_summary_embeddings(self):
        """Generate embeddings for document summaries."""
        if not self.document_summaries:
            return
        
        try:
            summary_texts = [doc['summary'] for doc in self.document_summaries]
            
            if self.use_google_embeddings:
                self.document_summary_embeddings = self._get_google_embeddings(summary_texts)
            elif self.model:
                self.document_summary_embeddings = self.model.encode(summary_texts)
            
            if hasattr(self, 'document_summary_embeddings') and self.document_summary_embeddings is not None:
                print(f"✅ Generated embeddings for {len(summary_texts)} document summaries")
            else:
                print("❌ Failed to generate summary embeddings")
                
        except Exception as e:
            print(f"Error generating summary embeddings: {e}")
            self.document_summary_embeddings = None
    
    def _build_sentence_structures(self):
        """Build sentence-level search structures."""
        if not self.legal_documents:
            return
        
        try:
            print("🔄 Building sentence window structures...")
            
            self.sentence_windows = []
            
            for doc_idx, doc in enumerate(self.legal_documents):
                content = doc.get('content', '')
                if len(content.strip()) < 50:  # Skip very short content
                    continue
                
                # Split into sentences (simple approach for Albanian)
                sentences = self._split_into_sentences(content)
                
                for sent_idx, sentence in enumerate(sentences):
                    if len(sentence.strip()) < 20:  # Skip very short sentences
                        continue
                    
                    # Create expanded context (sentence + surrounding sentences)
                    start_idx = max(0, sent_idx - 2)
                    end_idx = min(len(sentences), sent_idx + 3)
                    expanded_context = ' '.join(sentences[start_idx:end_idx])
                    
                    # Generate embedding for the sentence
                    if self.use_google_embeddings:
                        sentence_embedding = self._get_google_query_embedding(sentence)
                    elif self.model:
                        sentence_embedding = self.model.encode([sentence])[0]
                    else:
                        continue
                    
                    if sentence_embedding is not None:
                        window = {
                            'sentence': sentence,
                            'expanded_content': expanded_context,
                            'embedding': sentence_embedding,
                            'parent_doc_id': doc.get('id', f'doc_{doc_idx}'),
                            'parent_title': doc.get('title', 'Unknown'),
                            'source': doc.get('source', 'Unknown'),
                            'document_type': doc.get('document_type', 'unknown'),
                            'sentence_index': sent_idx,
                            'doc_index': doc_idx
                        }
                        self.sentence_windows.append(window)
            
            print(f"✅ Built sentence window structures: {len(self.sentence_windows)} sentence windows")
            
        except Exception as e:
            print(f"Error building sentence structures: {e}")
            self.sentence_windows = []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (Albanian-aware)."""
        import re
        
        # Albanian sentence endings
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _enhance_albanian_query(self, query: str) -> str:
        """Enhance Albanian queries with comprehensive legal synonyms and context"""
        enhanced_query = query.lower()
        
        # Comprehensive Albanian legal synonyms for better semantic matching
        legal_synonyms = {
            # Employment & Labor Law
            'punë': 'punë work employment job labor pune punim',
            'pagë': 'pagë salary wage rrogë compensation pagesë',
            'rrogë': 'rrogë salary wage pagë income të ardhura',
            'minimum': 'minimum minim minimale lowest më të ulët bazë',
            'pushim': 'pushim vacation leave holiday pushimi ditë',
            'punëtor': 'punëtor employee worker staff punonjës',
            'punëdhënës': 'punëdhënës employer boss company kompani',
            
            # Criminal Law - Enhanced for better matching
            'dënim': 'dënim punishment penalty sanksion gjykim burgim gjoba',
            'denimi': 'denimi punishment penalty sanksion gjykim burgim gjoba',
            'plagosje': 'plagosje injury wound attack assault sulm dhunë',
            'plagos': 'plagos injure wound attack assault sulm dhunë',
            'armë': 'armë weapon tool vegël instrument mjete ftohtë',
            'arme': 'arme weapon tool vegël instrument mjete ftohtë',
            'ftohtë': 'ftohtë cold steel metalike hekur thikë',
            'vjedhje': 'vjedhje theft stealing robbery grabitje marrje',
            'sanksion': 'sanksion penalty punishment dënim gjobë burgim',
            'gjobë': 'gjobë fine penalty sanksion dënim financiar',
            'krim': 'krim crime criminal penal vepër penale kundërvajtje',
            'penal': 'penal criminal crime kod ligj sanksion dënim',
            'burgim': 'burgim prison jail detention arrest paraburgim',
            'vrasje': 'vrasje murder killing homicide vdekje ekzekutim',
            'dhunë': 'dhunë violence force brutality aggression sulm',
            'sulm': 'sulm attack assault agression dhunë',
            
            # Business & Commercial Law
            'kompani': 'kompani company business shoqëri enterprise biznes',
            'biznes': 'biznes business company kompani shoqëri tregtare',
            'regjistroj': 'regjistroj register establish themelon krijoj',
            'shoqëri': 'shoqëri company business enterprise kompani',
            'tregtare': 'tregtare commercial business trading biznes',
            
            # Family Law
            'martesë': 'martesë marriage wedding bashkëshort familje',
            'divorcë': 'divorcë divorce separation ndarje',
            'fëmijë': 'fëmijë child children kids të mitur',
            'familje': 'familje family household shtëpi',
            'prindër': 'prindër parents mother father',
            
            # Civil Law
            'kontrata': 'kontrata contract agreement marrëveshje',
            'pronë': 'pronë property asset pasuri',
            'të drejta': 'të drejta rights legal ligjore',
            'detyrim': 'detyrim obligation duty responsibility',
            
            # General Legal Terms
            'ligj': 'ligj law legal kod juridik ligjor',
            'kod': 'kod code law ligj juridik',
            'juridik': 'juridik legal law ligj ligjor',
            'gjykatë': 'gjykatë court tribunal drejtësi',
            'avokat': 'avokat lawyer attorney jurist',
            
            # Tax & Finance
            'taksë': 'taksë tax duty detyrim tatim',
            'tatim': 'tatim tax taksë detyrim financiar',
            
            # Constitutional & Administrative
            'kushtetutë': 'kushtetutë constitution basic law',
            'shtet': 'shtet state government qeveri',
            'administrativ': 'administrativ administrative government publik'
        }
        
        # Add relevant synonyms to enhance query
        for albanian_term, synonyms in legal_synonyms.items():
            if albanian_term in enhanced_query:
                enhanced_query += f" {synonyms}"
        
        # Add legal context keywords based on query type
        if any(word in enhanced_query for word in ['sa', 'how much', 'shumë', 'amount']):
            enhanced_query += " sasi amount vlerë"
        
        if any(word in enhanced_query for word in ['si', 'how', 'procedura', 'process']):
            enhanced_query += " procedurë process hapa steps"
        
        if any(word in enhanced_query for word in ['çfarë', 'what', 'cilat', 'which']):
            enhanced_query += " përkufizim definition detaje specifics"
        
        return enhanced_query


# Export for backward compatibility
CloudEnhancedAlbanianLegalRAG = AlbanianLegalRAG
