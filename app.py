"""
Enhanced Albanian Legal RAG System with Cloud LLM Integration
Now uses Groq API for advanced response generation with environment variables
"""

import os
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from datetime import datetime
from scraper import AlbanianLegalScraper
from ai import CloudLLMClient
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

# Detect headless mode (FastAPI)
HEADLESS = os.getenv("RAG_HEADLESS", "0") == "1"

# Set environment variables for PyTorch compatibility
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class CloudEnhancedAlbanianLegalRAG:
    def __init__(self):
        """Initialize the cloud-enhanced RAG system with Google embeddings"""
        # Embedding configuration
        self.use_google_embeddings = True
        self.model_name = 'all-MiniLM-L6-v2'  # Fallback model
        self.model = None
        self.gemini_api_key = None

        # Document storage
        self.legal_documents = []
        self.document_embeddings = None
        # Super-chunking config (character-based)
        self.superchunk_chars = int(os.getenv('RAG_SUPERCHUNK_CHARS', '4500'))
        self.superchunk_overlap = int(os.getenv('RAG_SUPERCHUNK_OVERLAP', '500'))
        # Sparse retrieval
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.scraper = AlbanianLegalScraper(max_docs=10)
        self.cloud_llm = None

        # Configuration from environment variables
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.15'))
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))  # characters for packing
        self.max_chunks_to_return = int(os.getenv('MAX_CHUNKS_TO_RETURN', '5'))
        # Hybrid + reranking
        self.hybrid_alpha = float(os.getenv('RAG_HYBRID_ALPHA', '0.5'))
        self.mmr_lambda = float(os.getenv('RAG_MMR_LAMBDA', '0.5'))
        self.neighbor_expansion = int(os.getenv('RAG_NEIGHBOR_EXPANSION', '1'))
        # Conversation memory (in-process)
        self.memory_store = {}
        # Context packing (extractive) to reduce token usage
        self.context_packing = os.getenv('RAG_CONTEXT_PACKING', '1') == '1'

        # Cache file for embeddings
        self.embeddings_cache_file = 'document_embeddings_cache_google.json'

        # Initialize components
        self._setup_google_api()
        self._load_fallback_model()
        self._load_cloud_llm()
        self._load_all_documents()
        self._generate_embeddings()

    # -----------------------------
    # Index building helpers
    # -----------------------------
    def _build_sparse_index(self, document_texts: List[str]):
        """Build TF-IDF index for sparse retrieval."""
        try:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(document_texts)
            if not HEADLESS:
                st.success("✅ Built TF‑IDF index for hybrid retrieval")
        except Exception as e:
            if not HEADLESS:
                st.warning(f"⚠️ Could not build TF‑IDF index: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def _setup_google_api(self):
        """Setup Google API for embeddings"""
        try:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                if not HEADLESS:
                    st.info("🔄 Google API configured for embeddings...")
                self.use_google_embeddings = True
                if not HEADLESS:
                    st.success("✅ Google embeddings enabled!")
            else:
                if not HEADLESS:
                    st.warning("⚠️ GEMINI_API_KEY not found - using local embeddings")
                self.use_google_embeddings = False
        except Exception as e:
            if not HEADLESS:
                st.error(f"❌ Error setting up Google API: {e}")
            self.use_google_embeddings = False
    
    def _load_fallback_model(self):
        """Load the local sentence transformer model as fallback"""
        if not self.use_google_embeddings:
            try:
                # Lazy import to avoid torch dependency unless needed
                from sentence_transformers import SentenceTransformer  # type: ignore
                if not HEADLESS:
                    st.info("🔄 Loading local AI model...")
                self.model = SentenceTransformer(self.model_name)
                if not HEADLESS:
                    st.success("✅ Local AI model loaded successfully!")
            except Exception as e:
                if not HEADLESS:
                    st.error(f"❌ Error loading local model: {e}")
                    st.info("Local embeddings disabled. Using Google embeddings only.")
                self.model = None

    def _get_google_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings using Google's embedding-001 model"""
        if not self.use_google_embeddings:
            return None
            
        try:
            # Use Google's text-embedding-004 model (latest) — call per text
            embeddings: List[List[float]] = []
            for text in texts:
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                vec = res.get('embedding') if isinstance(res, dict) else getattr(res, 'embedding', None)
                if not vec:
                    raise ValueError("Empty embedding from Google API")
                embeddings.append(vec)

            return np.array(embeddings, dtype=float)
            
        except Exception as e:
            if not HEADLESS:
                st.error(f"❌ Google embeddings error: {e}")
                st.info("Falling back to local embeddings...")
            self.use_google_embeddings = False
            return None

    def _get_google_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for a query using Google's model"""
        if not self.use_google_embeddings:
            return None
            
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"  # Optimized for query embedding
            )
            
            vec = result.get('embedding') if isinstance(result, dict) else getattr(result, 'embedding', None)
            if not vec:
                return None
            return np.array([vec], dtype=float)
            
        except Exception as e:
            if not HEADLESS:
                st.error(f"❌ Google query embedding error: {e}")
            return None
    
    def _load_cloud_llm(self):
        """Initialize cloud LLM client"""
        try:
            if not HEADLESS:
                st.info("🌐 Connecting to cloud LLM...")
            self.cloud_llm = CloudLLMClient("gemini")  # Changed to Gemini as default
            
            if self.cloud_llm.api_key:
                if not HEADLESS:
                    st.success("✅ Cloud LLM connected successfully!")
                    st.info(f"🤖 Using model: {self.cloud_llm.model}")
            else:
                if not HEADLESS:
                    st.warning("⚠️ Cloud LLM API key not found - using fallback responses")
        except Exception as e:
            if not HEADLESS:
                st.error(f"❌ Cloud LLM connection error: {e}")
                st.info("Will use template-based responses as fallback")
            self.cloud_llm = None
    
    def _load_hardcoded_documents(self):
        """Load the original hardcoded legal documents"""
        hardcoded_docs = [
            {
                "id": "family_law_1",
                "title": "Kodi i Familjes - Family Code",
                "source": "Albanian Family Code",
                "content": "Kodi i Familjes rregullon marrëdhëniet familjare, martesën, divorci, dhe të drejtat e fëmijëve në Shqipëri.",
                "content_en": "The Family Code regulates family relationships, marriage, divorce, and children's rights in Albania.",
                "document_type": "hardcoded"
            },
            {
                "id": "property_law_1", 
                "title": "Ligji i Pronësisë - Property Law",
                "source": "Albanian Property Law",
                "content": "Ligji i pronësisë në Shqipëri rregullon të drejtën e pronësisë private, publike dhe transferimin e pronave.",
                "content_en": "Property law in Albania regulates private property rights, public property and property transfers.",
                "document_type": "hardcoded"
            },
            {
                "id": "labor_law_1",
                "title": "Kodi i Punës - Labor Code", 
                "source": "Albanian Labor Code",
                "content": "Kodi i Punës rregullon marrëdhëniet e punës, të drejtat dhe detyrimet e punëdhënësve dhe të punësuarve. Çdo punëtor ka të drejtë për pushim vjetor me pagesë. Pushimi vjetor është gjithsej 28 ditë kalendarike në vit, nga të cilat 14 ditë janë të detyrueshme dhe duhet të merren gjatë vitit. Ditët e tjera mund të transferohen në vitin tjetër me marrëveshje me punëdhënësin.",
                "content_en": "The Labor Code regulates employment relationships, rights and obligations of employers and employees. Every worker has the right to paid annual leave. Annual leave is a total of 28 calendar days per year, of which 14 days are mandatory and must be taken during the year. Other days can be transferred to the next year by agreement with the employer.",
                "document_type": "hardcoded"
            },
            {
                "id": "criminal_law_1",
                "title": "Kodi Penal - Criminal Code",
                "source": "Albanian Criminal Code", 
                "content": "Kodi Penal i Shqipërisë përcakton veprat penale dhe sanksionet përkatëse për to.",
                "content_en": "Albania's Criminal Code defines criminal acts and their corresponding sanctions.",
                "document_type": "hardcoded"
            },
            {
                "id": "business_law_1",
                "title": "Ligji për Biznesin - Business Law",
                "source": "Albanian Business Law",
                "content": "Ligji për biznesin rregullon themelimin, funksionimin dhe mbylljen e shoqërive tregtare në Shqipëri.",
                "content_en": "Business law regulates the establishment, operation and closure of commercial companies in Albania.",
                "document_type": "hardcoded"
            },
            {
                "id": "civil_code_1",
                "title": "Kodi Civil - Civil Code",
                "source": "Albanian Civil Code",
                "content": "Kodi Civil rregullon të drejtat dhe detyrimet civile, kontrata dhe përgjegjësinë civile.",
                "content_en": "The Civil Code regulates civil rights and obligations, contracts and civil liability.",
                "document_type": "hardcoded"
            },
            {
                "id": "constitution_1",
                "title": "Kushtetuta e Shqipërisë - Albanian Constitution", 
                "source": "Albanian Constitution",
                "content": "Kushtetuta është ligji më i lartë i vendit dhe garanton të drejtat dhe liritë themelore të qytetarëve.",
                "content_en": "The Constitution is the highest law of the country and guarantees fundamental rights and freedoms of citizens.",
                "document_type": "hardcoded"
            },
            {
                "id": "tax_law_1",
                "title": "Ligji për Taksat - Tax Law",
                "source": "Albanian Tax Law", 
                "content": "Ligji për taksat rregullon sistemin tatimor, detyrimet tatimore dhe procedurat e mbledhjes së taksave.",
                "content_en": "Tax law regulates the tax system, tax obligations and tax collection procedures.",
                "document_type": "hardcoded"
            }
        ]
        
        return hardcoded_docs
    
    def _load_scraped_documents(self):
        """Load documents scraped from qbz.gov.al"""
        try:
            scraped_docs = self.scraper.get_processed_documents_for_rag()
            if not HEADLESS:
                st.info(f"📄 Loaded {len(scraped_docs)} scraped document chunks")
            # Merge to super-chunks for speed and better cross-paragraph combinations
            merged = self._to_superchunks(scraped_docs, self.superchunk_chars, self.superchunk_overlap)
            if not HEADLESS:
                st.info(f"🧩 Built {len(merged)} super-chunks for scraped docs (≈{self.superchunk_chars} chars each)")
            return merged
        except Exception as e:
            if not HEADLESS:
                st.warning(f"⚠️ Could not load scraped documents: {e}")
            return []
    
    def _load_pdf_documents(self):
        """Load processed PDF documents"""
        try:
            pdf_file = os.path.join("legal_documents", "pdf_rag_documents.json")
            if os.path.exists(pdf_file):
                with open(pdf_file, 'r', encoding='utf-8') as f:
                    pdf_docs = json.load(f)
                
                # Normalize PDF documents to match expected format
                normalized_docs = []
                for doc in pdf_docs:
                    normalized_doc = {
                        'title': doc.get('title', 'Unknown Document'),
                        'content': doc.get('content', ''),
                        'source': doc.get('filename', doc.get('source_url', 'Local PDF')),
                        'url': doc.get('source_url', ''),
                        'document_type': 'local_pdf'
                    }
                    normalized_docs.append(normalized_doc)
                
                if not HEADLESS:
                    st.info(f"📄 Loaded {len(normalized_docs)} PDF document chunks")
                # Merge into super-chunks for better semantic coverage
                merged = self._to_superchunks(normalized_docs, self.superchunk_chars, self.superchunk_overlap)
                if not HEADLESS:
                    st.info(f"🧩 Built {len(merged)} super-chunks for PDF docs (≈{self.superchunk_chars} chars each)")
                return merged
            else:
                if not HEADLESS:
                    st.warning("⚠️ No processed PDF documents found")
                return []
        except Exception as e:
            st.warning(f"⚠️ Could not load PDF documents: {e}")
            return []
    
    def _load_all_documents(self):
        """Load hardcoded, scraped, and PDF documents"""
        # Load hardcoded documents
        hardcoded = self._load_hardcoded_documents()
        
        # Load scraped documents
        scraped = self._load_scraped_documents()
        
        # Load PDF documents
        pdf_docs = self._load_pdf_documents()
        
        # Combine all documents
        self.legal_documents = hardcoded + scraped + pdf_docs
        # Ensure IDs exist for neighbor expansion and caching determinism
        for i, d in enumerate(self.legal_documents):
            if 'id' not in d or not d.get('id'):
                prefix = (d.get('source') or d.get('url') or d.get('title') or 'doc').replace(' ', '_')
                d['id'] = f"{prefix}_chunk_{i}"
        
        if not HEADLESS:
            st.info(
                f"📚 Total documents loaded: {len(self.legal_documents)} "
                f"({len(hardcoded)} hardcoded + {len(scraped)} scraped + {len(pdf_docs)} PDF)"
            )
    
    def _load_embeddings_cache(self):
        """Load cached embeddings if available"""
        cache_file = self.embeddings_cache_file if self.use_google_embeddings else 'document_embeddings_cache_local.json'
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                
                # Check if cache is still valid
                if (cache.get('document_count') == len(self.legal_documents) and 
                    cache.get('embedding_type') == ('google' if self.use_google_embeddings else 'local') and
                    cache.get('chunking', {}) == {
                        'superchunk_chars': self.superchunk_chars,
                        'superchunk_overlap': self.superchunk_overlap
                    }):
                    self.document_embeddings = np.array(cache['embeddings'])
                    if not HEADLESS:
                        st.success(f"✅ Loaded cached {'Google' if self.use_google_embeddings else 'local'} embeddings")
                    return True
            except Exception as e:
                if not HEADLESS:
                    st.warning(f"⚠️ Could not load embedding cache: {e}")
        
        return False
    
    def _save_embeddings_cache(self):
        """Save embeddings to cache"""
        try:
            cache_file = self.embeddings_cache_file if self.use_google_embeddings else 'document_embeddings_cache_local.json'
            
            cache = {
                'embeddings': self.document_embeddings.tolist(),
                'document_count': len(self.legal_documents),
                'embedding_type': 'google' if self.use_google_embeddings else 'local',
                'created_at': datetime.now().isoformat(),
                'chunking': {
                    'superchunk_chars': self.superchunk_chars,
                    'superchunk_overlap': self.superchunk_overlap
                }
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
                
            if not HEADLESS:
                st.success(f"✅ {'Google' if self.use_google_embeddings else 'Local'} embeddings cached for faster loading")
        except Exception as e:
            if not HEADLESS:
                st.warning(f"⚠️ Could not save embedding cache: {e}")
    
    def _generate_embeddings(self):
        """Generate embeddings for all documents using Google or local model"""
        if not self.use_google_embeddings and not self.model:
            if not HEADLESS:
                st.error("❌ No embedding model available - cannot generate embeddings")
            return
        
        if not self.legal_documents:
            if not HEADLESS:
                st.error("❌ No documents loaded - cannot generate embeddings")
            return
        
        # Try to load from cache first
        if self._load_embeddings_cache():
            return
        
        try:
            if self.use_google_embeddings:
                if not HEADLESS:
                    st.info("🔄 Generating Google embeddings for better Albanian legal search...")
                
                # Prepare document texts with enhanced content
                document_texts = []
                for doc in self.legal_documents:
                    # Combine title and content for better context
                    enhanced_text = f"Dokument ligjor: {doc['title']}. Përmbajtja: {doc['content']}"
                    if doc.get('content_en'):
                        enhanced_text += f" English: {doc['content_en']}"
                    document_texts.append(enhanced_text)
                
                # Generate embeddings using Google's model
                self.document_embeddings = self._get_google_embeddings(document_texts)
                
                if self.document_embeddings is not None:
                    self._save_embeddings_cache()
                    if not HEADLESS:
                        st.success(f"✅ Generated Google embeddings for {len(self.legal_documents)} documents")
                else:
                    raise Exception("Failed to generate Google embeddings")
                    
            else:
                # Fallback to local embeddings
                if not HEADLESS:
                    st.info("🔄 Generating local embeddings...")
                
                document_texts = []
                for doc in self.legal_documents:
                    combined_text = f"{doc['content']} {doc.get('content_en', '')}"
                    document_texts.append(combined_text)
                
                self.document_embeddings = self.model.encode(document_texts)
                self._save_embeddings_cache()
                if not HEADLESS:
                    st.success(f"✅ Generated local embeddings for {len(self.legal_documents)} documents")

            # Build sparse index for hybrid retrieval using the same document_texts prepared above
            try:
                if 'document_texts' not in locals():
                    # Re-create document_texts if not present
                    document_texts = []
                    for doc in self.legal_documents:
                        enhanced_text = f"{doc['title']} — {doc['content']} {doc.get('content_en', '')}"
                        document_texts.append(enhanced_text)
                self._build_sparse_index(document_texts)
            except Exception:
                pass
            
        except Exception as e:
            if not HEADLESS:
                st.error(f"❌ Error generating embeddings: {e}")
            
            # Try fallback to local model if Google fails
            if self.use_google_embeddings and self.model:
                if not HEADLESS:
                    st.info("🔄 Trying local embeddings as fallback...")
                try:
                    self.use_google_embeddings = False
                    self._generate_embeddings()
                except Exception as fallback_error:
                    if not HEADLESS:
                        st.error(f"❌ Fallback also failed: {fallback_error}")
                    self.document_embeddings = None
    
    def search_documents(self, query: str, top_k: int = 3, mode: str = 'hybrid', session_id: Optional[str] = None, multi_query: bool = True):
        """Search for relevant documents using hybrid retrieval and MMR diversification.

        Args:
            query: user query
            top_k: number of chunks to return
            mode: 'hybrid' | 'embedding' | 'sparse'
            session_id: optional conversation id for context-aware expansion
            multi_query: expand query into variants and aggregate
        """
        if self.document_embeddings is None:
            return []
        
        try:
            # Enhance Albanian queries for better semantic search
            enhanced_query = self._enhance_albanian_query(query)

            # Include brief memory to disambiguate retrieval (last 2 user turns)
            memory_context = self.get_memory_context(session_id) if session_id else ""
            if memory_context:
                enhanced_query = f"{enhanced_query} \nKontekst bisede: {memory_context}"
            
            # Generate query embedding using appropriate method
            if self.use_google_embeddings:
                query_embedding = self._get_google_query_embedding(enhanced_query)
                if query_embedding is None:
                    # Fallback to local model if available
                    if self.model:
                        query_embedding = self.model.encode([enhanced_query])
                    else:
                        if not HEADLESS:
                            st.error("❌ No embedding method available for query")
                        return []
            else:
                if not self.model:
                    if not HEADLESS:
                        st.error("❌ Local model not loaded")
                    return []
                query_embedding = self.model.encode([enhanced_query])
            
            # Calculate dense similarities
            dense_sim = cosine_similarity(query_embedding, self.document_embeddings)[0]

            # Calculate sparse similarities if enabled
            sparse_sim = None
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None and mode in ('hybrid', 'sparse'):
                try:
                    q_vec = self.tfidf_vectorizer.transform([enhanced_query])
                    # cosine similarity for sparse vectors
                    # sklearn doesn't provide direct; use dot product as TF-IDF is L2-normalized by default
                    sparse_sim = (q_vec @ self.tfidf_matrix.T).toarray()[0]
                except Exception:
                    sparse_sim = None

            # Combine
            if mode == 'embedding' or sparse_sim is None:
                combined_sim = dense_sim
            elif mode == 'sparse':
                combined_sim = sparse_sim
            else:
                alpha = self.hybrid_alpha
                combined_sim = alpha * dense_sim + (1 - alpha) * sparse_sim
            
            # Multi-query expansion (lightweight): include original + 1-2 heuristic rewrites
            if multi_query:
                rewrites = self._expand_queries(enhanced_query)
                for rw in rewrites:
                    if rw.strip() == enhanced_query.strip():
                        continue
                    # Dense
                    if self.use_google_embeddings:
                        rw_emb = self._get_google_query_embedding(rw)
                        if rw_emb is None and self.model:
                            rw_emb = self.model.encode([rw])
                    else:
                        rw_emb = self.model.encode([rw]) if self.model else None
                    if rw_emb is not None:
                        rw_dense = cosine_similarity(rw_emb, self.document_embeddings)[0]
                        if mode == 'embedding' or (self.tfidf_vectorizer is None or self.tfidf_matrix is None):
                            combined_sim = np.maximum(combined_sim, rw_dense)
                        else:
                            # Sparse for rewrite
                            try:
                                qv = self.tfidf_vectorizer.transform([rw])
                                rw_sparse = (qv @ self.tfidf_matrix.T).toarray()[0]
                                alpha = self.hybrid_alpha if mode == 'hybrid' else 0.0
                                rw_comb = alpha * rw_dense + (1 - alpha) * rw_sparse
                                combined_sim = np.maximum(combined_sim, rw_comb)
                            except Exception:
                                combined_sim = np.maximum(combined_sim, rw_dense)

            # MMR selection to diversify
            selected_indices = self._mmr_select(query_embedding[0], self.document_embeddings, combined_sim, k=top_k, lambda_param=self.mmr_lambda)

            # Optionally expand with neighbors from same document (by chunk id suffix)
            expanded_indices = self._expand_neighbors(selected_indices, self.neighbor_expansion)
            
            results = []
            max_similarity = 0
            
            # Find the maximum similarity to make adaptive threshold decisions
            for idx in expanded_indices:
                similarity = float(combined_sim[idx])
                if similarity > max_similarity:
                    max_similarity = similarity
                
                # Use adaptive threshold with improved settings for Google embeddings
                base_threshold = self.similarity_threshold
                if self.use_google_embeddings:
                    # Google embeddings have different scale, adjust threshold
                    base_threshold = max(0.1, self.similarity_threshold * 0.8)
                
                adaptive_threshold = max(base_threshold, max_similarity * 0.7) if max_similarity > 0.2 else base_threshold
                
                if similarity > adaptive_threshold or (len(results) == 0 and idx == expanded_indices[0] and similarity > 0.1):
                    doc = self.legal_documents[idx].copy()
                    doc['similarity'] = similarity
                    results.append(doc)
            
            # Store max similarity for fallback decision
            self.last_search_max_similarity = max_similarity
            
            return results
            
        except Exception as e:
            if not HEADLESS:
                st.error(f"❌ Search error: {e}")
            return []

    def _expand_neighbors(self, indices: List[int], window: int) -> List[int]:
        """Expand selected indices by including +/- window neighbors within same document chunks when ids suggest ordering."""
        if window <= 0:
            return indices
        expanded: set[int] = set(indices)
        for idx in indices:
            base_id = self.legal_documents[idx].get('id', '')
            # Extract numeric chunk suffix if present
            chunk_num = None
            prefix = None
            if '_chunk_' in base_id:
                try:
                    chunk_num = int(base_id.split('_chunk_')[-1])
                    prefix = base_id.rsplit('_chunk_', 1)[0]
                except Exception:
                    chunk_num = None
            if chunk_num is not None and prefix is not None:
                # search neighbors in list for same prefix and num +/- window
                for j, d in enumerate(self.legal_documents):
                    did = d.get('id', '')
                    if did.startswith(prefix + '_chunk_'):
                        try:
                            n = int(did.split('_chunk_')[-1])
                            if abs(n - chunk_num) <= window:
                                expanded.add(j)
                        except Exception:
                            continue
        return list(expanded)

    def _mmr_select(self, query_vec: np.ndarray, doc_embs: np.ndarray, sim_scores: np.ndarray, k: int, lambda_param: float = 0.5) -> List[int]:
        """Maximal Marginal Relevance selection to diversify top-k results."""
        if k <= 0:
            return []
        selected: List[int] = []
        candidates = set(range(len(sim_scores)))
        # Precompute doc-doc similarities lazily when needed
        doc_norms = np.linalg.norm(doc_embs, axis=1) + 1e-10
        while len(selected) < k and candidates:
            if not selected:
                # pick max sim
                idx = int(np.argmax(sim_scores))
                selected.append(idx)
                candidates.discard(idx)
                continue
            best_idx: Optional[int] = None
            best_score = -1e9
            for i in list(candidates):
                # relevance
                rel = sim_scores[i]
                # redundancy: max similarity to already selected
                red = 0.0
                for s in selected:
                    # cosine between doc i and doc s using precomputed embs
                    dot = float(np.dot(doc_embs[i], doc_embs[s]))
                    sim = dot / (doc_norms[i] * doc_norms[s])
                    if sim > red:
                        red = sim
                mmr_score = lambda_param * rel - (1 - lambda_param) * red
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            if best_idx is None:
                break
            selected.append(best_idx)
            candidates.discard(best_idx)
        return selected

    def _expand_queries(self, enhanced_query: str) -> List[str]:
        """Heuristic multi-query rewrites to improve recall without extra LLM calls."""
        q = enhanced_query
        rewrites = []
        # Swap some common Albanian synonyms order
        rewrites.append(q.replace('procedurë', 'hapa'))
        rewrites.append(q.replace('dënim', 'sanksion'))
        rewrites.append(q.replace('pagë', 'rrogë'))
        # English hint variant
        rewrites.append(q + " in English: law code article section penalty definition procedure")
        # Deduplicate and limit
        uniq: List[str] = []
        for r in rewrites:
            if r not in uniq:
                uniq.append(r)
        return uniq[:3]
    
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
    
    def search_legal_documents(self, query: str, session_id: Optional[str] = None) -> str:
        """Main interface method for searching and responding to legal queries"""
        try:
            # Search for relevant documents
            relevant_docs = self.search_documents(query, top_k=self.max_chunks_to_return, mode='hybrid', session_id=session_id)
            
            # Compact context to fit budget and accelerate LLM
            docs_for_llm = self._compact_docs(relevant_docs, query, self.max_context_length) if self.context_packing else relevant_docs
            
            # Generate comprehensive response
            response = self.generate_response(query, docs_for_llm, session_id=session_id)

            # Persist in-memory conversation if session provided
            if session_id:
                self.add_to_memory(session_id, role='user', content=query)
                self.add_to_memory(session_id, role='assistant', content=response)
            
            return response
            
        except Exception as e:
            if not HEADLESS:
                st.error(f"❌ Search error: {e}")
            return f"❌ **Gabim në kërkim**: {e}\n\n⚖️ Ju lutem provoni përsëri ose kontaktoni një jurist."
    
    def _find_similar_documents(self, query: str, top_k: int = 5):
        """Find similar documents - wrapper around search_documents for compatibility"""
        docs = self.search_documents(query, top_k)
        
        # Return in the format expected by test files: (doc, similarity) tuples
        return [(doc, doc.get('similarity', 0.0)) for doc in docs]
    
    def generate_response(self, query: str, relevant_docs: list, session_id: Optional[str] = None) -> str:
        """Generate a comprehensive response using cloud LLM or fallback"""
        # Build memory context for LLM
        memory_context = self.get_memory_context(session_id) if session_id else ""
        # Trim memory context to keep prompts small
        if len(memory_context) > 1000:
            memory_context = memory_context[-1000:]
        if self.cloud_llm and self.cloud_llm.api_key:
            try:
                # Use cloud LLM for advanced response generation
                return self.cloud_llm.generate_response(query, relevant_docs, context=memory_context)
            except Exception as e:
                if not HEADLESS:
                    st.warning(f"⚠️ Cloud LLM error, using fallback: {e}")
                return self._fallback_response(query, relevant_docs)
        else:
            # Use fallback response
            return self._fallback_response(query, relevant_docs)

    # -----------------------------
    # Super-chunking and context packing helpers
    # -----------------------------
    def _to_superchunks(self, docs: List[Dict], chunk_chars: int, overlap_chars: int) -> List[Dict]:
        """Merge chunks into larger super-chunks, grouped per source/filename for coherence."""
        if chunk_chars <= 0 or not docs:
            return docs
        # Group docs by a stable key (source or title)
        groups: Dict[str, List[Dict]] = {}
        for d in docs:
            key = (d.get('source') or d.get('url') or d.get('title') or 'doc').strip()
            groups.setdefault(key, []).append(d)
        merged: List[Dict] = []
        for key, items in groups.items():
            # keep order
            buffer = ""
            meta = items[0]
            part = 0
            def flush(buf: str, m: Dict, p: int):
                if not m or not buf:
                    return
                item = {
                    'title': m.get('title', 'Document'),
                    'content': buf,
                    'source': m.get('source', key),
                    'url': m.get('url', ''),
                    'document_type': m.get('document_type', 'local_pdf'),
                    'id': f"{(key or 'doc').replace(' ', '_')}_chunk_{p}"
                }
                merged.append(item)
            for d in items:
                text = d.get('content', '')
                if not buffer:
                    buffer = text
                else:
                    buffer += "\n\n" + text
                while len(buffer) > chunk_chars:
                    to_take = buffer[:chunk_chars]
                    flush(to_take.strip(), meta, part)
                    part += 1
                    start = max(0, len(to_take) - overlap_chars)
                    buffer = buffer[start:]
            flush(buffer.strip(), meta, part)
        return merged

    def _compact_docs(self, docs: List[Dict], query: str, budget_chars: int) -> List[Dict]:
        """Extractive packing: select top sentences from each doc to fit within a character budget.

        Keeps document metadata; reduces content length to accelerate LLM and reduce tokens.
        """
        if budget_chars <= 0 or not docs:
            return docs
        # Split into sentences (simple heuristic)
        packed: List[Dict] = []
        remaining = max(500, budget_chars)  # ensure some reasonable space
        # Distribute budget roughly evenly across docs
        per_doc = max(250, remaining // max(1, len(docs)))
        for d in docs:
            content = d.get('content', '')
            excerpt = self._select_top_sentences(content, query, max_chars=per_doc)
            new_d = d.copy()
            if excerpt:
                new_d['content'] = excerpt
            packed.append(new_d)
        return packed

    def _select_top_sentences(self, text: str, query: str, max_chars: int = 600) -> str:
        """Score sentences by TF-IDF against the query and return the best within max_chars."""
        if not text:
            return text
        # Naive sentence split
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        if not sentences:
            return text[:max_chars]
        # Build corpus of sentences + query
        corpus = sentences + [query]
        try:
            vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)).fit_transform(corpus)
            q_vec = vec[-1]
            s_mat = vec[:-1]
            scores = (s_mat @ q_vec.T).toarray().ravel()
        except Exception:
            # Fallback: length-based
            scores = np.array([len(s) for s in sentences], dtype=float)
        order = np.argsort(-scores)
        picked: List[str] = []
        total = 0
        for idx in order:
            s = sentences[idx]
            if s.endswith('.'):
                sent = s
            else:
                sent = s + '.'
            if total + len(sent) > max_chars:
                if total < max_chars * 0.6:  # allow one longer sentence to avoid empty
                    picked.append(sent[: max(0, max_chars - total)])
                break
            picked.append(sent)
            total += len(sent)
            if total >= max_chars:
                break
        return ' '.join(picked) if picked else text[:max_chars]

    # -----------------------------
    # Conversation memory
    # -----------------------------
    def add_to_memory(self, session_id: str, role: str, content: str, max_turns: int = 10):
        if not session_id:
            return
        hist = self.memory_store.setdefault(session_id, [])
        hist.append({"role": role, "content": content})
        # cap memory
        if len(hist) > 2 * max_turns:
            self.memory_store[session_id] = hist[-2 * max_turns :]

    def get_memory_context(self, session_id: Optional[str], last_n: int = 4) -> str:
        if not session_id:
            return ""
        hist = self.memory_store.get(session_id, [])
        if not hist:
            return ""
        # Take last_n turns
        snippet = hist[-last_n:]
        formatted = []
        for h in snippet:
            role = 'Përdoruesi' if h['role'] == 'user' else 'Asistenti'
            formatted.append(f"{role}: {h['content']}")
        return "\n".join(formatted)
    
    def _fallback_response(self, query: str, relevant_docs: list) -> str:
        """Enhanced fallback response with intelligent general legal knowledge"""
        
        # If we have relevant documents with decent similarity, use them
        if relevant_docs and any(doc.get('similarity', 0) > 0.3 for doc in relevant_docs):
            return self._generate_document_based_response(query, relevant_docs)
        
        # If no good documents found, provide intelligent general legal guidance
        return self._generate_general_legal_response(query)
    
    def _generate_document_based_response(self, query: str, relevant_docs: list) -> str:
        """Generate response based on found documents"""
        response = "📋 **Përgjigje bazuar në dokumentet ligjore shqiptare:**\n\n"
        
        # Analyze query to provide more specific introduction
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['sa', 'how much', 'shumë']):
            response += "💰 **Informacion mbi sasi dhe vlera:**\n\n"
        elif any(word in query_lower for word in ['si', 'how', 'procedura']):
            response += "📋 **Procedura dhe hapa:**\n\n"
        elif any(word in query_lower for word in ['çfarë', 'what', 'cilat']):
            response += "📖 **Përkufizime dhe detaje:**\n\n"
        else:
            response += "⚖️ **Informacion ligjor:**\n\n"
        
        # Add document information
        for i, doc in enumerate(relevant_docs, 1):
            similarity = doc.get('similarity', 0)
            
            # Determine document source emoji
            if doc.get('document_type') == 'local_pdf':
                doc_emoji = "📄"
            elif doc.get('document_type') == 'scraped_legal_document':
                doc_emoji = "🌐"
            else:
                doc_emoji = "📚"
            
            response += f"{doc_emoji} **{doc.get('title', 'Dokument Ligjor')}** (Relevanca: {similarity:.0%})\n"
            
            # Add content excerpt
            content = doc.get('content', '')
            if content:
                # Truncate content intelligently
                if len(content) > 300:
                    content = content[:300] + "..."
                response += f"📖 {content}\n\n"
        
        response += "---\n"
        response += "⚖️ **Shënim**: Ky informacion është për qëllime informuese. "
        response += "Për këshilla të detajuara ligjore, konsultohuni me një jurist të kualifikuar."
        
        return response
    
    def _generate_general_legal_response(self, query: str) -> str:
        """Generate intelligent general legal response when no specific documents match"""
        query_lower = query.lower()
        
        # Salary/wage related queries
        if any(word in query_lower for word in ['salary', 'pagë', 'rrogë', 'minimum', 'minim']):
            return self._get_salary_general_response()
        
        # Employment/labor queries
        elif any(word in query_lower for word in ['punë', 'employment', 'work', 'job', 'pushim', 'vacation']):
            return self._get_employment_general_response()
        
        # Business registration queries
        elif any(word in query_lower for word in ['kompani', 'biznes', 'regjistroj', 'business', 'company']):
            return self._get_business_general_response()
        
        # Criminal law queries
        elif any(word in query_lower for word in ['vjedhje', 'theft', 'krim', 'crime', 'sanksion', 'penalty']):
            return self._get_criminal_general_response()
        
        # Family law queries
        elif any(word in query_lower for word in ['martesë', 'marriage', 'divorcë', 'divorce', 'familje', 'family']):
            return self._get_family_general_response()
        
        # General legal guidance
        else:
            return self._get_generic_legal_response(query)
    
    def _get_salary_general_response(self) -> str:
        """General response for salary/wage questions"""
        return """
💰 **Paga Minimale në Shqipëri - Informacion i Përgjithshëm**

📊 **Paga minimale aktuale në Shqipëri**:
• **34,000 lekë** në muaj (rreth 340 Euro)
• Kjo është paga minimale për punëtorët me kohë të plotë
• Përditësohet periodikisht nga Qeveria

📋 **Detaje të rëndësishme**:
• Të gjithë punëtorët kanë të drejtë për pagën minimale
• Punëdhënësi nuk mund të paguajë më pak se paga minimale
• Kjo përfshin të gjitha kontributet dhe taksat

⚖️ **Bazë ligjore**:
• Rregullohet nga Kodi i Punës i Shqipërisë
• Vendime specifike të Këshillit të Ministrave
• Ligje të veçanta për sektorë të ndryshëm

💡 **Për informacion të përditësuar**:
• Kontaktoni Inspektoratin e Punës
• Consultoni me një jurist të specializuar
• Vizitoni faqen zyrtare të Ministrisë së Financave

---
⚠️ **Shënim**: Informacioni mund të jetë ndryshuar. Verifikoni me burime zyrtare.
"""
    
    def _get_employment_general_response(self) -> str:
        """General response for employment questions"""
        return """
💼 **Të Drejtat e Punëtorëve në Shqipëri**

📝 **Të drejta themelore**:
• Kontrata e punës me shkrim
• Pagesë për punën e kryer
• Ambiente të sigurta pune
• Pushim vjetor me pagesë (28 ditë)
• Pushim javor (24 orë të njëpasnjëshme)

⏰ **Kohë pune**:
• Maksimumi 8 orë në ditë, 40 orë në javë
• Pauza gjatë ditës së punës
• Pagesa shtesë për orët jashtë programit

🏥 **Sigurime dhe mbrojtje**:
• Sigurimi shëndetësor
• Sigurime shoqërore
• Kompensim për aksidente në punë
• Lejet e sëmundjes

⚖️ **Bazë ligjore**: Kodi i Punës i Shqipërisë

💡 **Për probleme**: Kontaktoni Inspektoratin e Punës ose një jurist.
"""
    
    def _get_business_general_response(self) -> str:
        """General response for business registration questions"""
        return """
🏢 **Regjistrimi i Biznesit në Shqipëri**

📋 **Hapat kryesorë**:
1. **Zgjedhja e emrit** të biznesit
2. **Përcaktimi i formës juridike** (SHPK, SHA, etj.)
3. **Regjistrimi në QKB** (Qendra Kombëtare e Biznesit)
4. **Marrja e licencave** të nevojshme
5. **Regjistrimi tatimor** në Drejtorinë e Tatimeve

💰 **Kostot**:
• Taksa e regjistrimit: rreth 2,000-5,000 lekë
• Kosto notari (nëse nevojitet)
• Taksa për licencat specifike

📍 **Ku të shkoni**:
• **QKB** - Qendra Kombëtare e Biznesit
• **Zyrat e bashkive** për licencat lokale
• **Drejtoria e Tatimeve** për regjistrimin tatimor

⏱️ **Kohëzgjatja**: 1-3 ditë pune për procedurat bazë

💡 **Këshillë**: Konsultohuni me një kontabilist ose jurist për procedurat specifike.
"""
    
    def _get_criminal_general_response(self) -> str:
        """General response for criminal law questions"""
        return """
⚖️ **Ligji Penal në Shqipëri - Informacion i Përgjithshëm**

📖 **Kodi Penal** përcakton:
• Veprat penale dhe sanksionet
• Procedurat gjyqësore
• Të drejtat e të akuzuarit
• Llojet e dënimeve

🗡️ **Plagosje me Armë të Ftohtë**:
• **Plagosje e thjeshtë**: Burgim deri në 2 vjet ose gjobë
• **Plagosje e rëndë**: Burgim nga 2 deri në 8 vjet
• **Plagosje shumë të rëndë**: Burgim nga 5 deri në 15 vjet
• **Me pasoja vdekjeje**: Burgim nga 10 deri në 20 vjet

⚔️ **Mbajtja e Armëve të Ftohtë**:
• Gjoba ose burgim deri në 3 muaj për mbajtje pa leje
• Përdorimi në vepra penale rëndon dënimin

🏛️ **Llojet e dënimeve**:
• **Burgim** (nga 15 ditë deri në 25 vjet)
• **Gjoba** (sasi të ndryshme)
• **Punë në dobi të përgjithshme**
• **Heqja e të drejtave** civile

⚠️ **Vepra të rënda**:
• Vrasja, grabitja, trafikimi
• Korrupsioni, pastrimi i parave
• Vepra kundër sigurisë publike

🔍 **Procedura penale**:
• Hetimi, akuzimi, gjykimi
• E drejta për mbrojtje ligjore
• Ankimimi i vendimeve

💡 **Për ndihmë ligjore**: Kontaktoni një avokat penal të kualifikuar.

---
⚠️ **Shënim**: Dënimet e sakta varen nga rrethanat e veçanta të rastit. Konsultohuni me një jurist për raste specifike.
"""
    
    def _get_family_general_response(self) -> str:
        """General response for family law questions"""
        return """
👨‍👩‍👧‍👦 **E Drejta e Familjes në Shqipëri**

💍 **Martesa**:
• Mosha minimale: 18 vjet (me përjashtime 16 vjet)
• Dokumentet e nevojshme
• Ceremonia civile (e detyrueshme)
• Ceremonia fetare (opsionale)

💔 **Divorci**:
• **Me marrëveshje**: procedurë e thjeshtë
• **Me mosmarrëveshje**: procedurë gjyqësore
• Ndarje e pasurisë së përbashkët
• Kujdestaria e fëmijëve

👶 **Të drejtat e fëmijëve**:
• E drejta për kujdes dhe edukim
• Alimentet nga prindërit
• Mbrojtja nga dhuna
• Përfaqësimi ligjor

🏠 **Pasuria familjare**:
• Pasuria e përbashkët dhe individuale
• Trashëgimia dhe testamenti
• Të drejtat e banimit

⚖️ **Bazë ligjore**: Kodi i Familjes së Shqipërisë

💡 **Për ndihmë**: Konsultohuni me një jurist familjar.
"""
    
    def _get_generic_legal_response(self, query: str) -> str:
        """Generic legal response for unmatched queries"""
        return f"""
🤔 **Pyetja juaj**: "{query}"

⚖️ **Sistemit Ligjor Shqiptar** përfshin:

📚 **Kodet kryesore**:
• **Kodi Civil** - të drejtat civile, kontratat
• **Kodi Penal** - veprat penale, sanksionet  
• **Kodi i Punës** - marrëdhëniet e punës
• **Kodi i Familjes** - martesa, divorci, fëmijët
• **Kodi Doganor** - importi, eksporti
• **Kushtetuta** - të drejtat themelore

🏛️ **Institucionet kryesore**:
• Gjykatat e shkallës së parë dhe të dytë
• Gjykata e Lartë
• Gjykata Kushtetuese
• Prokuroria e Përgjithshme

💡 **Për pyetjen tuaj specifike**:
• Reformuloni pyetjen me fjalë kyçe të qarta
• Përdorni terma specifike ligjorë
• Kontaktoni një jurist të specializuar
• Vizitoni faqen qbz.gov.al për ligjet e plota

📞 **Ndihmë ligjore falas**: 0800 8080 (Qendra e Ndihmës Ligjore)

---
⚠️ **Shënim**: Për këshilla të detajuara ligjore, konsultohuni me një jurist të kualifikuar.
"""
    
    def _get_vacation_response(self, relevant_docs: list) -> str:
        """Generate specific response for vacation-related queries"""
        response = "🏖️ **Informacion mbi Pushimin Vjetor në Shqipëri**\n\n"
        
        response += "Sipas Kodit të Punës së Shqipërisë:\n\n"
        response += "📅 **Pushimi Vjetor Total**: 28 ditë kalendarike në vit\n"
        response += "⚠️ **Pushimi i Detyrueshëm**: 14 ditë që duhet të merren gjatë vitit\n"
        response += "🔄 **Pushimi Opsional**: 14 ditë të tjera mund të transferohen në vitin tjetër\n"
        response += "💰 **Pushimi me Pagesë**: Po, pushimi vjetor është me pagesë të plotë\n\n"
        
        response += "📋 **Detaje të Rëndësishme**:\n"
        response += "• Të gjithë punëtorët kanë të drejtë për pushim vjetor\n"
        response += "• Pushimi planifikohet në marrëveshje me punëdhënësin\n"
        response += "• 14 ditët e detyrueshme nuk mund të zëvendësohen me pagesë\n"
        response += "• Ditët e tjera mund të akumulohen me marrëveshje\n\n"
        
        # Add relevant document excerpts if available
        if relevant_docs:
            for doc in relevant_docs[:2]:
                if any(term in doc.get('content', '').lower() for term in ['pushim', 'vacation', 'punë', 'labor']):
                    doc_type = "🌐" if doc.get('document_type') == 'scraped_legal_document' else "📚"
                    response += f"{doc_type} **Burim**: {doc['title']}\n"
                    response += f"📖 {doc['content'][:200]}...\n\n"
        
        response += "\n---\n⚖️ **Këshillë Ligjore**: Për situata të veçanta ose mosmarrëveshje, "
        response += "konsultohuni me një jurist të specializuar në të drejtën e punës."
        
        return response
    
    def _get_no_results_response(self) -> str:
        """Response when no relevant documents are found"""
        return """
        🤔 **Nuk u gjetën dokumente specifike për këtë pyetje.**
        
        Ju lutem provoni:
        • Përdorni fjalë kyçe të tjera
        • Bëni pyetjen më të qartë
        • Kontrolloni drejtshkrimin
        
        📚 Ose eksploro temat e disponueshme:
        • Kodi i Familjes
        • Ligji i Pronësisë  
        • Kodi i Punës
        • Kodi Penal
        • Ligji për Biznesin
        • Kodi Civil
        • Kushtetuta e Shqipërisë
        • Ligji për Taksat
        
        Gjithashtu disponojmë dokumente të përditësuara nga qbz.gov.al
        """
    
    def scrape_new_documents(self):
        """Scrape new documents and update the system"""
        with st.spinner("🔄 Duke shkarkuar dokumente të reja nga qbz.gov.al..."):
            try:
                # Scrape new documents
                found_docs = self.scraper.scrape_qbz_search(['ligj', 'kod', 'vendim'])
                
                if found_docs:
                    st.success(f"📥 U shkarkuan {len(found_docs)} dokumente të reja")
                    
                    # Process the documents
                    with st.spinner("📖 Duke përpunuar dokumentet..."):
                        processed = self.scraper.process_downloaded_documents()
                        st.success(f"✅ U përpunuan {len(processed)} dokumente")
                    
                    # Reload all documents
                    self._load_all_documents()
                    
                    # Regenerate embeddings
                    self._generate_embeddings()
                    
                    st.success("🎉 Sistemi u përditësua me dokumente të reja!")
                    return True
                else:
                    st.info("ℹ️ Nuk u gjetën dokumente të reja")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Gabim gjatë shkarkimit: {e}")
                return False


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Albanian Legal RAG System", 
        page_icon="⚖️",
        layout="wide"
    )
    
    # Header
    st.title("⚖️ Sistemi Shqiptar i Ligjit me AI")
    st.markdown("### 🤖 Intelligent Legal Research Assistant for Albanian Law")
    st.markdown("#### 🌐 Now powered by advanced cloud AI (Groq API)")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        with st.spinner("🔄 Duke inicializuar sistemin..."):
            st.session_state.rag_system = CloudEnhancedAlbanianLegalRAG()
    
    rag_system = st.session_state.rag_system
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Kontrollet e Sistemit")
        
        # Document statistics
        total_docs = len(rag_system.legal_documents)
        hardcoded_count = len([d for d in rag_system.legal_documents if d.get('document_type') == 'hardcoded'])
        scraped_count = total_docs - hardcoded_count
        
        st.metric("📚 Total Dokumente", total_docs)
        st.metric("📖 Dokumente Bazë", hardcoded_count)
        st.metric("🌐 Dokumente të Shkarkuara", scraped_count)
        
        # Cloud LLM status
        st.divider()
        st.subheader("🌐 Cloud AI Status")
        if rag_system.cloud_llm and rag_system.cloud_llm.api_key:
            st.success("✅ Cloud AI Connected")
            st.info(f"🤖 Model: {rag_system.cloud_llm.model}")
        else:
            st.warning("⚠️ Using Fallback Responses")
        
        st.divider()
        
        # Scraping controls
        st.subheader("🔄 Përditësim Dokumentesh")
        
        if st.button("🌐 Shkarko Dokumente të Reja", type="primary"):
            rag_system.scrape_new_documents()
        
        st.caption("Shkarkon dokumente të reja nga qbz.gov.al")
        
        st.divider()
        
        # Help section
        st.subheader("❓ Si të Përdorni")
        st.write("""
        1. **Shkruani pyetjen tuaj** në shqip ose anglisht
        2. **Shtypni Enter** ose klikoni Kërko
        3. **Lexoni përgjigjen** e bazuar në ligjet shqiptare
        4. **Përdorni "Shkarko Dokumente"** për përditësime
        
        **🌐 E re**: Përgjigjet tani gjenerohen nga AI i avancuar!
        """)
        
        # Examples
        st.subheader("💡 Shembuj Pyetjesh")
        example_queries = [
            "Sa dit pushimi kam në punë të detyrueshme?",
            "Si themeloj një biznes në Shqipëri?",
            "Cilat janë të drejtat e fëmijëve?",
            "Si funksionon procedura e divorcit?",
            "Çfarë taksash duhet të paguaj?"
        ]
        
        for query in example_queries:
            if st.button(f"💬 {query}", key=f"example_{hash(query)}"):
                st.session_state.example_query = query
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Bëni Pyetjen Tuaj")
        
        # Query input
        default_query = st.session_state.get('example_query', '')
        query = st.text_area(
            "Shkruani pyetjen tuaj ligjore:",
            value=default_query,
            placeholder="P.sh. Sa dit pushimi kam në punë të detyrueshme?",
            height=100
        )
        
        # Clear example query after use
        if 'example_query' in st.session_state:
            del st.session_state.example_query
        
        # Search button
        if st.button("🔍 Kërko Përgjigje", type="primary") or query:
            if query.strip():
                with st.spinner("🔍 Duke kërkuar në dokumentet ligjore..."):
                    # Search for relevant documents
                    relevant_docs = rag_system.search_documents(query, top_k=3)
                    
                    # Generate response
                    with st.spinner("🤖 Duke gjeneruar përgjigje me AI..."):
                        response = rag_system.generate_response(query, relevant_docs)
                    
                    # Display response
                    st.subheader("📋 Përgjigja")
                    st.markdown(response)
                    
                    # Show source documents
                    if relevant_docs:
                        with st.expander("📚 Dokumentet e Përdorura", expanded=False):
                            for i, doc in enumerate(relevant_docs, 1):
                                # Determine document type with safe access
                                doc_type = "🌐 Dokument i shkarkuar" if doc.get('document_type') == 'scraped_legal_document' else \
                                          "� PDF Document" if doc.get('document_type') == 'local_pdf' else \
                                          "�📚 Dokument bazë"
                                
                                st.write(f"**{i}. {doc.get('title', 'Unknown Title')}** ({doc_type})")
                                st.write(f"Relevanca: {doc.get('similarity', 0):.2%}")
                                
                                # Safe access to source field
                                source = doc.get('source', doc.get('filename', doc.get('source_url', 'Unknown Source')))
                                st.write(f"Burimi: {source}")
                                
                                if doc.get('url'):
                                    st.write(f"URL: {doc['url']}")
                                st.write("---")
                    else:
                        # No relevant documents found
                        st.info("ℹ️ **Asnjë dokument specifik nuk u gjet për këtë pyetje**")
                        st.write("Përgjigja e mësipërme është bazuar në njohuritë e përgjithshme për të drejtën shqiptare.")
                        st.write("💡 **Sugjerim**: Kontaktoni një jurist të kualifikuar për këshilla të detajuara ligjore.")
            else:
                st.warning("⚠️ Ju lutem shkruani një pyetje")
    
    with col2:
        st.header("📊 Statistika Sistemi")
        
        # System status
        model_status = "✅ I ngarkuar" if rag_system.model else "❌ Gabim"
        embeddings_status = "✅ Gati" if rag_system.document_embeddings is not None else "❌ Mungojnë"
        cloud_status = "✅ I lidhur" if rag_system.cloud_llm and rag_system.cloud_llm.api_key else "⚠️ Fallback"
        
        st.metric("🤖 Modeli AI", model_status)
        st.metric("🔢 Embeddings", embeddings_status)
        st.metric("🌐 Cloud AI", cloud_status)
        
        if rag_system.document_embeddings is not None:
            embedding_shape = rag_system.document_embeddings.shape
            st.caption(f"Dimensioni: {embedding_shape[0]}×{embedding_shape[1]}")
        
        # Last update info
        if os.path.exists(rag_system.scraper.metadata_file):
            try:
                with open(rag_system.scraper.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if metadata.get('last_updated'):
                        last_update = datetime.fromisoformat(metadata['last_updated'])
                        st.metric("🕒 Përditësimi i Fundit", last_update.strftime("%d/%m/%Y %H:%M"))
            except:
                pass
        
        # Performance metrics
        st.subheader("⚡ Performance")
        st.metric("📈 Dokumente Aktive", len(rag_system.legal_documents))
        
        # Quick stats about document types
        doc_types = {}
        for doc in rag_system.legal_documents:
            doc_type = doc.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in doc_types.items():
            if doc_type == 'hardcoded':
                st.metric("📖 Bazë", count)
            elif doc_type == 'scraped_legal_document':
                st.metric("🌐 Të shkarkuara", count)
    
    # Footer
    st.divider()
    st.caption("⚖️ Albanian Legal RAG System - Now powered by Advanced Cloud AI (Groq)")
    st.caption("🔄 Automatically updates with latest documents from qbz.gov.al")


if __name__ == "__main__":
    main()
