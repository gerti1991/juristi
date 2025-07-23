"""
Enhanced Albanian Legal RAG System with Cloud LLM Integration
Now uses Groq API for advanced response generation with environment variables
"""

import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
from scraper import AlbanianLegalScraper
from ai import CloudLLMClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables for PyTorch compatibility
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class CloudEnhancedAlbanianLegalRAG:
    def __init__(self):
        """Initialize the cloud-enhanced RAG system with environment variables"""
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = None
        self.legal_documents = []
        self.document_embeddings = None
        self.scraper = AlbanianLegalScraper(max_docs=10)
        self.cloud_llm = None
        
        # Configuration from environment variables
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.08'))
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))
        self.max_chunks_to_return = int(os.getenv('MAX_CHUNKS_TO_RETURN', '5'))
        
        # Cache file for embeddings
        self.embeddings_cache_file = 'document_embeddings_cache.json'
        
        # Load model with error handling
        self._load_model()
        
        # Initialize cloud LLM
        self._load_cloud_llm()
        
        # Initialize document collection
        self._load_all_documents()
        
        # Generate embeddings
        self._generate_embeddings()
    
    def _load_model(self):
        """Load the sentence transformer model with error handling"""
        try:
            st.info("🔄 Loading AI model...")
            self.model = SentenceTransformer(self.model_name)
            st.success("✅ AI model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("Please ensure sentence-transformers is installed: pip install sentence-transformers")
            self.model = None
    
    def _load_cloud_llm(self):
        """Initialize cloud LLM client"""
        try:
            st.info("🌐 Connecting to cloud LLM...")
            self.cloud_llm = CloudLLMClient("groq")
            
            if self.cloud_llm.api_key:
                st.success("✅ Cloud LLM connected successfully!")
                st.info(f"🤖 Using model: {self.cloud_llm.model}")
            else:
                st.warning("⚠️ Cloud LLM API key not found - using fallback responses")
        except Exception as e:
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
            st.info(f"📄 Loaded {len(scraped_docs)} scraped document chunks")
            return scraped_docs
        except Exception as e:
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
                
                st.info(f"📄 Loaded {len(normalized_docs)} PDF document chunks")
                return normalized_docs
            else:
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
        
        st.info(f"📚 Total documents loaded: {len(self.legal_documents)} "
                f"({len(hardcoded)} hardcoded + {len(scraped)} scraped + {len(pdf_docs)} PDF)")
    
    def _load_embeddings_cache(self):
        """Load cached embeddings if available"""
        if os.path.exists(self.embeddings_cache_file):
            try:
                with open(self.embeddings_cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                
                # Check if cache is still valid (same number of documents)
                if cache.get('document_count') == len(self.legal_documents):
                    self.document_embeddings = np.array(cache['embeddings'])
                    st.success("✅ Loaded cached embeddings")
                    return True
            except Exception as e:
                st.warning(f"⚠️ Could not load embedding cache: {e}")
        
        return False
    
    def _save_embeddings_cache(self):
        """Save embeddings to cache"""
        try:
            cache = {
                'embeddings': self.document_embeddings.tolist(),
                'document_count': len(self.legal_documents),
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.embeddings_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
                
            st.success("✅ Embeddings cached for faster loading")
        except Exception as e:
            st.warning(f"⚠️ Could not save embedding cache: {e}")
    
    def _generate_embeddings(self):
        """Generate embeddings for all documents"""
        if not self.model:
            st.error("❌ Model not loaded - cannot generate embeddings")
            return
        
        if not self.legal_documents:
            st.error("❌ No documents loaded - cannot generate embeddings")
            return
        
        # Try to load from cache first
        if self._load_embeddings_cache():
            return
        
        try:
            st.info("🔄 Generating document embeddings...")
            
            # Combine Albanian and English content for better matching
            document_texts = []
            for doc in self.legal_documents:
                combined_text = f"{doc['content']} {doc.get('content_en', '')}"
                document_texts.append(combined_text)
            
            # Generate embeddings
            self.document_embeddings = self.model.encode(document_texts)
            
            # Save to cache
            self._save_embeddings_cache()
            
            st.success(f"✅ Generated embeddings for {len(self.legal_documents)} documents")
            
        except Exception as e:
            st.error(f"❌ Error generating embeddings: {e}")
            self.document_embeddings = None
    
    def search_documents(self, query: str, top_k: int = 3):
        """Search for relevant documents using semantic similarity with Albanian query enhancement"""
        if not self.model or self.document_embeddings is None:
            return []
        
        try:
            # Enhance Albanian queries for better semantic search
            enhanced_query = self._enhance_albanian_query(query)
            
            # Generate query embedding
            query_embedding = self.model.encode([enhanced_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            max_similarity = 0
            for idx in top_indices:
                similarity = float(similarities[idx])
                if similarity > max_similarity:
                    max_similarity = similarity
                if similarity > self.similarity_threshold:  # Configurable threshold for Albanian queries
                    doc = self.legal_documents[idx].copy()
                    doc['similarity'] = similarity
                    results.append(doc)
            
            # Store max similarity for fallback decision
            self.last_search_max_similarity = max_similarity
            
            return results
            
        except Exception as e:
            st.error(f"❌ Search error: {e}")
            return []
    
    def _enhance_albanian_query(self, query: str) -> str:
        """Enhance Albanian queries with synonyms and related terms"""
        enhanced_query = query
        
        # Add Albanian legal synonyms
        legal_synonyms = {
            'pushim': 'pushim leaves vacation dit',
            'punë': 'punë work employment job',
            'ligj': 'ligj law legal kod',
            'sanksion': 'sanksion penalty dënim gjobë',
            'kompani': 'kompani company business shoqëri',
            'kontrata': 'kontrata contract marrëveshje',
            'vjedhje': 'vjedhje theft stealing',
            'martesë': 'martesë marriage bashkëshort',
            'divorcë': 'divorcë divorce ndarje'
        }
        
        for albanian_term, synonyms in legal_synonyms.items():
            if albanian_term in query.lower():
                enhanced_query += f" {synonyms}"
        
        return enhanced_query
    
    def generate_response(self, query: str, relevant_docs: list) -> str:
        """Generate a comprehensive response using cloud LLM or fallback"""
        if self.cloud_llm and self.cloud_llm.api_key:
            try:
                # Use cloud LLM for advanced response generation
                return self.cloud_llm.generate_response(query, relevant_docs)
            except Exception as e:
                st.warning(f"⚠️ Cloud LLM error, using fallback: {e}")
                return self._fallback_response(query, relevant_docs)
        else:
            # Use fallback response
            return self._fallback_response(query, relevant_docs)
    
    def _fallback_response(self, query: str, relevant_docs: list) -> str:
        """Fallback response when cloud LLM is not available"""
        if not relevant_docs:
            return self._get_no_results_response()
        
        # Check for vacation-related queries
        vacation_keywords = ['pushim', 'pushimi', 'vacation', 'leave', 'dit']
        if any(keyword in query.lower() for keyword in vacation_keywords):
            return self._get_vacation_response(relevant_docs)
        
        # General response
        response = "Bazuar në dokumentet ligjore shqiptare:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            doc_type_indicator = "🌐" if doc.get('document_type') == 'scraped_legal_document' else "📚"
            response += f"{doc_type_indicator} **{doc['title']}** (Relevanca: {doc['similarity']:.2f})\n"
            response += f"{doc['content'][:300]}...\n\n"
        
        response += "\n---\n**Shënim**: Ky përgjigje është bazuar në dokumentet e disponueshme. "
        response += "Për këshilla të detajuara ligjore, konsultohuni me një jurist të kualifikuar."
        
        return response
    
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
        
        # Add relevant document excerpts
        for doc in relevant_docs[:2]:
            if 'labor' in doc.get('id', '').lower() or 'punë' in doc.get('content', '').lower():
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
