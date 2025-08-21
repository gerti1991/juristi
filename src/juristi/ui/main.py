"""
Streamlit User Interface for Albanian Legal RAG System

This module provides the web-based user interface using Streamlit,
extracted and refactored from the original app.py.
"""

import os
import sys
import time
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

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


class StreamlitInterface:
    """Streamlit-based web interface for the Albanian Legal RAG system."""
    
    def __init__(self):
        """Initialize the Streamlit interface."""
        self.setup_page_config()
        self.rag_system = None
        self.llm_client = None
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Juristi AI - Albanian Legal Research Assistant",
            page_icon="⚖️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #1f4e79;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        
        .search-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1f4e79;
            margin: 1rem 0;
        }
        
        .result-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border-left: 4px solid #28a745;
        }
        
        .law-reference {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 6px;
            font-style: italic;
            color: #2c3e50;
            margin: 0.5rem 0;
        }
        
        .similarity-score {
            background-color: #28a745;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_systems(self):
        """Initialize RAG and LLM systems with lazy loading."""
        if self.rag_system is None:
            with st.spinner("🔄 Initializing legal document system..."):
                self.rag_system = AlbanianLegalRAG(
                    quick_start=True,  # Enable quick start for better UX
                    rag_mode='traditional'  # Can be configured in sidebar
                )
        
        if self.llm_client is None:
            self.llm_client = CloudLLMClient()
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">⚖️ Juristi AI</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">'
            'Asistenti Inteligjent për Kërkime Ligjore Shqiptare</p>',
            unsafe_allow_html=True
        )
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Konfigurimet")
            
            # RAG Mode Selection
            st.subheader("🔍 Mënyra e Kërkimit")
            rag_mode = st.selectbox(
                "Zgjidhni mënyrën e kërkimit:",
                ["traditional", "hierarchical", "sentence_window"],
                index=0,
                help="Traditional: Kërkim i thjeshtë\nHierarchical: Kërkim hierarkik\nSentence Window: Kërkim në nivel fjalie"
            )
            
            # Search Parameters
            st.subheader("📊 Parametrat e Kërkimit")
            
            top_k = st.slider(
                "Numri i dokumenteve:",
                min_value=1,
                max_value=10,
                value=3,
                help="Sa dokumente të shfaqen në rezultat"
            )
            
            search_mode = st.selectbox(
                "Lloji i kërkimit:",
                ["hybrid", "embedding", "sparse"],
                index=0,
                help="Hybrid: Kombinon të gjitha\nEmbedding: Vetëm embedding\nSparse: Vetëm TF-IDF"
            )
            
            multi_query = st.checkbox(
                "Zgjerimi i pyetjes",
                value=True,
                help="Përdor sinonime dhe terma të ngjashëm"
            )
            
            # LLM Configuration
            st.subheader("🤖 Konfigurimi i AI")
            
            llm_provider = st.selectbox(
                "Ofrues i AI:",
                ["gemini", "groq", "huggingface"],
                index=0,
                help="Zgjidhni ofruеsin e modelit të AI"
            )
            
            use_ai_response = st.checkbox(
                "Përgjigje me AI",
                value=True,
                help="Gjenero përgjigje të plotë me AI"
            )
            
            # Advanced Options
            with st.expander("🔧 Opsione të Avancuara"):
                use_google_embeddings = st.checkbox(
                    "Google Embeddings",
                    value=True,
                    help="Përdor Google embeddings për performancë më të mirë"
                )
                
                hybrid_alpha = st.slider(
                    "Hybrid Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Balanci midis embedding dhe sparse search"
                )
            
            # System Status
            st.subheader("📊 Statusi i Sistemit")
            if self.rag_system:
                if hasattr(self.rag_system, 'legal_documents') and self.rag_system.legal_documents:
                    st.success(f"✅ {len(self.rag_system.legal_documents)} dokumente të ngarkuara")
                else:
                    st.info("📚 Dokumentet do të ngarkohen në kërkimin e parë")
                
                if hasattr(self.rag_system, 'document_embeddings') and self.rag_system.document_embeddings is not None:
                    st.success("✅ Embeddings të gatshme")
                else:
                    st.info("🔄 Embeddings do të krijohen në kërkimin e parë")
            
            return {
                'rag_mode': rag_mode,
                'top_k': top_k,
                'search_mode': search_mode,
                'multi_query': multi_query,
                'llm_provider': llm_provider,
                'use_ai_response': use_ai_response,
                'use_google_embeddings': use_google_embeddings,
                'hybrid_alpha': hybrid_alpha
            }
    
    def render_search_interface(self, config: Dict) -> Optional[str]:
        """Render the main search interface."""
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        
        # Search input
        query = st.text_area(
            "🔍 Shtypni pyetjen tuaj ligjore këtu:",
            height=100,
            placeholder="Shembull: Çfarë janë të drejtat e punëtorit në pushim vjetor?",
            help="Shtypni pyetjen tuaj në shqip ose anglisht"
        )
        
        # Search buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_clicked = st.button(
                "🔍 Kërko",
                type="primary",
                help="Filloni kërkimin",
                use_container_width=True
            )
        
        with col2:
            example_clicked = st.button(
                "💡 Shembull",
                help="Shfaq një pyetje shembull"
            )
        
        with col3:
            clear_clicked = st.button(
                "🗑️ Pastro",
                help="Pastro të gjitha"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle button clicks
        if example_clicked:
            example_queries = [
                "Çfarë është procedura për martesë civile në Shqipëri?",
                "Cilat janë të drejtat e punëtorit për pushim vjetor?",
                "Si përcaktohet dënimi për vrasje me dashje?",
                "Çfarë dokumentesh nevojiten për regjistrim biznesi?",
                "Cilat janë detyrimet e punëdhënësit ndaj punëtorit?"
            ]
            import random
            return random.choice(example_queries)
        
        if clear_clicked:
            st.rerun()
        
        if search_clicked and query.strip():
            return query.strip()
        
        return None
    
    def perform_search(self, query: str, config: Dict) -> tuple:
        """Perform the document search."""
        start_time = time.time()
        
        # Update RAG system configuration
        if self.rag_system:
            self.rag_system.rag_mode = config['rag_mode']
            self.rag_system.use_google_embeddings = config['use_google_embeddings']
            self.rag_system.hybrid_alpha = config['hybrid_alpha']
        
        # Perform search
        with st.spinner("🔍 Duke kërkuar në dokumentet ligjore..."):
            try:
                results = self.rag_system.search_documents(
                    query=query,
                    top_k=config['top_k'],
                    mode=config['search_mode'],
                    multi_query=config['multi_query']
                )
                
                search_time = time.time() - start_time
                
                return results, search_time, None
                
            except Exception as e:
                return [], 0, str(e)
    
    def generate_ai_response(self, query: str, results: List[Dict], config: Dict) -> Optional[str]:
        """Generate AI response based on search results."""
        if not config['use_ai_response'] or not results:
            return None
        
        try:
            with st.spinner("🤖 Duke gjeneruar përgjigje me AI..."):
                # Configure LLM client
                if config['llm_provider'] != self.llm_client.current_provider:
                    self.llm_client.current_provider = config['llm_provider']
                
                # Prepare context from search results
                context_parts = []
                for i, doc in enumerate(results[:3]):  # Use top 3 results
                    context_parts.append(f"""
Dokumenti {i+1}: {doc.get('title', 'Pa titull')}
Burimi: {doc.get('source', 'Pa burim')}
Përmbajtja: {doc.get('content', '')[:1000]}...
                    """)
                
                context = "\n".join(context_parts)
                
                # Generate response
                response = self.llm_client.generate_response(
                    query=query,
                    relevant_docs=results,
                    context=context
                )
                
                return response
                
        except Exception as e:
            st.error(f"❌ Gabim në gjenerimin e përgjigjes: {e}")
            return None
    
    def render_results(self, query: str, results: List[Dict], ai_response: Optional[str], 
                      search_time: float, config: Dict):
        """Render search results and AI response."""
        if not results and not ai_response:
            st.warning("❌ Nuk u gjetën rezultate për pyetjen tuaj. Provoni me fjalë kyçe të ndryshme.")
            return
        
        # Results header
        st.markdown("---")
        st.header("📋 Rezultatet e Kërkimit")
        
        # Search statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔍 Pyetja", "✓ E përpunuar")
        with col2:
            st.metric("📊 Rezultate", len(results))
        with col3:
            st.metric("⏱️ Koha", f"{search_time:.2f}s")
        
        # AI Response (if available)
        if ai_response:
            st.subheader("🤖 Përgjigja e AI")
            st.markdown(
                f'<div class="result-card" style="border-left-color: #007bff;">'
                f'{ai_response.replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Document Results
        if results:
            st.subheader("📚 Dokumentet e Gjetur")
            
            for i, doc in enumerate(results):
                with st.expander(
                    f"📄 {i+1}. {doc.get('title', 'Pa titull')} "
                    f"({doc.get('similarity_score', 0):.1%} përputhje)",
                    expanded=(i == 0)  # Expand first result by default
                ):
                    # Document metadata
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**📋 Burimi:** {doc.get('source', 'Pa burim')}")
                        if doc.get('document_type'):
                            st.markdown(f"**📁 Lloji:** {doc.get('document_type')}")
                    
                    with col2:
                        score = doc.get('similarity_score', 0)
                        st.markdown(
                            f'<span class="similarity-score">{score:.1%} përputhje</span>',
                            unsafe_allow_html=True
                        )
                    
                    # Document content
                    content = doc.get('content', 'Pa përmbajtje')
                    if len(content) > 1000:
                        st.markdown(f"**📝 Përmbajtja:** {content[:1000]}...")
                        
                        if st.button(f"Shfaq më shumë", key=f"show_more_{i}"):
                            st.markdown(f"**Përmbajtja e plotë:**\n\n{content}")
                    else:
                        st.markdown(f"**📝 Përmbajtja:** {content}")
                    
                    # Search metadata
                    if doc.get('search_method'):
                        st.caption(f"🔍 Metoda e kërkimit: {doc.get('search_method')}")
    
    def render_footer(self):
        """Render application footer."""
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; padding: 1rem;">
                <p>⚖️ <strong>Juristi AI</strong> - Asistenti Inteligjent për Kërkime Ligjore Shqiptare</p>
                <p style="font-size: 0.9rem;">
                    🏛️ Sistemi bazohet në legjislacionin shqiptar aktual<br>
                    ⚠️ <em>Ky sistem ofron ndihmë informative. Konsultohuni me një jurist të kualifikuar për çështje të rëndësishme ligjore.</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def run(self):
        """Run the main Streamlit application."""
        # Initialize systems
        self.initialize_systems()
        
        # Render interface components
        self.render_header()
        config = self.render_sidebar()
        query = self.render_search_interface(config)
        
        # Handle search
        if query:
            # Store query in session state
            st.session_state.last_query = query
            st.session_state.last_config = config
            
            # Perform search
            results, search_time, error = self.perform_search(query, config)
            
            if error:
                st.error(f"❌ Gabim në kërkim: {error}")
            else:
                # Generate AI response if requested
                ai_response = None
                if config['use_ai_response']:
                    ai_response = self.generate_ai_response(query, results, config)
                
                # Render results
                self.render_results(query, results, ai_response, search_time, config)
        
        # Show previous results if available
        elif hasattr(st.session_state, 'last_query') and st.session_state.last_query:
            st.info(f"💭 Rezultatet e fundit për: \"{st.session_state.last_query}\"")
            # Re-render previous results if they exist
            # (This would require storing results in session state as well)
        
        # Render footer
        self.render_footer()


def run_streamlit_app():
    """Entry point for running the Streamlit application."""
    app = StreamlitInterface()
    app.run()


if __name__ == "__main__":
    run_streamlit_app()
