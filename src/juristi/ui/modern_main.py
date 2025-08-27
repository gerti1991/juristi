"""
Modern Streamlit UI for Albanian Legal RAG System

Uses the new LangChain-based RAG engine with Google Generative AI
for better Albanian/English language support and modern retrieval.
"""

# Fix for PyTorch/Streamlit compatibility issue
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Set environment variable to fix torch classes issue
import os
os.environ["TORCH_CLASSES_PATCH"] = "1"

import streamlit as st
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the modern RAG engine and configuration
from src.juristi.core.rag_engine import AlbanianLegalRAG
from src.juristi.config import config

# Configure Streamlit page
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1f4e79, #2c5aa0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .search-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .response-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
    }
    
    .response-container.analyzed-mode {
        border-left: 5px solid #6f42c1;
        background: linear-gradient(135deg, #ffffff, #f8f9ff);
    }
    
    .response-container.precise-mode {
        border-left: 5px solid #28a745;
        background: linear-gradient(135deg, #ffffff, #f8fff8);
    }
    
    .source-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_rag_system():
    """Initialize the RAG system with session state management."""
    if 'rag_system' not in st.session_state:
        with st.spinner("🔄 Initializing Modern Albanian Legal RAG System (UI Mode)..."):
            try:
                # Initialize in UI-only mode - expects embeddings to be pre-processed
                st.session_state.rag_system = AlbanianLegalRAG(verbose=True, ui_only=True)
                
                # Check if system is ready for queries
                if hasattr(st.session_state.rag_system, 'vectorstore') and st.session_state.rag_system.vectorstore is not None:
                    st.session_state.system_initialized = True
                    st.success("✅ System initialized successfully in UI-only mode!")
                    st.info("📊 Using pre-processed embeddings from ChromaDB")
                else:
                    st.session_state.system_initialized = False
                    st.error("❌ No pre-processed embeddings found!")
                    st.error("Please run the embedding processing script first.")
                    return None
                
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                st.session_state.system_initialized = False
                return None
    
    return st.session_state.rag_system


def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">⚖️ Juristi AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Asistenti i Avancuar për Kërkime Ligjore Shqiptare</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align: center; margin-bottom: 2rem;">'
        '<span style="background: linear-gradient(90deg, #28a745, #20c997); color: white; '
        'padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">'
        '🚀 Powered by Google Generative AI & LangChain</span>'
        '</div>',
        unsafe_allow_html=True
    )


def render_sidebar(rag_system: AlbanianLegalRAG):
    """Render the sidebar with system status and configuration."""
    with st.sidebar:
        st.header("⚙️ Konfigurimi i Sistemit")
        
        # System Status
        st.subheader("📊 Statusi i Sistemit")
        status = rag_system.get_system_status()
        
        col1, col2 = st.columns(2)
        with col1:
            if status['documents_loaded']:
                st.markdown('<div class="status-good">✅ Dokumente</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-warning">⏳ Dokumente</div>', unsafe_allow_html=True)
        
        with col2:
            if status['chain_ready']:
                st.markdown('<div class="status-good">✅ AI Chain</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-warning">⏳ AI Chain</div>', unsafe_allow_html=True)
        
        st.metric("📚 Dokumente", status.get('total_documents', 0))
        st.metric("🤖 Model LLM", status.get('llm_model', 'N/A'))
        st.metric("🔤 Embedding Model", status.get('embedding_model', 'N/A').split('/')[-1])
        
        # Search Configuration
        st.subheader("🔍 Konfigurimi i Kërkimit")
        
        search_params = {
            'k': st.slider("Dokumentet për t'u kthyer", 3, 10, 5, 
                          help="Numri i dokumenteve më relevante"),
            'fetch_k': st.slider("Kandidatët për MMR", 15, 30, 20, 
                               help="Dokumentet e konsideruara për MMR"),
            'lambda_mult': st.slider("Balanci Relevancë/Diversitet", 0.0, 1.0, 0.7, 0.1,
                                   help="0.0 = vetëm diversitet, 1.0 = vetëm relevancë")
        }
        
        # Memory Management
        st.subheader("🧠 Menaxhimi i Memories")
        if st.button("🔄 Reset Memory"):
            rag_system.reset_memory()
            st.success("Memory u resetua!")
        
        # Advanced Options
        with st.expander("🔧 Opsione të Avancuara"):
            st.info("Konfigurimi i parametrave të avancuar")
            verbose_mode = st.checkbox("Verbose Logging", value=True)
            show_sources = st.checkbox("Shfaq Burimet", value=True)
        
        return {
            'search_params': search_params,
            'verbose_mode': verbose_mode,
            'show_sources': show_sources
        }


def render_search_interface():
    """Render the enhanced dual-mode search interface."""
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "🔍 Shkruani pyetjen tuaj ligjore këtu:",
            height=100,
            placeholder="P.sh. 'Sa dënohet nëse lëviz me makinë në krah të kundërt, në gjendje të dehur, bën aksident dhe vret dy persona?'",
            help="Mund të shkruani në shqip ose anglisht. Zgjidhni llojin e përgjigjes më poshtë."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        
        # Dual-mode buttons
        st.markdown("**🎯 Zgjidh llojin e përgjigjes:**")
        
        col2a, col2b = st.columns(2)
        with col2a:
            precise_clicked = st.button("� Precize", 
                                      type="secondary", 
                                      use_container_width=True,
                                      help="Përgjigje e drejtpërdrejtë nga ligji")
        
        with col2b:
            analyzed_clicked = st.button("🧠 E Analizuar", 
                                       type="primary", 
                                       use_container_width=True,
                                       help="Analizë e detajuar me kombinim informacionesh")
        
        st.markdown("---")
        
        # Example queries
        st.markdown("**Shembuj:**")
        example_queries = [
            "plagosje e rëndë dënim",
            "makinë krah kundërt + dehur + vrasje",
            "divorci procedura gjyqësore"
        ]
        
        for example in example_queries:
            if st.button(f"📋 {example}", key=f"example_{example}", use_container_width=True):
                st.session_state.example_query = example
                st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle example query selection
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        precise_clicked = True  # Default to precise for examples
        del st.session_state.example_query
    
    # Determine search mode and if search was clicked
    search_clicked = precise_clicked or analyzed_clicked
    query_mode = "analyzed" if analyzed_clicked else "precise"
    
    return query, search_clicked, query_mode


def render_response(response: Dict[str, Any], config: Dict[str, Any]):
    """Render the AI response and sources with enhanced query mode display."""
    if response.get('error'):
        st.error(f"❌ Gabim: {response['error']}")
        return
    
    answer = response.get('answer', '')
    sources = response.get('sources', [])
    query_mode = response.get('query_mode', 'precise')
    total_sources = len(sources)
    
    # Enhanced response metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        mode_emoji = "🧠" if query_mode == "analyzed" else "📍"
        mode_text = "E Analizuar" if query_mode == "analyzed" else "Precize"
        st.metric("🎯 Lloji i Përgjigjes", f"{mode_emoji} {mode_text}")
    with col2:
        st.metric("📄 Burimet e Përdorura", total_sources)
    with col3:
        coverage = "Gjërë" if query_mode == "analyzed" else "Specifike"
        st.metric("🔍 Mbulueshmëria", coverage)
    
    # AI Response with mode-specific styling
    if answer:
        if query_mode == "analyzed":
            st.markdown('<div class="response-container analyzed-mode">', unsafe_allow_html=True)
            st.subheader("� Analiza e Detajuar Ligjore")
            st.info("💡 Kjo përgjigje kombinon informacione nga shumë burime ligjore për një analizë të plotë.")
        else:
            st.markdown('<div class="response-container precise-mode">', unsafe_allow_html=True)
            st.subheader("📍 Përgjigja e Precizë Ligjore")
            st.info("🎯 Kjo përgjigje është bazuar në burimet më relevantet nga dokumentet ligjore.")
        
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced source documents display
    if sources and config.get('show_sources', True):
        source_title = f"📚 Burimet e Konsideruara ({total_sources} dokumente)"
        if query_mode == "analyzed":
            source_title += " - Analizë e Gjërë"
        
        st.subheader(source_title)
        
        for i, source in enumerate(sources, 1):
            # Show first 3 sources expanded in analyzed mode, first 1 in precise mode
            expanded = (i <= 3) if query_mode == "analyzed" else (i == 1)
            
            with st.expander(f"📄 Burimi {i}: {source.get('title', 'Pa titull')}", 
                           expanded=expanded):
                
                st.markdown('<div class="source-card">', unsafe_allow_html=True)
                
                # Source metadata
                metadata = source.get('metadata', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**📍 Burimi:** {metadata.get('source', 'I panjohur')}")
                    st.markdown(f"**📂 Tipi:** {metadata.get('type', 'Dokument ligjor')}")
                
                with col2:
                    if 'page' in metadata:
                        st.markdown(f"**📄 Faqja:** {metadata['page']}")
                    if 'title' in metadata:
                        st.markdown(f"**📋 Titulli:** {metadata['title']}")
                
                # Source content
                st.markdown("**📖 Përmbajtja:**")
                st.markdown(source.get('content', 'Pa përmbajtje'))
                
                st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    render_header()
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    if not rag_system:
        st.stop()
    
    # Render sidebar and get configuration
    config = render_sidebar(rag_system)
    
    # Render search interface
    query, search_clicked, query_mode = render_search_interface()
    
    # Process search if triggered
    if search_clicked and query.strip():
        
        # Display query mode indicator
        mode_emoji = "🧠" if query_mode == "analyzed" else "📍"
        mode_text = "E Analizuar" if query_mode == "analyzed" else "Precize"
        st.info(f"{mode_emoji} **Mënyra e Kërkimit:** {mode_text}")
        
        with st.spinner("🔍 Duke kërkuar dhe analizuar dokumentet..."):
            
            # Set up verbose mode for debugging
            if config.get('verbose_mode', True):
                st.info("🔧 Verbose mode aktiv - shfaqen detajet e procesimit")
            
            # Execute query with mode
            response = rag_system.query(
                question=query,
                session_state=st.session_state,
                query_mode=query_mode
            )
            
            # Store in session state for reference
            st.session_state.last_response = response
            st.session_state.last_query = query
            st.session_state.last_config = config
            st.session_state.last_query_mode = query_mode
        
        # Render response
        render_response(response, config)
    
    # Show previous response if available
    elif 'last_response' in st.session_state:
        mode_emoji = "🧠" if st.session_state.get('last_query_mode') == "analyzed" else "📍"
        mode_text = "E Analizuar" if st.session_state.get('last_query_mode') == "analyzed" else "Precize"
        st.info(f"💭 Rezultatet e fundit ({mode_emoji} {mode_text}): \"{st.session_state.last_query}\"")
        render_response(st.session_state.last_response, st.session_state.get('last_config', {}))
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>⚖️ <strong>Juristi AI</strong> - Sistemi Modern i Kërkimeve Ligjore Shqiptare</p>
            <p style="font-size: 0.9rem;">
                🔬 <strong>Teknologjia:</strong> Google Generative AI • LangChain • ChromaDB • MMR Retrieval<br>
                ⚠️ <em>Për çështje të rëndësishme ligjore, konsultohuni me një jurist të kualifikuar.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
