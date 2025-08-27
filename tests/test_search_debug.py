#!/usr/bin/env python3
"""
Search and Debug Test Script for Albanian Legal RAG System

Updated for new project structure and dual-mode querying.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.juristi.core.rag_engine import AlbanianLegalRAG
from src.juristi.config import config


def test_specific_queries():
    """Test specific legal queries in both modes."""
    
    # Set up logging
    config.setup_logging()
    
    print("🏛️ Albanian Legal RAG System - Enhanced Test")
    print("=" * 60)
    
    # Initialize system
    print("🔄 Initializing system...")
    try:
        rag = AlbanianLegalRAG(verbose=True, ui_only=True)
        
        if not hasattr(rag, 'vectorstore') or rag.vectorstore is None:
            print("❌ No embeddings found. Please run: python main.py process")
            return
            
        print("✅ System initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test queries
    test_queries = [
        "Cila eshte permbajtja e nenit 135 te Kodit te Familjes?",
        "Çfarë thotë neni 266 i Kodit të Familjes?",
        "Cila është permbajtja e nenit 1 të Kodit të Familjes?",
        "Sa dënohet vrasja me dashje?",
        "Cilat janë kushtet për martesë?"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"🔍 Query: '{query}'")
        print("-" * 60)
        
        # Test both modes
        for mode in ["precise", "analyzed"]:
            mode_emoji = "📍" if mode == "precise" else "🧠"
            print(f"\n{mode_emoji} **{mode.upper()} MODE:**")
            
            try:
                result = rag.query(query, query_mode=mode)
                
                if result.get('error'):
                    print(f"❌ Error: {result['error']}")
                else:
                    answer = result.get('answer', '')
                    sources = result.get('sources', [])
                    
                    print(f"✅ Answer ({len(sources)} sources):")
                    # Show first 500 characters of answer
                    display_answer = answer[:500] + "..." if len(answer) > 500 else answer
                    print(display_answer)
                    
                    if sources:
                        print(f"\n📚 First 3 Sources:")
                        for i, source in enumerate(sources[:3], 1):
                            metadata = source.get('metadata', {})
                            source_name = metadata.get('source', 'Unknown')
                            page = metadata.get('page', 'N/A')
                            content_preview = source.get('content', '')[:150] + "..."
                            print(f"  {i}. {source_name} (Page: {page})")
                            print(f"     {content_preview}")
            
            except Exception as e:
                print(f"❌ Query failed: {e}")


if __name__ == "__main__":
    test_specific_queries()
