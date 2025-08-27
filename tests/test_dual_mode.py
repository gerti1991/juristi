#!/usr/bin/env python3
"""
Test script for the enhanced dual-mode RAG system.
Tests both precise and analyzed query modes.
"""

import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.juristi.core.rag_engine import AlbanianLegalRAG

def test_dual_mode_queries():
    """Test both precise and analyzed query modes."""
    
    print("🚀 Initializing Enhanced Albanian Legal RAG System...")
    rag_system = AlbanianLegalRAG(verbose=True, ui_only=True)
    
    # Test query - complex legal scenario requiring analysis
    test_query = "Sa dënohet nëse lëviz me makinë në krah të kundërt, në gjendje të dehur, bën aksident dhe vret dy persona?"
    
    print("\n" + "="*80)
    print("🧪 TESTING DUAL MODE FUNCTIONALITY")
    print("="*80)
    print(f"📝 Test Query: {test_query}")
    
    # Test 1: Precise Mode
    print("\n🔍 TEST 1: PRECISE MODE")
    print("-" * 50)
    
    result_precise = rag_system.query(test_query, query_mode="precise")
    
    if result_precise.get('error'):
        print(f"❌ Error in precise mode: {result_precise['error']}")
    else:
        answer = result_precise.get('answer', '')[:500] + "..." if len(result_precise.get('answer', '')) > 500 else result_precise.get('answer', '')
        sources_count = len(result_precise.get('sources', []))
        
        print(f"✅ Precise Answer (first 500 chars):")
        print(answer)
        print(f"📊 Sources retrieved: {sources_count}")
    
    # Test 2: Analyzed Mode  
    print("\n🧠 TEST 2: ANALYZED MODE")
    print("-" * 50)
    
    result_analyzed = rag_system.query(test_query, query_mode="analyzed")
    
    if result_analyzed.get('error'):
        print(f"❌ Error in analyzed mode: {result_analyzed['error']}")
    else:
        answer = result_analyzed.get('answer', '')[:500] + "..." if len(result_analyzed.get('answer', '')) > 500 else result_analyzed.get('answer', '')
        sources_count = len(result_analyzed.get('sources', []))
        
        print(f"✅ Analyzed Answer (first 500 chars):")
        print(answer)
        print(f"📊 Sources retrieved: {sources_count}")
    
    # Comparison
    print("\n📊 COMPARISON")
    print("-" * 50)
    
    if not result_precise.get('error') and not result_analyzed.get('error'):
        precise_sources = len(result_precise.get('sources', []))
        analyzed_sources = len(result_analyzed.get('sources', []))
        
        print(f"📍 Precise Mode: {precise_sources} sources")
        print(f"🧠 Analyzed Mode: {analyzed_sources} sources")
        print(f"📈 Source difference: +{analyzed_sources - precise_sources} more in analyzed mode")
        
        # Check if analyzed mode provides more comprehensive response
        precise_length = len(result_precise.get('answer', ''))
        analyzed_length = len(result_analyzed.get('answer', ''))
        
        print(f"📝 Precise response length: {precise_length} characters")
        print(f"📝 Analyzed response length: {analyzed_length} characters")
        print(f"📏 Length difference: +{analyzed_length - precise_length} characters in analyzed mode")
    
    print("\n✅ Dual mode testing completed!")
    print("🌐 The enhanced system is ready for use at: http://localhost:8501")

if __name__ == "__main__":
    test_dual_mode_queries()
