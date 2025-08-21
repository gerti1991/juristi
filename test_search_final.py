#!/usr/bin/env python3
"""
Test script to verify the search functionality is working
"""
import sys
sys.path.append('.')
from src.juristi.core.rag_engine import AlbanianLegalRAG

def test_search():
    print('🔍 Testing search functionality...')
    engine = AlbanianLegalRAG(use_google_embeddings=True)
    
    # Test the original problematic query
    query = 'sa denohesh per plagosje'
    print(f'\n📋 Searching for: "{query}"')
    
    results = engine.search_documents(query, top_k=5)
    print(f'✅ Found {len(results)} results:')
    
    if len(results) > 0:
        for i, result in enumerate(results[:3], 1):
            print(f'\n{i}. 📊 Score: {result["similarity_score"]:.3f}')
            print(f'   📁 Source: {result["source"]}')
            print(f'   📄 Content preview: {result["content"][:150]}...')
    else:
        print('❌ No results found - there might still be an issue')
    
    # Test with another legal query
    print('\n' + '='*50)
    query2 = 'denim burgu ligj penal'
    print(f'📋 Searching for: "{query2}"')
    
    results2 = engine.search_documents(query2, top_k=3)
    print(f'✅ Found {len(results2)} results:')
    
    if len(results2) > 0:
        for i, result in enumerate(results2[:2], 1):
            print(f'\n{i}. 📊 Score: {result["similarity_score"]:.3f}')
            print(f'   📁 Source: {result["source"]}')
            print(f'   📄 Content preview: {result["content"][:150]}...')
    
    return len(results), len(results2)

if __name__ == "__main__":
    result1_count, result2_count = test_search()
    print(f'\n🎯 Summary:')
    print(f'   Query 1 ("sa denohesh per plagosje"): {result1_count} results')
    print(f'   Query 2 ("denim burgu ligj penal"): {result2_count} results')
    
    if result1_count > 0 and result2_count > 0:
        print('🎉 Search functionality is working correctly!')
    else:
        print('⚠️ Search functionality needs more investigation')
