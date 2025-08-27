"""
Development Utilities for Albanian Legal RAG System

Quick development and testing tools.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.juristi.config import config
from src.juristi.core.rag_engine import AlbanianLegalRAG


def quick_test():
    """Quick functionality test."""
    print("🧪 Running Quick Test...")
    
    try:
        # Set up logging
        config.setup_logging()
        logger = logging.getLogger(__name__)
        
        # Initialize RAG system
        print("🔄 Initializing RAG system...")
        rag = AlbanianLegalRAG(verbose=True, ui_only=True)
        
        if not hasattr(rag, 'vectorstore') or rag.vectorstore is None:
            print("❌ No embeddings found. Run: python main.py process")
            return False
        
        # Test queries
        test_queries = [
            ("Sa është denimi për vjedhje?", "precise"),
            ("Çfarë është martesa sipas ligjit?", "analyzed")
        ]
        
        print("\n🔍 Testing queries...")
        for query, mode in test_queries:
            print(f"\n📝 Query ({mode}): {query}")
            
            result = rag.query(query, query_mode=mode)
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                answer = result.get('answer', '')[:200] + "..." if len(result.get('answer', '')) > 200 else result.get('answer', '')
                sources_count = len(result.get('sources', []))
                print(f"✅ Answer ({sources_count} sources): {answer}")
        
        print("\n✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_embeddings():
    """Check embedding system status."""
    print("🔍 Checking Embedding Status...")
    
    try:
        from src.juristi.core.rag_engine import AlbanianLegalRAG
        
        # Try to initialize
        rag = AlbanianLegalRAG(verbose=True, ui_only=True)
        
        if hasattr(rag, 'vectorstore') and rag.vectorstore is not None:
            # Get collection info
            try:
                collection = rag.vectorstore._collection
                count = collection.count()
                print(f"✅ ChromaDB collection found with {count} documents")
                
                # Test a simple search
                test_docs = rag.vectorstore.similarity_search("ligj", k=3)
                print(f"✅ Search test successful - found {len(test_docs)} documents")
                
                return True
                
            except Exception as e:
                print(f"❌ ChromaDB access error: {e}")
                return False
        else:
            print("❌ No vectorstore found - run embedding processing first")
            return False
            
    except Exception as e:
        print(f"❌ Embedding check failed: {e}")
        return False


def benchmark():
    """Run performance benchmark."""
    print("⚡ Running Performance Benchmark...")
    
    try:
        import time
        from src.juristi.core.rag_engine import AlbanianLegalRAG
        
        rag = AlbanianLegalRAG(verbose=False, ui_only=True)
        
        if not hasattr(rag, 'vectorstore') or rag.vectorstore is None:
            print("❌ No embeddings found for benchmarking")
            return
        
        # Benchmark queries
        benchmark_queries = [
            "plagosje e rëndë",
            "martesa dhe divorci",
            "vrasje me dashje",
            "kontrata shitjeje",
            "procedura gjyqësore"
        ]
        
        results = []
        
        for query in benchmark_queries:
            print(f"🔍 Testing: {query}")
            
            # Precise mode
            start_time = time.time()
            result_precise = rag.query(query, query_mode="precise")
            precise_time = time.time() - start_time
            
            # Analyzed mode
            start_time = time.time()
            result_analyzed = rag.query(query, query_mode="analyzed")
            analyzed_time = time.time() - start_time
            
            results.append({
                'query': query,
                'precise_time': precise_time,
                'analyzed_time': analyzed_time,
                'precise_sources': len(result_precise.get('sources', [])),
                'analyzed_sources': len(result_analyzed.get('sources', []))
            })
            
            print(f"  📍 Precise: {precise_time:.2f}s ({result_precise.get('sources', []).__len__()} sources)")
            print(f"  🧠 Analyzed: {analyzed_time:.2f}s ({result_analyzed.get('sources', []).__len__()} sources)")
        
        # Summary
        avg_precise = sum(r['precise_time'] for r in results) / len(results)
        avg_analyzed = sum(r['analyzed_time'] for r in results) / len(results)
        
        print(f"\n📊 Benchmark Summary:")
        print(f"  📍 Average Precise Time: {avg_precise:.2f}s")
        print(f"  🧠 Average Analyzed Time: {avg_analyzed:.2f}s")
        print(f"  ⚡ Performance Ratio: {avg_analyzed/avg_precise:.1f}x")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def main():
    """Main development utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Development utilities")
    parser.add_argument("command", choices=["test", "embeddings", "benchmark"], help="Command to run")
    
    args = parser.parse_args()
    
    print("🔧 Albanian Legal RAG - Development Tools")
    print("=" * 50)
    
    if args.command == "test":
        success = quick_test()
        sys.exit(0 if success else 1)
    elif args.command == "embeddings":
        success = check_embeddings()
        sys.exit(0 if success else 1)
    elif args.command == "benchmark":
        benchmark()


if __name__ == "__main__":
    main()
