#!/usr/bin/env python3
"""
Quick test for number word conversion feature.
"""

import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.juristi.core.rag_engine import AlbanianLegalRAG

def test_number_conversion():
    """Test the number word conversion feature."""
    print("🔢 Testing Number Word Conversion...")
    
    rag = AlbanianLegalRAG(verbose=False, ui_only=True)
    
    # Test query about penalties
    result = rag.query(
        'Sa vite burgim parashikon ligji për vrasje të thjeshtë?', 
        query_mode='analyzed'
    )
    
    answer = result.get('answer', '').lower()
    
    # Check for Albanian number words
    number_words = ['një', 'dy', 'tre', 'katër', 'pesë', 'gjashtë', 'shtatë', 'tetë', 'nëntë', 'dhjetë']
    found_words = [word for word in number_words if word in answer]
    
    print(f"✅ Response length: {len(result.get('answer', ''))} characters")
    print(f"🔍 Albanian number words found: {found_words}")
    print(f"📊 Number conversion working: {len(found_words) > 0}")
    
    # Check for digits (should be minimal)
    import re
    digits = re.findall(r'\d+', answer)
    print(f"🔢 Digits still present: {digits}")
    
    return len(found_words) > 0

if __name__ == "__main__":
    test_number_conversion()
