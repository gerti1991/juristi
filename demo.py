"""
Albanian Legal RAG System Demo
Shows the integrated document collection and capabilities
"""

import json
import os

def show_integrated_documents():
    """Show all integrated Albanian legal documents"""
    
    print("🏛️ ALBANIAN LEGAL RAG SYSTEM")
    print("=" * 60)
    print("📚 Your system now includes:")
    
    # Load PDF documents
    pdf_file = os.path.join("legal_documents", "pdf_rag_documents.json")
    
    if os.path.exists(pdf_file):
        with open(pdf_file, 'r', encoding='utf-8') as f:
            pdf_docs = json.load(f)
        
        # Count documents by type
        doc_types = {}
        for doc in pdf_docs:
            title = doc['title']
            if title not in doc_types:
                doc_types[title] = 0
            doc_types[title] += 1
        
        print(f"\n📄 {len(pdf_docs)} PDF Document Chunks from {len(doc_types)} Albanian Legal Codes:")
        
        key_documents = [
            "Kodi Civil 2023",
            "Kodi I Punes Ligj 2024", 
            "Ligj Nr. 7895 1995 Kodi Penal I Përditësuar",
            "Ligj Nr. 9062 08052003 Kodi I Familjes",
            "Kodi Zjedhor Perditesim 2025",
            "Ligj 2014 07 31 102, Kodi Doganor I Rsh"
        ]
        
        for i, doc_title in enumerate(key_documents, 1):
            count = doc_types.get(doc_title, 0)
            if count > 0:
                print(f"  ✅ {i}. {doc_title} - {count} chunks")
        
        # Show additional documents
        other_docs = [title for title in doc_types if title not in key_documents]
        if other_docs:
            print(f"\n📋 Plus {len(other_docs)} additional legal documents:")
            for doc in other_docs[:5]:  # Show first 5
                print(f"  • {doc}")
            if len(other_docs) > 5:
                print(f"  • ... and {len(other_docs) - 5} more")
    
    else:
        print("❌ PDF documents not found!")
        return False
    
    # Show sample content
    print(f"\n🔍 Sample Content from Civil Code 2023:")
    civil_code_chunks = [doc for doc in pdf_docs if "Civil 2023" in doc['title']]
    if civil_code_chunks:
        sample = civil_code_chunks[0]['content'][:200]
        print(f"   📄 {sample}...")
    
    print(f"\n🔍 Sample Content from Labor Code 2024:")
    labor_code_chunks = [doc for doc in pdf_docs if "Punes" in doc['title']]
    if labor_code_chunks:
        sample = labor_code_chunks[0]['content'][:200]
        print(f"   📄 {sample}...")
    
    return True

def show_capabilities():
    """Show system capabilities"""
    
    print("\n🚀 SYSTEM CAPABILITIES")
    print("-" * 40)
    print("✅ Comprehensive Albanian Legal Database")
    print("✅ Semantic Search in Albanian Language")
    print("✅ Cloud AI Integration (Groq API)")
    print("✅ Real-time Legal Document Search")
    print("✅ Professional Legal Responses")
    
    print("\n💼 Can Answer Questions About:")
    print("   • Labor Laws & Employee Rights")
    print("   • Civil Code & Contracts")
    print("   • Criminal Law & Penalties")
    print("   • Family Law & Marriage")
    print("   • Business Registration & Corporate Law")
    print("   • Customs & Trade Regulations")
    print("   • Electoral & Constitutional Law")

def show_sample_queries():
    """Show sample queries the system can handle"""
    
    print("\n❓ SAMPLE QUERIES YOU CAN ASK")
    print("-" * 40)
    
    sample_queries = [
        "Sa dit pushimi kam në punë të detyrueshme?",
        "Çfarë sanksionesh ka për vjedhje në Kodin Penal?",
        "Si mund të regjistroj një kompani në Shqipëri?",
        "Çfarë rregullon Kodi Civil për kontratat?",
        "Sa është pagesa minimale sipas Kodit të Punës?",
        "Cilat janë procedurat e divorcit sipas Kodit të Familjes?",
        "Çfarë taksash duhet të paguaj për import?",
        "Si funksionon sistemi zgjedhor në Shqipëri?"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")

def main():
    """Main demo function"""
    
    success = show_integrated_documents()
    
    if success:
        show_capabilities()
        show_sample_queries()
        
        print("\n" + "=" * 60)
        print("🎉 YOUR ALBANIAN LEGAL RAG IS READY!")
        print("🚀 Start the application:")
        print("   streamlit run app.py")
        print("\n💡 The system will:")
        print("   1. Search through 251 document chunks")
        print("   2. Find the most relevant Albanian legal text")
        print("   3. Generate professional responses using AI")
        print("   4. Provide accurate legal information in Albanian")
    else:
        print("\n❌ Please run simple_pdf_processor.py first to process PDFs")

if __name__ == "__main__":
    main()
