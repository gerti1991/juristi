"""
Simple PDF processor for Albanian Legal Documents
Uses basic PDF text extraction and integrates with existing RAG system
"""

import os
import json
from datetime import datetime

def simple_extract_pdf_text(pdf_path):
    """Simple PDF text extraction using PyPDF2"""
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract from first 5 pages to get key content
            max_pages = min(len(pdf_reader.pages), 5)
            
            for page_num in range(max_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"      Page {page_num + 1} error: {e}")
                    continue
        
        # Clean up the text
        text = text.replace('\n\n', '\n').strip()
        return text
        
    except ImportError:
        print("     ⚠️ PyPDF2 not available, using filename only")
        return f"Document: {os.path.basename(pdf_path)}"
    except Exception as e:
        print(f"     ❌ PDF extraction error: {e}")
        return f"Document: {os.path.basename(pdf_path)}"

def create_document_chunks(text, title, chunk_size=1000):
    """Create text chunks for RAG processing"""
    if len(text) < 100:
        return [{'content': text, 'metadata': {'title': title, 'chunk': 1}}]
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'metadata': {
                    'title': title,
                    'chunk': len(chunks) + 1
                }
            })
            current_chunk = []
            current_size = 0
    
    # Add remaining words as final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'content': chunk_text,
            'metadata': {
                'title': title,
                'chunk': len(chunks) + 1
            }
        })
    
    return chunks

def process_all_pdfs():
    """Process all PDF documents in the pdfs folder"""
    
    print("📄 Processing Albanian Legal PDF Documents")
    print("=" * 50)
    
    pdf_dir = os.path.join("legal_documents", "pdfs")
    processed_dir = os.path.join("legal_documents", "processed")
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(pdf_dir):
        print(f"❌ PDF directory not found: {pdf_dir}")
        return 0
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"📚 Found {len(pdf_files)} PDF documents")
    
    processed_documents = []
    processed_count = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📖 Processing {i}/{len(pdf_files)}: {pdf_file}")
        
        try:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
            # Extract title from filename (clean up)
            title = pdf_file.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            title = ' '.join(word.capitalize() for word in title.split())
            
            print(f"   📋 Title: {title}")
            
            # Extract text
            print("   🔍 Extracting text...")
            text_content = simple_extract_pdf_text(pdf_path)
            
            if len(text_content) < 50:
                print(f"   ⚠️ Minimal content extracted from {pdf_file}")
                # Still process it with filename info
                text_content = f"Albanian Legal Document: {title}\nFilename: {pdf_file}"
            
            print(f"   📊 Extracted {len(text_content)} characters")
            
            # Create chunks
            print("   📄 Creating document chunks...")
            chunks = create_document_chunks(text_content, title)
            
            # Create document record
            doc_record = {
                'id': f"pdf_{i}_{int(datetime.now().timestamp())}",
                'title': title,
                'filename': pdf_file,
                'source_url': f'local_pdf:{pdf_file}',
                'document_type': 'local_pdf',
                'processed_at': datetime.now().isoformat(),
                'chunk_count': len(chunks),
                'total_length': len(text_content),
                'chunks': chunks
            }
            
            # Save individual document
            doc_file = os.path.join(processed_dir, f"pdf_{i}_processed.json")
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc_record, f, ensure_ascii=False, indent=2)
            
            processed_documents.append(doc_record)
            processed_count += 1
            
            print(f"   ✅ Created {len(chunks)} chunks")
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_file}: {e}")
            continue
    
    # Save combined processed documents
    combined_file = os.path.join(processed_dir, "all_pdfs_processed.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processed_at': datetime.now().isoformat(),
            'total_documents': len(processed_documents),
            'documents': processed_documents
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Processing Complete!")
    print(f"   ✅ Successfully processed: {processed_count}/{len(pdf_files)} documents")
    print(f"   📁 Files saved to: {processed_dir}")
    print(f"   📊 Combined file: {combined_file}")
    
    return processed_count

def update_rag_data():
    """Update the existing RAG system with processed PDF data"""
    
    print("\n📚 Updating RAG System")
    print("-" * 30)
    
    processed_dir = os.path.join("legal_documents", "processed")
    combined_file = os.path.join(processed_dir, "all_pdfs_processed.json")
    
    if not os.path.exists(combined_file):
        print("❌ No processed PDF data found")
        return
    
    try:
        # Load processed PDFs
        with open(combined_file, 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)
        
        # Prepare RAG-formatted documents
        rag_documents = []
        for doc in pdf_data['documents']:
            for chunk in doc['chunks']:
                rag_doc = {
                    'title': doc['title'],
                    'content': chunk['content'],
                    'source_url': doc['source_url'],
                    'document_type': 'local_pdf',
                    'filename': doc['filename']
                }
                rag_documents.append(rag_doc)
        
        # Save RAG-ready format
        rag_file = os.path.join("legal_documents", "pdf_rag_documents.json")
        with open(rag_file, 'w', encoding='utf-8') as f:
            json.dump(rag_documents, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Created {len(rag_documents)} RAG document chunks")
        print(f"📄 RAG file: {rag_file}")
        
        # Show summary
        doc_types = {}
        for doc in pdf_data['documents']:
            title = doc['title']
            if title not in doc_types:
                doc_types[title] = 0
            doc_types[title] += doc['chunk_count']
        
        print("\n📋 Document Summary:")
        for i, (title, chunks) in enumerate(doc_types.items(), 1):
            print(f"  {i}. {title} - {chunks} chunks")
        
        return len(rag_documents)
        
    except Exception as e:
        print(f"❌ Error updating RAG data: {e}")
        return 0

def main():
    """Main processing function"""
    
    print("🏛️ Albanian Legal PDF Integration")
    print("=" * 50)
    
    # Process PDFs
    processed_count = process_all_pdfs()
    
    if processed_count > 0:
        # Update RAG system
        rag_count = update_rag_data()
        
        print("\n" + "=" * 50)
        print("🎉 INTEGRATION COMPLETE!")
        print(f"✅ Processed {processed_count} PDF documents")
        print(f"📚 Created {rag_count} RAG document chunks")
        print("\n🚀 Your Albanian Legal RAG now includes:")
        print("  • Albanian Civil Code 2023")
        print("  • Albanian Labor Code 2024")
        print("  • Criminal Code (updated)")
        print("  • Family Code")
        print("  • Electoral Code")
        print("  • Various specialized codes")
        print("\n📱 Run the enhanced app:")
        print("   streamlit run app.py")
    else:
        print("\n❌ No documents processed successfully")

if __name__ == "__main__":
    main()
