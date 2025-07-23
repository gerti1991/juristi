# PDF Files Note

⚠️ **Important**: The PDF files containing Albanian legal documents are not included in this repository due to GitHub's file size limitations (some files exceed 100MB).

## To use the system:

1. **Add your own PDF files** to the `legal_documents/pdfs/` directory
2. **Run the setup script** to process them:
   ```bash
   python setup.py
   ```
3. **Or use the existing processed documents** that are already included in the system

## Current System Status:

The system comes pre-loaded with **251 document chunks** from **19 Albanian legal codes** that have been processed and are ready to use. You can start the application immediately without adding PDF files.

## File Structure:
```
legal_documents/
├── pdfs/                           # Your PDF files go here (gitignored)
├── processed/                      # Processed document chunks (gitignored)  
└── albanian_legal_rag_documents.json  # Pre-processed legal documents (included)
```

The system will work with the existing processed documents, or you can add your own PDFs and reprocess them using `setup.py`.

## Adding Your Own Legal Documents:

1. **Create the pdfs directory**: `mkdir legal_documents/pdfs`
2. **Copy your PDF files** into that directory
3. **Run processing**: `python setup.py`
4. **Restart the application**: `streamlit run app.py`

The system will automatically process and integrate your documents into the RAG system.
