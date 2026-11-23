# Retrieval-Augmented Generation (RAG) System (`RAG_sys`)

A modular Python Retrieval-Augmented Generation (RAG) pipeline using ChromaDB for vector storage, PDF/text ingestion, and advanced retrieval strategies including Multi-Query Retrieval and Maximal Marginal Relevance (MMR).

## Project Structure

```text
RAG_sys/
├── vector_folder/
│   └── DB.py               # Vector database configuration and utilities
├── Arvix_my.py             # arXiv research paper retrieval and querying
├── create_database.py      # Document chunking, embedding, and ChromaDB creation
├── lbdl.pdf                # Sample PDF document for ingestion
├── main.py                 # Primary RAG pipeline execution script
├── main3.py                # Alternative / extended RAG workflow
├── mmr.py                  # Maximal Marginal Relevance (MMR) retriever implementation
├── multiquery.py           # Multi-Query Retriever for enhanced document search
├── notes.txt               # Sample text document for ingestion
├── notes2.txt              # Additional sample text notes
├── page.py                 # Web page loading and text extraction
├── pdf.py                  # PDF document loader and parser
├── requirement.txt         # Project Python dependencies
└── test.py                 # Pipeline testing and retrieval verification