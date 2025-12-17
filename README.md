# Course RAG Assistant (SBERT + FAISS)

A robust Retrieval-Augmented Generation (RAG) system designed to query university course materials (PDFs) using advanced semantic search and LLM-based answer generation.

This system ingests course PDFs, constructs a dense vector index using Sentence-BERT and FAISS, retrieves the most relevant passages, and generates highly grounded answers using Groq (LLaMA 3.1) with explicit source citations.

## Key Features

**PDF Ingestion & Chunking:** Efficiently parses and segments course documents.
**Semantic Retrieval:** Uses all-MiniLM-L6-v2 (Sentence-BERT) and FAISS for fast, dense vector search.
**Hybrid Search Capable:** Designed with robust retrieval logic supporting both BM25 (keyword) and Dense (semantic) search, merged via Reciprocal Rank Fusion (RRF).
**End-to-End Pipeline:** Handling the full lifecycle from raw data to LLM inference
**Grounded Answers:** Answers are based only on retrieved context, minimizing hallucinations.
**Interactive UI:** Clean, user-friendly interface built with Streamlit.

## Tech Stack

**Language:** Python 3.10+
**Retrieval:** Sentence-transformers (SBERT), FAISS, Rank-BM25
**Generation (LLM):** Groq API (LLaMA 3.1) 
**Data Processing:** PyPDF2, NLTK

## Repository Structure

```text
RAG/
│
├── app/
│   └── streamlit_app.py        # Streamlit user interface
│
├── data/
│   └── pdfs/                   # Directory for input course PDFs
│
├── outputs/
│   ├── chunks.json             # Processed text chunks (JSON)
│   └── index/                  # FAISS index and embedding IDs
│
├── src/
│   ├── ingest.py               # PDF ingestion and text chunking logic
│   ├── embed_index.py          # Vector embedding generation and FAISS indexing
│   ├── retriever.py            # Dense/BM25 retrieval & Hybrid RRF logic
│   └── pipeline.py             # End-to-end RAG pipeline (Retrieval + Generation)
│
├── .env                        # Example environment variables file
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
Setup and Usage
1. Installation
Clone the repository and install dependencies:

Bash

git clone <repo_url>
cd RAG
pip install -r requirements.txt

2. Configuration
Create a .env file in the root directory and add your API key:

GROQ_API_KEY=gsk_your_key_here
# Optional: OPENAI_API_KEY=sk_...

3. Add Data
Place your course PDFs (lecture notes, slides, textbooks) into the data folder:

RAG/data/

4. Build the Index
Run the ingestion and embedding scripts to prepare your data:

Bash

# Step 1: Extract text and chunk documents
python src/ingest.py

# Step 2: Generate embeddings and build FAISS index
python src/embed_index.py

5. Run the Application
You can run the pipeline directly in the terminal for testing:

Bash

python -m src.pipeline
Or launch the full interactive web interface:

Bash

streamlit run app/streamlit_app.py