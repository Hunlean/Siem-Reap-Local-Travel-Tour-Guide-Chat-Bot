# Siem Reap Local Travel & Tour Guide
*CS382 Final Project - RAG-Based AI Search System*

This is a complete Retrieval-Augmented Generation (RAG) system acting as a local travel and tour guide for Siem Reap. It retrieves context from a provided local document corpus (24 documents) and answers user questions intelligently using Google's Gemini LLMs.

## Features & Architecture

The system's architecture is modularized into four distinct layers:

1. **Ingestion (`rag/ingest.py`)**: Loads `24` plain text documents detailing locations in Siem Reap. Uses a fixed-size word-count chunking strategy with dynamic overlaps to split the documents into retrievable pieces.
2. **Retrieval & Embeddings (`rag/embed_store.py`)**: Uses Google's state-of-the-art `gemini-embedding-001` via REST API to turn chunks into dense semantic vectors. 
3. **Vector Store**: The embeddings are loaded into a high-performance **FAISS** index (Facebook AI Similarity Search) utilizing inner product similarity to instantly retrieve the top-k most relevant chunks.
4. **Generation (`rag/generate.py`)**: Passes the query and retrieved context to `gemini-3.1-flash-lite`. The LLM has strict system guardrails to prevent prompt injection and hallucinations, citing exactly where it got its information.

## Interface
The Streamlit interface (`app.py`) provides:
- A modern chat UI with conversation history.
- Dynamic settings (Chunk Size, Top-K Retrieval) via the sidebar.
- File uploads: drop a new `.txt` document to instantly re-index the FAISS database.
- A built-in Automated Evaluation Suite to grade semantic search performance.
- Exact-match highlighting and latency metrics on every generation.

## Run it

1. Make sure you install the dependencies:
```bash
pip install -r requirements.txt
```
*(Requires `faiss-cpu`, `google-genai`, `streamlit`, `requests`, and `numpy`)*

2. Set your Gemini API key in your terminal:
```bash
$env:GEMINI_API_KEY="your-api-key"
```

3. Run the Streamlit app:
```bash
py -m streamlit run app.py
```

## Known Limitations

- **Text only**: The ingestion engine currently only handles `.txt` files. Support for PDFs or Markdown files would require importing `pypdf` or `BeautifulSoup`.
- **In-memory indexing**: The FAISS index is generated entirely in-memory on application boot. While fast for a few thousand documents, scaling to millions would require persisting the FAISS index to disk.
- **Word-count chunking**: The chunker splits aggressively on word count, meaning a chunk might occasionally split midway through a sentence. A sentence-aware chunker using NLTK or SpaCy would improve contextual boundary retention.
