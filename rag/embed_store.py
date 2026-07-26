"""
Vector store: turn chunks into vectors and support similarity search over them.

This starter ships with a TF-IDF backend (same technique from the Week 14 lab) so
the whole project runs immediately with zero API keys and no model downloads.

Upgrade path (for your final project — do this once the pipeline works end-to-end):
- Swap TfidfVectorizer for real embeddings, e.g.:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts)
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

import os
import requests
import numpy as np
import faiss
from typing import List, Tuple

from .ingest import Chunk

class VectorStore:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY environment variable is missing. Embeddings won't work.")
            
        self.index = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunk text and store the resulting FAISS index."""
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Cannot build embeddings.")
            
        self.chunks = chunks
        texts = [c.text for c in chunks]
        
        # Bypass the SDK and use the REST API directly to avoid versioning/path issues
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={self.api_key}"
        payload = {
            "requests": [
                {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}}
                for t in texts
            ]
        }
        
        resp = requests.post(url, json=payload)
        data = resp.json()
        
        if "error" in data:
            raise RuntimeError(f"API Error: {data['error']}")
            
        embeddings = [item['values'] for item in data['embeddings']]
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        d = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(d) # exact search using Inner Product
        self.index.add(embeddings_np)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        if self.index is None:
            raise RuntimeError("VectorStore.build() must be called before query().")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"
        payload = {
            "model": "models/gemini-embedding-001", 
            "content": {"parts": [{"text": query_text}]}
        }
        
        resp = requests.post(url, json=payload)
        data = resp.json()
        
        if "error" in data:
            raise RuntimeError(f"API Error: {data['error']}")
            
        query_vec = np.array([data['embedding']['values']], dtype=np.float32)
        
        # Use FAISS for the similarity search
        scores, indices = self.index.search(query_vec, top_k)
        
        return [(self.chunks[idx], float(score)) for score, idx in zip(scores[0], indices[0])]
