"""
Ingestion: load raw documents from disk and split them into overlapping chunks.

Upgrade path (for your final project):
- Add PDF/HTML/Markdown loaders (e.g. pypdf, BeautifulSoup) alongside plain .txt
- Swap the naive word-count chunker below for a sentence- or token-aware chunker
- Store document metadata (source URL, author, date) alongside each chunk
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str


def load_documents(folder: str) -> List[dict]:
    """Load every .txt file in `folder` into {"title": ..., "text": ...} dicts."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        title = os.path.splitext(filename)[0].replace("_", " ").title()
        docs.append({"title": title, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping word-count chunks (simple, dependency-free)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunk_records(docs: List[dict], chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            records.append(Chunk(chunk_id=f"{doc['title']}::{i}", doc_title=doc["title"], text=piece))
    return records
