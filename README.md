# Siem Reap Local Travel & Tour Guide
*CS382 Final Project — RAG-Based AI Search System*

A complete Retrieval-Augmented Generation (RAG) system acting as a local travel and tour
guide for Siem Reap, Cambodia. It retrieves context from a local corpus of plain-text
documents covering temples, markets, transport, weather, and practical travel advice,
then answers questions using Google's Gemini models. Answers are grounded in the corpus
and cite their sources; questions the corpus cannot answer are declined rather than
guessed at.

The corpus is bilingual — most practical-information documents carry a Khmer summary
alongside the English text, so the assistant answers questions asked in either language.

## Setup

**Requirements:** Python 3.10+ (developed on 3.13) and a Gemini API key.

**1. Install dependencies**

```bash
py -m pip install -r requirements.txt
```

**2. Add your Gemini API key**

Create a file named `.env` in the project root containing:

```
GEMINI_API_KEY=your-api-key-here
```

Use exactly this format — a bare `KEY=value` line, with no `export` and no `$env:`
prefix. Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
`.env` is gitignored and must never be committed.

The key is required even in extractive mode, because the embeddings themselves are
generated through the Gemini API.

*Alternatively*, set it as an environment variable instead of using `.env`:

```powershell
$env:GEMINI_API_KEY = "your-api-key-here"   # PowerShell, current session only
```

**3. Run the app**

```bash
py -m streamlit run app.py
```

The app opens at `http://localhost:8501`. First load takes a few seconds while every
chunk is embedded and the FAISS index is built.

**4. Run the evaluation (optional)**

```bash
py evaluate.py
```

Prints Hit@1, Hit@3 and MRR over the ten test queries described in
[EVALUATION.md](EVALUATION.md).

## Architecture

The pipeline is split into four layers, each in its own module:

| Layer | Module | What it does |
|---|---|---|
| **Ingestion** | [rag/ingest.py](rag/ingest.py) | Loads every `.txt` file from `data/sample_docs/` and splits it into fixed-size word-count chunks with a 25% overlap. Chunk size is adjustable from the sidebar. |
| **Embedding & retrieval** | [rag/embed_store.py](rag/embed_store.py) | Turns chunks into 3072-dimension dense vectors via `gemini-embedding-001`, called through the REST API directly to avoid SDK version drift. |
| **Vector store** | [rag/embed_store.py](rag/embed_store.py) | A FAISS `IndexFlatIP` index performing exact inner-product search to return the top-k most similar chunks. |
| **Generation** | [rag/generate.py](rag/generate.py) | Passes the question plus retrieved chunks to `gemini-3.1-flash-lite`. The system prompt restricts it to the supplied sources, requires citations, and refuses prompt-injection and source-dumping attempts. Unanswerable questions return a `[NO_SOURCES]` marker. |

Two answer modes are available. **LLM mode** (default) produces fluent cited prose.
**Extractive mode** skips generation and returns the retrieved passages verbatim, which
is useful for inspecting retrieval quality on its own.

### Data

`data/sample_docs/` holds the indexed corpus: 24 location documents (each with English
and Khmer names, category, coordinates, and a description) plus practical-information
documents on transport, weather, and mobile networks.

`data/extra sample to feed/` holds four further documents — temple etiquette, currency
and payments, essential Khmer phrases, and the Angkor Pass ticket guide. These are **not
indexed at startup**; they exist to be added through the sidebar uploader to demonstrate
live re-indexing.

### Interface

The Streamlit interface ([app.py](app.py)) provides a chat UI with conversation history,
sidebar controls for chunk size and top-k, `.txt` upload with instant re-indexing,
retrieved-source inspection with exact-match highlighting, per-answer latency metrics,
and a built-in evaluation tab.

## Evaluation

Ten paraphrased test queries scored **90% Hit@1, 100% Hit@3, MRR 0.950**. Full
methodology, per-query results, guardrail tests, and discussion are in
[EVALUATION.md](EVALUATION.md).

## Known limitations

- **Text files only.** The loader handles `.txt` and nothing else. PDF or Markdown
  support would require `pypdf` or `BeautifulSoup`.
- **In-memory index.** The FAISS index is rebuilt from scratch on every boot and on every
  chunk-size change, costing an embedding API call for the whole corpus each time.
  Scaling beyond a few thousand documents would require persisting the index to disk.
- **Word-count chunking.** The chunker splits on a fixed word count with no awareness of
  sentence or list boundaries, so a fact can be separated from the context that
  introduces it. This measurably cost one rank-1 result during evaluation.
- **Filename-derived titles.** Source titles come from filenames, so citations display as
  "Document 3" instead of "Ta Prohm". The real name sits unparsed inside the file body.
  One file is also misnamed: a Khmer-titled file about "basic Khmer phrases" actually
  contains the SIM card guide.
- **Similarity scores are not a confidence signal.** Out-of-scope queries score in the
  same range as correct answers (see [EVALUATION.md](EVALUATION.md)), so no threshold can
  flag an unanswerable question. Only the LLM's `[NO_SOURCES]` marker is reliable.
- **No conversational memory in retrieval.** Chat history is displayed, but each query is
  embedded on its own, so follow-ups like "how much does it cost?" do not inherit the
  previous subject.
- **Network dependent.** Both embedding and generation are remote API calls; the app
  cannot run offline and is subject to Gemini rate limits.
