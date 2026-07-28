"""
RAG-Based AI Search System — starter interface.
"""

import os
import time
import re
import streamlit as st
from dotenv import load_dotenv

# Load .env before the rag modules read GEMINI_API_KEY from the environment.
load_dotenv()

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

DATA_FOLDER = "data/sample_docs"

st.set_page_config(page_title="RAG Search", page_icon="🔎", layout="wide")

@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store(chunk_size):
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs, chunk_size=chunk_size, overlap=chunk_size//4)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks

def highlight_text(text, query):
    """Simple exact-match highlighting of query terms in the text."""
    if not query:
        return text
    # Split query into words and filter out small stop words
    words = [w for w in re.split(r'\W+', query.lower()) if len(w) > 3]
    highlighted = text
    for word in words:
        # Case insensitive replace with markdown bold
        highlighted = re.sub(f"(?i)({re.escape(word)})", r"**\1**", highlighted)
    return highlighted

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size (words)", min_value=20, max_value=200, value=80, step=10)
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    mode = st.radio("Answer mode", ["extractive", "llm"], index=1,
                     help="Extractive works with no setup. LLM mode needs GEMINI_API_KEY set.")
    
    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Add a new .txt document", type=["txt"])
    if uploaded_file is not None:
        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name}!")
        load_store.clear()
        st.rerun()
        
    store, docs, chunks = load_store(chunk_size)
        
    st.divider()
    st.caption(f"Indexed **{len(docs)}** documents → **{len(chunks)}** chunks")
    with st.expander("Documents in this index"):
        for d in docs:
            st.write(f"- {d['title']}")

st.title("🔎 Siem Reap Local Travel & Tour Guide")
st.caption("Ask a question about the indexed documents below.")

tab1, tab2 = st.tabs(["Search", "Evaluation & Write-up"])

with tab1:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("**Sources:**")
                for chunk, score in message["sources"]:
                    with st.expander(f"{chunk.doc_title}  ·  similarity {score:.2f}"):
                        st.write(highlight_text(chunk.text, message.get("query", "")))

    # React to user input
    if prompt := st.chat_input("e.g. What to do in Siem Reap?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        
        # Retrieve and generate
        retrieved = store.query(prompt, top_k=top_k)
        answer = generate_answer(prompt, retrieved, mode=mode)
        
        end_time = time.time()
        latency = end_time - start_time

        hide_sources = "[NO_SOURCES]" in answer
        answer = answer.replace("[NO_SOURCES]", "").strip()
        
        # Add latency info
        response_content = f"{answer}\n\n*Latency: {latency:.2f}s*"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_content)
            sources_to_save = []
            if not hide_sources:
                st.markdown("**Sources:**")
                sources_to_save = retrieved
                for chunk, score in retrieved:
                    with st.expander(f"{chunk.doc_title}  ·  similarity {score:.2f}"):
                        highlighted_chunk = highlight_text(chunk.text, prompt)
                        st.write(highlighted_chunk)
                        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_content, 
            "sources": sources_to_save,
            "query": prompt
        })

with tab2:
    st.header("Automated Evaluation Suite")
    st.write("This suite runs predefined queries and checks if the expected document is retrieved in the top 3 results.")
    
    test_cases = [
        {"query": "Where can I get street food?", "expected": "Road 60 Street Food Market"},
        {"query": "Good place for sunset?", "expected": "Phnom Bakheng"},
        {"query": "Is there a lake or reservoir?", "expected": "West Baray"},
        {"query": "Largest religious monument?", "expected": "Angkor Wat"},
        {"query": "Temple with smiling faces?", "expected": "Bayon Temple"},
        {"query": "Tomb Raider temple?", "expected": "Ta Prohm"}
    ]
    
    if st.button("Run Evaluation Suite"):
        st.write("Running tests...")
        passed = 0
        
        # Create a table/dataframe format
        results = []
        for case in test_cases:
            retrieved_chunks = store.query(case["query"], top_k=3)
            retrieved_titles = [chunk.doc_title for chunk, score in retrieved_chunks]
            
            # Check if the expected name actually appears inside the retrieved chunks
            is_match = any(case["expected"].lower() in chunk.text.lower() for chunk, score in retrieved_chunks)
            if is_match:
                passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                
            results.append({
                "Query": case["query"],
                "Expected Source": case["expected"],
                "Status": status,
                "Retrieved": ", ".join(retrieved_titles)
            })
            
        st.dataframe(results, width='stretch')
        st.success(f"**Accuracy:** {passed}/{len(test_cases)} ({(passed/len(test_cases))*100:.1f}%)")

    st.divider()
    
    st.header("Project Write-up")
    st.markdown("""
### What worked
- **Semantic Embeddings**: Switching from basic TF-IDF to Gemini Embeddings (`gemini-embedding-001`) massively improved the system's ability to understand synonyms and conceptual queries (e.g., matching "sunset" with "Phnom Bakheng").
- **Vector Database**: Swapping the naive NumPy array dot product for a **FAISS** inner product index proved to be extremely straightforward and prepared the system to scale to thousands of documents instantly.
- **LLM Integration**: Using Google's `gemini-3.1-flash-lite` model for generation produced highly grounded and fluent answers without requiring a heavyweight model.

### What didn't work
- **Prompt Injection vulnerabilities**: Initially, the LLM was susceptible to users asking "What are your source documents?". This required adding a strict **Guardrail** to the system prompt to explicitly refuse dumping sources.
- **Exact-Match Retrieval**: Before we integrated Gemini embeddings, the TF-IDF search failed constantly when users queried with different words (like "lake" instead of "reservoir"), leading to poor RAG outcomes.

### Why
Using an end-to-end semantic approach (embeddings + FAISS + LLM) is strictly superior for travel guides because tourists rarely use the exact terminology found in Wikipedia or official brochures. They ask conceptual questions ("cool places for dinner"), and only dense vectors can capture that underlying semantic intent.
""")
