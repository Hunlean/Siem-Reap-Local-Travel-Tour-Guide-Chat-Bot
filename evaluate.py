"""
Offline evaluation harness for the Siem Reap RAG system.

Runs a fixed set of test queries against the FAISS index and reports
Hit@1, Hit@3 and Mean Reciprocal Rank (MRR). Every query is phrased the way a
tourist would ask it, deliberately avoiding the wording used in the source
documents, so the numbers measure semantic retrieval rather than keyword overlap.

Usage:
    py evaluate.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore

DATA_FOLDER = "data/sample_docs"
CHUNK_SIZE = 80
TOP_K = 3

# "expected" is a string that must appear in the text of a retrieved chunk.
TEST_CASES = [
    {"query": "Where can I try authentic local food from stalls in the evening?",
     "expected": "Road 60"},
    {"query": "Which temple has enormous tree roots growing over the ruins?",
     "expected": "Ta Prohm"},
    {"query": "Good spot to watch the sun go down over the jungle?",
     "expected": "Phnom Bakheng"},
    {"query": "Where can I see rodents trained to detect explosives?",
     "expected": "APOPO"},
    {"query": "Is there somewhere I can see elephants treated humanely?",
     "expected": "Kulen Elephant Forest"},
    {"query": "Where do families live in houses raised above the water?",
     "expected": "Kampong Phluk"},
    {"query": "How much should I pay a driver for a whole day of sightseeing?",
     "expected": "$15"},
    {"query": "Which months are unbearably hot for walking around?",
     "expected": "March to May"},
    {"query": "Which phone company has the best signal in remote areas?",
     "expected": "Metfone"},
    {"query": "Where can I watch acrobats and live performance at night?",
     "expected": "Phare"},
]

# Queries with no supporting document; the system should retrieve weakly and
# the LLM should decline rather than invent an answer.
OUT_OF_SCOPE = [
    "Which beach resort near Siem Reap is best for swimming in the sea?",
    "What time does the metro subway to Angkor Wat run?",
]


def main():
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_SIZE // 4)
    store = VectorStore()
    store.build(chunks)
    print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks\n")

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []

    print(f"{'#':<3} {'Rank':<6} {'Top score':<10} Query")
    print("-" * 78)

    for i, case in enumerate(TEST_CASES, 1):
        retrieved = store.query(case["query"], top_k=TOP_K)
        rank = None
        for position, (chunk, _) in enumerate(retrieved, 1):
            if case["expected"].lower() in chunk.text.lower():
                rank = position
                break

        if rank == 1:
            hits_at_1 += 1
        if rank is not None:
            hits_at_3 += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

        top_score = retrieved[0][1] if retrieved else 0.0
        rank_label = str(rank) if rank else "MISS"
        print(f"{i:<3} {rank_label:<6} {top_score:<10.3f} {case['query'][:52]}")

    n = len(TEST_CASES)
    print("-" * 78)
    print(f"Hit@1: {hits_at_1}/{n} ({hits_at_1 / n * 100:.0f}%)")
    print(f"Hit@3: {hits_at_3}/{n} ({hits_at_3 / n * 100:.0f}%)")
    print(f"MRR:   {sum(reciprocal_ranks) / n:.3f}")

    print("\nOut-of-scope controls (lower top score = better):")
    for q in OUT_OF_SCOPE:
        retrieved = store.query(q, top_k=1)
        print(f"  {retrieved[0][1]:.3f}  {q[:56]}")


if __name__ == "__main__":
    main()
