#!/usr/bin/env python3
"""Embed a few test texts and print pairwise distance/similarity matrices.

Usage (from project root, with venv activated):
    python test_distances.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import ollama
from rag_demo import DEFAULT_EMBEDDING_MODEL

# ------------------------------------------------------------------
# 1. Define test texts
# ------------------------------------------------------------------
texts = ["Hawaii", "pineapple", "apple"]

print(f"Embedding model : {DEFAULT_EMBEDDING_MODEL}")
print(f"Texts           : {texts}\n")

# ------------------------------------------------------------------
# 2. Embed
# ------------------------------------------------------------------
response = ollama.embed(model=DEFAULT_EMBEDDING_MODEL, input=texts)
embeddings = response["embeddings"]
dim = len(embeddings[0])
print(f"Embedding dim   : {dim}\n")

# ------------------------------------------------------------------
# 3. Compute pairwise cosine similarity & L2 distance
# ------------------------------------------------------------------
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(dot(a, a))

def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def l2_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

n = len(texts)

# ------------------------------------------------------------------
# 4. Print cosine similarity matrix
# ------------------------------------------------------------------
col_w = max(len(t) for t in texts) + 2

print("=== Cosine Similarity (1.0 = identical, 0.0 = unrelated) ===")
print(f"{'':>{col_w}}", "  ".join(f"{t:>{col_w}}" for t in texts))
for i in range(n):
    row = "  ".join(f"{cosine_sim(embeddings[i], embeddings[j]):>{col_w}.4f}" for j in range(n))
    print(f"{texts[i]:>{col_w}}  {row}")

# ------------------------------------------------------------------
# 5. Print L2 distance matrix (what ChromaDB uses by default)
# ------------------------------------------------------------------
print(f"\n=== L2 Distance (0.0 = identical, higher = more distant) ===")
print(f"{'':>{col_w}}", "  ".join(f"{t:>{col_w}}" for t in texts))
for i in range(n):
    row = "  ".join(f"{l2_distance(embeddings[i], embeddings[j]):>{col_w}.4f}" for j in range(n))
    print(f"{texts[i]:>{col_w}}  {row}")

# ------------------------------------------------------------------
# 6. Also store in ChromaDB so you can query them
# ------------------------------------------------------------------
import chromadb
from rag_demo import VECTOR_DB_DIR, COLLECTION_NAME

client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
col = client.get_or_create_collection(COLLECTION_NAME)

ids = [f"test_{t.lower()}" for t in texts]
col.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"company": "test", "source": "distance_test"} for _ in texts],
)
print(f"\nUpserted {len(texts)} test vectors into '{COLLECTION_NAME}'.")

# ------------------------------------------------------------------
# 7. Query ChromaDB to verify distances match
# ------------------------------------------------------------------
print(f"\n=== ChromaDB query: 'apple' vs all test vectors ===")
q_emb = embeddings[texts.index("apple")]
results = col.query(
    query_embeddings=[q_emb],
    n_results=len(texts),
    where={"source": {"$eq": "distance_test"}},
    include=["documents", "distances"],
)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  {doc:<15} distance = {dist:.4f}")

# Cleanup hint
print(f"\nTo remove test vectors:")
print(f'  col.delete(where={{"source": {{"$eq": "distance_test"}}}})')
