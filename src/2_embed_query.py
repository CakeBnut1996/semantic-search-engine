import numpy as np
from collections import defaultdict
import json, os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

root = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
db_path = os.path.join(root, "data science kdf", "chroma_db")

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection("kdf_embeddings")

print(f"✅ Connected to Chroma DB at {db_path}")
print(f"Total vectors in DB: {collection.count()}\n")

# openai_client = OpenAI()   # assumes your OPENAI_API_KEY is set in env

# ----------------------------------------------------
# (1) Load all embeddings and metadata from DB
# ----------------------------------------------------
records = collection.get(include=["embeddings", "documents", "metadatas"])
embeddings = np.array(records["embeddings"])
chunks = records["documents"]
dataset_ids = [m["dataset"] for m in records["metadatas"]]

print(f"Loaded {len(embeddings)} total chunks from {len(set(dataset_ids))} datasets.\n")

# ----------------------------------------------------
# (2) Load embedding model
# ----------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------------------------------
# (3) Enter user query
# ----------------------------------------------------
query = input("Enter your query: ")
query_embedding = model.encode([query])
similarities = cosine_similarity(query_embedding, embeddings)[0]

# ----------------------------------------------------
# (4) Aggregate by dataset
# ----------------------------------------------------
dataset_scores = defaultdict(list)
for idx, dataset_id in enumerate(dataset_ids):
    dataset_scores[dataset_id].append((similarities[idx], chunks[idx]))

# Rank datasets by their top chunk score
dataset_rankings = []
for dataset_id, entries in dataset_scores.items():
    entries.sort(key=lambda x: x[0], reverse=True)
    top_score = entries[0][0]      # best chunk similarity
    dataset_rankings.append((dataset_id, top_score, entries[:10]))

dataset_rankings.sort(key=lambda x: x[1], reverse=True)

# ----------------------------------------------------
# (5) Display top-k datasets with their top chunks
# ----------------------------------------------------
top_n_datasets = 3
print(f"\nTop {top_n_datasets} relevant datasets:\n")

for rank, (dataset_id, score, top_chunks) in enumerate(dataset_rankings[:top_n_datasets], start=1):
    print(f"\n[{rank}] Dataset: {dataset_id} | Top similarity: {score:.4f}")
    print("-" * 90)
    for i, (sim, chunk) in enumerate(top_chunks, start=1):
        print(f"  Chunk {i}: Similarity={sim:.4f}")
        print(f"  {chunk[:400]}{'...' if len(chunk) > 400 else ''}\n")

# ----------------------------------------------------
# (5) Alternative. Using Instruct model
# ----------------------------------------------------
top_n_datasets = 3
top_dataset_id, top_score, top_chunks = dataset_rankings[0]
related_datasets = [d for d, _, _ in dataset_rankings[1:top_n_datasets]]

context = "\n\n".join([c for _, c in top_chunks])
print(f"\n🔎 Top dataset: {top_dataset_id}")
print(f"Related datasets: {', '.join(related_datasets)}\n")

prompt = f"""
You are a helpful assistant summarizing information from bioenergy datasets.

User question:
{query}

Context (from the most relevant dataset: {top_dataset_id}):
{context[:6000]}

Instructions:
1. Use only the context above to produce a concise and factual answer to the user's question.
2. At the end, list the related datasets that might also contain useful information.
3. If information is missing, say so rather than guessing.

Related datasets: {', '.join(related_datasets) if related_datasets else 'None'}
"""

# --- Query OpenAI instruct/chat model ---
response = openai_client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": "You are an expert research assistant on bioenergy datasets."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.3,
)

answer = response.choices[0].message.content

print("\n💬 ======= FINAL ANSWER =======\n")
print(answer)
print("\n📘 Main dataset:", top_dataset_id)
print("📚 Other relevant datasets:", ", ".join(related_datasets))