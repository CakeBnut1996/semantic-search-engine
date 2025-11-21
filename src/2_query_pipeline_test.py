import numpy as np
from collections import defaultdict
from google import genai
from typing import List
from pydantic import BaseModel, Field
import re, json, os, enum, yaml
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
query = input("Enter your query: ") # How can biomass feedstocks be used to provide non-fuel ecosystem goods and services?
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
    dataset_rankings.append((dataset_id, top_score, entries[:8]))

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


top_n_datasets = 3
top_dataset_id, top_score, top_chunks = dataset_rankings[0]
related_datasets = [d for d, _, _ in dataset_rankings[1:top_n_datasets]]


# ----------------------------------------------------
# Using Instruct model
# ----------------------------------------------------
prompt = f"""
You are a helpful assistant summarizing information from bioenergy datasets.

User question:
{query}

Context (retrieved datasets and their most relevant text chunks):
{json.dumps(dataset_rankings[:top_n_datasets], indent=2)}

Instructions:
1. Use the context above to produce a concise and factual answer to the user's question.
2. If information is missing, say so rather than guessing.
"""

# --- Query OpenAI instruct/chat model ---
class KDFEntry(BaseModel):
    name_data: str = Field(description="The name of the first-ranked dataset")
    summary: str = Field(description="A concise summary of the content from the first-ranked dataset")
    other_data_name: List[str] = Field(description="Names of other relevant datasets")


class DatasetSummary(BaseModel):
    name: str = Field(..., description="Name of the dataset")
    summary: str = Field(..., description="A summary of the dataset content relevant to the query")
    quote: str = Field(..., description="Representative quote or key snippet from the dataset")

class KDFResponse(BaseModel):
    answer: str = Field(..., description="A concise and factual answer to the user's question")
    supporting_datasets: List[DatasetSummary] = Field(
        ..., description="List of datasets that support the answer, with summaries and representative quotes"
    )

key_path = os.path.join(root, "data science kdf", "keys", "Gemini.yaml")
with open(key_path, "r") as f:
    config = yaml.safe_load(f)

api_key = config["api_key"]
client_gem = genai.Client(api_key=api_key)

response = client_gem.models.generate_content(
    model='gemini-2.0-flash',
    contents = prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': KDFResponse,
    },)
print(response.text)