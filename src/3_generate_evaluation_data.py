from src.semantic_search_engine import retrieve_data, generate_answers, _collection, _load_gemini_client
import json, os
import chromadb
from enum import Enum
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict
import time
from sentence_transformers import SentenceTransformer

ROOT = r"C:\Users\mmh\Documents\data science kdf"
DB_PATH = os.path.join(ROOT, "chroma_db")
KEY_PATH = os.path.join(ROOT, "keys", "Gemini.yaml")
OUTPUT_JSON = os.path.join(ROOT, "eval","chunk_training_examples.json")
OUTPUT_CSV = os.path.join(ROOT, "eval","chunk_training_examples.csv")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("kdf_embeddings")

_client_gem = _load_gemini_client()

### Create evaluation dataset and save
records = _collection.get(include=["embeddings", "documents", "metadatas"])
chunks = records["documents"]
dataset_ids = [m["dataset"] for m in records["metadatas"]]

class QAItem(BaseModel):
    question: str
    answer: str

class HardNegativeItem(BaseModel):
    question: str

class ChunkType(str, Enum):
    content = "content"      # continuous narrative text
    metadata = "metadata"    # structured fields: names, emails, dates, IDs

class ChunkTrainingExample(BaseModel):
    chunk_type: ChunkType = Field(..., description="Type of chunk: content or metadata")
    positives: List[QAItem]
    hard_negatives: List[HardNegativeItem]

def make_prompt(chunk: str) -> str:
    prompt = f"""
    You are generating evaluation data for semantic retrieval.

STEP 1 — Classify the chunk as:
• "content" — narrative/explanatory text where semantic questions are appropriate.
• "metadata" — structured fields (names, organizations, dates, emails, phone numbers, project titles).

STEP 2 — Generate TWO positive QA pairs:
• Questions must be fully self-contained.
• For CONTENT chunks: ask semantic, meaning-based questions answerable from the text.
• For METADATA chunks: ask explicit metadata questions that reference the field type 
  (e.g., “According to the metadata, what is the contact email for Henriette Jager?”).
• Answers must come strictly from the chunk.

STEP 3 — Generate TWO hard negative questions:
• Must be topically related, but unanswerable from the chunk.
• Must be self-contained.
• Should not rely on document structure or metadata fields not present in the chunk.

    
    Return your output in the specified JSON format.

    CHUNK:
    \"\"\"{chunk}\"\"\"
    """

    return prompt


def generate_examples(chunk: str) -> Dict:
    prompt = make_prompt(chunk)
    resp = _client_gem.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ChunkTrainingExample,   # <— SCHEMA ENFORCED
        }
    )

    # response.text already obeys the schema; parse to dict
    return json.loads(resp.text)


results = []

for i, chunk in enumerate(chunks):
    print(f"\n=== Processing chunk {i+1}/{len(chunks)} ===")

    examples = generate_examples(chunk)
    time.sleep(4.2)

    results.append({
        "chunk_index": i,
        "chunk_text": chunk,
        "dataset_id": dataset_ids[i],
        "examples": examples
    })


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Done. Saved RAG training examples.")


####### Convert it to csv
flat_rows = []

for item in results:
    dataset_id = item["dataset_id"]
    chunk_index = item["chunk_index"]
    chunk_text = item["chunk_text"]
    examples = item["examples"]

    # --- Positive examples ---
    for qa in examples["positives"]:
        flat_rows.append({
            "dataset_id": dataset_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "question_type": "positive",
            "question": qa["question"],
            "answer": qa["answer"]
        })

    # --- Hard negative examples ---
    for hn in examples["hard_negatives"]:
        flat_rows.append({
            "dataset_id": dataset_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "question_type": "hard_negative",
            "question": hn["question"],
            "answer": ""   # empty because hard negatives have no answer
        })

# Convert to DataFrame
df = pd.DataFrame(flat_rows)
df.to_csv(OUTPUT_CSV, index=False)
