import os, json, yaml, re, logging
import streamlit as st
from collections import defaultdict
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel, Field
from google import genai


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = os.path.dirname(os.path.abspath(__name__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DB_PATH = os.path.join(ROOT, "chroma_db")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_N_DATASETS = 5
TOP_N_CHUNKS = 5


# -------------------------------------------------------------------------
# Pydantic schema for structured output
# -------------------------------------------------------------------------
class DatasetSummary(BaseModel):
    name: str = Field(..., description="Name of the dataset")
    summary: str = Field(..., description="A summary of the dataset content relevant to the query based on its returned chunks")
    quote: str = Field(..., description="Top-ranked chunk quoted from the dataset")

class KDFResponse(BaseModel):
    answer: str = Field(..., description="A concise and factual answer to the user's question")
    name_top: str = Field(..., description="Name of the top-ranked dataset")
    supporting_datasets: List[DatasetSummary] = Field(
        ..., description="List of datasets that support the answer, with summaries and representative quotes"
    )
# -------------------------------------------------------------------------
# Initialization (done once)
# -------------------------------------------------------------------------
def _load_chroma_collection() -> chromadb.Collection:
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("kdf_embeddings")
        logging.info(f"✅ Connected to Chroma DB at {DB_PATH}")
        logging.info(f"Total vectors: {collection.count()}")
        return collection
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load Chroma collection: {e}")


def _load_gemini_client() -> genai.Client:
    # api_key = config.get("keys")['gemini']
    api_key = st.secrets["gkeys"]["gemini"]
    if not api_key:
        raise ValueError("Gemini API key missing in YAML file.")
    return genai.Client(api_key=api_key)


# Global in-memory cache (so Streamlit doesn’t re-init repeatedly)
_collection = _load_chroma_collection()
_model = SentenceTransformer(MODEL_NAME)
_client_gem = _load_gemini_client()


# -------------------------------------------------------------------------
# Core search function
# -------------------------------------------------------------------------
def retrieve_data(user_query: str, sim_thre: float = 0.1): # Maximum number of documents
    """Perform semantic retrieval and summarization for a given user query."""
    if not user_query.strip():
        return {"error": "Empty query."}

    # --- Encode query ---
    query_emb = _model.encode(user_query).tolist()

    # --- Retrieve top-K chunks using Chroma ---
    res = _collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_N_DATASETS * TOP_N_CHUNKS,
        include=["documents", "metadatas", "embeddings", "distances"]
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    embs = res["embeddings"][0]

    # Convert distances to similarity (Chroma returns cosine distance)
    # cosine_distance = 1 - cosine_similarity
    sim_scores = [1 - d for d in res["distances"][0]]

    # --- Aggregate scores by dataset ---
    dataset_scores = defaultdict(list)

    for sim, doc, meta in zip(sim_scores, docs, metas):
        ds_id = meta["dataset"]
        if sim >= sim_thre:
            dataset_scores[ds_id].append((sim, doc))

    # If nothing exceeds threshold
    if not dataset_scores:
        return []

    # --- Rank datasets ---
    rankings = []

    for ds_id, entries in dataset_scores.items():
        entries.sort(reverse=True, key=lambda x: x[0])
        top_score = entries[0][0]
        top_chunks = entries[:TOP_N_CHUNKS]
        rankings.append({
            "dataset_id": ds_id,
            "top_score": top_score,
            "top_chunks": top_chunks
        })

    # Sort by highest similarity
    rankings.sort(key=lambda x: x["top_score"], reverse=True)
    return rankings

def generate_answers(user_query, rankings):
    # --- Prepare LLM prompt ---
    prompt = f"""
    You are a helpful assistant summarizing information from bioenergy datasets.

    User question:
    {user_query}

    Context (retrieved datasets and their most relevant text chunks):
    {json.dumps(rankings)}

    Instructions:
    1. Use the context above to produce a concise and factual answer to the user's question.
    2. If information is missing, say so rather than guessing.
    """

    # --- Call Gemini ---
    try:
        response = _client_gem.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": KDFResponse,
            },
        )
        output = json.loads(response.text)
    except Exception as e:
        logging.error(f"Gemini call failed: {e}")
        output = {"error": str(e)}

    return output
