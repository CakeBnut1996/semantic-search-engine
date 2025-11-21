import re, json, os, enum, yaml
import hashlib
from bs4 import BeautifulSoup
import tiktoken
from nltk.tokenize import sent_tokenize
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
import numpy as np
from sentence_transformers import SentenceTransformer  # or use OpenAI API?
import chromadb

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__name__))
CONFIG_PATH = os.path.join(BASE_DIR, "config")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

root = config.get("paths")['ROOT']
data_dir = os.path.join(root, "data")

# https://github.com/chroma-core/chroma
db_path = os.path.join(root, "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("kdf_embeddings")

# client.delete_collection("kdf_embeddings") # Delete all records in the collection
# collection = client.get_or_create_collection("kdf_embeddings")

enc = tiktoken.get_encoding("cl100k_base")  # OpenAI-compatible tokenizer

# === TEXT EXTRACTION ===
def extract_text_from_html(path):
    """
    Extract clean readable text from a saved HTML file.
    Removes scripts, styles, and tags.
    """
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()

    # Get main text content
    text = soup.get_text(separator="\n", strip=True)

    # Optional: collapse multiple newlines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def clean_text(text):
    """Remove excess whitespace and control characters."""
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\f', '', text)  # remove page breaks
    return text.strip()

# === FILTERING ===
def filter_noise(text):
    """Remove common noisy sections (references, tables, etc.)."""
    lines = text.split("\n")
    clean_lines = []
    for ln in lines:
        # skip lines that are mostly numbers or short headers
        if re.match(r'^\d+[\.\)]', ln.strip()):
            continue
        if len(ln.strip()) < 30 and ln.strip().isupper():
            continue
        if "REFERENCES" in ln.upper() or "TABLE" in ln.upper():
            continue
        clean_lines.append(ln)
    return "\n".join(clean_lines)

# === CHUNKING ===
def num_tokens(text):
    return len(enc.encode(text))

def chunk_text(text, max_tokens=256, overlap=40):
    # overlap: When you create the next chunk, repeat the last 40 tokens from the previous chunk
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = num_tokens(sent)

        # If a single sentence is too large, split it hard
        if sent_tokens > max_tokens:
            continue

        if current_tokens + sent_tokens > max_tokens:
            # close the current chunk
            chunk = " ".join(current)
            chunks.append(chunk)

            # start new chunk with overlap
            overlap_text = " ".join(current)[-overlap:]
            current = [overlap_text] if overlap_text else []
            current_tokens = num_tokens(" ".join(current))

        current.append(sent)
        current_tokens += sent_tokens

    # leftover
    if current:
        chunks.append(" ".join(current))

    return chunks

# === DEDUPLICATION ===
def deduplicate_chunks(chunks):
    """Skip duplicate or near-duplicate text chunks."""
    seen = set()
    unique_chunks = []
    for c in chunks:
        h = hashlib.md5(c.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_chunks.append(c)
    return unique_chunks

# === EMBEDDING ===
def compute_embeddings(chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

# === PROCESS ALL HTML FILES ===

for filename in os.listdir(data_dir):
    if not filename.endswith(".html"):
        continue

    base = os.path.splitext(filename)[0]
    file_path = os.path.join(data_dir, filename)

    print(f"\n📄 Processing: {filename}")
    text = extract_text_from_html(file_path)
    text = clean_text(text)
    text = filter_noise(text)
    chunks = chunk_text(text)
    chunks = deduplicate_chunks(chunks)
    print(f"Extracted {len(chunks)} chunks from texts")

    embeddings = compute_embeddings(chunks)
    print(f"Computed embeddings shape: {np.array(embeddings).shape}")


    collection.add(
        ids=[f"{base}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"dataset": base}] * len(chunks)
    )

    print(f"✅ Inserted {len(chunks)} chunks for {base} into Chroma DB.")