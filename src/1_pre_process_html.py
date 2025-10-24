import fitz  # PyMuPDF
import re, json, os
import hashlib
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
import numpy as np
from sentence_transformers import SentenceTransformer  # or use OpenAI API?
import chromadb

# === CONFIG ===
root = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
data_dir = os.path.join(root, "data science kdf", "data")

# https://github.com/chroma-core/chroma
db_path = os.path.join(root, "data science kdf", "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("kdf_embeddings")

# client.delete_collection("kdf_embeddings") # Delete all records in the collection
# collection = client.get_or_create_collection("kdf_embeddings")

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
def chunk_text(text, max_passages=6, min_len=100):
    """
    Split by paragraphs or sentences into coherent chunks.
    Keeps 4–6 sentences or paragraphs per chunk, skips very short ones.
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buffer = []

    for para in paras:
        # Split long paragraphs into sentences if needed
        sentences = sent_tokenize(para)
        buffer.extend(sentences)
        if len(buffer) >= max_passages:
            chunk = " ".join(buffer[:max_passages])
            if len(chunk) > min_len:
                chunks.append(chunk)
            buffer = buffer[max_passages:]
    # final leftover
    if buffer:
        chunk = " ".join(buffer)
        if len(chunk) > min_len:
            chunks.append(chunk)
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
    print(f"Extracted {len(chunks)} chunks from PDF")

    embeddings = compute_embeddings(chunks)
    print(f"Computed embeddings shape: {np.array(embeddings).shape}")


    collection.add(
        ids=[f"{base}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"dataset": base}] * len(chunks)
    )

    print(f"✅ Inserted {len(chunks)} chunks for {base} into Chroma DB.")