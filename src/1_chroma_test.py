import chromadb
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
db_path = os.path.join(root, "data science kdf", "chroma_db")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("kdf_embeddings")

# list all collections
collections = client.list_collections()
print("Collections in DB:")
for c in collections:
    print(" -", c.name)

# my collection
collection = client.get_collection("kdf_embeddings")
print("Collection name:", collection.name)
print("Count:", collection.count())  # total vectors stored

# Peek at a few stored records
sample = collection.get(limit=3)
print("IDs:", sample["ids"])
print("Metadata:", sample["metadatas"])
print("Docs:", [d[:200] for d in sample["documents"]])