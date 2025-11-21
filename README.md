# Local Semantic Search and Summarization

I develoepd a local semantic search function for a website using a Retrieval-Augmented Generation (RAG) system. It provides end-to-end data ingestion pipelines, embedding generation, vector database management, and LLM-powered retrieval. 
It allows users to:
- Embed text datasets into a vector database (Chroma).
- Query semantically, i.e., ask natural-language questions instead of using keywords.
- Retrieve and summarize the most relevant dataset using Gemini API.
- While the example data are crawled HTML pages, the approach works with any text corpus (research papers, reports, knowledge bases, etc.).

## Repository structure
```
semantic-search-engine/
├── src/
| ├── ingest_to_db.py # offline data ingestion pipeline
│ ├── semantic_search_engine.py # functions of retrieval and and generation
│ ├── app.py # visualization
| ├── Rest of the files are for testing purposes
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Usage
1. Clone the repository
```
cd src
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Configure your environment

Add your API keys (e.g., Gemini, OpenAI) into the config file (e.g., config.yaml). Ensure your data or vector database directories exist (e.g., data/, chroma_db/).

4. Run the application
```
streamlit run app.py
```
## Output description
1. AI Summary: The system generates a concise LLM-powered summary of the extracted content.
2. Reference: The engine identifies the exact source location in the dataset or document that matches the user’s query.
3. Direct Quote: Once a relevant reference is found, the system extracts the exact msot relevant sentence or paragraph from the original document.
   
## Illustration
<img width="613" height="733" alt="image" src="https://github.com/user-attachments/assets/62e0ef9c-233e-43c9-9820-54103b017425" />
