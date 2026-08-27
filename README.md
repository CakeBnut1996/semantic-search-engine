# SEED: Semantic Energy Exploration and Discovery

SEED is a Retrieval-Augmented Generation (RAG) tool designed to query complex energy and bioenergy document repositories using natural language. Instead of relying solely on keyword matching, SEED performs semantic search and generates concise, factual AI summaries grounded directly in your source documents, complete with citations and exact quotes.

## Key Features

- **Semantic Retrieval**: Finds relevant content based on conceptual meaning rather than exact keywords.
- **Grounded AI Answers**: Generates answers using an LLM based strictly on retrieved document excerpts.
- **Source Citation & Excerpts**: Provides document titles, original URLs, and direct quotes for verification.
- **Dual Interfaces**: Offers an interactive Web UI (Streamlit) and a CLI script that outputs formatted Markdown files.

---

## How It Works

1. **Retrieval**: Documents are split into chunks, converted into vector embeddings, and stored in ChromaDB (`chroma_db/`). For any query, the system retrieves the most semantically relevant passages.
2. **Generation**: A Large Language Model (LLM) reads the retrieved context and formulates a concise, factual summary.

---

## Getting Started

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd semantic-search-engine
pip install uv
uv sync
```

### 2. Configure System Settings

Edit `config.yaml` to set your active embedding model, vector database, and LLM provider. Set the required API key for your chosen LLM as an environment variable (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`, or `GOOGLE_API_KEY`).

### 3. Data Ingestion

To ingest your own HTML files:

1. Place HTML files into the `data_raw/` directory (or the directory specified by `data.data_to_db` in `config.yaml`).
2. Run the ingestion pipeline to extract, chunk, embed, and store document vectors into `chroma_db/`:

```bash
python -m ingestion_utils.pre_processor
```

### 4. Running Search

#### Option A: Interactive Web UI (Streamlit)

```bash
streamlit run app.py
```

A browser window will open displaying the search interface:

![Illustration](./images/illustration_annotated.png)

#### Option B: Command-Line Interface (CLI)

Run `main.py` interactively or pass a query directly as an argument:

```bash
# Interactive mode
python main.py

# Direct query mode
python main.py "What are the forestry biomass production costs?"
```

Results will be automatically formatted and saved as Markdown files in the `output/` directory.

---

## Evaluation Framework

For measuring retrieval accuracy and generation quality in RAG systems, see the evaluation framework repository: [RAG Evaluator](http://github.com/CakeBnut1996/rag-evaluator). 
