# SEED: Semantic Energy Exploration and Discovery

SEED is an AI-powered search gateway that turns complex document repositories into actionable answers. It sits between your users and a **Solr** search server, using Large Language Models to summarize search results into natural language answers.

## Quick Start

### 1. Installation
This project uses `uv` or `pip` for dependency management.
```bash
# Using uv (recommended)
uv sync
source .venv/bin/activate

# Or using pip
pip install .
```

### 2. Configure Your Environment
Create a `.env` file in the root directory:
```bash
# Application Security
SEED_API_KEY=your_generated_shared_secret

# Provider Keys
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
```

### 3. Setup config.yaml
SEED is highly configurable. An example for **Ollama** (Local Inference):
```yaml
retrieval:
  solr_url: "http://localhost:8983/solr/biokdf"
  num_docs: 5

generation:
  active_student: "ollama_llama"

llm:
  ollama_llama:
    provider: "ollama"
    model: "llama3"
    base_url: "http://localhost:11434/v1"
```

---

## Architecture & Integration

SEED follows a gateway pattern. Your CMS (Drupal, WordPress) or Frontend calls SEED, and SEED orchestrates the RAG (Retrieval-Augmented Generation) flow.

```text
Drupal/WP -> SEED (FastAPI) -> Apache Solr + LLM (Ollama/OpenAI)
```

### API Endpoints
- **Health Check:** `GET /v1/search/health` (Verifies Solr + LLM)
- **Search:** `POST /v1/search/semantic` (Requires Bearer Token)

### CLI Mode
Test retrieval and generation directly from your terminal:
```bash
python -m seed.main "What are the latest biomass production costs?"
```

---

## Tech Stack
- **API Framework:** FastAPI
- **Search Engine:** Apache Solr
- **AI Providers:** Ollama (Local), Groq, OpenAI, Gemini, Anthropic
- **Configuration:** Pydantic / YAML
- **Integration:** Drupal & WordPress ready

---
