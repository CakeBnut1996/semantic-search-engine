# SEED: Semantic Energy Exploration and Discovery

SEED is an AI-powered search gateway that turns complex document repositories into actionable answers. It sits between your users and a **Solr** search server, using Large Language Models to summarize search results into natural language answers.

## Quick Start

### 1. Installation
This project uses \`uv\` for dependency management.
\`\`\`bash
pip install uv
uv sync
source .venv/bin/activate
\`\`\`

### 2. Configure Your Environment
Create a \`.env\` file in the root directory and add your API keys:
\`\`\`bash
# Example for Groq (default)
GROQ_API_KEY=your_api_key_here
\`\`\`

### 3. Setup config.yaml
Open \`config.yaml\` and point it to your Solr server:
\`\`\`yaml
retrieval:
  solr_url: "http://solr:8983/solr/biokdf" # The URL of your Solr core
  num_docs: 5

generation:
  active_student: "groq_llama"
\`\`\`

---

## Usage Steps

### A. Indexing Your Data (Optional)
If your Solr server is already populated, skip this step. If you need to index local HTML files into Solr:
1.  Place your HTML files in the \`data_raw/\` folder.
2.  Run the sync script:
    \`\`\`bash
    python io_utils/sync_ingestion_solr.py
    \`\`\`
    *This script uses MD5 hashing to only upload new or modified files.*

### B. Launch the Search API
Run the FastAPI gateway. This is the endpoint your frontend (Drupal, React, etc.) will talk to.
\`\`\`bash
python server.py
\`\`\`
- **Endpoint:** \`POST /v1/search/semantic\`
- **Health Check:** \`GET /v1/search/health\`

### C. Test via Command Line
You can test the retrieval and AI generation without the API:
\`\`\`bash
python main.py "What are the latest biomass production costs?"
\`\`\`

---

## Tech Stack & Architecture
- **API Framework:** FastAPI
- **Search Engine:** Apache Solr (Keyword search with \`edismax\`)
- **AI Layer:** Llama (Summarization and Synthesis)
- **Data Sync:** Incremental sync using MD5 state tracking

---
