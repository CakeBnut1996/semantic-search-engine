# Local Embedding-Based Search and Summarization

This project demonstrates how to build a lightweight semantic search engine on your local machine using embeddings, a vector database, and an LLM for natural-language answers.
It allows users to:
- Embed text datasets into a vector database (Chroma).
- Query semantically — ask natural-language questions instead of using keywords.
- Retrieve and summarize the most relevant dataset using a small instruction model or OpenAI API.
- While the example data originates from crawled HTML pages, the approach works with any text corpus (research papers, reports, knowledge bases, etc.).

## Repository structure
```
semantic-search-engine/
├── src/
│ ├── 0_crawl_data_for_testing.py # (Optional) Example data collection
│ ├── 1_process_html_chroma_ingest.py # Core: text cleaning, chunking, embedding ('all-MiniLM-L6-v2'), and Chroma ingestion
│ ├── 2_query_pipeline.py # Core: semantic search and summarization
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Example Output

Enter your query: What are the key feedstocks studied for marine bioenergy?

Top 3 relevant datasets:

[1] data-emerging-resources-macroalgae | Similarity: 0.82

[2] inl-biomass-feedstock-library-bfl | Similarity: 0.77

[3] algae-energy-resource-summary | Similarity: 0.75

💬 Answer:

The most relevant dataset focuses on macroalgae as a marine bioenergy resource...
