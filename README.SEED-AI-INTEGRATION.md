# SEED Semantic Search Integration Guide for AI Agent

## Purpose
This guide tells an implementation agent how BioKDF search currently works and how to integrate SEED (Semantic Energy Exploration and Discovery) as a semantic layer without coupling to Drupal internals.

Primary target site:
- Dev: https://biokdfdev3.ornl.gov/
- Search page: https://biokdfdev3.ornl.gov/biokdf-search

Reference project:
- OSTI entry: https://www.osti.gov/doecode/biblio/178057
- Repository: https://github.com/CakeBnut1996/semantic-exploration-discovery

## Current Search System (Observed in live config)

### Page and View Contract
- View ID: biokdf_solr_search
- Path: /biokdf-search
- Base table: search_api_index_documents
- Exposed text input query key: search_api_fulltext
- Exposed submit label: Apply
- Pager defaults:
  - default size: 20
  - allowed sizes: 20, 50, 100, 200
- Built-in filters:
  - status = 1 (published only)
  - content types = biokdf_landing_page, documents
- Default sort priority:
  1. search_api_relevance desc
  2. field_publication_year desc
  3. created desc

### Search API Index and Solr Backend
- Search API index ID: documents
- Search API server ID: biokdf_documents
- Solr core/collection in use: biokdf
- Connector: basic_auth over http
- Solr host/port in current runtime: solr:8983
- Important note: credentials must be read from secrets/env, never committed.

### Facet URL Contract on /biokdf-search
Facet filters are passed via repeated f[] query parameters using this pattern:
- f[0]=facet_alias:value
- f[1]=facet_alias:value

Example seen on live page:
- /biokdf-search?f%5B0%5D=biokdf_facet_keywords%3A106

Primary facet aliases attached to this view:
- add_doe_flag
- doe_funded
- biokdf_facet_bioenergy_category
- biokdf_facet_keywords
- biokdf_facet_lab
- biokdf_facet_organization

## Non-Goals
- Do not implement as a Drupal module.
- Do not put model calls or Solr credentials in browser code.
- Do not tightly couple semantic flow to current CMS templates.

## Recommended Integration Strategy (Best for future CMS migration)
Use an external Semantic Search Gateway service as the stable contract.

### Why this is the best fit
- Keeps semantic/RAG logic independent from Drupal.
- Works now with Drupal and later with a non-Drupal frontend.
- Allows Python-native SEED code reuse.
- Keeps Solr/auth/network policy in one backend service.

### Service shape
Build a standalone service (Python FastAPI preferred, since SEED is Python):

1) POST /v1/search/semantic
- Request body:
  - query: string
  - facets: string[] (same alias:value values used in current f[] contract)
  - page: integer (1-based)
  - page_size: integer
  - mode: semantic | hybrid | keyword
  - include_summary: boolean
- Response body:
  - query: string
  - mode_used: semantic | hybrid | keyword
  - fallback_used: boolean
  - total: integer
  - results: array of
    - id
    - title
    - url
    - snippet
    - score
    - source_type
    - supporting_quotes[]
    - metadata {}
  - summary: string
  - diagnostics:
    - latency_ms
    - solr_query_id
    - reranker_model

2) GET /v1/search/health
- Returns service health and dependency status (LLM provider, embedding store, Solr reachability).

3) Optional: POST /v1/search/explain
- Returns retrieval and rerank trace for debugging and eval.

## Retrieval Design
Use hybrid retrieval so existing precision and facet behavior are preserved.

Pipeline:
1. Parse incoming query and facet filters.
2. Retrieve lexical candidates from Solr (BM25/edismax) with facet constraints.
3. Retrieve semantic candidates from vector index (or embedding cache).
4. Merge and rerank candidates (cross-encoder or weighted rank fusion).
5. Return ranked records plus supporting quotes.
6. If semantic stack errors/timeouts, fallback to lexical-only from Solr.

## Solr Integration Rules
- Use server-side Solr access only.
- Translate incoming facets directly from current alias:value format.
- Keep page and page_size behavior consistent with current UI.
- Preserve current public URL behavior so old links still work.
- Add timeout budgets and circuit-breaker fallback:
  - semantic stage timeout (example: 2-4s)
  - hard fallback to keyword path if semantic stage fails

## Minimal Migration-Safe Frontend Contract
Any frontend (Drupal now, other framework later) should only depend on:
- endpoint: /v1/search/semantic
- stable JSON result schema
- existing URL params: search_api_fulltext, f[], page, items_per_page

## How to wrap SEED code
Recommended wrapper style:
- Keep SEED repository as a library dependency or git submodule inside gateway service.
- Create adapter classes in the gateway:
  - SeedSemanticAdapter: handles query understanding, embeddings, answer synthesis.
  - SolrRetrieverAdapter: handles lexical retrieval and facet filtering.
  - RankFusionAdapter: merges lexical + semantic candidates.
- Expose only gateway API to clients.

## Security and Operations
- Secrets via env/secret manager only.
- Never log raw secrets.
- Log request IDs and latency for each stage.
- Add rate limiting for semantic endpoint.
- Add canary flag to compare semantic vs keyword click behavior.

## Agent Implementation Checklist
1. Clone SEED code and run smoke tests in isolated environment.
2. Implement standalone FastAPI gateway with endpoint contracts above.
3. Implement Solr adapter using current core and facet/query conventions.
4. Implement fallback policy to lexical keyword search.
5. Add integration tests for:
   - plain query
   - query + one facet
   - query + multiple facets
   - timeout fallback
6. Add eval harness (nDCG/MRR + answer groundedness).
7. Provide Docker image and compose profile for dev deployment.

## Suggested rollout
Phase 1:
- Hybrid retrieval in gateway, no summary generation by default.
- Return top semantic + lexical fused results.

Phase 2:
- Add generated summary with quote grounding.
- Add explain endpoint and observability dashboards.

Phase 3:
- UI-agnostic search app consumes gateway directly (CMS-independent).

## Hand-off notes for the SEED developer
- Keep API schema stable and versioned from day one.
- Avoid CMS-specific assumptions in retrieval layer.
- Treat Solr as authoritative lexical source and metadata source.
- Favor hybrid mode as default; semantic-only can reduce precision on niche technical terms.
