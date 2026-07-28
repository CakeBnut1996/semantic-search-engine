UPSTREAM_IMPLEMENTATION_PLAN.md
# SEED Upstream Implementation Plan

## Purpose

This document is an execution brief for improving the
`CakeBnut1996/semantic-exploration-discovery` repository.

The goal is to make SEED deployable as a general semantic-search gateway with:

- Apache Solr retrieval.
- Local inference through Ollama.
- Any OpenAI-compatible model provider.
- Optional Drupal integration.
- Optional WordPress integration.
- Safe, documented production and internal-test deployment patterns.

An implementation agent should execute the work in the phases below, preserve
backward compatibility where practical, and submit the result as focused commits
or pull requests.

Do not add organization-specific hostnames, IP addresses, credentials, API keys,
Solr passwords, Drupal node IDs, or private data to the repository.

---

## Current problems to address

The current repository has several deployment blockers:

1. The README describes Llama-based generation, but the application only
   initializes selected cloud-provider clients.
2. Ollama is not documented or exposed as a first-class configuration.
3. Solr query and result fields are hard-coded.
4. The hard-coded Solr fields do not match common Drupal Search API Solr schemas.
5. Solr multivalued fields can be passed into Pydantic string fields without
   normalization.
6. Document URL construction is not configurable.
7. FastAPI and Uvicorn are used at runtime but are not correctly represented in
   the minimal runtime dependency set.
8. Installing the API downloads unrelated notebook, visualization, embedding,
   Torch, and CUDA packages.
9. The semantic-search endpoint has no application-level authentication.
10. LLM exceptions are swallowed and converted into empty response models.
11. The health endpoint verifies Solr but does not prove that the LLM is usable.
12. The Docker and service deployment paths are incomplete.
13. Python cache files are present in the repository.
14. There are no maintained CMS integrations.

---

## Target architecture

```text
Drupal or WordPress
        |
        | server-to-server HTTP request
        | Authorization: Bearer <token>
        v
SEED FastAPI gateway
        |
        +----> Solr core or collection
        |
        +----> Ollama or another OpenAI-compatible provider
```

The CMS must call SEED server-to-server. The browser must not receive the SEED
API key or call Ollama directly.

Ollama should normally remain bound to `127.0.0.1:11434`. SEED may also remain
on localhost when Apache or Nginx is used as a reverse proxy.

---

## Proposed repository structure

Use a structure similar to:

```text
semantic-exploration-discovery/
├── seed/
│   ├── api/
│   ├── generation/
│   ├── retrieval/
│   ├── config.py
│   └── models.py
├── integrations/
│   ├── drupal/
│   │   └── seed_search/
│   └── wordpress/
│       └── seed-search/
├── examples/
│   ├── config.example.yaml
│   ├── config.ollama.yaml
│   ├── config.drupal-solr.yaml
│   ├── seed.service
│   ├── apache-seed.conf
└── docs/
    └── UPSTREAM_IMPLEMENTATION_PLAN.md
```

---

## Detailed Implementation Phases

### Phase 1: Hybrid Retrieval & Gateway Foundation
- Implement standalone FastAPI gateway with endpoint contracts.
- Implement Solr adapter using current core and facet/query conventions.
- Implement hybrid retrieval (lexical + semantic fused results).
- Implement fallback policy to lexical keyword search if semantic stack fails.
- Add integration tests for plain queries, facets, and timeout fallback.

### Phase 2: Generation & Observability
- Add generated summary with quote grounding using GPT-OSS (Ollama).
- Implement the `explain` endpoint for retrieval and rerank tracing.
- Add observability dashboards (latency, request IDs).
- Add eval harness (nDCG/MRR + answer groundedness).

### Phase 3: CMS Agnostic Deployment
- UI-agnostic search app consumes gateway directly.
- Finalize Docker image and compose profile for production deployment.
- Documentation for Drupal/WordPress integration.

---

## Agent Implementation Checklist

1. **Environment & Core**:
   - [x] Clone SEED code and run smoke tests.
   - [x] Set up FastAPI structure in `seed/api/`.
   - [x] Configure `config.yaml` for `ornl_gpt_oss` via Ollama.

2. **Search Logic**:
   - [x] Implement `SolrRetrieverAdapter` (as `seed/retrieval/retriever_solr.py`).
   - [x] Implement `RankFusionAdapter` for hybrid merging.
   - [x] Ensure facet mapping (`f[]` parameters) matches Drupal conventions.

3. **Generation**:
   - [x] Integrate `generation_utils/generator.py` with FastAPI endpoints.
   - [x] Implement summary generation with source attribution (Quote grounding prompt).

4. **Testing & Ops**:
   - [x] Add integration tests for facets and fallbacks.
   - [x] Create `Dockerfile` and `docker-compose.yml` for the gateway.
   - [x] Verify health endpoint (`/v1/search/health`) checks all dependencies.
