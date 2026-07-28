import os
import time
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from seed.retrieval.retriever_solr import SolrRetriever
from seed.generation.generator import StudentGenerator
from seed.generation.schema import Response as LLMResponseSchema
from seed.config import load_config
from seed.models import SearchRequest, SearchResponse, SearchResult, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('seed-api')

app = FastAPI(title='SEED Semantic Search Gateway', version='2.1.0')
security = HTTPBearer()

# Load configuration (Problem 3, 4, 6)
cfg = load_config()

# Initialize components
active_provider = cfg.llm_providers.get(cfg.active_student)
if not active_provider:
    raise ValueError(f"Active student '{cfg.active_student}' not found in providers.")

student = StudentGenerator(
    provider=active_provider.provider, 
    model_name=active_provider.model,
    base_url=active_provider.base_url
)

# Setup Retrieval (Solr only)
retriever = SolrRetriever(
    url=cfg.solr.url,
    query_fields=cfg.solr.query_fields,
    result_fields=cfg.solr.result_fields,
    sort_order=cfg.solr.sort_order,
    url_template=cfg.solr.url_template
)

def verify_token(auth: HTTPAuthorizationCredentials = Security(security)):
    """Problem 9: Application-level authentication."""
    if cfg.api_key and auth.credentials != cfg.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return auth.credentials

@app.get('/')
async def root(): return {'message': 'SEED Search Gateway is active.'}

@app.get('/v1/search/health', response_model=HealthResponse)
async def health_check():
    """Problem 11: Verification that LLM is usable."""
    solr_ok = retriever.health_check()
    llm_ok = student.health_check()
    
    return {
        'status': 'ok' if (solr_ok and llm_ok) else 'degraded', 
        'dependencies': {
            'solr': 'reachable' if solr_ok else 'unreachable', 
            'llm': 'functional' if llm_ok else 'failed'
        }
    }

@app.post('/v1/search/semantic', response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest = Body(...),
    token: str = Depends(verify_token)
):
    start_time = time.time()
    try:
        f_dict = {}
        if request.facets:
            for f in request.facets:
                if ':' in f:
                    k, v = f.split(':', 1)
                    f_dict[k] = v
        
        # Retrieval (Solr)
        res = retriever.search(query=request.query, facets=f_dict, rows=cfg.solr.num_docs)
        
        if not res:
            return SearchResponse(
                query=request.query, 
                answer="No relevant documents found.", 
                results=[], 
                processing_time_ms=(time.time()-start_time)*1000
            )

        # Context construction
        context_str = "\n".join([f"Source [{r.id}]: {r.snippet}" for r in res])
        
        # Generation (Problem 10: Exceptions now propagate to 500 automatically or handled here)
        ans = student.generate(query=request.query, context=context_str, schema=LLMResponseSchema)
        
        return SearchResponse(
            query=request.query, 
            answer=ans.answer if hasattr(ans, 'answer') else str(ans), 
            results=[SearchResult(
                id=r.id, 
                title=r.title or r.id, 
                url=r.url, 
                snippet=r.snippet or "", 
                score=r.score
            ) for r in res], 
            processing_time_ms=(time.time()-start_time)*1000
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        # Problem 10: No longer swallowing exceptions into empty models
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
