import os
import time
import yaml
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from retrieval_utils.retriever_solr import SolrRetriever, SolrResult
from generation_utils.generator import StudentGenerator
from generation_utils.schema import Response as LLMResponseSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('seed-solr-gateway')

app = FastAPI(title='SEED Solr Search Gateway', version='2.0.0')

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()
active_stu = cfg['llm'][cfg['generation']['active_student']]
sys_cfg = {'SOLR_URL': cfg['retrieval'].get('solr_url', 'http://solr:8983/solr/biokdf'), 'NUM_DOCS': cfg['retrieval'].get('num_docs', 5)}

student = StudentGenerator(provider=active_stu['provider'], model_name=active_stu['model'])
solr_client = SolrRetriever(url=sys_cfg['SOLR_URL'])

class SearchRequest(BaseModel):
    query: str
    facets: Optional[List[str]] = None
    mode: str = 'lexical'

class SearchResult(BaseModel):
    id: str
    title: str
    url: Optional[str] = None
    snippet: str
    score: float
    quotes: List[str] = []

class SearchResponse(BaseModel):
    query: str
    answer: str
    results: List[SearchResult]
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    dependencies: Dict[str, str]

@app.get('/')
async def root(): return {'message': 'SEED Solr Search Gateway is running.'}

@app.get('/v1/search/health', response_model=HealthResponse)
async def health_check():
    solr_ok = solr_client.health_check()
    return {'status': 'ok' if solr_ok else 'unreachable', 'dependencies': {'solr': 'reachable' if solr_ok else 'unreachable', 'llm_provider': active_stu['provider']}}

@app.post('/v1/search/semantic', response_model=SearchResponse)
async def semantic_search(request: SearchRequest = Body(...)):
    start_time = time.time()
    try:
        f_dict = {f.split(':')[0]: f.split(':')[1] for f in request.facets if ':' in f} if request.facets else None
        res = solr_client.search(query=request.query, facets=f_dict, rows=sys_cfg['NUM_DOCS'])
        formatted = []
        for s in res:
            c = s.metadata.get('content', '')
            formatted.append({'id': s.id, 'title': s.title or s.id, 'url': s.url, 'text': s.snippet or (c[:500] if c else ''), 'score': s.score})
        ans = student.generate(query=request.query, context=str(formatted), schema=LLMResponseSchema)
        return SearchResponse(query=request.query, answer=ans.answer if hasattr(ans, 'answer') else str(ans), results=[SearchResult(id=r['id'], title=r['title'], url=r['url'], snippet=r['text'], score=r['score'], quotes=[r['text']]) for r in formatted], processing_time_ms=(time.time()-start_time)*1000)
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
