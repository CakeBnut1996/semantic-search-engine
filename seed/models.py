from typing import List, Optional, Dict, Any
from pydantic import BaseModel

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
