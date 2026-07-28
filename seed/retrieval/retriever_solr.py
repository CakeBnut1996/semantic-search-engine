import pysolr
import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger("seed-solr")

class SolrResult(BaseModel):
    id: str
    title: Optional[str] = None
    url: Optional[str] = None
    score: float
    snippet: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SolrRetriever:
    def __init__(self, 
                 url: str = "http://solr:8983/solr/biokdf", 
                 timeout: int = 10,
                 auth: Optional[tuple] = None,
                 query_fields: str = "search_api_fulltext",
                 result_fields: str = "id,title,url,score",
                 sort_order: str = "score desc",
                 url_template: str = "{url}"):
        """
        Initializes the Solr client. 
        """
        self.solr = pysolr.Solr(url, timeout=timeout, auth=auth)
        self.query_fields = query_fields
        self.result_fields = result_fields
        self.sort_order = sort_order
        self.url_template = url_template
        logger.info(f"🔌 Solr Retriever connected to {url}")

    def _normalize_field(self, value: Any) -> Optional[str]:
        """Problem 5: Normalize multivalued fields into strings."""
        if isinstance(value, list):
            return " ".join([str(v) for v in value]) if value else None
        return str(value) if value is not None else None

    def search(self, 
               query: str, 
               facets: Dict[str, Any] = None, 
               rows: int = 20, 
               start: int = 0) -> List[SolrResult]:
        """
        Performs a lexical search on Solr.
        """
        fq = []
        if facets:
            for alias, value in facets.items():
                fq.append(f"{alias}:\"{value}\"")

        search_params = {
            'fq': fq,
            'rows': rows,
            'start': start,
            'fl': self.result_fields, 
            'sort': self.sort_order
        }

        try:
            results = self.solr.search(f"{{!edismax qf={self.query_fields}}}{query}", **search_params)
            
            parsed_results = []
            for doc in results:
                # Problem 6: URL Template construction
                raw_url = self._normalize_field(doc.get('url'))
                formatted_url = self.url_template.format(url=raw_url) if raw_url else None
                
                parsed_results.append(SolrResult(
                    id=self._normalize_field(doc.get('id')),
                    title=self._normalize_field(doc.get('title')),
                    url=formatted_url,
                    score=float(doc.get('score', 0.0)),
                    snippet=self._normalize_field(doc.get('content')) or self._normalize_field(doc.get('body')),
                    metadata=doc
                ))
            return parsed_results
        except Exception as e:
            logger.error(f"Solr search error: {str(e)}")
            return []

    def health_check(self) -> bool:
        """Verifies Solr reachability."""
        try:
            # A simple ping to check if the server is up
            self.solr.ping()
            return True
        except Exception:
            return False
