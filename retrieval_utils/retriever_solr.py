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
                 auth: Optional[tuple] = None):
        """
        Initializes the Solr client. 
        Credentials should be passed via auth parameter (user, pass).
        """
        self.solr = pysolr.Solr(url, timeout=timeout, auth=auth)
        logger.info(f"🔌 Solr Retriever connected to {url}")

    def search(self, 
               query: str, 
               facets: Dict[str, Any] = None, 
               rows: int = 20, 
               start: int = 0) -> List[SolrResult]:
        """
        Performs a lexical search on Solr.
        Translates facets into Solr filter queries.
        """
        fq = []
        if facets:
            # Basic translation of facets into Solr filter queries
            # f[0]=facet_alias:value
            for alias, value in facets.items():
                fq.append(f"{alias}:\"{value}\"")

        # Basic search parameters mapping from README requirements
        search_params = {
            'fq': fq,
            'rows': rows,
            'start': start,
            'fl': 'id,title,url,score,search_api_relevance,field_publication_year,created', 
            'sort': 'search_api_relevance desc, field_publication_year desc, created desc'
        }

        try:
            # Using edismax for better full-text matching as suggested in README
            results = self.solr.search(f"{{!edismax qf=search_api_fulltext}}{query}", **search_params)
            
            parsed_results = []
            for doc in results:
                parsed_results.append(SolrResult(
                    id=doc.get('id'),
                    title=doc.get('title'),
                    url=doc.get('url'),
                    score=doc.get('score', 0.0),
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
