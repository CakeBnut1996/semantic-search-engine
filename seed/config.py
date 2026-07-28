import os
import yaml
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class SolrConfig(BaseModel):
    url: str = Field(default="http://solr:8983/solr/biokdf")
    num_docs: int = Field(default=5)
    query_fields: str = Field(default="search_api_fulltext")
    result_fields: str = Field(default="id,title,url,score,search_api_relevance,field_publication_year,created")
    sort_order: str = Field(default="search_api_relevance desc, field_publication_year desc, created desc")
    url_template: str = Field(default="{url}") # For configurable URL construction

class LLMProviderConfig(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None # For Ollama or OpenAI compatibles

class AppConfig(BaseModel):
    solr: SolrConfig = SolrConfig()
    active_student: str = "groq_llama"
    llm_providers: Dict[str, LLMProviderConfig] = {}
    api_key: Optional[str] = os.getenv("SEED_API_KEY") # Problem 9

def load_config(config_path: str = "config.yaml") -> AppConfig:
    if not os.path.exists(config_path):
        return AppConfig()
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Map old config structure to new one if needed, or just use new structure
    # For now, let's just parse it.
    
    solr_data = data.get('retrieval', {})
    llm_data = data.get('llm', {})
    gen_data = data.get('generation', {})
    
    providers = {}
    for name, p_data in llm_data.items():
        providers[name] = LLMProviderConfig(
            provider=p_data.get('provider'),
            model=p_data.get('model'),
            api_key=os.getenv(f"{p_data.get('provider').upper()}_API_KEY"),
            base_url=p_data.get('base_url')
        )
    
    return AppConfig(
        solr=SolrConfig(
            url=solr_data.get('solr_url', "http://solr:8983/solr/biokdf"),
            num_docs=solr_data.get('num_docs', 5),
            query_fields=solr_data.get('query_fields', "search_api_fulltext"),
            result_fields=solr_data.get('result_fields', "id,title,url,score,search_api_relevance,field_publication_year,created"),
            sort_order=solr_data.get('sort_order', "search_api_relevance desc, field_publication_year desc, created desc"),
            url_template=solr_data.get('url_template', "{url}")
        ),
        active_student=gen_data.get('active_student', "groq_llama"),
        llm_providers=providers
    )
