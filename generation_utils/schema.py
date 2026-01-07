from pydantic import BaseModel, Field
from typing import List, Optional

class DatasetSummary(BaseModel):
    """Summarizes a specific dataset and its most relevant evidence."""
    name: str = Field(..., description="dataset_id from the retrieval")
    summary: str = Field(..., description="A summary of the dataset content relevant to the query based on its returned chunks")
    quote: str = Field(..., description="Top-ranked chunk quoted from the dataset")

class KDFResponse(BaseModel):
    """The structured final response containing the answer and supporting evidence."""
    answer: str = Field(..., description="A concise and factual answer to the user's question based on retrieval")
    name_top: str = Field(..., description="dataset_id from the top retrieval")
    supporting_datasets: List[DatasetSummary]