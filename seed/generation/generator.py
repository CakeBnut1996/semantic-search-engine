from .llm_client import LLMClient
from typing import Optional, Type, Union, Any
from pydantic import BaseModel

class StudentGenerator:
    def __init__(self, provider: str, model_name: str, base_url: Optional[str] = None):
        """
        The Student uses the LLMClient for text generation.
        """
        self.llm = LLMClient(provider, model_name, base_url=base_url)

    def generate(self, query: str, context: str, schema: Optional[Type[BaseModel]] = None) -> Union[str, Any]:
        system_instr = """You are a highly accurate search discovery assistant for the BioEnergy Knowledge Discovery Framework (KDF).
Base your answer ONLY on the provided context snippets.
For every factual claim, provide a direct quote and cite the Source ID in brackets, e.g., [source_id].
If the evidence is insufficient, state that clearly.
Format your response as a concise summary followed by grounded evidence."""
        
        prompt = f"### CONTEXT:\n{context}\n\n### USER QUERY: {query}\n"

        try:
            if schema:
                return self.llm.generate_structured(prompt, schema)

            response = self.llm.generate_text(prompt, system_instruction=system_instr)
            return response
        except Exception as e:
            # Re-raise or return a structured error
            raise RuntimeError(f"Generation failed: {str(e)}")

    def health_check(self) -> bool:
        return self.llm.health_check()