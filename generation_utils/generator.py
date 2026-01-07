from generation_utils.llm_client import LLMClient
from typing import Optional, Type, Union
from pydantic import BaseModel

class StudentGenerator:
    def __init__(self, provider: str, model_name: str):
        """
        The Student uses the LLMClient for text generation.
        """
        self.llm = LLMClient(provider, model_name)

    def generate(self, query: str, context: str, schema: Optional[Type[BaseModel]] = None) -> Union[str, BaseModel]:
        """
        High-level generate method.
        - If schema is provided: returns a validated Pydantic object.
        - If schema is None: returns a raw string.
        """
        system_instr = "You are a helpful assistant. Answer based strictly on the context provided."

        # Construct the user prompt
        prompt = (
            f"Context (retrieved datasets and chunks):\n{context}\n\n"
            f"User question: {query}\n"
        )

        if schema:
            # for now only handles Gemini/OpenAI.
            structured_prompt = prompt + "\nProvide your response as a structured JSON object."
            return self.llm.generate_structured(structured_prompt, schema)

        else:
            # text only
            return self.llm.generate_text(prompt, system_instruction=system_instr)