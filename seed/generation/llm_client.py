import os, json
from typing import Type, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from .schema import Response

# --- IMPORTS FOR PROVIDERS ---
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


class LLMClient:
    def __init__(self, provider: str, model_name: str, base_url: Optional[str] = None):
        load_dotenv()
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url
        self._init_client()
        print(f"🔌 LLM Client connected: {self.provider}/{self.model_name}")

    def _init_client(self):
        # Helper to fetch key from env
        def get_key(env_name):
            return os.getenv(env_name)

        if self.provider == "gemini":
            if not HAS_GEMINI: raise ImportError("Run `pip install google-genai`")
            api_key = get_key("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)

        elif self.provider == "openai" or self.provider == "ollama":
            if not HAS_OPENAI: raise ImportError("Run `pip install openai`")
            api_key = get_key("OPENAI_API_KEY") if self.provider == "openai" else "ollama"
            # If ollama, we usually use http://localhost:11434/v1
            url = self.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1") if self.provider == "ollama" else self.base_url
            self.client = OpenAI(api_key=api_key, base_url=url)

        elif self.provider == "anthropic":
            if not HAS_ANTHROPIC: raise ImportError("Run `pip install anthropic`")
            api_key = get_key("ANTHROPIC_API_KEY")
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "groq":
            if not HAS_GROQ: raise ImportError("Run `pip install groq`")
            api_key = get_key("GROQ_API_KEY")
            self.client = Groq(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def health_check(self) -> bool:
        """Verifies if the LLM provider is reachable."""
        try:
            if self.provider in ["openai", "ollama", "groq"]:
                # Simple request to list models or just a tiny completion
                self.client.models.list()
                return True
            elif self.provider == "gemini":
                # Check models list
                self.client.models.list()
                return True
            elif self.provider == "anthropic":
                # Anthropic doesn't have a simple list models in the same way, but we can try a tiny message
                return True # Placeholder for now as it's harder to check without cost
            return True
        except Exception:
            return False

    def generate_text(self, prompt: str, system_instruction: str = None) -> str:
        """Standard text generation for the Student."""
        try:
            # --- GEMINI ---
            if self.provider == "gemini":
                return self.client.models.generate_content(
                    model=self.model_name, contents=prompt
                ).text

            # --- OPENAI / OLLAMA ---
            elif self.provider in ["openai", "ollama", "groq"]:
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    messages.insert(0, {"role": "system", "content": system_instruction})
                res = self.client.chat.completions.create(
                    model=self.model_name, messages=messages
                )
                return res.choices[0].message.content

            # --- ANTHROPIC ---
            elif self.provider == "anthropic":
                res = self.client.messages.create(
                    model=self.model_name, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return res.content[0].text

        except Exception as e:
            raise e # Raising so it can be handled by the caller (Problem 10)

    def generate_structured(self, prompt: str, schema_model: Type[BaseModel]) -> Any:
        """Structured output """
        try:
            # --- GEMINI STRUCTURED ---
            if self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema_model
                    }
                )
                return schema_model.model_validate_json(response.text)

            # --- OPENAI / OLLAMA STRUCTURED ---
            elif self.provider in ["openai", "ollama"]:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=schema_model
                )
                return completion.choices[0].message.parsed

            elif self.provider == "groq":
                # https://console.groq.com/docs/structured-outputs
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Response",
                            "schema": schema_model.model_json_schema()
                        }
                    }
                )
                return schema_model.model_validate_json(response.choices[0].message.content)
            
            else:
                # Fallback for providers that don't support structured output directly yet
                text = self.generate_text(prompt)
                try:
                    # Try to extract JSON from markdown block if present
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    return schema_model.model_validate_json(text)
                except Exception as e:
                    raise ValueError(f"Failed to parse structured output from {self.provider}: {e}\nRaw output: {text}")

        except Exception as e:
            raise e