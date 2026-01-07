import os
import json
import yaml
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv


# === DATA SCHEMAS ===

class DocStatus(str, Enum):
    valid = "valid"
    skip = "skip"  # File is empty, noise, or just contact info


class QAItem(BaseModel):
    question: str
    answer: str
    context_quote: str = Field(..., description="Verbatim text snippet from the document that supports this answer.")


class HardNegativeItem(BaseModel):
    question: str


class DatasetTrainingExample(BaseModel):
    status: DocStatus = Field(..., description="Skip if document is just noise/metadata.")
    positives: List[QAItem] = Field(default_factory=list)
    hard_negatives: List[HardNegativeItem] = Field(default_factory=list)


# === GENERATOR CLASS ===

class QAGenerator:
    def __init__(self, key_path=None):
        self._setup_gemini(key_path)

    def _setup_gemini(self, key_path):
        """Load API key and setup client."""
        # Priority: explicit path -> env var
        api_key = None

        if key_path and os.path.exists(key_path):
            with open(key_path, 'r') as f:
                creds = yaml.safe_load(f)
                api_key = creds.get("api_key")

        if not api_key:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("❌ No API key found in YAML or .env")

        self.client = genai.Client(api_key=api_key)

    def _make_prompt(self, text: str) -> str:
            # We limit text sent to prompt to avoid extreme token usage if file is huge
            preview = text[:25000]

            return f"""Task: Generate RAG evaluation data from this document.

    1. ANALYZE: Is this content (narrative, reports, data) valid or noise?
       - If noise, set status="skip".

    2. GENERATE (if valid):
       - Create 3 POSITIVE Q&A pairs.
       - **QUERY STYLE**: Simulate a user typing into a search engine to FIND this dataset. 
         - **Imprecision**: Use layman terms or synonyms for at least 1 query (e.g., use "car wrecks" instead of "vehicle collisions").
         - **Discovery Mode**: The user does NOT know the dataset at all; they are searching for the *topic* or *entities* inside it.
       - **CONTEXT**: For each answer, provide the `context_quote` (EXACT, VERBATIM snippet).
       - Create 3 HARD NEGATIVE questions: Plausible queries for this domain that are NOT answered by this specific text.

    OUTPUT JSON SCHEMA:
    {{
        "status": "valid" | "skip",
        "positives": [ {{"question": "...", "answer": "...", "context_quote": "..."}} ],
        "hard_negatives": [ {{"question": "..."}} ]
    }}

    DOCUMENT TEXT (Truncated):
    {preview}"""

    def generate_examples(self, full_text: str) -> Dict:
        """Calls Gemini to generate QA pairs from the raw file."""
        prompt = self._make_prompt(full_text)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": DatasetTrainingExample
                }
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ Error processing file: {e}")
            return {"status": "skip", "positives": [], "hard_negatives": []}