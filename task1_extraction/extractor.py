"""Extraction pipeline: prompt -> Gemini -> parse JSON -> validate with Pydantic."""
import json
import re

from pydantic import BaseModel, ValidationError

from llm_client import GeminiClient, LLMError
from .prompts import SYSTEM_INSTRUCTION, build_prompt
from .schemas import SCHEMAS


class ExtractionError(Exception):
    """Raised when valid structured output could not be obtained."""


def _parse_json(raw: str) -> dict:
    """Parse JSON, tolerating markdown code fences around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object at the top level.")
    return data


def extract(doc_type: str, text: str, client: GeminiClient | None = None,
            max_attempts: int = 3) -> BaseModel:
    """Extract structured data from invoice/resume text. Returns a validated Pydantic model."""
    if doc_type not in SCHEMAS:
        raise ValueError(f"Unknown document type '{doc_type}'. Use: {list(SCHEMAS)}")
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    client = client or GeminiClient()
    schema = SCHEMAS[doc_type]
    prompt = build_prompt(doc_type, text)
    last_error = "unknown"

    for attempt in range(1, max_attempts + 1):
        try:
            raw = client.generate(prompt, system_instruction=SYSTEM_INSTRUCTION, json_mode=True)
        except LLMError as e:
            raise ExtractionError(f"API failure: {e}") from e

        try:
            return schema.model_validate(_parse_json(raw))
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"Invalid JSON: {e}"
        except ValidationError as e:
            last_error = f"Schema validation failed: {e}"

        # Re-prompt with the error so the model can correct itself
        prompt = (
            build_prompt(doc_type, text)
            + f"\n\nYour previous answer was rejected: {last_error}\n"
              "Return corrected JSON only."
        )

    raise ExtractionError(f"Could not get valid output after {max_attempts} attempts. Last error: {last_error}")