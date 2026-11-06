"""Shared Gemini client wrapper. Reused by all four tasks."""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RETRYABLE_CODES = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 6):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY not found. Add it to your .env file.")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60000),  # 60 seconds, in milliseconds
        )
        self.model = model
        self.max_retries = max_retries

    def generate(self, prompt: str, system_instruction: str | None = None,
                 json_mode: bool = False, temperature: float = 0.0) -> str:
        """Send a prompt and return the response text. Retries on transient API errors."""
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                if not response.text:
                    raise LLMError("The model returned an empty response.")
                return response.text
            except errors.APIError as e:
                last_error = e
                if e.code in RETRYABLE_CODES and attempt < self.max_retries:
                    time.sleep(5 ** attempt)  # 2s, 4s, ...
                    continue
                raise LLMError(f"Gemini API error ({e.code}): {e.message}") from e
            except LLMError:
                raise
            except Exception as e:  # network problems, etc.
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(5 ** attempt)
                    continue
                raise LLMError(f"Unexpected error calling Gemini: {e}") from e
        raise LLMError(f"Failed after {self.max_retries} attempts: {last_error}")