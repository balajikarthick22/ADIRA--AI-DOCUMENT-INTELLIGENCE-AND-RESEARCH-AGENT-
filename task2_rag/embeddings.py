"""Gemini embeddings: turn text into vectors."""
import os
import re
import time

from google.genai import errors, types

from llm_client import GeminiClient  # importing this also loads your .env file

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
# The free tier allows about 100 embedding requests per minute, and every chunk counts
# as one. 20 chunks followed by a 13-second pause stays safely under that limit.
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "20"))
PAUSE_SECONDS = float(os.getenv("EMBED_PAUSE_SECONDS", "13"))
RETRYABLE_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6

_client = None


class EmbeddingError(Exception):
    """Raised when embeddings could not be created."""


def _get_client():
    global _client
    if _client is None:
        _client = GeminiClient().client
    return _client


def _wait_time(error, attempt: int) -> float:
    """Seconds to wait before retrying. Uses Google's own 'retry in Ns' hint if present."""
    match = re.search(r"retry in ([\d.]+)s", str(error.message or ""), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2
    return 10 * attempt


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    client = _get_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                batch_vectors = [list(e.values) for e in response.embeddings]
                if len(batch_vectors) != len(batch):
                    raise EmbeddingError("Embedding count did not match the input count.")
                vectors.extend(batch_vectors)
                break
            except errors.APIError as e:
                if e.code in RETRYABLE_CODES and attempt < MAX_ATTEMPTS:
                    wait = _wait_time(e, attempt)
                    print(f"  API busy or rate limited, waiting {wait:.0f}s...", flush=True)
                    time.sleep(wait)
                    continue
                hint = " Run list_embed_models.py to see valid names." if e.code == 404 else ""
                raise EmbeddingError(
                    f"Embedding API error ({e.code}): {e.message}{hint}"
                ) from e
            except EmbeddingError:
                raise
            except Exception as e:
                if attempt < MAX_ATTEMPTS:
                    time.sleep(5 * attempt)
                    continue
                raise EmbeddingError(f"Unexpected embedding error: {e}") from e

        if len(texts) > BATCH_SIZE:
            print(f"  embedded {len(vectors)}/{len(texts)} chunks", flush=True)
        if start + BATCH_SIZE < len(texts):
            time.sleep(PAUSE_SECONDS)  # stay under the free-tier per-minute limit
    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document chunks (for storing)."""
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a user question (for searching)."""
    return _embed([text], "RETRIEVAL_QUERY")[0]