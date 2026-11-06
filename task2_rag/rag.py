"""The RAG pipeline: ingest_file() stores documents, ask() answers questions about them."""
from llm_client import GeminiClient, LLMError
from google.genai import types

from . import vectorstore
from .embeddings import embed_documents, embed_query
from .loader import load_file
from .splitter import split_pages

NOT_FOUND_MESSAGE = "I could not find the answer in the uploaded documents."
TOP_K = 5
# Cosine distance of the best chunk. Above this, the question is treated as unrelated.
# Tune it: raise it if good questions are rejected, lower it if off-topic ones get through.
MAX_DISTANCE = 0.65

SYSTEM_INSTRUCTION = (
    "You answer questions using ONLY the document excerpts provided in the context. "
    "Do not use outside knowledge. Cite the sources you used in the form (file name, page N); "
    "for text files, cite only the file name. "
    f"If the context does not contain the answer, reply exactly: {NOT_FOUND_MESSAGE}"
)

PROMPT_TEMPLATE = """Context excerpts from the uploaded documents:

{context}

Question: {question}

Answer using only the context above, with citations:"""


def ingest_file(path: str) -> dict:
    """Load, split, embed and store one document. Returns a small summary."""
    pages = load_file(path)
    chunks = split_pages(pages)
    if not chunks:
        raise ValueError("The document produced no text chunks.")
    embeddings = embed_documents([c["text"] for c in chunks])   # embed first, so a failure keeps old data
    vectorstore.delete_source(chunks[0]["source"])
    vectorstore.add_chunks(chunks, embeddings)
    return {"source": chunks[0]["source"], "pages": len(pages), "chunks": len(chunks)}


def _format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = f"{h['source']}, page {h['page']}" if h["page"] else h["source"]
        blocks.append(f"[Excerpt {i} | {where}]\n{h['text']}")
    return "\n\n".join(blocks)


def _unique_sources(hits: list[dict]) -> list[dict]:
    seen, sources = set(), []
    for h in hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": h["source"], "page": h["page"]})
    return sources


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    return vectorstore.search(embed_query(question), k=k)


def ask(question: str, k: int = TOP_K, client: GeminiClient | None = None) -> dict:
    """Answer a question from the stored documents.

    Returns {"answer": str, "sources": [{"source", "page"}], "found": bool}.
    """
    if not question or not question.strip():
        raise ValueError("Question is empty.")

    hits = retrieve(question, k)
    if not hits or hits[0]["distance"] > MAX_DISTANCE:
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "found": False}

    prompt = PROMPT_TEMPLATE.format(context=_format_context(hits), question=question.strip())
    client = client or GeminiClient()
    answer = client.generate(prompt, system_instruction=SYSTEM_INSTRUCTION).strip()

    if NOT_FOUND_MESSAGE.lower() in answer.lower():
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "found": False}
    return {"answer": answer, "sources": _unique_sources(hits), "found": True}


def ask_stream(question: str, k: int = TOP_K, client: GeminiClient | None = None) -> dict:
    """Like ask(), but the answer arrives in pieces.

    Returns {"found": bool, "sources": [...], "stream": iterator of text pieces}.
    """
    if not question or not question.strip():
        raise ValueError("Question is empty.")

    hits = retrieve(question, k)
    if not hits or hits[0]["distance"] > MAX_DISTANCE:
        return {"found": False, "sources": [], "stream": iter([NOT_FOUND_MESSAGE])}

    prompt = PROMPT_TEMPLATE.format(context=_format_context(hits), question=question.strip())
    client = client or GeminiClient()
    config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0.0)

    def _pieces():
        try:
            for chunk in client.client.models.generate_content_stream(
                model=client.model, contents=prompt, config=config
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(f"Streaming failed: {e}") from e

    return {"found": True, "sources": _unique_sources(hits), "stream": _pieces()}