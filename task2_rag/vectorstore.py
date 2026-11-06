"""ChromaDB storage and search. Data is saved to disk in the chroma_db folder."""
import os

import chromadb

DB_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION = "documents"
WRITE_BATCH = 500


_cached_client = None


def _client():
    global _cached_client
    if _cached_client is None:
        _cached_client = chromadb.PersistentClient(path=DB_PATH)
    return _cached_client


def _collection():
    # cosine space: a smaller distance means a closer match (distance = 1 - similarity)
    return _client().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    col = _collection()
    for start in range(0, len(chunks), WRITE_BATCH):
        part = chunks[start:start + WRITE_BATCH]
        col.upsert(
            ids=[f"{c['source']}::p{c['page']}::c{c['chunk']}" for c in part],
            documents=[c["text"] for c in part],
            embeddings=embeddings[start:start + WRITE_BATCH],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in part],
        )


def delete_source(source: str) -> None:
    """Remove all chunks of one file, so re-uploading it does not leave duplicates."""
    _collection().delete(where={"source": source})


def search(query_embedding: list[float], k: int = 5) -> list[dict]:
    """Return the k closest chunks as {"text", "source", "page", "distance"}."""
    col = _collection()
    total = col.count()
    if total == 0:
        return []
    result = col.query(
        query_embeddings=[query_embedding],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": text, "source": meta["source"], "page": meta["page"], "distance": dist})
    return hits


def list_sources() -> dict[str, int]:
    """Return {file name: number of chunks}."""
    data = _collection().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts


def clear() -> None:
    """Delete everything that has been stored."""
    try:
        _client().delete_collection(COLLECTION)
    except Exception:
        pass