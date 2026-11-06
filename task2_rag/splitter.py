"""Recursive text splitting. Each chunk keeps its source file and page number."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 2000      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks


def split_pages(pages: list[dict]) -> list[dict]:
    """Split loaded pages into chunks: {"text", "source", "page", "chunk"}."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # try big breaks first, then smaller
    )
    chunks = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page["text"])):
            if piece.strip():
                chunks.append({
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": i,
                })
    return chunks