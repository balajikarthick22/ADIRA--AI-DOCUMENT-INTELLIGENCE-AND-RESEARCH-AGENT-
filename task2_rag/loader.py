"""Load PDF and text files. Every page keeps its file name and page number."""
import os

from pypdf import PdfReader

SUPPORTED = (".pdf", ".txt", ".md")


class LoadError(Exception):
    """Raised when a document cannot be read."""


def load_file(path: str) -> list[dict]:
    """Return a list of {"text": ..., "source": file name, "page": int}.

    page is the 1-based PDF page number, or 0 for plain text files.
    """
    if not os.path.isfile(path):
        raise LoadError(f"File not found: {path}")

    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED:
        raise LoadError(f"Unsupported file type '{ext}'. Use: {', '.join(SUPPORTED)}")

    pages = []
    if ext == ".pdf":
        try:
            reader = PdfReader(path)
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"text": text, "source": name, "page": number})
        except Exception as e:
            raise LoadError(f"Could not read PDF '{name}': {e}") from e
        if not pages:
            raise LoadError(
                f"No text found in '{name}'. It may be a scanned PDF (images only), "
                "which needs OCR."
            )
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        except OSError as e:
            raise LoadError(f"Could not read '{name}': {e}") from e
        if not text:
            raise LoadError(f"'{name}' is empty.")
        pages.append({"text": text, "source": name, "page": 0})

    return pages