"""Command-line interface for Task 2 (run from the project root).

    python -m task2_rag.main ingest docs
    python -m task2_rag.main ingest docs/report.pdf notes.txt
    python -m task2_rag.main ask "What is the refund policy?"
    python -m task2_rag.main chat
    python -m task2_rag.main list
    python -m task2_rag.main clear
"""
import argparse
import os

from llm_client import LLMError

from . import vectorstore
from .embeddings import EmbeddingError
from .loader import SUPPORTED, LoadError
from .rag import ask, ingest_file


def _collect_files(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in sorted(names):
                    if n.lower().endswith(SUPPORTED):
                        files.append(os.path.join(root, n))
        else:
            files.append(p)
    return files


def _print_result(result: dict) -> None:
    print("\nAnswer:\n" + result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - {s['source']}, page {s['page']}" if s["page"] else f"  - {s['source']}")


def _safe_ask(question: str) -> None:
    try:
        _print_result(ask(question))
    except (LLMError, EmbeddingError, ValueError) as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG over your PDF and text documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Add files or folders")
    p_ingest.add_argument("paths", nargs="+")
    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question")
    sub.add_parser("chat", help="Ask questions interactively")
    sub.add_parser("list", help="Show stored documents")
    sub.add_parser("clear", help="Delete all stored documents")
    args = parser.parse_args()

    if args.command == "ingest":
        files = _collect_files(args.paths)
        if not files:
            print("No .pdf, .txt or .md files found.")
        for f in files:
            try:
                info = ingest_file(f)
                print(f"Added {info['source']}: {info['pages']} page(s), {info['chunks']} chunks")
            except (LoadError, EmbeddingError, LLMError, ValueError) as e:
                print(f"Skipped {f}: {e}")

    elif args.command == "ask":
        _safe_ask(args.question)

    elif args.command == "chat":
        print("Ask questions about your documents. Type 'exit' to quit.")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                _safe_ask(q)

    elif args.command == "list":
        sources = vectorstore.list_sources()
        if not sources:
            print("Nothing stored yet.")
        for name, count in sources.items():
            print(f"{name}: {count} chunks")

    elif args.command == "clear":
        vectorstore.clear()
        print("All stored documents deleted.")


if __name__ == "__main__":
    main()