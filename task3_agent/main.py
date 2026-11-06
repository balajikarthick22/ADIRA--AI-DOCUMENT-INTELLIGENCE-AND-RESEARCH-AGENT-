"""Command-line interface for Task 3 (run from the project root).

    python -m task3_agent.main 'Find the population of India and calculate 2% of it'
    python -m task3_agent.main            (interactive mode)
"""
import argparse

from llm_client import LLMError

from .agent import AgentError, run_agent


def _run(question: str) -> None:
    try:
        run_agent(question)
    except (AgentError, LLMError, ValueError) as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="AI agent with web search, calculator and file reader.")
    parser.add_argument("question", nargs="?", help="Question to answer. Leave out for interactive mode.")
    args = parser.parse_args()

    if args.question:
        _run(args.question)
        return

    print("Ask the agent something. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            _run(q)


if __name__ == "__main__":
    main()