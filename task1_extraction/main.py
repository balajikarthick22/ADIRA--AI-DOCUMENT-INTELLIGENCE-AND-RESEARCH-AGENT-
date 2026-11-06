"""CLI for Task 1.

Examples (run from the project root):
    python -m task1_extraction.main --type invoice --file sample_invoice.txt
    python -m task1_extraction.main --type resume --text "Jane Doe, jane@x.com, Skills: Python"
    python -m task1_extraction.main --demo
"""
import argparse
import sys

from .extractor import ExtractionError, extract

DEMO_INVOICE = """TAX INVOICE
Bright Solutions Ltd
Invoice #: BS-7788
Issued: 5th Jan 2025
Consulting services .... 12,000.00
VAT ............ 2,400.00
Grand Total: EUR 14,400.00"""

DEMO_RESUME = """Ananya Rao
ananya.rao@example.com
Skills: Python, Machine Learning, SQL, Streamlit
Education: M.Sc Data Science, Anna University, 2023
Experience: ML Intern at DataWorks, Jun 2022 - Dec 2022. Built a churn prediction model."""


def main():
    parser = argparse.ArgumentParser(description="Extract structured data from invoices/resumes.")
    parser.add_argument("--type", choices=["invoice", "resume"])
    parser.add_argument("--file", help="Path to a .txt file")
    parser.add_argument("--text", help="Raw text to extract from")
    parser.add_argument("--demo", action="store_true", help="Run on built-in sample texts")
    args = parser.parse_args()

    jobs = []
    if args.demo:
        jobs = [("invoice", DEMO_INVOICE), ("resume", DEMO_RESUME)]
    else:
        if not args.type or not (args.file or args.text):
            parser.error("Provide --type and either --file or --text (or use --demo).")
        if args.file:
            try:
                with open(args.file, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError as e:
                sys.exit(f"Could not read file: {e}")
        else:
            text = args.text
        jobs = [(args.type, text)]

    for doc_type, text in jobs:
        print(f"\n=== {doc_type.upper()} ===")
        try:
            result = extract(doc_type, text)
            print(result.model_dump_json(indent=2))
        except (ExtractionError, ValueError) as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()