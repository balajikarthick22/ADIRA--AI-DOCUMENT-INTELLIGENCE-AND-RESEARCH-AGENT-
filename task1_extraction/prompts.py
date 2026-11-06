"""Reusable prompt templates with few-shot examples."""
import json

SYSTEM_INSTRUCTION = (
    "You are a precise information-extraction engine. "
    "Return ONLY valid JSON that matches the requested fields. "
    "Never invent values. If a field is not present in the text, use null "
    "(or an empty list for list fields)."
)

# NOTE: the template itself has no JSON braces, so str.format is safe.
# Examples are rendered separately and inserted as plain text.
TEMPLATE = """Task: Extract {doc_label} information from the text below.

Fields to extract:
{field_guide}

Here are examples of correct extraction:

{examples}

Now extract from this text. Return only the JSON object.

Text:
\"\"\"
{text}
\"\"\"
JSON:"""

INVOICE_FIELDS = """- invoice_number: string or null
- date: string in YYYY-MM-DD format or null
- vendor_name: string or null (the company issuing the invoice)
- total_amount: number or null (final amount due, no currency symbol)
- currency: 3-letter code like USD, INR, EUR, or null"""

RESUME_FIELDS = """- candidate_name: string or null
- email: string or null
- skills: list of strings
- education: list of objects with degree, institution, year
- experience: list of objects with job_title, company, duration, description"""

INVOICE_EXAMPLES = [
    (
        "INVOICE\nAcme Supplies Pvt Ltd\nInvoice No: INV-2041\nDate: 14 March 2024\n"
        "Items: Printer paper x10 ... 2,000.00\nGST 18% ... 360.00\nTotal Due: INR 2,360.00",
        {"invoice_number": "INV-2041", "date": "2024-03-14",
         "vendor_name": "Acme Supplies Pvt Ltd", "total_amount": 2360.00, "currency": "INR"},
    ),
    (
        "Bill from Greenfield Cafe\nDate: 2024-07-02\nAmount: $45.50\n(no invoice number printed)",
        {"invoice_number": None, "date": "2024-07-02",
         "vendor_name": "Greenfield Cafe", "total_amount": 45.50, "currency": "USD"},
    ),
]

RESUME_EXAMPLES = [
    (
        "Priya Sharma | priya.sharma@example.com\nSkills: Python, SQL, Pandas\n"
        "Education: B.Tech Computer Science, VIT University, 2022\n"
        "Experience: Data Analyst at TechNova, Jan 2022 - Present. Built sales dashboards.",
        {"candidate_name": "Priya Sharma", "email": "priya.sharma@example.com",
         "skills": ["Python", "SQL", "Pandas"],
         "education": [{"degree": "B.Tech Computer Science", "institution": "VIT University", "year": "2022"}],
         "experience": [{"job_title": "Data Analyst", "company": "TechNova",
                         "duration": "Jan 2022 - Present", "description": "Built sales dashboards."}]},
    ),
    (
        "Rahul Verma\nFresher, looking for opportunities.\nSkills: Java, HTML",
        {"candidate_name": "Rahul Verma", "email": None, "skills": ["Java", "HTML"],
         "education": [], "experience": []},
    ),
]

CONFIG = {
    "invoice": {"label": "invoice", "fields": INVOICE_FIELDS, "examples": INVOICE_EXAMPLES},
    "resume": {"label": "resume", "fields": RESUME_FIELDS, "examples": RESUME_EXAMPLES},
}


def _render_examples(examples) -> str:
    blocks = []
    for i, (text, output) in enumerate(examples, 1):
        blocks.append(f"Example {i}\nText:\n\"\"\"\n{text}\n\"\"\"\nJSON:\n{json.dumps(output, indent=2)}")
    return "\n\n".join(blocks)


def build_prompt(doc_type: str, text: str) -> str:
    if doc_type not in CONFIG:
        raise ValueError(f"Unknown document type '{doc_type}'. Use: {list(CONFIG)}")
    cfg = CONFIG[doc_type]
    return TEMPLATE.format(
        doc_label=cfg["label"],
        field_guide=cfg["fields"],
        examples=_render_examples(cfg["examples"]),
        text=text.strip(),
    )