"""Agent tools: web search, calculator, file reader. Each returns a string, never raises."""
import ast
import operator
import os

from task2_rag.loader import LoadError, load_file

DOCS_DIR = os.path.abspath(os.getenv("DOCS_DIR", "docs"))
MAX_CHARS = 4000  # cap on text returned to the model

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large.")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only numbers and + - * / // % ** ( ) are allowed.")


def calculator(expression: str) -> str:
    """Safely evaluate a math expression (no eval, no function calls)."""
    try:
        expr = str(expression).replace(",", "").replace("^", "**").strip()
        if not expr:
            return "Error: empty expression."
        return str(_eval(ast.parse(expr, mode="eval")))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, OverflowError) as e:
        return f"Error: {e}"


def web_search(query: str) -> str:
    """Search the web with DuckDuckGo and return the top results."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: the ddgs package is not installed. Run: pip install ddgs"
    try:
        results = DDGS().text(str(query), max_results=5)
    except Exception as e:
        return f"Error: web search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.get('title', '')}\n   {r.get('body', '')}\n   {r.get('href', '')}")
    return "\n".join(lines)


def read_file(filename: str) -> str:
    """Read a .pdf, .txt or .md file from the docs folder."""
    try:
        path = os.path.abspath(os.path.join(DOCS_DIR, str(filename)))
        if os.path.commonpath([path, DOCS_DIR]) != DOCS_DIR:
            return "Error: only files inside the docs folder can be read."
        if not os.path.isfile(path):
            available = ", ".join(sorted(os.listdir(DOCS_DIR))) if os.path.isdir(DOCS_DIR) else "none"
            return f"Error: file '{filename}' not found. Available files: {available}"
        pages = load_file(path)
    except LoadError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: could not read file: {e}"

    parts = [f"[page {p['page']}]\n{p['text']}" if p["page"] else p["text"] for p in pages]
    text = "\n\n".join(parts)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n...[truncated]"
    return text


def list_files(_: str = "") -> str:
    """List the files in the docs folder, with the page count for PDFs."""
    try:
        if not os.path.isdir(DOCS_DIR):
            return "Error: the docs folder does not exist."
        lines = []
        for name in sorted(os.listdir(DOCS_DIR)):
            path = os.path.join(DOCS_DIR, name)
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            if name.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    pages = len(PdfReader(path).pages)
                    lines.append(f"{name}: PDF, {pages} pages, {size} bytes")
                except Exception as e:
                    lines.append(f"{name}: PDF, could not count pages ({e})")
            else:
                lines.append(f"{name}: {size} bytes")
        return "\n".join(lines) if lines else "The docs folder is empty."
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "web_search": {
        "function": web_search,
        "description": "Search the web for current information. Input: a search query.",
    },
    "calculator": {
        "function": calculator,
        "description": ("Evaluate a math expression using numbers and + - * / // % ** ( ). "
                        "Input: an expression like '0.02 * 1400000000'. Write percentages as decimals."),
    },
    "read_file": {
        "function": read_file,
        "description": "Read a file from the docs folder. Input: a file name like 'policy.txt'.",
    },
    
    "list_files": {
        "function": list_files,
        "description": ("List the files in the docs folder with their sizes and the page count "
                        "of PDFs. Use this to count pages. Input: any text, such as 'all'."),
    },
}