"""ADIRA: glass-style Streamlit chat app (documents RAG or tool-using agent).

Run from the project root:   streamlit run task4_app/app.py
"""
import html
import os
import sys
import time

# Streamlit puts this script's folder on the path, not the project root, so add the root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from llm_client import LLMError
from task2_rag import vectorstore
from task2_rag.embeddings import EmbeddingError
from task2_rag.loader import LoadError
from task2_rag.rag import NOT_FOUND_MESSAGE, ask_stream, ingest_file
from task3_agent.agent import AgentError, run_agent
from task3_agent.tools import DOCS_DIR, TOOLS

RAG_MODE = "📄 Documents (RAG)"
AGENT_MODE = "🛠️ Agent (tools)"
SPEEDS = {"Letter by letter": 1, "Fast": 3, "Off": 0}   # characters shown per tick

RAG_EXAMPLES = [
    "Summarize the main topics in my documents",
    "How long do refunds take?",
    "What is the capital of Mars?",
]
AGENT_EXAMPLES = [
    "What is 15% of 2480 plus 37?",
    "Find the current population of India and calculate 2% of it",
    "Read docs/policy.txt and tell me how much express shipping costs",
]

st.set_page_config(page_title="ADIRA", page_icon="✨", layout="centered",
                   initial_sidebar_state="expanded")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--ink:#0F172A;--muted:#475569;--blue:#2563EB;--blue-d:#1D4ED8;
  --glass:rgba(255,255,255,.62);--edge:rgba(255,255,255,.88);
  --shadow:0 10px 32px rgba(15,23,42,.10);}
.stApp,.stApp p,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,
.stApp button,.stApp input,.stApp textarea{
  font-family:'Inter',-apple-system,'SF Pro Text','Segoe UI',Roboto,sans-serif;}
#MainMenu,footer{visibility:hidden}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2.2rem;padding-bottom:6rem;max-width:860px}

/* ---------- animated glass background ---------- */
.stApp{background:#E8ECF3;isolation:isolate}
.stApp::before{content:"";position:fixed;inset:-25%;z-index:-1;pointer-events:none;
  background:
   radial-gradient(40% 35% at 20% 25%,rgba(99,102,241,.30),transparent 70%),
   radial-gradient(35% 40% at 80% 20%,rgba(56,189,248,.28),transparent 70%),
   radial-gradient(40% 40% at 65% 85%,rgba(167,139,250,.26),transparent 70%);
  animation:drift 26s ease-in-out infinite alternate}
.stApp::after{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
  background-image:linear-gradient(rgba(15,23,42,.05) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(15,23,42,.05) 1px,transparent 1px);
  background-size:38px 38px;
  -webkit-mask-image:radial-gradient(ellipse at center,#000 30%,transparent 82%);
  mask-image:radial-gradient(ellipse at center,#000 30%,transparent 82%)}
@keyframes drift{from{transform:translate(-3%,-2%) scale(1)}to{transform:translate(3%,2%) scale(1.08)}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes pulse{70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}

/* ---------- header ---------- */
.hero{display:flex;align-items:center;gap:14px;margin:4px 0 18px;animation:rise .6s ease backwards}
.logo{width:50px;height:50px;border-radius:15px;display:grid;place-items:center;color:#fff;
  font-weight:800;font-size:23px;background:linear-gradient(135deg,#2563EB,#7C3AED);
  box-shadow:0 8px 22px rgba(37,99,235,.35)}
.brand{font-size:31px;font-weight:800;letter-spacing:-.02em;color:var(--ink);line-height:1.1}
.tag{color:var(--muted);font-size:14px;font-weight:500;display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:#22C55E;
  box-shadow:0 0 0 0 rgba(34,197,94,.6);animation:pulse 2s infinite}
.welcome{padding:22px 24px;border-radius:22px;background:var(--glass);
  backdrop-filter:blur(18px) saturate(160%);-webkit-backdrop-filter:blur(18px) saturate(160%);
  border:1px solid var(--edge);box-shadow:var(--shadow);margin-bottom:14px;animation:rise .6s ease backwards}
.welcome h3{margin:0 0 4px;font-size:20px;font-weight:700;color:var(--ink)}
.welcome p{margin:0;color:var(--muted);font-size:15px}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:var(--glass);backdrop-filter:blur(22px) saturate(170%);
  -webkit-backdrop-filter:blur(22px) saturate(170%);border-right:1px solid var(--edge);box-shadow:var(--shadow)}
[data-testid="stSidebar"]>div:first-child{background:transparent}
.side-label{font-size:12px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin:20px 0 8px}
[data-testid="stRadio"] [role="radiogroup"]{gap:8px}
[data-testid="stRadio"] label{background:rgba(255,255,255,.72);border:1px solid rgba(15,23,42,.14);
  border-radius:14px;padding:10px 14px;width:100%;transition:transform .18s ease,border-color .18s,background .18s}
[data-testid="stRadio"] label:hover{transform:translateY(-1px);border-color:var(--blue)}
[data-testid="stRadio"] label:has(input:checked){background:rgba(37,99,235,.15);border-color:#2563EB}
[data-testid="stRadio"] label p{font-weight:600;color:var(--ink)}

/* ---------- chat bubbles ---------- */
[data-testid="stChatMessage"]{background:var(--glass);backdrop-filter:blur(16px) saturate(160%);
  -webkit-backdrop-filter:blur(16px) saturate(160%);border:1px solid var(--edge);border-radius:20px;
  padding:14px 18px;margin-bottom:12px;box-shadow:var(--shadow);
  animation:rise .4s ease backwards;transition:transform .2s ease,box-shadow .2s ease}
[data-testid="stChatMessage"]:hover{transform:translateY(-1px);box-shadow:0 14px 38px rgba(15,23,42,.14)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  flex-direction:row-reverse;width:fit-content;max-width:86%;margin-left:auto;
  background:rgba(37,99,235,.14);border-color:rgba(37,99,235,.30)}
[data-testid="stChatMessageAvatarUser"]{background:#2563EB}
[data-testid="stChatMessageAvatarAssistant"]{background:#0F172A}
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li{color:var(--ink);font-size:16px;line-height:1.65}

/* ---------- chat input ---------- */
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{
  background:transparent !important}
[data-testid="stChatInput"]{background:rgba(255,255,255,.78);backdrop-filter:blur(18px);
  -webkit-backdrop-filter:blur(18px);border:1px solid rgba(37,99,235,.30);border-radius:22px;
  box-shadow:0 12px 34px rgba(15,23,42,.12);transition:box-shadow .25s,border-color .25s}
[data-testid="stChatInput"]>div{background:transparent}
[data-testid="stChatInput"]:focus-within{border-color:#2563EB;
  box-shadow:0 0 0 4px rgba(37,99,235,.18),0 12px 34px rgba(15,23,42,.14)}
[data-testid="stChatInput"] textarea{font-size:16px;color:var(--ink)}

/* ---------- buttons ---------- */
.stButton>button{min-height:46px;border-radius:14px;font-weight:600;font-size:15px;color:var(--ink);
  background:rgba(255,255,255,.85);border:1px solid rgba(15,23,42,.14);
  box-shadow:0 2px 8px rgba(15,23,42,.06);
  transition:transform .18s ease,box-shadow .18s ease,border-color .18s,background .18s,color .18s}
.stButton>button p{color:inherit !important;font-weight:600}
.stButton>button:hover{transform:translateY(-2px);border-color:var(--blue);color:var(--blue-d);
  box-shadow:0 10px 24px rgba(37,99,235,.20)}
.stButton>button:active{transform:translateY(0)}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#2563EB,#4F46E5);color:#fff;border:none;
  box-shadow:0 8px 20px rgba(37,99,235,.35)}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover{
  color:#fff;box-shadow:0 12px 28px rgba(37,99,235,.45)}

/* ---------- upload ---------- */
[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.58);border:2px dashed rgba(37,99,235,.45);
  border-radius:18px;transition:background .2s,border-color .2s,transform .2s}
[data-testid="stFileUploaderDropzone"]:hover{background:rgba(255,255,255,.88);border-color:#2563EB;transform:translateY(-1px)}
[data-testid="stFileUploaderDropzone"] button{border-radius:12px;font-weight:600;color:var(--ink);
  background:#fff;border:1px solid rgba(15,23,42,.18)}
[data-testid="stAlert"]{border-radius:14px}

/* ---------- chips and agent steps ---------- */
.chip{display:inline-block;padding:5px 12px;margin:4px 6px 0 0;border-radius:999px;font-size:13px;
  font-weight:600;color:#1E3A8A;background:rgba(37,99,235,.12);border:1px solid rgba(37,99,235,.28)}
.step{display:flex;gap:12px;padding:12px 14px;margin:8px 0;border-radius:16px;
  background:rgba(255,255,255,.72);border:1px solid var(--edge);animation:rise .35s ease backwards}
.step-n{flex:0 0 28px;height:28px;border-radius:50%;background:#2563EB;color:#fff;font-weight:700;
  display:grid;place-items:center;font-size:13px}
.step-thought{font-weight:600;color:var(--ink);margin-bottom:6px}
.step code{display:block;background:#0F172A;color:#E2E8F0;padding:6px 10px;border-radius:10px;
  font-size:13px;overflow-x:auto;white-space:pre-wrap}
.step-obs{color:var(--muted);font-size:13.5px;margin-top:6px}

@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none !important;transition:none !important}}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="logo">A</div><div><div class="brand">ADIRA</div>'
    '<div class="tag"><span class="dot"></span>AI DOCUMENT INTELLIGENCE - RESEARCH AGENT</div></div></div>',
    unsafe_allow_html=True,
)

for key, default in (("messages", []), ("saved", set()), ("indexed", set()),
                     ("mode", RAG_MODE), ("speed", "Letter by letter")):
    if key not in st.session_state:
        st.session_state[key] = default


# ---------- helpers ----------
def esc(text) -> str:
    """Escape text for HTML. Newlines become <br> so the markdown block is never split."""
    return html.escape(str(text)).replace("\n", "<br>")


def source_chips(sources: list[dict]) -> str:
    out = []
    for s in sources:
        label = f'{s["source"]} · page {s["page"]}' if s["page"] else s["source"]
        out.append(f'<span class="chip">📄 {esc(label)}</span>')
    return "".join(out)


def step_card(s: dict) -> str:
    obs = str(s.get("observation", ""))
    obs = obs if len(obs) <= 400 else obs[:400] + "..."
    call = f'{s.get("action", "")}({s.get("input", "")!r})'
    return (
        f'<div class="step"><div class="step-n">{esc(s.get("step", ""))}</div>'
        f'<div style="min-width:0"><div class="step-thought">{esc(s.get("thought", ""))}</div>'
        f'<code>{esc(call)}</code><div class="step-obs">{esc(obs)}</div></div></div>'
    )


def typed(source):
    """Wrap a stream of text so it appears a few letters at a time, like typing."""
    per_tick = SPEEDS[st.session_state.speed]
    if not per_tick:
        return iter(source)

    def _gen():
        for piece in source:
            for i in range(0, len(piece), per_tick):
                yield piece[i:i + per_tick]
                time.sleep(0.012)
    return _gen()


def show_steps(steps: list[dict]) -> None:
    """Show the agent's actions and observations in a collapsible box."""
    with st.expander(f"Agent steps ({len(steps)})"):
        if not steps:
            st.write("The agent answered without using a tool.")
        for s in steps:
            st.markdown(step_card(s), unsafe_allow_html=True)


def save_upload(up) -> str:
    """Save an uploaded file into the docs folder (the agent can read it from there)."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    path = os.path.join(DOCS_DIR, os.path.basename(up.name))
    with open(path, "wb") as f:
        f.write(up.getbuffer())
    return path


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="side-label">Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", [RAG_MODE, AGENT_MODE], label_visibility="collapsed")
    if mode != st.session_state.mode:       # switching mode starts a fresh chat
        st.session_state.mode = mode
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="side-label">Typing effect</div>', unsafe_allow_html=True)
    st.selectbox("Typing effect", list(SPEEDS), key="speed", label_visibility="collapsed")

    st.markdown('<div class="side-label">Upload documents</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "PDF, TXT or MD files", type=["pdf", "txt", "md"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    refresh = False
    for up in uploads or []:
        key = f"{up.name}:{up.size}"
        try:
            if key not in st.session_state.saved:
                save_upload(up)
                st.session_state.saved.add(key)
            if mode == RAG_MODE and key not in st.session_state.indexed:
                path = os.path.join(DOCS_DIR, os.path.basename(up.name))
                with st.spinner(f"Indexing {up.name} (large files can take a few minutes)..."):
                    info = ingest_file(path)
                st.session_state.indexed.add(key)
                st.success(f"Indexed {info['source']}: {info['pages']} page(s), {info['chunks']} chunks")
                refresh = True
            elif mode == AGENT_MODE:
                st.caption(f"{up.name} saved to the docs folder. The agent can read it.")
        except (LoadError, EmbeddingError, LLMError, ValueError, OSError) as e:
            st.error(f"Could not process {up.name}: {e}")

    # Read the stored-document list once, and again only after something was indexed.
    if refresh or "stored" not in st.session_state:
        st.session_state.stored = vectorstore.list_sources()
    stored = st.session_state.stored

    if mode == RAG_MODE:
        st.markdown('<div class="side-label">Indexed documents</div>', unsafe_allow_html=True)
        if stored:
            st.markdown("".join(f'<span class="chip">📄 {esc(n)} · {c}</span>' for n, c in stored.items()),
                        unsafe_allow_html=True)
        else:
            st.caption("Nothing indexed yet.")
    else:
        st.markdown('<div class="side-label">Agent tools</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<span class="chip">🔧 {esc(n)}</span>' for n in TOOLS), unsafe_allow_html=True)
        st.caption("Each step is one Gemini call, so free-tier limits apply.")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("✨ New chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Welcome screen (only while the chat is empty) ----------
if not st.session_state.messages:
    if mode == RAG_MODE and not stored:
        st.markdown('<div class="welcome"><h3>Welcome 👋</h3><p>Upload a document in the sidebar, '
                    'then ask a question about it.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="welcome"><h3>What would you like to know?</h3><p>Type below, '
                    'or try one of these.</p></div>', unsafe_allow_html=True)
        tag = "rag" if mode == RAG_MODE else "agent"
        for i, example in enumerate(RAG_EXAMPLES if mode == RAG_MODE else AGENT_EXAMPLES):
            if st.button(example, key=f"example_{tag}_{i}", use_container_width=True):
                st.session_state.pending = example
                st.rerun()

# ---------- Chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("steps") is not None:
            show_steps(m["steps"])
        st.markdown(m["content"])
        if m.get("sources"):
            st.markdown(source_chips(m["sources"]), unsafe_allow_html=True)

# ---------- New question ----------
question = st.chat_input(
    "Ask a question about your documents" if mode == RAG_MODE else "Ask the agent something"
)
if not question and "pending" in st.session_state:
    question = st.session_state.pop("pending")      # an example button was clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if mode == RAG_MODE:
                with st.spinner("Searching your documents..."):
                    result = ask_stream(question)
                answer = st.write_stream(typed(result["stream"]))
                found = result["found"] and NOT_FOUND_MESSAGE.lower() not in answer.lower()
                sources = result["sources"] if found else []
                if sources:
                    st.markdown(source_chips(sources), unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            else:
                with st.status("The agent is working...", expanded=True) as status:
                    def on_step(s):
                        st.markdown(step_card(s), unsafe_allow_html=True)
                    result = run_agent(question, verbose=False, on_step=on_step,
                                        history=st.session_state.messages[:-1])
                    status.update(label=f"Done in {len(result['steps'])} step(s)",
                                  state="complete", expanded=False)
                st.write_stream(typed([result["answer"]]))
                if not result["finished"]:
                    st.caption("The agent stopped at its step limit, so this answer may be incomplete.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "steps": result["steps"]}
                )
        except (AgentError, LLMError, EmbeddingError, ValueError) as e:
            message = f"Something went wrong: {e}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})