"""AI Study Assistant — a beginner LangGraph project.

The agent can:
  1. list_notes     — see what PDF files are in ./notes/
  2. search_notes   — keyword-search PDFs you drop into ./notes/
  3. read_note      — read full text from a chosen PDF, in chunks of pages
  4. calculator     — solve / simplify math with sympy
  5. web_search     — look things up on DuckDuckGo
  6. remember       — the MemorySaver checkpointer keeps the convo across turns
"""

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import sympy
import base64
import shutil
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from pypdf import PdfReader
from ddgs import DDGS
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

NOTES_DIR = Path(__file__).parent / "notes"
PLOTS_DIR = Path(__file__).parent / "plots"
MAX_READ_CHARS = 2500

# Plots produced during the current graph.invoke; app.py clears this after each turn.
_pending_plots: list[str] = []


def drain_pending_plots() -> list[str]:
    """Return and clear plot file paths saved since the last drain."""
    global _pending_plots
    paths = _pending_plots[:]
    _pending_plots = []
    return paths


# We chunk every PDF, embed the chunks locally and store them in an in-memory FAISS index. search_notes then retrieves the most 
# relevant chunks from the documents.

from fastembed import TextEmbedding  # ONNX-based; no torch — saves ~1.5 GB RAM
from langchain_community.vectorstores import FAISS  # still needed for the FAISS wrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class FastEmbedAdapter(Embeddings):
    """Thin LangChain Embeddings wrapper around FastEmbed (ONNX)."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, next(self._model.embed([text]))))


_embeddings = FastEmbedAdapter()
_vectorstore = None
_indexed_signature = None  # (filename, mtime, size) per PDF — detects changes


def _notes_signature():
    if not NOTES_DIR.exists():
        return ()
    return tuple(
        (p.name, p.stat().st_mtime, p.stat().st_size)
        for p in sorted(NOTES_DIR.glob("*.pdf"))
    )


def build_index(force: bool = False):
    """Build (or rebuild) the FAISS index over all PDFs."""
    global _vectorstore, _indexed_signature
    sig = _notes_signature()
    if not force and sig == _indexed_signature and _vectorstore is not None:
        return _vectorstore

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs: list[Document] = []
    for pdf in sorted(NOTES_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf))
        except Exception:
            continue
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for chunk in splitter.split_text(text):
                docs.append(
                    Document(page_content=chunk,
                             metadata={"source": pdf.name, "page": page_num})
                )

    _vectorstore = FAISS.from_documents(docs, _embeddings) if docs else None
    _indexed_signature = sig
    return _vectorstore


@tool
def list_notes() -> str:
    """List all PDF files in the student's ./notes/ folder, with page counts.
    Call this first when the user asks anything about their notes."""
    if not NOTES_DIR.exists():
        return f"No notes folder at {NOTES_DIR}. Ask the student to upload PDFs first."
    pdfs = sorted(NOTES_DIR.glob("*.pdf"))
    if not pdfs:
        return "Notes folder is empty. Ask the student to upload PDFs."
    lines = []
    for p in pdfs:
        try:
            n = len(PdfReader(str(p)).pages)
            lines.append(f"- {p.name} ({n} pages)")
        except Exception as e:
            lines.append(f"- {p.name} (unreadable: {e})")
    return "\n".join(lines)


@tool
def search_notes(query: str) -> str:
    """Semantically search the student's PDF notes and return the most
    relevant passages from anywhere in the documents (not just keyword matches).
    Use this for any question about the content of the notes."""
    vs = build_index()
    if vs is None:
        return ("No readable text found in your notes. The PDFs may be scanned "
                "images (which need OCR) or the folder is empty.")
    results = vs.similarity_search(query, k=4)
    return "\n\n".join(
        f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content.strip()}"
        for d in results
    )


@tool
def read_note(filename: str, start_page: int = 1, num_pages: int = 2) -> str:
    """Read pages from a PDF in ./notes/. Use this to get content for quiz
    generation or summaries. Also use this to generate flashcards when asked 
    for flashcards from notes."""
    path = NOTES_DIR / filename
    if not path.exists():
        return f"No such file: {filename}. Call list_notes to see available files."
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        return f"Could not open {filename}: {e}"

    total = len(reader.pages)
    s = max(1, start_page)
    e = min(total, s + num_pages - 1)
    chunks = []
    for i in range(s - 1, e):
        text = (reader.pages[i].extract_text() or "").strip()
        chunks.append(f"--- {filename} page {i+1} ---\n{text}")
    out = "\n\n".join(chunks)
    if len(out) > MAX_READ_CHARS:
        out = out[:MAX_READ_CHARS] + f"\n\n[truncated; call again with start_page={e+1} for more]"
    out += f"\n\n[showed pages {s}-{e} of {total}]"
    return out


@tool
def calculator(expression: str) -> str:
    """Evaluate or simplify a math expression with sympy."""
    try:
        return str(sympy.sympify(expression))
    except Exception as e:
        return f"Could not evaluate {expression!r}: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo) for an explanation of a topic.
    Use this when the answer isn't in the student's notes."""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Web search failed: {e}"
    if not hits:
        return "No web results."
    return "\n\n".join(
        f"{h.get('title','')}\n{h.get('body','')}\n{h.get('href','')}"
        for h in hits
    )
@tool
def execute_plot_code(code: str) -> str:
    """Execute Python code that draws a plot and save it as a PNG shown to the
    student. Use this whenever the student asks to plot, graph, sketch, or
    visualize a function or some data.
    The code runs with `plt` (matplotlib.pyplot) and `np` (numpy) in scope. Build
    the figure with `plt.plot(...)`, `plt.title(...)`, etc. — do NOT call
    `plt.savefig` or `plt.show` yourself; this tool saves and closes the figure
    after your code runs. After the tool returns, write a one-line description
    of the graph; the image is appended to the chat automatically.
    """
    PLOTS_DIR.mkdir(exist_ok=True)
    filename = f"plot_{int(time.time() * 1000)}.png"
    path = PLOTS_DIR / filename
    try:
        exec(code, {"plt": plt, "np": np})
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        plt.close("all")
        return f"Error executing plot code: {type(e).__name__}: {e}"

    _pending_plots.append(str(path))
    return f"Plot generated and saved as {filename}. It is shown to the student below this message."

tools = [list_notes, search_notes, read_note, calculator, web_search, execute_plot_code]


SYSTEM_PROMPT = (
    "You are a study assistant. You have these tools:\n"
    "- list_notes: see PDF files in the student's notes folder\n"
    "- search_notes(query): semantic search across the full content of all "
    "notes — returns the most relevant passages from anywhere in the PDFs. "
    "This is your main tool for any question about the notes' content.\n"
    "- read_note(filename, start_page, num_pages): read specific pages "
    "verbatim when the student references a page or wants a literal passage\n"
    "- calculator(expression): math via sympy\n"
    "- web_search(query): DuckDuckGo for general explanations\n"
    "- execute_plot_code(code): run a short matplotlib snippet and show the "
    "resulting graph to the student. `plt` (matplotlib.pyplot) and `np` (numpy) "
    "are already in scope. Use plt.plot / plt.title / plt.xlabel / plt.legend "
    "etc., but do NOT call plt.savefig or plt.show — the tool handles saving. "
    "Use this whenever the student asks to plot, graph, sketch, or visualize "
    "a function or some data. After the tool returns, write one short sentence "
    "describing what the graph shows; the image is appended to the chat "
    "automatically.\n\n"
    "Answer directly by default. If the student asks a general question — math, "
    "a definition, an explanation — just answer it using your own knowledge "
    "(and the calculator for computation). "
    "Only use the note tools when the student explicitly refers to the notes "
    "(e.g. 'from my notes', 'quiz me on my notes', 'what's on page 5'). In that "
    "case call search_notes(topic) first and write from the returned passages; "
    "don't ask which file. "
    "When asked for flashcards from the notes, call search_notes(topic) to find relevant "
    "material (use read_note for deeper content if needed), then format the output as plain "
    "question / answer pairs, one per line:\n"
    "Q: <question>\n"
    "A: <answer>\n"
    "Aim for 5-10 cards unless the student asks otherwise.\n"
    "Use web_search only for current/external info you don't know.\n\n"
    "After every tool call returns, you must write a clear text "
    "reply to the student summarizing what you found. Never return an empty message. DO NOT LEAVE A SENTENCE UNFINSHED ALWAYS END" \
    "WITH FULLY SEMANTICALLY MEANINGFUL SENTENCES."
    "Math formatting:\n"
    "- Use $...$ for inline math and $$...$$ for displayed equations. Pick ONE. ALWAYS FINISH A DELIMITER"
    "style and always close every delimiter you open.\n"
    "- For simple logic/set symbols, prefer plain Unicode (∨, ∧, ¬, →, ↔, ∈) "
    "rather than LaTeX, so short statements stay readable.\n"
    "- don't put math inside a Markdown table. Format questions and answer options as a plain numbered or "
    "lettered list (1., 2., a), b)), one item per line.\n"
    "- Don't write bare LaTeX commands like \\lambda outside $ delimiters."
)

VISION_PROMPT = (
    "You are a patient math tutor reviewing a student's handwritten solution shown in the attached image. Do these three things"
     " in order:\n\n "
     "1. **Transcribe** what the student wrote — convert their handwriting (math + words) into clean text. Number each step.\n"
    "2. **Spot errors** for each step, say whether it's correct. If it's wrong, explain specifically what's wrong "
    "(arithmetic slip, wrong rule, missing case, logic gap, bad notation). Be specific about the step.\n"
    "3. **Show the corrected version** rewrite the solution properly from the first wrong step.\n\n"
    "Be honest. If the whole solution is correct, say so clearly. Format all math in $...$ for inline or $$...$$ for display, " 
    "and always close every delimiter you open."
)

def _system_message():
    """Build the system prompt with the current note filenames injected, so the
    bot always knows what files exist even after history trimming."""
    files = (
        ", ".join(p.name for p in sorted(NOTES_DIR.glob("*.pdf")))
        if NOTES_DIR.exists()
        else ""
    )
    suffix = f"\n\nThe student's note files right now are: {files or '(none uploaded yet)'}."
    return SystemMessage(content=SYSTEM_PROMPT + suffix)

def check_solution(image_path, chat_history):
    """Send a handwritten-solution image to a vision LLM and append the
    transcription + critique to the chat as a new turn. Standalone —
    the text agent does NOT see this in its memory."""
    if not image_path:
        return chat_history, None

    # Build a multimodal message (text prompt + inline base64 image).
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else (ext or "png")
    msg = HumanMessage(content=[
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url",
         "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
    ])

    try:
        reply = visionllm.invoke([msg]).content
    except Exception as e:
        reply = f"Sorry — couldn't analyse the image. ({type(e).__name__}: {e})"

    chat_history = chat_history + [
        {"role": "user", "content": "Please check my handwritten solution."},
        {"role": "assistant", "content": reply},
    ]
    return chat_history, None

groq = ChatGroq(model="openai/gpt-oss-20b").bind_tools(tools)
visionllm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_retries=0).bind_tools(tools)


def _is_blank(resp) -> bool:
    """A response with no text and no tool call is a dead end."""
    return not str(resp.content).strip() and not getattr(resp, "tool_calls", None)


def _approx_tokens(messages) -> int:
    return sum(len(str(m.content)) for m in messages) // 4

_trimmer = trim_messages(
    max_tokens=2500,
    strategy="last",
    token_counter=_approx_tokens,
    start_on="human",
    include_system=False,
    allow_partial=False,
)


def agent_node(state: MessagesState):
    msgs = [_system_message()] + _trimmer.invoke(state["messages"])
    try:
        resp = groq.invoke(msgs)
        if _is_blank(resp):
            print("[Groq returned blank] retrying on Gemini")
            resp = gemini.invoke(msgs)
    except Exception as e:
        print(f"[Groq unavailable: {type(e).__name__}] falling back to Gemini")
        resp = gemini.invoke(msgs)
    return {"messages": [resp]}


builder = StateGraph(MessagesState) #It is the main graph class to use 
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # → "tools" or END
builder.add_edge("tools", "agent")  # loop back after a tool runs

graph = builder.compile(checkpointer=MemorySaver()) #graph has to be compiled before it can be used, breakpointer is an additional field that can be added

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "student-1"}}

    questions = [
        "Hi! I'm Lucy and I'm studying for a math exam.",
        "What's the derivative of x^3 + 2*x?",
        "Search the web for a one-sentence explanation of the chain rule.",
        "Can you check my notes for anything about logic?",
        "Just to confirm what was my name again?",
    ]

    for q in questions:
        print(f"\n>>> {q}")
        result = graph.invoke({"messages": [("user", q)]}, config=config)
        print(result["messages"][-1].content)
