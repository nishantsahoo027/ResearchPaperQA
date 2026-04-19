"""
agent.py — Research Paper Q&A Agent
Author : Nishant Sahoo
Course : Agentic AI Hands-On | Dr. Kanthi Kiran Sirra

Shared agent module. Import build_agent() from this file
to use the compiled LangGraph app in any context.
"""

import os, re
import fitz
import chromadb
from typing import TypedDict, List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── Constants ────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 0
SLIDING_WINDOW         = 6
CHUNK_SIZE             = 800
EMBED_MODEL            = "all-MiniLM-L6-v2"
LLM_MODEL              = "llama-3.3-70b-versatile"

ROUTER_PROMPT = (
    "Route this question. Reply ONE word: retrieve, tool, or memory_only.\n"
    "- retrieve    : paper content, methods, results, architecture\n"
    "- tool        : needs live web info (recent papers, citations)\n"
    "- memory_only : greeting or follow-up, no new retrieval needed"
)

SYSTEM_PROMPT = (
    "You are a Research Paper Q&A assistant.\n"
    "You MUST answer the question using the context provided below.\n"
    "The context contains excerpts from the paper — use them to give a complete answer.\n"
    "Only say 'I don't have that in the knowledge base' if the context is truly empty or completely unrelated.\n"
    "Never refuse to answer when relevant context is present."
)

EVAL_PROMPT = (
    "Rate faithfulness 0.0–1.0. Reply ONE decimal number only.\n"
    "1.0=fully grounded, 0.7=minor inference, 0.5=partial, 0.0=fabricated"
)

# ── State ────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    paper_filter: str


# ── PDF helpers ──────────────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\x0c", "", text)


def extract_pdf_chunks(pdf_path: str, paper_name: str,
                       chunk_size: int = CHUNK_SIZE) -> list[dict]:
    with fitz.open(pdf_path) as doc:
        full_text = _clean("".join(page.get_text() for page in doc))
    sentences = [s.strip() for s in re.split(r'\n+', full_text) if len(s.strip()) > 40]
    documents, buf, idx = [], "", 1
    for sentence in sentences:
        if len(buf) + len(sentence) < chunk_size:
            buf += " " + sentence
        else:
            if buf.strip():
                documents.append({
                    "id":    f"{paper_name}_{idx:03d}",
                    "paper": paper_name,
                    "topic": f"Section {idx}",
                    "text":  buf.strip()
                })
                idx += 1
            buf = sentence
    if buf.strip():
        documents.append({
            "id":    f"{paper_name}_{idx:03d}",
            "paper": paper_name,
            "topic": f"Section {idx}",
            "text":  buf.strip()
        })
    return documents


# ── Build agent ──────────────────────────────────────────────
def build_agent(pdf_papers: list[str]):
    """
    Build and return a compiled LangGraph agent.

    Args:
        pdf_papers: list of PDF file paths to ingest into the knowledge base.

    Returns:
        (app, embedder, collection) — compiled graph, embedder, ChromaDB collection.
    """
    llm      = ChatGroq(model=LLM_MODEL, temperature=0)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Build ChromaDB
    chroma = chromadb.Client()
    try:    chroma.delete_collection("research_kb")
    except: pass
    collection = chroma.create_collection("research_kb")

    documents: list[dict] = []
    for path in pdf_papers:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            documents.extend(extract_pdf_chunks(path, name))
        except Exception as exc:
            print(f"  ⚠️  Could not load {path}: {exc}")

    if documents:
        texts = [d["text"]  for d in documents]
        ids   = [d["id"]    for d in documents]
        metas = [{"topic": d["topic"], "paper": d["paper"]} for d in documents]
        collection.add(
            documents=texts,
            embeddings=embedder.encode(texts).tolist(),
            ids=ids,
            metadatas=metas
        )
        print(f"✅ Knowledge base: {collection.count()} chunks from {len(pdf_papers)} PDFs")

    # ── Retrieval helper ─────────────────────────────────────
    def retrieve(question: str, n: int = 3):
        q_emb   = embedder.encode([question]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=n)
        return results["documents"][0], results["metadatas"][0]

    # ── LLM helper with rate-limit retry ────────────────────
    import time
    def _invoke(messages_or_str, *, max_retries: int = 4) -> str:
        if isinstance(messages_or_str, str):
            messages_or_str = [HumanMessage(content=messages_or_str)]
        for attempt in range(max_retries):
            try:
                return llm.invoke(messages_or_str).content
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "rate_limit" in msg.lower():
                    wait = [15, 30, 45, 60][attempt]
                    print(f"  ⚠️  Rate limit — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("LLM call failed after max retries.")

    # ── Node functions ───────────────────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        return {"messages": msgs[-SLIDING_WINDOW:]}

    def router_node(state: CapstoneState) -> dict:
        prompt = f"{ROUTER_PROMPT}\n\nQuestion: {state['question']}"
        route  = _invoke(prompt).strip().lower()
        if route not in ("retrieve", "tool", "memory_only"):
            route = "retrieve"
        return {"route": route}

    def retrieval_node(state: CapstoneState) -> dict:
        docs, metas = retrieve(state["question"])
        chunks  = [f"[{m['topic']}]\n{d}" for d, m in zip(docs, metas)]
        sources = [m["topic"] for m in metas]
        return {"retrieved": "\n\n---\n\n".join(chunks), "sources": sources}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(state["question"] + " research paper", max_results=3))
            if not results:
                return {"tool_result": "No web results found."}
            snippets = [f"[{r.get('title','')}]\n{r.get('body','')[:200]}" for r in results]
            return {"tool_result": "\n\n".join(snippets)}
        except Exception as exc:
            return {"tool_result": f"Web search unavailable: {exc}"}

    def answer_node(state: CapstoneState) -> dict:
        parts = []
        if state.get("retrieved"):
            parts.append(f"CONTEXT:\n{state['retrieved'][:1500]}")
        if state.get("tool_result"):
            parts.append(f"WEB:\n{state['tool_result'][:400]}")
        context = "\n\n".join(parts) or "No context."
        history_msgs = state.get("messages", [])[:-1][-1:]
        history = "\n".join(f"{m['role'].upper()}: {m['content'][:80]}" for m in history_msgs)
        retries    = state.get("eval_retries", 0)
        retry_note = f"\n[Retry #{retries}: be specific.]" if retries else ""
        user_prompt = (
            f"{context}\n\n"
            f"History:\n{history}\n\n"
            f"Q: {state['question']}{retry_note}\nA:"
        )
        response = _invoke([SystemMessage(content=SYSTEM_PROMPT),
                            HumanMessage(content=user_prompt)])
        return {"answer": response}

    def eval_node(state: CapstoneState) -> dict:
        if not state.get("retrieved") and not state.get("tool_result"):
            return {"faithfulness": 1.0, "eval_retries": state.get("eval_retries", 0)}
        ctx    = (state.get("retrieved") or state.get("tool_result", ""))[:300]
        answer = state["answer"][:150]
        prompt = f"{EVAL_PROMPT}\n\nCtx: {ctx}\nAns: {answer}\nScore:"
        raw    = _invoke(prompt).strip()
        try:
            score = max(0.0, min(1.0, float(raw)))
        except ValueError:
            score = 0.5
        retries = state.get("eval_retries", 0) + 1
        return {"faithfulness": score, "eval_retries": retries}

    def save_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", []) + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": msgs}

    # ── Routing functions ────────────────────────────────────
    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        return "tool" if r == "tool" else "skip" if r == "memory_only" else "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score < FAITHFULNESS_THRESHOLD and retries < MAX_EVAL_RETRIES:
            return "answer"
        return "save"

    # ── Graph assembly ───────────────────────────────────────
    graph = StateGraph(CapstoneState)
    for name, fn in [("memory",   memory_node),
                     ("router",   router_node),
                     ("retrieve", retrieval_node),
                     ("skip",     skip_retrieval_node),
                     ("tool",     tool_node),
                     ("answer",   answer_node),
                     ("eval",     eval_node),
                     ("save",     save_node)]:
        graph.add_node(name, fn)

    graph.set_entry_point("memory")
    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)
    graph.add_conditional_edges("router", route_decision,
                                {"retrieve": "retrieve", "tool": "tool", "skip": "skip"})
    graph.add_conditional_edges("eval",   eval_decision,
                                {"answer": "answer", "save": "save"})

    app = graph.compile(checkpointer=MemorySaver())
    return app, embedder, collection
