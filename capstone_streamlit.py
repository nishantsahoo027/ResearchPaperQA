import streamlit as st
import uuid, os, re
import fitz
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="Research Paper Q&A", layout="centered")
st.title("📄 Research Paper Q&A Agent")
st.caption("Ask questions about Deep Reinforcement Learning research.")

def _clean(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    return re.sub(r'[ \t]+', ' ', text)

def _chunk_pdf(pdf_path, paper_name, chunk_size=800):
    with fitz.open(pdf_path) as doc:
        raw = _clean(''.join(p.get_text() for p in doc))
    lines = [s.strip() for s in re.split(r'\n+', raw) if len(s.strip()) > 40]
    docs, buf, idx = [], '', 1
    for line in lines:
        if len(buf) + len(line) < chunk_size:
            buf += ' ' + line
        else:
            if buf.strip():
                docs.append({'id': f'{paper_name}_{idx:03d}', 'paper': paper_name,
                              'topic': f'Section {idx}', 'text': buf.strip()})
                idx += 1
            buf = line
    if buf.strip():
        docs.append({'id': f'{paper_name}_{idx:03d}', 'paper': paper_name,
                     'topic': f'Section {idx}', 'text': buf.strip()})
    return docs

@st.cache_resource
def load_agent():
    llm      = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    PDF_PAPERS = [
        'papers/1312.5602v1.pdf',    # DQN original
        'papers/1509.06461v3.pdf',   # Double DQN
        'papers/1511.06581v3.pdf',   # Dueling DQN
        'papers/1511.05952v4.pdf',   # Prioritized Replay
        'papers/1710.02298v1.pdf',   # Rainbow
        'papers/1602.01783v2.pdf',   # A3C
    ]

    cli = chromadb.Client()
    try:    cli.delete_collection('research_kb')
    except: pass
    col = cli.create_collection('research_kb')

    all_docs = []
    for path in PDF_PAPERS:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            all_docs.extend(_chunk_pdf(path, name))
        except Exception as e:
            st.warning(f'Could not load {path}: {e}')

    if not all_docs:
        st.error('No PDFs loaded. Please add papers to the papers/ folder.')
        st.stop()

    texts = [d['text'] for d in all_docs]
    col.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d['id'] for d in all_docs],
        metadatas=[{'topic': d['topic'], 'paper': d['paper']} for d in all_docs]
    )

    THRESHOLD, MAX_RETRIES = 0.7, 0

    class S(TypedDict):
        question: str; messages: List[dict]; route: str
        retrieved: str; sources: List[str]; tool_result: str
        answer: str; faithfulness: float; eval_retries: int; paper_filter: str

    SYSTEM_PROMPT = (
        'You are a Research Paper Q&A assistant.\n'
        'You MUST answer the question using the context provided below.\n'
        'The context contains excerpts from the paper — use them to give a complete answer.\n'
        'Only say "I do not have that in the knowledge base" if the context is truly empty or completely unrelated.\n'
        'Never refuse to answer when relevant context is present.'
    )

    def memory(s):
        msgs = s.get('messages', []) + [{'role': 'user', 'content': s['question']}]
        return {'messages': msgs[-6:]}

    def router(s):
        r = llm.invoke(
            f"Route: retrieve / tool / memory_only.\nQuestion: {s['question']}\nOne word:"
        ).content.strip().lower()
        if 'memory' in r: r = 'memory_only'
        elif 'tool' in r: r = 'tool'
        else: r = 'retrieve'
        return {'route': r}

    def retrieval(s):
        q   = embedder.encode([s['question']]).tolist()
        res = col.query(query_embeddings=q, n_results=3)
        chunks  = [f"[{m['topic']}]\n{d}" for d, m in zip(res['documents'][0], res['metadatas'][0])]
        sources = [m['topic'] for m in res['metadatas'][0]]
        return {'retrieved': '\n\n---\n\n'.join(chunks), 'sources': sources}

    def skip(s):
        return {'retrieved': '', 'sources': []}

    def tool(s):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                hits = list(ddgs.text(s['question'] + ' research paper', max_results=3))
            return {'tool_result': '\n\n'.join(f"[{r['title']}]\n{r['body'][:200]}" for r in hits)}
        except Exception as exc:
            return {'tool_result': f'Search unavailable: {exc}'}

    def answer(s):
        parts = []
        if s.get('retrieved'):
            parts.append(f"CONTEXT:\n{s['retrieved'][:1500]}")
        if s.get('tool_result'):
            parts.append(f"WEB SEARCH:\n{s['tool_result'][:300]}")
        ctx    = '\n\n'.join(parts) or 'No context.'
        system = SYSTEM_PROMPT + f'\n\n{ctx}'
        if s.get('eval_retries', 0) > 0:
            system += '\n\nIMPORTANT: Use ONLY explicit context.'
        msgs = [SystemMessage(content=system)]
        for m in s.get('messages', [])[:-1]:
            cls = HumanMessage if m['role'] == 'user' else AIMessage
            msgs.append(cls(content=m['content']))
        msgs.append(HumanMessage(content=s['question']))
        return {'answer': llm.invoke(msgs).content}

    def eval_(s):
        ctx = s.get('retrieved', '')[:500]
        if not ctx:
            return {'faithfulness': 1.0, 'eval_retries': s.get('eval_retries', 0) + 1}
        try:
            score = max(0.0, min(1.0, float(
                llm.invoke(
                    f"Rate faithfulness 0.0-1.0. ONE number only.\nContext: {ctx}\nAnswer: {s.get('answer','')[:300]}"
                ).content.strip().split()[0])))
        except Exception:
            score = 0.5
        return {'faithfulness': score, 'eval_retries': s.get('eval_retries', 0) + 1}

    def save(s):
        return {'messages': s.get('messages', []) + [{'role': 'assistant', 'content': s['answer']}]}

    def route_dec(s):
        r = s.get('route', 'retrieve')
        return 'tool' if r == 'tool' else 'skip' if r == 'memory_only' else 'retrieve'

    def eval_dec(s):
        return ('save'
                if s.get('faithfulness', 1.0) >= THRESHOLD or s.get('eval_retries', 0) >= MAX_RETRIES
                else 'answer')

    g = StateGraph(S)
    for name, fn in [('memory', memory), ('router', router), ('retrieve', retrieval),
                     ('skip', skip), ('tool', tool), ('answer', answer),
                     ('eval', eval_), ('save', save)]:
        g.add_node(name, fn)
    g.set_entry_point('memory')
    g.add_edge('memory', 'router')
    g.add_conditional_edges('router', route_dec,
                             {'retrieve': 'retrieve', 'skip': 'skip', 'tool': 'tool'})
    for n in ('retrieve', 'skip', 'tool'):
        g.add_edge(n, 'answer')
    g.add_edge('answer', 'eval')
    g.add_conditional_edges('eval', eval_dec, {'answer': 'answer', 'save': 'save'})
    g.add_edge('save', END)

    agent_app = g.compile(checkpointer=MemorySaver())
    return agent_app, col.count()


with st.spinner('Loading knowledge base...'):
    agent_app, doc_count = load_agent()

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("Ask questions about 6 foundational Deep Reinforcement Learning papers.")
    st.divider()
    st.write("**Papers covered:**")
    papers = [
        "DQN (1312.5602)",
        "Double DQN (1509.06461)",
        "Prioritized Replay (1511.05952)",
        "Dueling DQN (1511.06581)",
        "A3C (1602.01783)",
        "Rainbow (1710.02298)",
    ]
    for p in papers:
        st.write(f"• {p}")
    st.divider()
    st.write(f"**KB:** {doc_count} chunks loaded")
    st.write(f"**Session:** {st.session_state.get('thread_id', 'not started')}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# Session state
for key, default in [('messages', []), ('thread_id', str(uuid.uuid4())[:8])]:
    if key not in st.session_state:
        st.session_state[key] = default

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# Chat input
if prompt := st.chat_input('Ask about the papers...'):
    with st.chat_message('user'):
        st.write(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            result = agent_app.invoke(
                {'question': prompt,
                 'messages': st.session_state.messages[:-1],
                 'route': '', 'retrieved': '', 'sources': [], 'tool_result': '',
                 'answer': '', 'faithfulness': 0.0, 'eval_retries': 0, 'paper_filter': ''},
                config={'configurable': {'thread_id': st.session_state.thread_id}},
            )
            answer = result.get('answer', 'Sorry, could not generate an answer.')
        st.write(answer)
        faith = result.get('faithfulness', 0.0)
        if faith:
            st.caption(f"Route: {result.get('route','?')} | Faithfulness: {faith:.2f}")

    st.session_state.messages.append({'role': 'assistant', 'content': answer})
