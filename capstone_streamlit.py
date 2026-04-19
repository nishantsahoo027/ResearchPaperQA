import streamlit as st
import uuid, os, re, time
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
st.title("Research Paper Q&A Agent")
st.caption("Ask questions about Deep Reinforcement Learning research.")

# ── PDF helpers ──────────────────────────────────────────────────────────────

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

# ── Agent ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_agent():
    llm      = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    PDF_PAPERS = [
        'papers/1312.5602v1.pdf',
        'papers/1509.06461v3.pdf',
        'papers/1511.06581v3.pdf',
        'papers/1511.05952v4.pdf',
        'papers/1710.02298v1.pdf',
        'papers/1602.01783v2.pdf',
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


# ── Dynamic question generator ───────────────────────────────────────────────

PAPER_CONTEXTS = {
    "All papers": (
        "DQN, Double DQN, Prioritized Experience Replay, Dueling DQN, A3C, and Rainbow "
        "— six foundational deep reinforcement learning papers"
    ),
    "DQN (1312.5602)": (
        "the original DQN paper: playing Atari with deep Q-networks, experience replay, "
        "convolutional architecture, Q-learning, reward clipping, target network"
    ),
    "Double DQN (1509.06461)": (
        "Double DQN: overestimation bias in Q-learning, decoupled action selection and "
        "evaluation, target network reuse, Atari benchmark results"
    ),
    "Prioritized Replay (1511.05952)": (
        "Prioritized Experience Replay: TD-error priority, stochastic prioritization, "
        "importance sampling bias correction, sum-tree data structure, Blind Cliffwalk experiment"
    ),
    "Dueling DQN (1511.06581)": (
        "Dueling Network Architectures: separate value and advantage streams, aggregation "
        "module, saliency maps, corridor environment, policy evaluation"
    ),
    "A3C (1602.01783)": (
        "A3C: asynchronous advantage actor-critic, parallel workers replacing experience "
        "replay, entropy regularization, multi-step returns, MuJoCo continuous control"
    ),
    "Rainbow (1710.02298)": (
        "Rainbow: combining Double DQN, prioritized replay, dueling networks, multi-step "
        "returns, distributional RL, and Noisy Nets into one integrated agent"
    ),
}

# Fallback questions used if LLM call fails
FALLBACK_QUESTIONS = {
    "All papers": [
        "What is experience replay and why does DQN require it?",
        "How does Double DQN fix the overestimation bias in Q-learning?",
        "What makes the dueling architecture better for states with many similar actions?",
        "How does A3C achieve stability without an experience replay buffer?",
        "Which component contributed the most to Rainbow's performance gains?",
    ],
    "DQN (1312.5602)": [
        "What convolutional architecture did the original DQN use?",
        "How did DQN handle the non-stationarity problem during training?",
        "Why did DQN clip rewards to [-1, 1] and what are the trade-offs?",
        "On which Atari games did DQN surpass human-level performance?",
        "What is the role of the target network in stabilising DQN training?",
    ],
    "Double DQN (1509.06461)": [
        "What causes Q-learning to overestimate action values?",
        "How does Double DQN use the target network differently from standard DQN?",
        "What does Theorem 1 prove about estimation errors and overoptimism?",
        "On which games did Double DQN most dramatically outperform DQN?",
        "Did Double DQN require additional networks or parameters beyond DQN?",
    ],
    "Prioritized Replay (1511.05952)": [
        "Why is TD error used as a proxy for learning priority?",
        "What is the difference between rank-based and proportional prioritization?",
        "How does importance sampling correct the bias introduced by prioritized replay?",
        "What does the Blind Cliffwalk experiment demonstrate about uniform replay?",
        "How is the sum-tree data structure used to efficiently sample transitions?",
    ],
    "Dueling DQN (1511.06581)": [
        "What is the advantage function and why is it useful to estimate separately?",
        "Why is the mean advantage subtracted in the aggregation module?",
        "What do the saliency maps reveal about the value versus advantage streams?",
        "In which game settings does the dueling architecture provide the biggest gains?",
        "How does the corridor experiment isolate the benefit of the dueling architecture?",
    ],
    "A3C (1602.01783)": [
        "How do parallel workers in A3C decorrelate gradient updates?",
        "What is the n-step return and how does A3C use it in the forward view?",
        "How does entropy regularisation discourage premature convergence in A3C?",
        "What hardware did A3C use and how did it compare in speed to GPU-based DQN?",
        "How did A3C extend to continuous action spaces in MuJoCo tasks?",
    ],
    "Rainbow (1710.02298)": [
        "How does Rainbow integrate distributional RL with multi-step returns?",
        "Which ablation caused the largest drop in Rainbow's median performance?",
        "How many frames did Rainbow need to match DQN's final performance?",
        "What is distributional Q-learning and how does it differ from expected Q-learning?",
        "How do Noisy Nets replace epsilon-greedy exploration in Rainbow?",
    ],
}

QUESTION_PROMPT = """You are helping students explore deep reinforcement learning research papers.

Generate exactly 5 distinct, specific questions about: {context}

Rules:
- Each question must be a complete sentence ending with ?
- Questions should span different aspects: architecture, training, results, limitations, comparisons
- Make them genuinely interesting and answerable from the paper(s)
- No numbering, no bullet points — just 5 lines, one question per line
- Vary question types: some factual, some conceptual, some comparative

Output only the 5 questions, nothing else."""


@st.cache_data(ttl=300, show_spinner=False)
def generate_sidebar_questions(paper_filter: str, seed: int) -> list:
    """Call the LLM to generate 5 context-aware questions. Cached for 5 min."""
    context = PAPER_CONTEXTS.get(paper_filter, PAPER_CONTEXTS["All papers"])
    prompt  = QUESTION_PROMPT.format(context=context)
    try:
        llm_q = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.85, max_tokens=400)
        raw   = llm_q.invoke([HumanMessage(content=prompt)]).content.strip()
        qs    = [q.strip() for q in raw.split('\n') if q.strip().endswith('?')]
        if len(qs) >= 3:
            return qs[:5]
    except Exception:
        pass
    return FALLBACK_QUESTIONS.get(paper_filter, FALLBACK_QUESTIONS["All papers"])


# ── Load agent ───────────────────────────────────────────────────────────────

with st.spinner('Loading knowledge base...'):
    agent_app, doc_count = load_agent()

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in [('messages', []), ('thread_id', str(uuid.uuid4())[:8]),
                     ('q_seed', 1), ('pending_question', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    # Session + new conversation
    st.markdown(
        f"<div style='font-size:12px;color:gray;margin-bottom:4px;'>"
        f"🟢 Session: <code>{st.session_state.thread_id}</code></div>",
        unsafe_allow_html=True,
    )
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.thread_id       = str(uuid.uuid4())[:8]
        st.session_state.q_seed          = int(time.time())
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    # About
    st.markdown("**About**")
    st.caption(
        "An agentic Q&A system over 6 foundational Deep RL papers. "
        "Uses RAG, memory, self-reflection eval, and live web search."
    )

    st.divider()

    # Papers covered
    st.markdown("**Papers covered**")
    papers_list = [
        "DQN",
        "Double DQN",
        "Prioritized Replay",
        "Dueling DQN",
        "A3C",
        "Rainbow",
    ]
    for p in papers_list:
        st.markdown(f"&nbsp;&nbsp;· {p}", unsafe_allow_html=True)

    st.divider()

    # ── Dynamic questions ─────────────────────────────────────────────────────
    st.markdown("**Try asking**")

    paper_filter = st.selectbox(
        "Filter by paper",
        options=["All papers"] + papers_list,
        label_visibility="collapsed",
        key="sidebar_paper_filter",
    )

    # Refresh button bumps seed → cache miss → new questions
    if st.button("🔄 Refresh suggestions", use_container_width=True):
        st.session_state.q_seed = int(time.time())
        st.rerun()

    with st.spinner("Generating questions..."):
        questions = generate_sidebar_questions(paper_filter, st.session_state.q_seed)

    for q in questions:
        if st.button(q, use_container_width=True, key=f"sq_{hash(q)}"):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()
    st.caption(f"KB: {doc_count} chunks · {len(papers_list)} PDFs")

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# ── Helper: run agent and display answer ──────────────────────────────────────

def run_agent(question: str):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.write(question)
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            result = agent_app.invoke(
                {
                    'question':    question,
                    'messages':    st.session_state.messages[:-1],
                    'route':       '',
                    'retrieved':   '',
                    'sources':     [],
                    'tool_result': '',
                    'answer':      '',
                    'faithfulness': 0.0,
                    'eval_retries': 0,
                    'paper_filter': '',
                },
                config={'configurable': {'thread_id': st.session_state.thread_id}},
            )
            ans = result.get('answer', 'Sorry, could not generate an answer.')
        st.write(ans)
        faith = result.get('faithfulness', 0.0)
        if faith:
            st.caption(f"Route: {result.get('route','?')} | Faithfulness: {faith:.2f}")
    st.session_state.messages.append({'role': 'assistant', 'content': ans})

# ── Handle sidebar question click ─────────────────────────────────────────────

if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    run_agent(q)

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input('Ask about the papers...'):
    run_agent(prompt)# import streamlit as st
