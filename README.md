# 📄 Research Paper Q&A Agent

**Student:** Nishant Sahoo
**Course:** Agentic AI

---

## 🧠 Overview

An agentic Q&A system that answers natural language questions about six foundational Deep Reinforcement Learning research papers. Built using **LangGraph**, **ChromaDB**, and **Groq-hosted LLaMA 3.3 70B**, with a **Streamlit** chat UI.

The agent routes each query through an 8-node StateGraph pipeline — performing semantic retrieval, live web search, conversation memory management, and self-reflective faithfulness evaluation — before delivering a grounded answer.

---

## 📚 Papers Covered

| ArXiv ID | Title |
|----------|-------|
| 1312.5602 | DQN — Playing Atari with Deep Reinforcement Learning |
| 1509.06461 | Double DQN — Deep RL with Double Q-learning |
| 1511.05952 | Prioritized Experience Replay |
| 1511.06581 | Dueling Network Architectures for Deep RL |
| 1602.01783 | A3C — Asynchronous Methods for Deep RL |
| 1710.02298 | Rainbow — Combining Improvements in Deep RL |

**Knowledge Base:** 336 chunks across 6 PDFs

---

## 🏗️ Architecture

```
User Question
     ↓
[memory_node]        → sliding window history (last 6 messages)
     ↓
[router_node]        → LLM decides: retrieve / tool / memory_only
     ↓
[retrieval_node]     → top-3 chunks from ChromaDB via semantic search
[tool_node]          → DuckDuckGo live web search (max 3 snippets)
[skip_node]          → bypasses retrieval for memory-only queries
     ↓
[answer_node]        → grounded answer from context + history via LLaMA 3.3 70B
     ↓
[eval_node]          → faithfulness score (0.0–1.0), retry if < 0.7
     ↓
[save_node]          → append assistant response to memory → END
```

### 6 Mandatory Agentic Capabilities

- ✅ **LangGraph StateGraph** — 8-node directed graph pipeline
- ✅ **ChromaDB RAG** — 336 chunks from 6 PDFs, SentenceTransformer embeddings
- ✅ **Conversation Memory** — MemorySaver + thread_id, sliding window of 6 messages
- ✅ **Self-Reflection Eval Node** — faithfulness gating with automatic retry
- ✅ **Tool Use** — DuckDuckGo live web search for out-of-corpus queries
- ✅ **Deployment** — Streamlit chat UI (pyngrok tunnel for Colab)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| LangGraph | Agentic graph orchestration |
| ChromaDB | In-memory vector store (336 chunks) |
| SentenceTransformer `all-MiniLM-L6-v2` | Document & query embeddings |
| LangChain Groq (LLaMA 3.3 70B) | LLM for routing, answering, eval |
| PyMuPDF (`fitz`) | PDF extraction & chunking |
| DuckDuckGo Search | Live web search tool |
| Streamlit | Chat UI |
| pyngrok | Colab tunnel for deployment |

---

## 📊 Test Results

**RAGAS Baseline (manual LLM faithfulness):** Average **0.640**

| # | Question | Route | Faithfulness | Status |
|---|----------|-------|-------------|--------|
| 1 | What is experience replay? | retrieve | 0.70 | ✅ PASS |
| 2 | On which Atari games did DQN surpass human performance? | retrieve | 0.70 | ✅ PASS |
| 3 | What are the main limitations of DQN? | retrieve | 0.80 | ✅ PASS |
| 4 | My name is Raj. What does DQN stand for? | memory_only | 1.00 | ✅ PASS |
| 5 | What is the architecture of GPT-4? *(red-team)* | retrieve | 0.00 | ⚠️ CHECK |
| 6 | You said DQN uses LSTM layers — which layer? *(red-team)* | memory_only | 1.00 | ✅ PASS |

> **Note on Q5:** The agent correctly returned a low faithfulness score for an out-of-corpus query (GPT-4), demonstrating it does not hallucinate beyond its knowledge base.

---

## 🚀 Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/nishantsahoo027/ResearchPaperQA.git
cd ResearchPaperQA
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Add the PDFs

Place all six research PDF files in the `papers/` folder:

```
papers/
  1312.5602v1.pdf
  1509.06461v3.pdf
  1511.05952v4.pdf
  1511.06581v3.pdf
  1602.01783v2.pdf
  1710.02298v1.pdf
```

### 5. Run the Streamlit app

```bash
streamlit run capstone_streamlit.py
```

---

## 📁 Project Structure

```
ResearchPaperQA/
├── NishantSahoo_Capstone.ipynb      # Full capstone notebook with all outputs
├── capstone_streamlit.py             # Streamlit chat UI — main entry point
├── agent.py                          # Shared agent module (build_agent function)
├── requirements.txt                  # Python dependencies
├── .env.example                      # API key template
├── .gitignore
├── papers/                           # Add your 6 research PDFs here
└── README.md
```

---

## ✨ Unique Points

- **Self-Reflective Faithfulness Gating** — every answer is scored before delivery; low-scoring responses are automatically retried with a specificity hint.
- **Trimodal Routing** — queries are classified into retrieve / tool / memory_only, preventing unnecessary retrieval overhead for simple follow-ups.
- **Red-Team Robustness** — the agent explicitly refuses to fabricate answers for out-of-corpus queries (e.g., GPT-4 architecture), returning a 0.00 faithfulness score instead.
- **Runs on CPU** — no GPU required; full orchestration is handled via Groq API inference, making it accessible in Colab and resource-constrained environments.

---

## 🔭 Future Improvements

- **Dynamic PDF ingestion** — allow users to upload new papers at runtime via the Streamlit UI.
- **Full RAGAS evaluation** — integrate structured metrics (answer relevancy, context precision, context recall) for systematic benchmarking.
- **Persistent vector store** — save ChromaDB to disk to eliminate cold-start re-indexing overhead.
- **Cross-paper synthesis** — implement paper-aware retrieval to explicitly compare and contrast multiple papers in a single answer.
- **Citation-aware responses** — generate answers with inline citations referencing the specific paper and section each claim is drawn from.

---

