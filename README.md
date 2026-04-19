# 📄 Research Paper Q&A Agent

**Agentic AI Capstone Project**
**Student:** Nishant Sahoo
**Course:** Agentic AI Hands-On | Dr. Kanthi Kiran Sirra | 2026

---

## Overview

An agentic Q&A system that answers natural language questions about six foundational Deep Reinforcement Learning research papers. Built using LangGraph, ChromaDB, and Groq-hosted LLaMA 3.3 70B, with a Streamlit chat UI.

---

## Papers Covered

| Paper | Title |
|-------|-------|
| 1312.5602 | DQN — Playing Atari with Deep Reinforcement Learning |
| 1509.06461 | Double DQN |
| 1511.05952 | Prioritized Experience Replay |
| 1511.06581 | Dueling Network Architectures |
| 1602.01783 | A3C — Asynchronous Methods for Deep RL |
| 1710.02298 | Rainbow — Combining Improvements in DRL |

---

## Architecture

```
User Question
     ↓
[memory_node]    → sliding window history (last 6 messages)
     ↓
[router_node]    → LLM decides: retrieve / tool / memory_only
     ↓
[retrieval_node / tool_node / skip_node]
     ↓
[answer_node]    → grounded answer from context
     ↓
[eval_node]      → faithfulness score (0.0–1.0), retry if < 0.7
     ↓
[save_node]      → append to memory → END
```

**6 Mandatory Capabilities:**
- ✅ LangGraph StateGraph (8 nodes)
- ✅ ChromaDB RAG (336 chunks from 6 PDFs)
- ✅ Conversation memory (MemorySaver + thread_id)
- ✅ Self-reflection eval node (faithfulness gating)
- ✅ Tool use — DuckDuckGo live web search
- ✅ Deployment — Streamlit UI

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangGraph | Agentic graph orchestration |
| ChromaDB | Vector store (336 chunks) |
| SentenceTransformer `all-MiniLM-L6-v2` | Document & query embeddings |
| LangChain Groq (LLaMA 3.3 70B) | LLM for routing, answering, eval |
| PyMuPDF (`fitz`) | PDF extraction & chunking |
| DuckDuckGo Search | Live web search tool |
| Streamlit | Chat UI |
| pyngrok | Colab tunnel |

---

## Results

**Knowledge Base:** 336 chunks across 6 PDFs

**Test Results (6/6 run):**

| # | Question | Route | Faithfulness | Status |
|---|----------|-------|-------------|--------|
| 1 | What is experience replay? | retrieve | 0.70 | ✅ PASS |
| 2 | On which Atari games did DQN surpass human performance? | retrieve | 0.70 | ✅ PASS |
| 3 | What are the main limitations of DQN? | retrieve | 0.80 | ✅ PASS |
| 4 | My name is Raj. What does DQN stand for? | memory_only | 1.00 | ✅ PASS |
| 5 | What is the architecture of GPT-4? *(red-team)* | retrieve | 0.00 | ⚠️ CHECK |
| 6 | You said DQN uses LSTM layers — which layer? *(red-team)* | memory_only | 1.00 | ✅ PASS |

**RAGAS Baseline (manual LLM faithfulness):** Average **0.640**

---

## Setup & Run

### 1. Clone the repo

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

Place your research PDF files in the `papers/` folder:
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

## Project Files

```
NishantSahoo_ResearchPaperQA/
├── NishantSahoo_Capstone.ipynb   # Full capstone notebook with outputs
├── capstone_streamlit.py          # Streamlit chat UI
├── agent.py                       # Shared agent module
├── requirements.txt               # Python dependencies
├── .env.example                   # API key template
├── .gitignore
├── papers/                        # Add your PDFs here
└── README.md
```

---

## Submission

**Deadline:** April 21, 2026 | 11:59 PM
**Submitted via:** Google Form
