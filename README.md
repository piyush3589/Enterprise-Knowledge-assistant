# 🏭 Enterprise Knowledge Assistant

> Ask questions about industrial equipment in plain English. Get precise, cited answers from technical documentation — instantly.

A **RAG pipeline with LLM routing** built for industrial IoT environments. Three specialized AI agents work in sequence: a router classifies your query, a retriever finds the most relevant documents using hybrid search, and an answer agent synthesises a structured response with sources and confidence scores.

---

## 🎥 Demo

**Query:** *"Pump vibration is at 7 mm/s — what should I do?"*

```json
{
  "routing_decision": "VECTOR_SEARCH",
  "answer": "Pump P-03 vibration at 7 mm/s exceeds the alarm threshold of 6.0 mm/s.
             Immediate actions: (1) Reduce pump speed by 10% via VFD-01.
             (2) If vibration exceeds 7.5 mm/s — start standby pump P-03B immediately
             and shut down P-03 gradually to avoid water hammer.
             Spectrum analysis: 1x RPM = imbalance, 2x RPM = misalignment,
             >1000 Hz = bearing defect. See procedure MP-PUMP-007 for bearing replacement.",
  "sources": ["doc_troubleshoot_002", "doc_pump_maintenance_001"],
  "confidence": 0.81,
  "follow_up_suggestions": [
    "What is the bearing replacement procedure for pump P-03?",
    "How do I perform laser alignment on the pump-motor coupling?"
  ],
  "duration_ms": 8420
}
```

**Query:** *"What is the company leave policy?"*
```json
{
  "routing_decision": "OUT_OF_SCOPE",
  "answer": "This question is outside the scope of the industrial IoT knowledge base."
}
```
> The router agent prevents hallucination by rejecting out-of-scope queries instead of making something up.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Modular RAG pipeline                     │
│                                                             │
│  ┌──────────────┐                                           │
│  │ Router Agent │ → VECTOR_SEARCH / CLARIFICATION /         │
│  │  (Groq LLM)  │   OUT_OF_SCOPE                           │
│  └──────┬───────┘                                           │
│         │ (if VECTOR_SEARCH)                                │
│         ▼                                                   │
│  ┌──────────────────────────────┐                           │
│  │      Retriever Agent         │                           │
│  │  BM25 keyword search         │ ──→ top 5 candidates      │
│  │  + ChromaDB vector search    │ ──→ top 5 candidates      │
│  │  + Cohere reranker           │ ──→ top 3 final           │
│  └──────┬───────────────────────┘                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ Answer Agent │ → structured answer + sources +           │
│  │  (Groq LLM)  │   confidence + follow-up questions        │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
FastAPI  →  JSON response
```

---

## ✨ Why hybrid search beats naive RAG

Most RAG demos use pure vector search. This project combines three retrieval methods:

| Method | What it catches |
|---|---|
| **BM25 keyword search** | Exact matches — equipment IDs (`CV-201`), part numbers, procedure codes (`MP-PUMP-007`) |
| **Vector semantic search** | Meaning — *"pump is making noise"* → vibration troubleshooting docs |
| **Cohere reranker** | Re-scores the combined pool by true relevance to the query |

This combination dramatically reduces missed retrievals that pure vector search fails on.

---

## 📚 Knowledge Base

8 industrial IoT documents across 5 categories:

| Document | Category |
|---|---|
| Boiler B-01 — equipment specification | Equipment spec |
| Boiler heat exchanger cleaning procedure | Maintenance |
| Pump P-03 — equipment specification | Equipment spec |
| Pump bearing replacement — MP-PUMP-007 | Maintenance |
| Pipeline P-104 — specification | Equipment spec |
| Lockout/Tagout (LOTO) safety procedure | Safety |
| Pressure vessel compliance — IS 2825 | Compliance |
| Low coolant flow troubleshooting guide | Troubleshooting |
| High pump vibration troubleshooting guide | Troubleshooting |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3-70b-versatile |
| Keyword search | BM25 (rank-bm25) |
| Vector store | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| Reranking | Cohere rerank-english-v3.0 |
| Agent orchestration | Custom multi-agent pipeline |
| Structured outputs | Pydantic v2 |
| REST API | FastAPI + Uvicorn |
| Containerisation | Docker |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- Groq API key — free at [console.groq.com](https://console.groq.com)
- Cohere API key — free at [cohere.com](https://cohere.com) 

### Setup

```bash
# 1. Clone and create virtual environment
git clone https://github.com/yourusername/enterprise-knowledge-assistant
cd enterprise-knowledge-assistant

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Open .env and add your GROQ_API_KEY and COHERE_API_KEY

# 4. Seed the knowledge base (run once)
python data/seed_knowledge.py

# 5. Start the API
uvicorn api.main:app --reload

# 6. Open Swagger UI
# http://localhost:8000/docs
```

---

## 🧪 Try these queries

```bash
# Equipment specification
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the maximum operating pressure for boiler B-01?"}'

# Troubleshooting
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Pump vibration is at 7 mm/s — what should I do immediately?"}'

# Maintenance procedure
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How often should the heat exchanger be cleaned?"}'

# Safety
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the LOTO procedure before working on electrical equipment?"}'

# Compliance
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What inspections are required for pressure vessels under IS 2825?"}'

# Router rejects out-of-scope — no hallucination
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the company holiday policy?"}'
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/documents` | List all documents in knowledge base |
| `POST` | `/query` | Run full multi-agent pipeline |
| `POST` | `/query/batch` | Run multiple queries at once |

### Request schema
```json
{
  "query": "Your question here",
  "top_k": 3
}
```

### Response schema
```json
{
  "query": "string",
  "routing_decision": "VECTOR_SEARCH | CLARIFICATION | OUT_OF_SCOPE",
  "answer": "string",
  "sources": ["doc_id_1", "doc_id_2"],
  "confidence": 0.0,
  "low_confidence_warning": "string | null",
  "follow_up_suggestions": ["question 1", "question 2"],
  "duration_ms": 0
}
```

---

## 🐳 Docker

```bash
docker build -t enterprise-knowledge-assistant .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e COHERE_API_KEY=your_key \
  enterprise-knowledge-assistant
```

---

## 📁 Project Structure

```
enterprise-knowledge-assistant/
├── agents/
│   ├── pipeline.py      # Router → Retriever → Answer agent pipeline
│   ├── retriever.py     # Hybrid BM25 + vector + Cohere reranker
│   └── models.py        # Pydantic models for all agent I/O
├── api/
│   └── main.py          # FastAPI endpoints
├── data/
│   └── seed_knowledge.py # Seeds 8 industrial documents into ChromaDB
├── tests/
│   └── test_all.py      # Unit + integration tests
├── Dockerfile
├── .env.example
└── README.md
```

---

## 🔑 Key concepts demonstrated

- **Agentic routing** — query classification prevents hallucination on irrelevant questions
- **Hybrid search** — BM25 + vector catches both exact keywords and semantic meaning
- **Reranking** — Cohere re-scores candidates by true relevance, not embedding similarity
- **Multi-agent architecture** — each agent has a single, well-defined responsibility
- **Pydantic v2 structured outputs** — every agent input/output is typed and validated
- **Production API design** — proper error handling, CORS, batch endpoint, health check
