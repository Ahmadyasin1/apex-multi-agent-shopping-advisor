# APEX — Intelligent Shopping Advisor
### Elite Multi-Agent AI System | GEN-AI Assignment 2 | UCP

> **Author:** Ahmad Yasin | **University:** University of Central Punjab (UCP) | **Semester:** 8th  
> **Course:** Generative AI | **Assignment:** 2 — Intelligent Shopping Advisor

---

## Project Overview

APEX is a **production-grade Intelligent Shopping Advisor** built with a hierarchical multi-agent architecture orchestrated via **LangGraph**. It accepts natural language product queries and returns deeply-reasoned, personalized recommendations across 7 product categories and 51 real-world products.

**Example Queries:**
- *"Best gaming laptop under $2000"*
- *"Noise cancelling headphones for frequent flyers"*
- *"Smartphone with the best camera under PKR 100,000"*
- *"Laptop for AI and machine learning research"*

---

## Assignment Requirements Fulfillment

| Requirement | Status | Implementation |
|---|---|---|
| Natural language query input | ✅ | Hero search bar + voice search |
| Extract budget, category, preferences | ✅ | Preference Agent (NLP parsing) |
| ≥ 4 LangGraph agents | ✅ | **8 agents**: Supervisor, Preference, Retrieval, Enrichment, Comparison, Critique, Recommendation, Learning |
| Preference Agent | ✅ | `agents/preference.py` — priority vector + budget extraction |
| Product Retrieval Agent | ✅ | `agents/retrieval.py` — FAISS semantic search + RapidAPI boost |
| Comparison Agent | ✅ | `agents/comparison.py` — 5-axis Pareto-optimal scoring |
| Recommendation Agent | ✅ | `agents/recommendation.py` — Gemini/MockLLM premium report |
| Structured dataset | ✅ | `data_layer/dataset.json` — 51 products, 7 categories |
| Vector database | ✅ | FAISS (`data_layer/products.index`) + SentenceTransformers |
| API integration | ✅ | `data_layer/rapidapi_client.py` — RapidAPI real-time product boost |
| State management | ✅ | `state.py` TypedDict across all 8 agents |
| Conditional transitions | ✅ | Critique → retry loop (max 2 revisions) or Recommendation |
| Error handling & fallback | ✅ | MockLLM fallback, graceful exception handling everywhere |
| ≥ 50 test queries | ✅ | **60 queries** in `evaluate.py` (10 edge cases) |
| Edge cases | ✅ | Missing budget, conflicting constraints, garbled input, no-budget |
| Accuracy/relevance/time metrics | ✅ | `evaluate.py` — category accuracy, budget compliance, response time |
| System architecture diagram | ✅ | See below |
| Data privacy / bias considerations | ✅ | See Ethical Considerations section |

---

## Architecture — 8-Agent LangGraph Pipeline

```
User Query (Natural Language)
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│                    LANGRAPH STATE GRAPH                    │
│                                                           │
│  ┌─────────────┐     ┌──────────────┐     ┌────────────┐ │
│  │  Supervisor │────▶│  Preference  │────▶│ Retrieval  │ │
│  │   Agent     │     │    Agent     │     │   Agent    │ │
│  │  (Router)   │     │ (NLP/Budget) │     │ (FAISS+API)│ │
│  └─────────────┘     └──────────────┘     └─────┬──────┘ │
│                                                  │        │
│  ┌──────────────┐    ┌──────────────┐     ┌─────▼──────┐ │
│  │  Comparison  │◀───│  Enrichment  │◀────│ Enrichment │ │
│  │    Agent     │    │    Agent     │     │   Agent    │ │
│  │  (Pareto 5D) │    │ (Normalize)  │     │(Value Tier)│ │
│  └──────┬───────┘    └──────────────┘     └────────────┘ │
│         │                                                  │
│  ┌──────▼───────┐                                         │
│  │   Critique   │──── PASS ──────────────────────────┐   │
│  │    Agent     │                                     │   │
│  │  (QA Audit)  │──── FAIL ──▶ Preference (retry ≤2) │   │
│  └──────────────┘                                     │   │
│                                                        │   │
│  ┌────────────────┐    ┌──────────────┐               │   │
│  │    Learning    │◀───│Recommendation│◀──────────────┘   │
│  │     Agent      │    │    Agent     │                   │
│  │ (Persist/Adapt)│    │(Gemini/LLM)  │                   │
│  └────────┬───────┘    └──────────────┘                   │
│           │                                               │
│           ▼  END                                          │
└───────────────────────────────────────────────────────────┘
    │
    ▼
WebSocket JSON → Frontend (Real-time streaming)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph 0.2+ (StateGraph) |
| **LLM** | Google Gemini 2.0 Flash → Ollama (local) → MockLLM |
| **Embeddings** | SentenceTransformers `all-MiniLM-L6-v2` |
| **Vector DB** | FAISS (L2 index, 51 products) |
| **External APIs** | RapidAPI (real-time product boost) |
| **Backend** | FastAPI + uvicorn (async, WebSocket streaming) |
| **Frontend** | Vanilla JS + Three.js + Chart.js (luxury UI) |
| **Data** | JSON dataset (51 products, 7 categories) |

---

## Quick Start (Local)

```bash
# 1. Clone / download project
cd "Assignement 2"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY

# 4. Run the server (port 9000)
python run.py

# 5. Open browser
# http://127.0.0.1:9000
```

---

## Running the Evaluation Suite

```bash
# Quick smoke test (10 queries)
python evaluate.py --quick

# Full evaluation — all 60 queries
python evaluate.py

# Category-specific
python evaluate.py --group laptop
python evaluate.py --group edge

# Save JSON + CSV report
python evaluate.py --report

# Specific query IDs
python evaluate.py --id 1,11,21,31,41
```

**Expected metrics:**
- Pipeline success rate: ≥90%
- Category detection accuracy: ≥85%
- Budget compliance: ≥95%
- Avg response time: 20–60s (Gemini) / <5s (MockLLM fallback)

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the luxury frontend |
| `/ws/query` | WebSocket | Real-time 8-agent pipeline |
| `/api/stats` | GET | Product count, sessions, learned preferences |
| `/api/personas` | GET | 6 pre-configured user personas |
| `/api/categories` | GET | Category list with product counts |
| `/api/products/explore` | GET | All 51 products with pre-computed scores |
| `/api/budget-optimizer` | GET | Top 6 products for a given budget |
| `/api/price-trend` | GET | 13-month simulated price history |
| `/api/market-pulse` | GET | Category-level market intelligence |
| `/api/explain-score` | GET | Why a product got a specific dimension score |
| `/media/{filename}` | GET | Serves MP4 video files |
| `/health` | GET | Health check |

---

## Dataset

**Location:** `data_layer/dataset.json`

| Category | Count | Price Range |
|---|---|---|
| Laptop | 12 | $299 – $3,499 |
| Smartphone | 10 | $299 – $1,399 |
| Headphones | 8 | $29 – $599 |
| Desktop | 6 | $699 – $7,999 |
| Tablet | 5 | $129 – $1,899 |
| Smartwatch | 5 | $199 – $799 |
| Camera | 5 | $399 – $3,499 |
| **Total** | **51** | — |

Each product contains: id, name, brand, category, price, rating (1–5), specs, features, description, ideal_for.

---

## Deployment

### ✅ Recommended: Railway (Best for this project)

Railway supports Docker, WebSockets, and large memory instances.

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Set environment variables in Railway dashboard:
- `GEMINI_API_KEY` → your Gemini API key
- `RAPIDAPI_KEY` → your RapidAPI key

### ✅ Alternative: Render

```bash
# Connect your GitHub repo to Render
# Render will auto-detect render.yaml
# Set env vars in Render dashboard
```

### ❌ Not Recommended: PythonAnywhere

| Reason | Detail |
|---|---|
| No WebSocket | Free/Hacker tier does not support WebSocket connections |
| Memory limit | SentenceTransformers + FAISS requires ~1.5GB RAM (PythonAnywhere free = 512MB) |
| FAISS install | C extension compilation often fails on shared hosting |

### ❌ Not Recommended: Vercel

| Reason | Detail |
|---|---|
| Serverless | No persistent memory (FAISS index must reload every request) |
| No WebSocket | Vercel functions are stateless HTTP only |
| Timeout | 10s max function timeout; pipeline takes 20–60s |
| Python limits | No C extension support in serverless functions |

### Docker (Self-hosted / Any VPS)

```bash
docker build -t apex-advisor .
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e RAPIDAPI_KEY=your_key \
  apex-advisor
```

---

## Project Structure

```
Assignement 2/
├── backend/
│   └── main.py            # FastAPI app — all HTTP + WebSocket routes (v3.0.0)
├── agents/
│   ├── supervisor.py      # Complexity detection, LLM routing
│   ├── preference.py      # NLP budget/category/priority extraction
│   ├── retrieval.py       # FAISS semantic search + RapidAPI boost
│   ├── enrichment.py      # Spec normalization, value tier inference
│   ├── comparison.py      # 5-axis Pareto-optimal scoring
│   ├── critique.py        # Quality audit, budget validation
│   ├── recommendation.py  # LLM-powered advisory report
│   └── learning.py        # Adaptive preference persistence
├── data_layer/
│   ├── dataset.json       # 51 products — source of truth
│   ├── vector_db.py       # FAISS + SentenceTransformer wrapper
│   ├── rapidapi_client.py # External product API integration
│   ├── products.index     # FAISS index (auto-generated)
│   └── metadata.json      # FAISS metadata (auto-generated)
├── frontend/
│   └── index.html         # Single-page luxury UI (~130KB)
├── state.py               # AgentState TypedDict (shared state)
├── graph.py               # LangGraph StateGraph definition
├── config.py              # Central configuration (env-backed)
├── llm_factory.py         # LLM routing: Gemini → Ollama → MockLLM
├── evaluate.py            # Formal evaluation: 60 queries, 5 metrics
├── run.py                 # Local development launcher
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build for deployment
├── Procfile               # Railway / Render start command
├── render.yaml            # Render auto-deploy configuration
├── railway.toml           # Railway deployment configuration
├── .env.example           # Environment variable template
└── .gitignore             # Git exclusion rules
```

---

## Ethical Considerations

### Data Privacy
- No user PII is collected — only aggregate query patterns stored in `user_memory.json`
- User memory is local-only and not transmitted to any external service
- All product data is static; no personal purchase history is tracked

### Bias in Recommendations
- Premium brand weighting (`PREMIUM_BRANDS` in `comparison.py`) may disadvantage lesser-known brands
- Price-performance scoring is computed deterministically — no LLM hallucination in scores
- Pareto optimality analysis surfaces genuinely non-dominated choices, reducing bias toward any single dimension

### Limitations
- Dataset is static (51 products) — real prices change daily
- LLM (Gemini) may occasionally hallucinate product names in recommendation text; factual scores are computed deterministically to mitigate this
- Budget filter has 10% tolerance — edge cases near budget limit may show slightly over-budget products
- RapidAPI integration is best-effort; network failures fall back to local dataset

### Accuracy of Product Data
- All product specs are manually curated and cross-referenced with manufacturer specifications
- Ratings reflect aggregated real-world reviews (not LLM-generated)
- Price data represents typical retail pricing and may not reflect current market

---

## Author

**Ahmad Yasin**  
8th Semester — Generative AI  
University of Central Punjab (UCP)  
GEN-AI Assignment 2
