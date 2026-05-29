# ══════════════════════════════════════════════════════════════════════════════
# MASTER PROMPT — INTELLIGENT SHOPPING ADVISOR (GEN-AI ASSIGNMENT 2)
# Author: Ahmad Yasin | Top-Tier Implementation Guide
# Stack: Python · LangGraph · Gemini 2.0 · FastAPI · PostgreSQL · FAISS
# ══════════════════════════════════════════════════════════════════════════════

## ROLE

You are a senior AI engineer and full-stack developer. Build a complete,
production-grade Intelligent Shopping Advisor multi-agent system that will
score 100/100 on a university assignment. Every line of code must be clean,
professional, and well-commented. The system must be genuinely impressive to
a professor evaluating it.

---

## PROJECT OVERVIEW

Build an AI-powered **Intelligent Shopping Advisor** named **NexusShop** that:

1. Accepts natural language shopping queries in English (and optionally Urdu)
2. Orchestrates 4+ specialized AI agents using **LangGraph**
3. Uses **Google Gemini 2.0 Flash** as the LLM backbone
4. Retrieves and ranks products from a rich local dataset + FAISS vector DB
5. Returns top 3–5 ranked product recommendations with full justifications
6. Streams agent status in real-time via **WebSocket**
7. Serves a **luxury, premium frontend** (dark gold aesthetic, Three.js 3D)

---

## COMPLETE PROJECT STRUCTURE

```
nexusshop/
├── backend/
│   ├── main.py                    # FastAPI app + WebSocket endpoint
│   ├── config.py                  # Settings, API keys, DB connection strings
│   ├── requirements.txt
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py          # Abstract base class for all agents
│   │   ├── preference_agent.py    # Agent 1: Preference extraction
│   │   ├── retrieval_agent.py     # Agent 2: Product retrieval (DB + FAISS)
│   │   ├── comparison_agent.py    # Agent 3: Multi-dimensional scoring
│   │   ├── recommendation_agent.py# Agent 4: Final ranked recommendations
│   │   └── price_tracker_agent.py # Agent 5 (Bonus): Price history analysis
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── workflow.py            # LangGraph StateGraph definition
│   │   ├── state.py               # Shared ShoppingState TypedDict
│   │   └── router.py              # Conditional edge routing logic
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py              # SQLAlchemy ORM models
│   │   ├── connection.py          # DB engine, session factory
│   │   ├── crud.py                # CRUD operations
│   │   └── seed_data.py           # Script to populate database
│   │
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── faiss_store.py         # FAISS index creation + querying
│   │   └── embeddings.py          # Gemini embedding generation
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── product.py             # Pydantic schemas for products
│   │   ├── query.py               # Pydantic schemas for user queries
│   │   └── response.py            # Pydantic schemas for API responses
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── gemini_client.py       # Gemini API wrapper with retry logic
│   │   └── logger.py              # Structured logging
│   │
│   └── tests/
│       ├── test_agents.py
│       ├── test_workflow.py
│       ├── test_queries.py        # 50+ test query cases
│       └── evaluation.py          # Accuracy/relevance evaluator
│
├── frontend/
│   ├── index.html                 # Main luxury frontend (Three.js + Chart.js)
│   ├── assets/
│   │   ├── style.css
│   │   └── app.js
│   └── README.md
│
├── data/
│   ├── products.json              # 200+ products dataset
│   ├── products.csv               # Same data as CSV
│   └── test_queries.json          # 50 test queries with expected results
│
├── docs/
│   ├── architecture_diagram.png   # LangGraph flow diagram
│   ├── project_report.md          # 8-page project report
│   └── api_docs.md                # API documentation
│
├── docker-compose.yml             # PostgreSQL + backend + frontend
├── .env.example                   # Environment variable template
└── README.md                      # Setup and run instructions
```

---

## STEP 1: ENVIRONMENT SETUP

### requirements.txt
```
fastapi==0.111.0
uvicorn[standard]==0.30.0
websockets==12.0
langchain==0.2.11
langchain-google-genai==1.0.8
langgraph==0.1.19
google-generativeai==0.7.2
sqlalchemy==2.0.31
psycopg2-binary==2.9.9
pydantic==2.8.0
pydantic-settings==2.3.4
faiss-cpu==1.8.0
numpy==1.26.4
pandas==2.2.2
python-dotenv==1.0.1
httpx==0.27.0
aiofiles==23.2.1
pytest==8.2.2
pytest-asyncio==0.23.7
rich==13.7.1
```

### .env
```
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/nexusshop
FAISS_INDEX_PATH=./vector_store/faiss_index
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

---

## STEP 2: LANGGRAPH STATE (graph/state.py)

```python
"""
Shared state object that flows between all agents in the LangGraph pipeline.
Each agent reads from and writes to this state immutably.
"""
from typing import TypedDict, Optional, List, Dict, Any


class UserPreferences(TypedDict):
    """Structured output from the Preference Agent."""
    budget_min: Optional[float]        # Minimum budget in PKR
    budget_max: Optional[float]        # Maximum budget in PKR
    category: str                       # Product category (e.g., "smartphone")
    subcategory: Optional[str]          # e.g., "flagship", "mid-range"
    priority_features: List[str]        # e.g., ["camera", "battery", "performance"]
    use_case: Optional[str]             # e.g., "gaming", "travel", "photography"
    brand_preferences: List[str]        # Preferred brands (empty = no preference)
    brand_exclusions: List[str]         # Brands to exclude
    form_factor: Optional[str]          # e.g., "compact", "large screen"
    other_constraints: List[str]        # Any other parsed constraints


class ProductScore(TypedDict):
    """Scored product from the Comparison Agent."""
    product_id: str
    name: str
    brand: str
    price: float
    category: str
    specs: Dict[str, Any]
    composite_score: float              # Weighted final score (0-100)
    dimension_scores: Dict[str, float]  # Per-dimension breakdown
    within_budget: bool
    pros: List[str]
    cons: List[str]


class Recommendation(TypedDict):
    """Final recommendation with justification from Recommendation Agent."""
    rank: int
    product: ProductScore
    confidence: float                   # Agent's confidence 0.0–1.0
    justification: str                  # Detailed reasoning
    trade_off_analysis: str             # What you gain/lose vs alternatives
    best_for: str                       # Who this is ideal for


class ShoppingState(TypedDict):
    """
    The single shared state object flowing through the entire LangGraph pipeline.
    Agents read inputs from earlier fields and write outputs to their own fields.
    """
    # ── Input ──────────────────────────────────────────────────────────────
    raw_query: str                       # Original user query
    session_id: str                      # Unique session identifier

    # ── Agent 1 Output ─────────────────────────────────────────────────────
    preferences: Optional[UserPreferences]
    preference_confidence: float         # How confidently query was parsed
    is_valid_query: bool                 # False if query is too vague

    # ── Agent 2 Output ─────────────────────────────────────────────────────
    retrieved_products: List[Dict]       # Raw products from DB/FAISS
    retrieval_method: str                # "database", "vector", or "hybrid"
    total_retrieved: int

    # ── Agent 3 Output ─────────────────────────────────────────────────────
    scored_products: List[ProductScore]
    scoring_weights: Dict[str, float]   # Feature weights used

    # ── Agent 4 Output ─────────────────────────────────────────────────────
    recommendations: List[Recommendation]
    summary: str                         # Overall recommendation summary

    # ── Pipeline Metadata ──────────────────────────────────────────────────
    current_agent: str                   # For real-time status streaming
    errors: List[str]                    # Non-fatal errors encountered
    execution_times: Dict[str, float]    # Per-agent timing in seconds
    total_time: float
```

---

## STEP 3: ALL FOUR AGENTS

### Agent 1 — Preference Agent (agents/preference_agent.py)

```python
"""
Preference Agent: Parses natural language queries using Gemini 2.0 Flash
to extract structured shopping preferences and constraints.

Handles: budget ranges, product categories, priority features,
         brand preferences, use-case context, and edge cases.
"""
import json
import time
import logging
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from .base_agent import BaseAgent
from ..graph.state import ShoppingState, UserPreferences

logger = logging.getLogger(__name__)

PREFERENCE_SYSTEM_PROMPT = """You are a precision shopping preference parser. 
Extract structured information from the user's shopping query.

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{
  "budget_min": null or number (in PKR),
  "budget_max": null or number (in PKR),
  "category": "string (e.g. smartphone, laptop, earbuds, monitor, tablet)",
  "subcategory": null or "string (e.g. gaming laptop, flagship phone)",
  "priority_features": ["list", "of", "features"],
  "use_case": null or "string (e.g. gaming, photography, travel, office)",
  "brand_preferences": ["preferred brands, empty if none"],
  "brand_exclusions": ["excluded brands, empty if none"],
  "form_factor": null or "string",
  "other_constraints": ["any other constraints"],
  "confidence": 0.0 to 1.0,
  "is_valid": true or false,
  "clarification_needed": null or "string (what info is missing)"
}

Rules:
- Convert budget ranges to PKR (1 USD ≈ 278 PKR, 1 GBP ≈ 352 PKR)
- "under X" → budget_max = X, budget_min = 0
- "around X" → budget_min = X*0.85, budget_max = X*1.15
- If no budget mentioned, set both to null
- Extract implicit features (e.g. "for gaming" implies high performance, GPU)
- "best" or "top" implies no budget constraint (budget_max = null)
- is_valid = false ONLY if query has zero product context"""


class PreferenceAgent(BaseAgent):
    """
    Agent 1: Extracts structured user preferences from natural language.
    
    Input:  raw_query (string)
    Output: preferences (UserPreferences), preference_confidence (float)
    """

    def __init__(self, gemini_api_key: str):
        super().__init__(name="PreferenceAgent")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1,   # Low temperature for consistent parsing
            max_tokens=1024
        )

    def run(self, state: ShoppingState) -> ShoppingState:
        """
        Parse the raw user query and populate preferences in state.
        Uses Gemini 2.0 Flash with structured output prompting.
        """
        start = time.time()
        logger.info(f"PreferenceAgent processing: '{state['raw_query']}'")

        try:
            messages = [
                SystemMessage(content=PREFERENCE_SYSTEM_PROMPT),
                HumanMessage(content=f"Parse this shopping query: {state['raw_query']}")
            ]

            response = self.llm.invoke(messages)
            raw = response.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw)

            preferences: UserPreferences = {
                "budget_min": parsed.get("budget_min"),
                "budget_max": parsed.get("budget_max"),
                "category": parsed.get("category", "electronics"),
                "subcategory": parsed.get("subcategory"),
                "priority_features": parsed.get("priority_features", []),
                "use_case": parsed.get("use_case"),
                "brand_preferences": parsed.get("brand_preferences", []),
                "brand_exclusions": parsed.get("brand_exclusions", []),
                "form_factor": parsed.get("form_factor"),
                "other_constraints": parsed.get("other_constraints", [])
            }

            elapsed = round(time.time() - start, 2)
            logger.info(f"PreferenceAgent done in {elapsed}s: category={preferences['category']}, budget_max={preferences['budget_max']}")

            return {
                **state,
                "preferences": preferences,
                "preference_confidence": float(parsed.get("confidence", 0.8)),
                "is_valid_query": bool(parsed.get("is_valid", True)),
                "current_agent": "retrieval",
                "execution_times": {**state.get("execution_times", {}), "preference": elapsed}
            }

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"PreferenceAgent parse error: {e}")
            return {
                **state,
                "errors": [*state.get("errors", []), f"PreferenceAgent: {str(e)}"],
                "is_valid_query": False,
                "current_agent": "error"
            }
```

---

### Agent 2 — Retrieval Agent (agents/retrieval_agent.py)

```python
"""
Retrieval Agent: Finds relevant products using a hybrid approach:
  1. FAISS vector similarity search (semantic matching)
  2. PostgreSQL filtered query (hard constraints: budget, category)
  3. Merge and deduplicate results

Falls back gracefully if FAISS index is unavailable.
"""
import time
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from .base_agent import BaseAgent
from ..graph.state import ShoppingState
from ..database.crud import get_products_by_filters
from ..vector_store.faiss_store import FAISSProductStore

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    """
    Agent 2: Retrieves candidate products from PostgreSQL + FAISS.
    
    Strategy: Hybrid retrieval — vector search for semantic relevance,
              database filter for hard constraints (budget, category).
    
    Input:  preferences (UserPreferences)
    Output: retrieved_products (List[Dict]), retrieval_method (str)
    """

    def __init__(self, db_session: Session, faiss_store: FAISSProductStore):
        super().__init__(name="RetrievalAgent")
        self.db = db_session
        self.faiss = faiss_store

    def run(self, state: ShoppingState) -> ShoppingState:
        start = time.time()
        prefs = state["preferences"]

        # === Step 1: FAISS Semantic Search ===
        vector_results = []
        try:
            query_text = self._build_search_query(prefs)
            vector_results = self.faiss.search(query_text, top_k=30)
            logger.info(f"FAISS returned {len(vector_results)} results")
        except Exception as e:
            logger.warning(f"FAISS search failed, falling back to DB only: {e}")

        # === Step 2: Database Hard-Filter Query ===
        db_results = get_products_by_filters(
            db=self.db,
            category=prefs.get("category"),
            budget_max=prefs.get("budget_max"),
            budget_min=prefs.get("budget_min"),
            brand_preferences=prefs.get("brand_preferences", []),
            brand_exclusions=prefs.get("brand_exclusions", []),
            limit=50
        )
        logger.info(f"Database returned {len(db_results)} results")

        # === Step 3: Merge and Deduplicate ===
        all_products = self._merge_results(vector_results, db_results)

        # === Step 4: Apply budget filter strictly ===
        if prefs.get("budget_max"):
            all_products = [p for p in all_products if p["price"] <= prefs["budget_max"] * 1.1]

        elapsed = round(time.time() - start, 2)
        retrieval_method = "hybrid" if vector_results else "database"

        logger.info(f"RetrievalAgent: {len(all_products)} products after merge, {elapsed}s")

        return {
            **state,
            "retrieved_products": all_products[:50],
            "retrieval_method": retrieval_method,
            "total_retrieved": len(all_products),
            "current_agent": "comparison",
            "execution_times": {**state.get("execution_times", {}), "retrieval": elapsed}
        }

    def _build_search_query(self, prefs: Dict) -> str:
        """Build a natural language search query from structured preferences."""
        parts = []
        if prefs.get("category"):
            parts.append(prefs["category"])
        if prefs.get("use_case"):
            parts.append(f"for {prefs['use_case']}")
        if prefs.get("priority_features"):
            parts.append("with " + ", ".join(prefs["priority_features"][:3]))
        if prefs.get("subcategory"):
            parts.append(prefs["subcategory"])
        return " ".join(parts) if parts else prefs.get("category", "electronics")

    def _merge_results(self, vector: List[Dict], database: List[Dict]) -> List[Dict]:
        """Merge vector and database results, keeping highest relevance scores."""
        seen_ids = set()
        merged = []
        for p in vector:
            if p["id"] not in seen_ids:
                p["relevance_source"] = "semantic"
                merged.append(p)
                seen_ids.add(p["id"])
        for p in database:
            if p["id"] not in seen_ids:
                p["relevance_source"] = "database"
                merged.append(p)
                seen_ids.add(p["id"])
        return merged
```

---

### Agent 3 — Comparison Agent (agents/comparison_agent.py)

```python
"""
Comparison Agent: Scores each retrieved product across 8 weighted dimensions.
Uses Gemini for intelligent feature extraction, then applies a configurable
weighted scoring matrix aligned with user priorities.

Dimensions: Performance, Camera, Battery, Value, Build Quality,
            Software, Connectivity, Availability
"""
import time
import json
import logging
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from .base_agent import BaseAgent
from ..graph.state import ShoppingState, ProductScore

logger = logging.getLogger(__name__)

# Default scoring weights (adjust per use_case in run())
DEFAULT_WEIGHTS = {
    "performance": 0.20,
    "camera": 0.15,
    "battery": 0.15,
    "value": 0.20,
    "build_quality": 0.10,
    "software": 0.10,
    "connectivity": 0.05,
    "availability": 0.05
}

# Use-case weight overrides
USE_CASE_WEIGHTS = {
    "gaming": {"performance": 0.35, "battery": 0.20, "camera": 0.05, "value": 0.15, "build_quality": 0.10, "software": 0.10, "connectivity": 0.03, "availability": 0.02},
    "photography": {"camera": 0.40, "performance": 0.15, "battery": 0.15, "value": 0.15, "build_quality": 0.08, "software": 0.05, "connectivity": 0.01, "availability": 0.01},
    "travel": {"battery": 0.30, "performance": 0.15, "camera": 0.20, "value": 0.15, "build_quality": 0.10, "software": 0.05, "connectivity": 0.03, "availability": 0.02},
    "office": {"performance": 0.25, "battery": 0.20, "value": 0.25, "software": 0.15, "build_quality": 0.10, "camera": 0.02, "connectivity": 0.02, "availability": 0.01}
}

SCORING_SYSTEM_PROMPT = """You are a product scoring expert. Given a product and user preferences,
score the product on 8 dimensions from 0-100. Return ONLY valid JSON:
{
  "performance": 0-100,
  "camera": 0-100,
  "battery": 0-100,
  "value": 0-100,
  "build_quality": 0-100,
  "software": 0-100,
  "connectivity": 0-100,
  "availability": 0-100,
  "pros": ["list of top 3 strengths"],
  "cons": ["list of top 2 weaknesses"]
}
Base scores on real-world knowledge of this product. Be critical and accurate."""


class ComparisonAgent(BaseAgent):
    """
    Agent 3: Multi-dimensional product scoring with weighted composite score.
    
    For each product: calls Gemini to score dimensions, applies preference-
    aligned weights, computes composite score, ranks top candidates.
    
    Input:  retrieved_products, preferences
    Output: scored_products (List[ProductScore])
    """

    def __init__(self, gemini_api_key: str):
        super().__init__(name="ComparisonAgent")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_tokens=512
        )

    def run(self, state: ShoppingState) -> ShoppingState:
        start = time.time()
        prefs = state["preferences"]
        products = state["retrieved_products"]

        # Select appropriate weights based on use-case
        weights = USE_CASE_WEIGHTS.get(
            prefs.get("use_case", "").lower(),
            DEFAULT_WEIGHTS.copy()
        )

        # Adjust weights for priority features
        for feature in prefs.get("priority_features", []):
            feature_lower = feature.lower()
            if feature_lower in weights:
                # Boost priority features by 50%, redistribute from lowest
                weights[feature_lower] = min(weights[feature_lower] * 1.5, 0.45)

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        weights = {k: round(v / total, 3) for k, v in weights.items()}

        scored = []
        # Score top 15 candidates (optimization: skip obviously wrong ones)
        candidates = self._pre_filter(products, prefs)[:15]

        for product in candidates:
            score = self._score_product(product, prefs, weights)
            if score:
                scored.append(score)

        # Sort by composite score descending
        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        elapsed = round(time.time() - start, 2)
        logger.info(f"ComparisonAgent scored {len(scored)} products in {elapsed}s")

        return {
            **state,
            "scored_products": scored,
            "scoring_weights": weights,
            "current_agent": "recommendation",
            "execution_times": {**state.get("execution_times", {}), "comparison": elapsed}
        }

    def _pre_filter(self, products: List[Dict], prefs: Dict) -> List[Dict]:
        """Fast pre-filter to remove obviously irrelevant products."""
        budget_max = prefs.get("budget_max")
        exclusions = [b.lower() for b in prefs.get("brand_exclusions", [])]
        filtered = []
        for p in products:
            if budget_max and p.get("price", 0) > budget_max * 1.2:
                continue
            if p.get("brand", "").lower() in exclusions:
                continue
            filtered.append(p)
        return filtered

    def _score_product(self, product: Dict, prefs: Dict, weights: Dict) -> ProductScore | None:
        """Score a single product using Gemini + weighted matrix."""
        try:
            product_desc = json.dumps({
                "name": product.get("name"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "price_pkr": product.get("price"),
                "specs": product.get("specs", {}),
                "rating": product.get("rating")
            }, indent=2)

            messages = [
                SystemMessage(content=SCORING_SYSTEM_PROMPT),
                HumanMessage(content=f"Product:\n{product_desc}\n\nUser priorities: {prefs.get('priority_features')}\nUse case: {prefs.get('use_case')}\nBudget max: {prefs.get('budget_max')} PKR")
            ]

            response = self.llm.invoke(messages)
            raw = response.content.strip().lstrip("```json").rstrip("```")
            scores = json.loads(raw)

            # Compute weighted composite score
            dims = ["performance", "camera", "battery", "value", "build_quality", "software", "connectivity", "availability"]
            composite = sum(scores.get(d, 50) * weights.get(d, 0.1) for d in dims)
            composite = round(composite, 1)

            budget_max = prefs.get("budget_max")
            within_budget = True if not budget_max else product.get("price", 0) <= budget_max

            return ProductScore(
                product_id=str(product.get("id")),
                name=product.get("name", ""),
                brand=product.get("brand", ""),
                price=float(product.get("price", 0)),
                category=product.get("category", ""),
                specs=product.get("specs", {}),
                composite_score=composite,
                dimension_scores={d: scores.get(d, 50) for d in dims},
                within_budget=within_budget,
                pros=scores.get("pros", []),
                cons=scores.get("cons", [])
            )
        except Exception as e:
            logger.warning(f"Failed to score {product.get('name')}: {e}")
            return None
```

---

### Agent 4 — Recommendation Agent (agents/recommendation_agent.py)

```python
"""
Recommendation Agent: The final stage. Takes scored products and generates
a rich, explainable recommendation with ranked results, justifications,
trade-off analysis, and an executive summary.

Uses Gemini 2.0 Flash for chain-of-thought reasoning to produce
human-readable, trustworthy recommendation text.
"""
import time
import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from .base_agent import BaseAgent
from ..graph.state import ShoppingState, Recommendation

logger = logging.getLogger(__name__)

RECOMMENDATION_SYSTEM_PROMPT = """You are a trusted expert shopping advisor.
Given scored products and user preferences, generate final recommendations.

Return ONLY valid JSON with this structure:
{
  "recommendations": [
    {
      "rank": 1,
      "confidence": 0.0-1.0,
      "justification": "3-4 sentence detailed explanation of why this is the best choice",
      "trade_off_analysis": "What you gain and what you sacrifice vs other options",
      "best_for": "Who/what scenario this product is ideal for"
    }
    // repeat for each top product (3-5 items)
  ],
  "summary": "2-3 sentence overall advisory summary"
}

Write justifications as a knowledgeable human advisor would. Be specific.
Mention actual specs. Reference the user's stated use case and budget."""


class RecommendationAgent(BaseAgent):
    """
    Agent 4: Generates final ranked recommendations with rich justifications.
    
    Input:  scored_products, preferences
    Output: recommendations (List[Recommendation]), summary (str)
    """

    def __init__(self, gemini_api_key: str):
        super().__init__(name="RecommendationAgent")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.3,   # Slightly higher for natural language quality
            max_tokens=2048
        )

    def run(self, state: ShoppingState) -> ShoppingState:
        start = time.time()
        top_products = state["scored_products"][:5]
        prefs = state["preferences"]

        if not top_products:
            return {
                **state,
                "recommendations": [],
                "summary": "No matching products found for your criteria. Try broadening your budget or adjusting preferences.",
                "current_agent": "done",
                "execution_times": {**state.get("execution_times", {}), "recommendation": 0}
            }

        # Build product summary for the LLM
        products_summary = json.dumps([
            {
                "rank_by_score": i + 1,
                "name": p["name"],
                "brand": p["brand"],
                "price_pkr": p["price"],
                "composite_score": p["composite_score"],
                "within_budget": p["within_budget"],
                "dimension_scores": p["dimension_scores"],
                "pros": p["pros"],
                "cons": p["cons"],
                "specs": p["specs"]
            }
            for i, p in enumerate(top_products)
        ], indent=2)

        prefs_summary = {
            "query": state["raw_query"],
            "budget_max": prefs.get("budget_max"),
            "category": prefs.get("category"),
            "priorities": prefs.get("priority_features"),
            "use_case": prefs.get("use_case")
        }

        messages = [
            SystemMessage(content=RECOMMENDATION_SYSTEM_PROMPT),
            HumanMessage(content=f"User Preferences:\n{json.dumps(prefs_summary, indent=2)}\n\nScored Products:\n{products_summary}")
        ]

        response = self.llm.invoke(messages)
        raw = response.content.strip().lstrip("```json").rstrip("```")
        result = json.loads(raw)

        recommendations = []
        raw_recs = result.get("recommendations", [])
        for i, (product, rec_data) in enumerate(zip(top_products, raw_recs)):
            recommendations.append(Recommendation(
                rank=i + 1,
                product=product,
                confidence=float(rec_data.get("confidence", 0.8)),
                justification=rec_data.get("justification", ""),
                trade_off_analysis=rec_data.get("trade_off_analysis", ""),
                best_for=rec_data.get("best_for", "")
            ))

        elapsed = round(time.time() - start, 2)
        total = sum(state.get("execution_times", {}).values()) + elapsed

        return {
            **state,
            "recommendations": recommendations,
            "summary": result.get("summary", ""),
            "current_agent": "done",
            "total_time": round(total, 2),
            "execution_times": {**state.get("execution_times", {}), "recommendation": elapsed}
        }
```

---

## STEP 4: LANGGRAPH WORKFLOW (graph/workflow.py)

```python
"""
LangGraph StateGraph: Orchestrates the 4-agent pipeline with conditional
routing, error recovery, and real-time WebSocket status broadcasting.

Flow:
  START → preference_agent → [valid?] → retrieval_agent
                           → [invalid] → error_handler
  retrieval_agent → [results found?] → comparison_agent
                  → [no results] → fallback_handler  
  comparison_agent → recommendation_agent → END
"""
import asyncio
import logging
from langgraph.graph import StateGraph, END
from ..agents.preference_agent import PreferenceAgent
from ..agents.retrieval_agent import RetrievalAgent
from ..agents.comparison_agent import ComparisonAgent
from ..agents.recommendation_agent import RecommendationAgent
from .state import ShoppingState
from ..graph.router import route_after_preference, route_after_retrieval

logger = logging.getLogger(__name__)


def build_shopping_graph(
    preference_agent: PreferenceAgent,
    retrieval_agent: RetrievalAgent,
    comparison_agent: ComparisonAgent,
    recommendation_agent: RecommendationAgent,
    websocket_broadcast=None
) -> StateGraph:
    """
    Constructs the LangGraph StateGraph for the shopping advisor pipeline.
    
    Args:
        *_agent: Initialized agent instances
        websocket_broadcast: Optional async function for real-time status updates
    
    Returns:
        Compiled StateGraph ready to invoke
    """

    def wrap_with_status(agent, status_msg: str):
        """Wraps an agent's run() with WebSocket status broadcasting."""
        def wrapped(state: ShoppingState) -> ShoppingState:
            if websocket_broadcast:
                asyncio.create_task(websocket_broadcast({
                    "event": "agent_start",
                    "agent": agent.name,
                    "message": status_msg,
                    "session_id": state.get("session_id")
                }))
            result = agent.run(state)
            if websocket_broadcast:
                asyncio.create_task(websocket_broadcast({
                    "event": "agent_done",
                    "agent": agent.name,
                    "time": result.get("execution_times", {}).get(agent.name.lower().replace("agent", ""), 0),
                    "session_id": state.get("session_id")
                }))
            return result
        return wrapped

    # ── Build Graph ──────────────────────────────────────────────────────────
    workflow = StateGraph(ShoppingState)

    # Add agent nodes
    workflow.add_node("preference_agent",    wrap_with_status(preference_agent, "Extracting preferences..."))
    workflow.add_node("retrieval_agent",     wrap_with_status(retrieval_agent,  "Retrieving products..."))
    workflow.add_node("comparison_agent",    wrap_with_status(comparison_agent, "Scoring products..."))
    workflow.add_node("recommendation_agent",wrap_with_status(recommendation_agent, "Generating recommendations..."))

    # Error handler node
    workflow.add_node("error_handler", lambda s: {**s, "current_agent": "error"})
    workflow.add_node("fallback_handler", lambda s: {
        **s,
        "scored_products": [],
        "current_agent": "recommendation"
    })

    # ── Define Edges ─────────────────────────────────────────────────────────
    workflow.set_entry_point("preference_agent")

    # After preference: route based on query validity
    workflow.add_conditional_edges(
        "preference_agent",
        route_after_preference,
        {
            "valid":   "retrieval_agent",
            "invalid": "error_handler"
        }
    )

    # After retrieval: route based on whether products were found
    workflow.add_conditional_edges(
        "retrieval_agent",
        route_after_retrieval,
        {
            "found":    "comparison_agent",
            "not_found": "fallback_handler"
        }
    )

    workflow.add_edge("comparison_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", END)
    workflow.add_edge("error_handler", END)
    workflow.add_edge("fallback_handler", "recommendation_agent")

    return workflow.compile()
```

---

## STEP 5: FASTAPI + WEBSOCKET BACKEND (backend/main.py)

```python
"""
FastAPI application with:
  - REST endpoint for shopping queries (/api/query)
  - WebSocket endpoint for real-time agent streaming (/ws/{session_id})
  - Static file serving for the frontend
  - CORS configuration for development
"""
import uuid
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from .config import settings
from .graph.workflow import build_shopping_graph
from .agents.preference_agent import PreferenceAgent
from .agents.retrieval_agent import RetrievalAgent
from .agents.comparison_agent import ComparisonAgent
from .agents.recommendation_agent import RecommendationAgent
from .database.connection import get_db_session
from .vector_store.faiss_store import FAISSProductStore

logger = logging.getLogger(__name__)

# Active WebSocket connections by session_id
active_connections: dict[str, WebSocket] = {}


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents and database on startup."""
    logger.info("Initializing NexusShop agents...")
    app.state.faiss_store = FAISSProductStore(settings.FAISS_INDEX_PATH)
    await app.state.faiss_store.load_or_build(settings.DATABASE_URL)
    logger.info("FAISS store ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="NexusShop — Intelligent Shopping Advisor",
    description="Multi-agent AI shopping system powered by LangGraph + Gemini 2.0",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time agent status streaming."""
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info(f"WebSocket connected: {session_id}")
    try:
        while True:
            await websocket.receive_text()  # Keep-alive
    except WebSocketDisconnect:
        active_connections.pop(session_id, None)
        logger.info(f"WebSocket disconnected: {session_id}")


async def broadcast_to_session(session_id: str, data: dict):
    """Send a real-time event to a specific WebSocket session."""
    ws = active_connections.get(session_id)
    if ws:
        try:
            await ws.send_json(data)
        except Exception:
            active_connections.pop(session_id, None)


@app.post("/api/query")
async def process_query(request: QueryRequest):
    """
    Main endpoint: Process a shopping query through the 4-agent pipeline.
    
    Returns ranked product recommendations with justifications.
    Real-time agent status is broadcast via WebSocket to session_id.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    # Bind broadcast function to this session
    async def broadcast(data: dict):
        await broadcast_to_session(session_id, data)

    # Initialize agents
    db = get_db_session()
    preference_agent    = PreferenceAgent(settings.GEMINI_API_KEY)
    retrieval_agent     = RetrievalAgent(db, app.state.faiss_store)
    comparison_agent    = ComparisonAgent(settings.GEMINI_API_KEY)
    recommendation_agent = RecommendationAgent(settings.GEMINI_API_KEY)

    # Build and run the LangGraph pipeline
    graph = build_shopping_graph(
        preference_agent, retrieval_agent,
        comparison_agent, recommendation_agent,
        websocket_broadcast=broadcast
    )

    initial_state = {
        "raw_query": request.query,
        "session_id": session_id,
        "preferences": None,
        "preference_confidence": 0.0,
        "is_valid_query": True,
        "retrieved_products": [],
        "retrieval_method": "",
        "total_retrieved": 0,
        "scored_products": [],
        "scoring_weights": {},
        "recommendations": [],
        "summary": "",
        "current_agent": "preference",
        "errors": [],
        "execution_times": {},
        "total_time": 0.0
    }

    await broadcast({"event": "pipeline_start", "session_id": session_id})

    final_state = await asyncio.to_thread(graph.invoke, initial_state)

    await broadcast({"event": "pipeline_done", "session_id": session_id})

    return {
        "session_id": session_id,
        "query": request.query,
        "preferences": final_state.get("preferences"),
        "recommendations": final_state.get("recommendations", []),
        "summary": final_state.get("summary", ""),
        "total_products_retrieved": final_state.get("total_retrieved", 0),
        "execution_times": final_state.get("execution_times", {}),
        "total_time_seconds": final_state.get("total_time", 0),
        "errors": final_state.get("errors", [])
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "NexusShop API", "version": "1.0.0"}


# Serve the luxury frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
```

---

## STEP 6: DATABASE MODELS (database/models.py)

```python
"""
SQLAlchemy ORM models for the NexusShop product database.
PostgreSQL is the primary store; FAISS handles vector similarity.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, JSON, DateTime, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Product(Base):
    """Core product table with full specifications."""
    __tablename__ = "products"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(255), nullable=False)
    brand       = Column(String(100), nullable=False, index=True)
    category    = Column(String(100), nullable=False, index=True)
    subcategory = Column(String(100), nullable=True)
    price       = Column(Float, nullable=False, index=True)
    currency    = Column(String(10), default="PKR")
    rating      = Column(Float, nullable=True)
    review_count = Column(Integer, default=0)
    specs       = Column(JSON, nullable=True)   # Flexible key-value specs
    description = Column(Text, nullable=True)
    image_url   = Column(String(500), nullable=True)
    availability = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite index for common query pattern
    __table_args__ = (
        Index("ix_category_price", "category", "price"),
        Index("ix_brand_category", "brand", "category"),
    )

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "brand": self.brand,
            "category": self.category, "subcategory": self.subcategory,
            "price": self.price, "currency": self.currency,
            "rating": self.rating, "review_count": self.review_count,
            "specs": self.specs or {}, "description": self.description,
            "availability": self.availability
        }


class PriceHistory(Base):
    """Track price changes over time for trend analysis."""
    __tablename__ = "price_history"

    id         = Column(Integer, primary_key=True)
    product_id = Column(Integer, nullable=False, index=True)
    price      = Column(Float, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    source     = Column(String(100), default="manual")
```

---

## STEP 7: PRODUCT DATASET (data/products.json format)

Create a JSON file with **200+ products** across these categories:

```json
[
  {
    "id": 1,
    "name": "Samsung Galaxy S24",
    "brand": "Samsung",
    "category": "smartphone",
    "subcategory": "flagship",
    "price": 89999,
    "currency": "PKR",
    "rating": 4.5,
    "review_count": 1240,
    "specs": {
      "display": "6.2-inch Dynamic AMOLED 2X",
      "processor": "Snapdragon 8 Gen 3",
      "ram": "8GB",
      "storage": "128GB",
      "camera_main": "50MP",
      "camera_front": "12MP",
      "battery": "4000mAh",
      "charging": "25W Fast + 15W Wireless",
      "os": "Android 14 / One UI 6.1",
      "5g": true,
      "nfc": true
    },
    "description": "Samsung's 2024 flagship with AI-powered features...",
    "availability": true
  }
  // Add 200+ products: smartphones, laptops, earbuds, monitors, tablets, cameras
]
```

**Categories to include (minimum 20 products each):**
- `smartphone` — Budget, mid-range, flagship segments
- `laptop` — Gaming, ultrabook, business, student
- `earbuds` — TWS, over-ear, sports
- `monitor` — Gaming, 4K, ultra-wide, office
- `tablet` — iPad alternatives, Android tablets
- `camera` — DSLR, mirrorless, action cameras

---

## STEP 8: FAISS VECTOR STORE (vector_store/faiss_store.py)

```python
"""
FAISS-based semantic vector store for product similarity search.
Uses Gemini's text-embedding-004 model to embed product descriptions,
enabling semantic queries like "phone for photography enthusiast" to
match relevant products beyond keyword matching.
"""
import os
import json
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict
import google.generativeai as genai

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768


class FAISSProductStore:
    """
    Semantic vector store for product retrieval.
    
    Embeds product descriptions using Gemini embeddings,
    stores in FAISS index for fast approximate nearest-neighbor search.
    """

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index: faiss.Index | None = None
        self.products: List[Dict] = []
        self.id_map: Dict[int, Dict] = {}

    async def load_or_build(self, db_url: str):
        """Load existing FAISS index or build from database products."""
        index_file = f"{self.index_path}.faiss"
        meta_file  = f"{self.index_path}.pkl"

        if os.path.exists(index_file) and os.path.exists(meta_file):
            logger.info("Loading existing FAISS index...")
            self.index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                self.products = pickle.load(f)
            self.id_map = {i: p for i, p in enumerate(self.products)}
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        else:
            logger.info("Building FAISS index from database...")
            await self._build_index(db_url)

    async def _build_index(self, db_url: str):
        """Fetch products from DB, embed them, build FAISS index."""
        from ..database.connection import get_all_products
        products = get_all_products(db_url)

        texts = [
            f"{p['name']} {p['brand']} {p['category']} {p.get('description', '')} {json.dumps(p.get('specs', {}))}"
            for p in products
        ]

        logger.info(f"Generating embeddings for {len(texts)} products...")
        embeddings = []
        for i, text in enumerate(texts):
            result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
            embeddings.append(result["embedding"])
            if i % 50 == 0:
                logger.info(f"  Embedded {i}/{len(texts)} products")

        matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)  # Inner product = cosine similarity
        self.index.add(matrix)

        self.products = products
        self.id_map = {i: p for i, p in enumerate(products)}

        # Persist to disk
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(products, f)

        logger.info(f"FAISS index built and saved: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Semantic search: embed query, find top-k similar products."""
        if not self.index:
            return []
        result = genai.embed_content(model=EMBEDDING_MODEL, content=query)
        q_vec = np.array([result["embedding"]], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.id_map:
                product = self.id_map[idx].copy()
                product["similarity_score"] = float(score)
                results.append(product)
        return results
```

---

## STEP 9: TESTING (50+ Test Queries)

### tests/test_queries.py

```python
"""
Comprehensive test suite with 50+ queries covering all evaluation criteria.
Tests: budget parsing, category detection, edge cases, conflicting preferences.
"""
import pytest
from ..graph.workflow import build_shopping_graph
# ... (initialize agents)

TEST_QUERIES = [
    # ── Budget Tests ──────────────────────────────────────────────────────
    {"query": "Best smartphone under 100,000 PKR",            "expect_budget_max": 100000, "expect_category": "smartphone"},
    {"query": "Laptop for gaming and programming",             "expect_category": "laptop",  "expect_use_case": "gaming"},
    {"query": "Wireless earbuds under 10,000 rupees",         "expect_budget_max": 10000,   "expect_category": "earbuds"},
    {"query": "Best phone regardless of price",               "expect_budget_max": None,    "expect_category": "smartphone"},
    {"query": "Gaming monitor around 50k",                    "expect_budget_max": 57500,   "expect_budget_min": 42500},

    # ── Category Detection ─────────────────────────────────────────────────
    {"query": "4K display for video editing",                 "expect_category": "monitor"},
    {"query": "Noise cancelling headphones for office",       "expect_category": "earbuds"},
    {"query": "Mirrorless camera for travel photography",     "expect_category": "camera"},
    {"query": "iPad alternative for students",               "expect_category": "tablet"},
    {"query": "Ultrabook for business trips",                 "expect_category": "laptop"},

    # ── Feature Extraction ─────────────────────────────────────────────────
    {"query": "Phone with best battery life under 60k",       "expect_features": ["battery"]},
    {"query": "Laptop with RTX 4070 for gaming",              "expect_features": ["performance", "gpu"]},
    {"query": "Waterproof earbuds for swimming",              "expect_features": ["waterproof"]},
    {"query": "5G phone with 12GB RAM",                       "expect_features": ["5g", "ram"]},

    # ── Brand Preferences ──────────────────────────────────────────────────
    {"query": "Best Apple iPhone under 200k",                 "expect_brands": ["Apple"]},
    {"query": "Samsung or OnePlus phone under 80k",           "expect_brands": ["Samsung", "OnePlus"]},
    {"query": "Laptop not from HP or Lenovo",                 "expect_exclusions": ["HP", "Lenovo"]},

    # ── Edge Cases ─────────────────────────────────────────────────────────
    {"query": "phone",                                        "expect_valid": True},   # Minimal query
    {"query": "I want something good",                        "expect_valid": False},  # Too vague
    {"query": "Best everything under 0 PKR",                  "expect_valid": True},   # Zero budget
    {"query": "Laptop for gaming, programming, video editing, design", "expect_valid": True},  # Many features

    # ── Pakistani Context ──────────────────────────────────────────────────
    {"query": "Best mobile phone under 1 lakh",               "expect_budget_max": 100000},
    {"query": "Sasti laptop under 50 hazar",                  "expect_budget_max": 50000},  # Urdu
    {"query": "Cheapest smartphone with good camera",         "expect_features": ["camera"]},

    # ... (add 25+ more test cases)
]

@pytest.mark.parametrize("test_case", TEST_QUERIES)
async def test_preference_extraction(test_case):
    # Test that preference agent correctly extracts each field
    ...
```

---

## STEP 10: EXTRA PREMIUM ENHANCEMENTS (For 100% Marks)

### Enhancement 1: Bonus Agent — Price Tracker
```python
# agents/price_tracker_agent.py
# Analyzes historical price data, identifies deals, alerts on price drops
# Adds "price_trend": "dropping" | "stable" | "rising" to each recommendation
```

### Enhancement 2: Voice Input (Frontend)
```javascript
// In frontend: Web Speech API for voice queries
const recognition = new webkitSpeechRecognition();
recognition.lang = 'en-US';
recognition.onresult = e => {
  document.getElementById('query-input').value = e.results[0][0].transcript;
};
```

### Enhancement 3: Export to PDF
```python
# API endpoint: GET /api/export/{session_id}
# Uses reportlab or weasyprint to generate a PDF report of recommendations
# Include: product cards, comparison chart, justifications, system diagram
```

### Enhancement 4: Multi-Language Support
```python
# Detect if query is in Urdu using langdetect
# Translate to English for processing using Gemini
# Return response in same language as input
```

### Enhancement 5: Confidence Calibration
```python
# If recommendation_confidence < 0.7, trigger a clarification request
# "I'm not certain about your exact requirements. Did you mean X or Y?"
# Implements a feedback loop back to the preference agent
```

---

## STEP 11: PROJECT REPORT OUTLINE (8 pages)

```
1. Executive Summary (0.5 page)
   - What was built, key achievements, results

2. System Architecture (1.5 pages)
   - LangGraph flow diagram (include the actual .png)
   - Agent interaction model
   - Data flow from query to recommendation

3. Agent Design Decisions (2 pages)
   - Why LangGraph over simple sequential calls
   - Why Gemini 2.0 Flash (speed vs quality trade-off)
   - FAISS vs pure SQL for retrieval
   - Weighted scoring vs pure LLM ranking

4. Implementation Challenges (1 page)
   - JSON parsing reliability from LLM
   - FAISS index memory management
   - WebSocket connection lifecycle
   - Solutions applied

5. Evaluation Results (1.5 pages)
   - Table: 50 test queries, accuracy, response time
   - Precision/Recall metrics
   - Edge case handling results
   - Performance benchmarks

6. Ethical Considerations (0.5 page)
   - Bias in product rankings
   - Data freshness / stale price issues
   - Privacy (no user data stored)
   - Limitations of LLM-based systems

7. Future Enhancements (0.5 page)
   - Real-time price scraping integration
   - Personalization via user history
   - Multi-modal product image analysis

8. Conclusion (0.5 page)
```

---

## STEP 12: SYSTEM ARCHITECTURE DIAGRAM

Include a diagram showing:

```
User Query (Natural Language)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                    NEXUSSHOP BACKEND                       │
│                                                           │
│   FastAPI REST + WebSocket                                │
│        │                                                  │
│        ▼                                                  │
│   ┌─────────────────────────────────────────────────┐    │
│   │              LangGraph StateGraph                │    │
│   │                                                 │    │
│   │  ┌──────────────┐    ┌──────────────────────┐  │    │
│   │  │  Preference  │───▶│   Retrieval Agent    │  │    │
│   │  │    Agent     │    │ PostgreSQL + FAISS    │  │    │
│   │  │ Gemini 2.0   │    │  Vector Similarity   │  │    │
│   │  └──────────────┘    └──────────────────────┘  │    │
│   │                               │                │    │
│   │                               ▼                │    │
│   │  ┌──────────────┐    ┌──────────────────────┐  │    │
│   │  │Recommendation│◀───│  Comparison Agent    │  │    │
│   │  │    Agent     │    │  8-dim Scoring       │  │    │
│   │  │ Gemini 2.0   │    │  Weighted Matrix     │  │    │
│   │  └──────────────┘    └──────────────────────┘  │    │
│   └─────────────────────────────────────────────────┘    │
│        │                                                  │
│        ▼                                                  │
│   JSON Response + WebSocket Events                        │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│      LUXURY FRONTEND           │
│  Three.js 3D · Chart.js Radar  │
│  Real-time Agent Status        │
│  Animated Product Cards        │
└────────────────────────────────┘
```

---

## STEP 13: HOW TO RUN

```bash
# 1. Clone and setup
git clone https://github.com/ahmad-yasin/nexusshop
cd nexusshop
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: add your GEMINI_API_KEY

# 3. Start PostgreSQL (Docker)
docker run -d --name nexusshop-db \
  -e POSTGRES_DB=nexusshop \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 postgres:16

# 4. Initialize database & seed data
cd backend
python -m database.seed_data

# 5. Build FAISS index
python -m vector_store.faiss_store --build

# 6. Start backend
uvicorn main:app --reload --port 8000

# 7. Open frontend
open http://localhost:8000
```

---

## FINAL CHECKLIST FOR 100% MARKS

- [x] ✅ All 4 agents implemented with clear docstrings
- [x] ✅ LangGraph StateGraph with conditional routing
- [x] ✅ Real Gemini 2.0 Flash API calls (not mock)
- [x] ✅ PostgreSQL database with proper schema
- [x] ✅ FAISS vector store for semantic search
- [x] ✅ WebSocket real-time agent status streaming
- [x] ✅ 200+ product dataset (structured JSON + CSV)
- [x] ✅ 50+ test queries with edge cases
- [x] ✅ Evaluation metrics (accuracy, response time)
- [x] ✅ System architecture diagram (PNG)
- [x] ✅ 8-page project report
- [x] ✅ Luxury premium frontend (Three.js, radar chart, animations)
- [x] ✅ Ethical considerations addressed
- [x] ✅ Bonus: Price tracker agent (5th agent)
- [x] ✅ Bonus: Voice input support
- [x] ✅ Bonus: PDF export of recommendations
- [x] ✅ Clean, commented code throughout
- [x] ✅ README with complete setup instructions
- [x] ✅ Demo video script (show live query → agents → results)
