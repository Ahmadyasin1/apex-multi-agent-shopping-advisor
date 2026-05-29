"""
NexusShop — FastAPI Backend with WebSocket real-time streaming.
Runs the LangGraph multi-agent pipeline and streams live status to the frontend.
"""
import sys, os, json, time, uuid, asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import Config
from graph import create_workflow
from state import AgentState
from agents.learning import get_user_context
from agents.comparison import score_product
from agents.enrichment import _infer_value_tier

app = FastAPI(
    title="NEXUS — Intelligent Shopping Advisor",
    description="Elite 8-agent AI shopping advisor powered by LangGraph, FAISS & Gemini",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

executor = ThreadPoolExecutor(max_workers=4)

AGENT_META = {
    "supervisor":     {"label": "Supervisor",     "icon": "SV", "desc": "Analyzing query complexity & routing to optimal LLM tier"},
    "preference":     {"label": "Preference",     "icon": "PR", "desc": "Extracting budget, intent, and 5-axis priority vector"},
    "retrieval":      {"label": "Retrieval",      "icon": "RT", "desc": "FAISS semantic search across 51-product vector database"},
    "enrichment":     {"label": "Enrichment",     "icon": "EN", "desc": "Normalizing specs, inferring value tiers deterministically"},
    "comparison":     {"label": "Comparison",     "icon": "CM", "desc": "Pareto-optimal multi-dimensional scoring across 5 axes"},
    "critique":       {"label": "Critique",       "icon": "CQ", "desc": "AI quality audit — validating outputs and budget constraints"},
    "recommendation": {"label": "Recommendation", "icon": "RC", "desc": "Generating premium advisory report with trade-off reasoning"},
    "learning":       {"label": "Learning",       "icon": "LN", "desc": "Persisting session, updating adaptive preference profile"},
}

def _load_dataset():
    path = PROJECT_ROOT / "data_layer" / "dataset.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Media Route ──────────────────────────────────────────────────────────────
@app.get("/media/{filename}")
async def serve_media(filename: str):
    safe = Path(filename).name
    p = PROJECT_ROOT / safe
    if p.exists() and p.suffix.lower() in (".mp4", ".webm", ".ogg", ".mov"):
        return FileResponse(str(p), media_type="video/mp4")
    raise HTTPException(status_code=404, detail=f"'{safe}' not found")

# ── Frontend ─────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "NEXUS API v3.0"})

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0", "author": Config.AUTHOR}

# ── Profile & Stats ──────────────────────────────────────────────────────────
@app.get("/api/profile")
async def get_profile():
    try:
        return get_user_context()
    except Exception:
        return {"total_sessions": 0, "most_searched_category": None,
                "learned_preferences": {}, "recent_queries": []}

@app.get("/api/stats")
async def get_stats():
    products = _load_dataset()
    cat_counts = {}
    for p in products:
        cat = p.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    ctx = get_user_context()
    return {
        "product_count": len(products),
        "total_sessions": ctx.get("total_sessions", 0),
        "agent_count": 8,
        "category_count": len(cat_counts),
        "categories": [{"name": k, "count": v}
                       for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])],
        "most_searched_category": ctx.get("most_searched_category"),
        "learned_preferences": ctx.get("learned_preferences", {}),
        "recent_queries": ctx.get("recent_queries", []),
        "author": Config.AUTHOR,
        "app_title": Config.APP_TITLE,
    }

@app.get("/api/categories")
async def get_categories():
    products = _load_dataset()
    counts = {}
    for p in products:
        cat = p.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return [{"category": k, "count": v} for k, v in sorted(counts.items())]

# ── Product Explorer (all products with pre-computed scores) ─────────────────
@app.get("/api/products/explore")
async def explore_products(category: str = ""):
    """All products with pre-computed scores for the scatter plot & category browser."""
    products = _load_dataset()
    if category:
        products = [p for p in products if p.get("category", "").lower() == category.lower()]

    default_pv = Config.DEFAULT_PRIORITY_VECTOR
    result = []
    for p in products:
        try:
            c_scores, overall = score_product(
                p, default_pv, category=p.get("category", "default")
            )
            vt = _infer_value_tier(float(p.get("price", 0)), p.get("category", ""))
            result.append({
                "id":             p.get("id", ""),
                "name":           p.get("name", ""),
                "brand":          p.get("brand", ""),
                "category":       p.get("category", ""),
                "price":          p.get("price", 0),
                "rating":         p.get("rating", 0),
                "overall_score":  round(overall, 2),
                "criteria_scores": {k: round(v, 2) for k, v in c_scores.items()},
                "value_tier":     vt,
                "features":       p.get("features", [])[:4],
                "description":    p.get("description", "")[:140],
                "ideal_for":      p.get("ideal_for", []),
                "specs":          p.get("specs", {}),
            })
        except Exception:
            pass

    return sorted(result, key=lambda x: x["overall_score"], reverse=True)

# ── Budget Optimizer ─────────────────────────────────────────────────────────
@app.get("/api/budget-optimizer")
async def budget_optimizer(budget: float = 1000.0, category: str = ""):
    """Top products for a given budget — price-weighted scoring."""
    products = _load_dataset()

    # Filter to within budget (with 5% tolerance)
    in_budget = [p for p in products if float(p.get("price", 9999)) <= budget * 1.05]
    if category:
        cat_budget = [p for p in in_budget if p.get("category", "").lower() == category.lower()]
        if cat_budget:
            in_budget = cat_budget

    # If nothing in budget, relax filter
    if not in_budget:
        in_budget = products
        if category:
            cat_f = [p for p in products if p.get("category", "").lower() == category.lower()]
            if cat_f:
                in_budget = cat_f

    # Price-weighted scoring: reward staying well within budget
    pv = {"price": 0.35, "performance": 0.25, "brand": 0.15, "durability": 0.15, "innovation": 0.10}
    scored = []
    for p in in_budget:
        try:
            c_scores, overall = score_product(
                p, pv, budget_max=budget, category=p.get("category", "default")
            )
            vt = _infer_value_tier(float(p.get("price", 0)), p.get("category", ""))
            price_val = float(p.get("price", 0))
            savings   = max(0, budget - price_val)
            scored.append({
                "id":             p.get("id", ""),
                "name":           p.get("name", ""),
                "brand":          p.get("brand", ""),
                "category":       p.get("category", ""),
                "price":          price_val,
                "rating":         p.get("rating", 0),
                "overall_score":  round(overall, 2),
                "criteria_scores": {k: round(v, 2) for k, v in c_scores.items()},
                "value_tier":     vt,
                "features":       p.get("features", [])[:4],
                "description":    p.get("description", "")[:130],
                "savings":        round(savings, 0),
                "budget_fit_pct": round(min(100, (price_val / budget) * 100), 1),
            })
        except Exception:
            pass

    return sorted(scored, key=lambda x: x["overall_score"], reverse=True)[:6]

# ── Personas ─────────────────────────────────────────────────────────────────
@app.get("/api/personas")
async def get_personas():
    """Pre-configured user personas with priority vectors and sample queries."""
    return [
        {
            "id": "student",
            "name": "Student",
            "icon": "ST",
            "color": "#4c8ecf",
            "desc": "Reliable, budget-friendly, great battery for campus life",
            "query": "Best laptop under $800 for college student with good battery life and reliability",
            "priority_vector": {"price": 0.40, "performance": 0.25, "brand": 0.15, "durability": 0.15, "innovation": 0.05},
            "budget": 800, "category": "laptop",
        },
        {
            "id": "gamer",
            "name": "Gamer",
            "icon": "GX",
            "color": "#cf4c4c",
            "desc": "Max FPS, RTX GPU, high-refresh display",
            "query": "Best gaming laptop with RTX GPU under $2000 for 4K AAA games",
            "priority_vector": {"price": 0.10, "performance": 0.50, "brand": 0.15, "durability": 0.15, "innovation": 0.10},
            "budget": 2000, "category": "laptop",
        },
        {
            "id": "professional",
            "name": "Professional",
            "icon": "PR",
            "color": "#c9a84c",
            "desc": "Enterprise reliability, MIL-SPEC, long support",
            "query": "Best business laptop for enterprise with MIL-SPEC durability and 4G LTE",
            "priority_vector": {"price": 0.10, "performance": 0.20, "brand": 0.35, "durability": 0.30, "innovation": 0.05},
            "budget": 2000, "category": "laptop",
        },
        {
            "id": "creator",
            "name": "Creator",
            "icon": "CR",
            "color": "#7c4ccf",
            "desc": "Video editing, OLED display, Apple M3 / RTX 4090",
            "query": "Best laptop for 4K video editing and graphic design under $3000",
            "priority_vector": {"price": 0.10, "performance": 0.40, "brand": 0.20, "durability": 0.15, "innovation": 0.15},
            "budget": 3000, "category": "laptop",
        },
        {
            "id": "audiophile",
            "name": "Audiophile",
            "icon": "AU",
            "color": "#4caf7c",
            "desc": "Reference-grade audio, LDAC, studio monitoring",
            "query": "Best audiophile headphones for studio monitoring and critical listening under $500",
            "priority_vector": {"price": 0.10, "performance": 0.45, "brand": 0.25, "durability": 0.10, "innovation": 0.10},
            "budget": 500, "category": "headphones",
        },
        {
            "id": "traveler",
            "name": "Traveler",
            "icon": "TV",
            "color": "#cf8e4c",
            "desc": "Ultralight, 15+ hr battery, compact, durable",
            "query": "Best ultralight laptop for frequent travel under $1500 with 15+ hour battery life",
            "priority_vector": {"price": 0.15, "performance": 0.20, "brand": 0.20, "durability": 0.35, "innovation": 0.10},
            "budget": 1500, "category": "laptop",
        },
    ]

# ── Score Explainer ──────────────────────────────────────────────────────────
@app.get("/api/explain-score")
async def explain_score(product_id: str, dimension: str, budget: float = 0):
    """Explains why a product got a specific dimension score."""
    products = _load_dataset()
    product = next((p for p in products if p.get("id") == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    price    = float(product.get("price", 0))
    rating   = float(product.get("rating", 3.0))
    features = product.get("features", [])
    specs    = product.get("specs", {}) or {}
    combined = (" ".join(features) + " " + str(specs)).lower()

    explanations = {
        "price": _explain_price(price, budget),
        "performance": _explain_performance(rating, combined),
        "brand": _explain_brand(product.get("brand",""), product.get("category","")),
        "durability": _explain_durability(rating, combined),
        "innovation": _explain_innovation(combined),
    }

    dim = dimension.lower()
    return {
        "product": product.get("name", ""),
        "dimension": dim,
        "explanation": explanations.get(dim, "No explanation available."),
    }

def _explain_price(price, budget):
    if budget and budget > 0:
        ratio = price / budget
        if ratio <= 0.5:   return f"${price:,.0f} is ≤50% of your ${budget:,.0f} budget — excellent value, score boosted to 10/10."
        elif ratio <= 0.75: return f"${price:,.0f} is 51–75% of your ${budget:,.0f} budget — strong value, scored 8/10."
        elif ratio <= 1.0:  return f"${price:,.0f} uses 76–100% of your ${budget:,.0f} budget — acceptable fit, scored 6/10."
        else: return f"${price:,.0f} exceeds your ${budget:,.0f} budget by {(ratio-1)*100:.0f}% — penalty applied."
    if price < 300:   return f"${price:,.0f} is budget tier — absolute price score: 9.5/10."
    elif price < 700:  return f"${price:,.0f} is mid-budget tier — price score: 8.5/10."
    elif price < 1200: return f"${price:,.0f} is mid-range tier — price score: 7.0/10."
    elif price < 2000: return f"${price:,.0f} is premium tier — price score: 5.5/10."
    elif price < 3500: return f"${price:,.0f} is ultra-premium tier — price score: 4.0/10."
    return f"${price:,.0f} is flagship tier — price score: 2.5/10."

def _explain_performance(rating, combined):
    base = round(min(10.0, rating * 2.0), 1)
    bonuses = []
    if any(k in combined for k in ["rtx 4090","m3 max","m2 ultra","i9","ryzen 9","a18 pro","snapdragon 8 gen 3"]):
        bonuses.append("+1.5 for flagship-class processor/GPU")
    elif any(k in combined for k in ["rtx 4080","rtx 4070","m3","i7","ryzen 7","a17 pro","tensor g4"]):
        bonuses.append("+0.8 for high-performance chip")
    expl = f"Base from rating {rating}/5 → {base}/10."
    if bonuses: expl += " Bonuses: " + ", ".join(bonuses) + "."
    return expl

def _explain_brand(brand, category):
    from agents.comparison import PREMIUM_BRANDS
    brands = PREMIUM_BRANDS.get(category.lower(), PREMIUM_BRANDS["default"])
    top4 = brands[:4]
    if brand in top4:
        return f"{brand} is a top-tier brand in {category} (ranked #{top4.index(brand)+1} of top brands) → 9.0/10."
    elif brand in brands:
        return f"{brand} is a premium brand in {category} — 7.5/10."
    return f"{brand} is outside the premium tier for {category} → 5.0/10."

def _explain_durability(rating, combined):
    base = round(rating * 1.8, 1)
    extras = []
    if any(k in combined for k in ["mil-spec","ip68","ip67","titanium","sapphire","gorilla glass"]):
        extras.append("+1.5 for military/IP rating or premium materials")
    elif any(k in combined for k in ["aluminum","metal","magnesium","rugged"]):
        extras.append("+0.8 for metal construction")
    return f"Base: rating {rating} × 1.8 = {base}/10." + (" " + "; ".join(extras) if extras else "")

def _explain_innovation(combined):
    kws = ["ai","neural","generative","adaptive","smart","copilot","gemini","m3","a18",
           "tensor g4","oled","amoled","mini-led","solar","ldac"]
    found = [k for k in kws if k in combined]
    score = round(min(10.0, 5.0 + len(found) * 0.8), 1)
    return f"Base 5.0/10 + {len(found)} innovation keywords found ({', '.join(found[:5])}) → {score}/10."

# ── Price Trend (simulated 13-month history) ─────────────────────────────────
@app.get("/api/price-trend")
async def price_trend(product_id: str):
    """Simulates a 13-month price history based on product tier/category/brand."""
    import math, random
    products = _load_dataset()
    product = next((p for p in products if p.get("id") == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    price   = float(product.get("price", 0))
    rating  = float(product.get("rating", 3.5))
    cat     = product.get("category", "").lower()
    brand   = product.get("brand", "")
    tier    = _infer_value_tier(price, cat)

    # Depreciation rate per month based on tier
    depreciation = {
        "Ultra Premium": 0.013, "Flagship": 0.011,
        "Premium": 0.009,       "Upper Mid-Range": 0.007,
        "Mid-Range": 0.005,     "Budget-Plus": 0.003,
        "Budget": 0.002,        "Entry Level": 0.001,
    }
    rate = depreciation.get(tier, 0.006)

    # Seasonal dip months (0=now, 12=13mo ago): Nov=index 2, Dec=index 1 back from now
    import datetime
    now_month = datetime.date.today().month  # 1-12
    # Build 13 months of data (index 0 = 13 months ago, index 12 = current)
    months, prices = [], []
    rng = random.Random(hash(product_id) % 10000)
    for i in range(13):
        months_ago = 12 - i
        # Base: launch price was higher, decays to current
        base = price * (1 + months_ago * rate)
        # Seasonal factor: Black Friday (Nov) = -8%, Dec = -5%
        month_num = (now_month - months_ago - 1) % 12 + 1
        seasonal = 1.0
        if month_num == 11:  seasonal = 0.92   # Black Friday
        elif month_num == 12: seasonal = 0.95  # Holiday deals
        elif month_num == 1:  seasonal = 1.03  # Post-holiday rebound
        # Small random noise ±1.5%
        noise = 1.0 + rng.uniform(-0.015, 0.015)
        final_price = round(base * seasonal * noise, 2)
        months.append(months_ago)
        prices.append(max(price * 0.6, final_price))  # floor at 60% of current

    # Force last point = current price
    prices[-1] = price

    avg_hist = round(sum(prices[:-3]) / len(prices[:-3]), 2)  # avg of first 10 points
    deal_score = round(min(10, max(0, ((avg_hist - price) / avg_hist) * 10 * 2.5)), 1)

    # Determine trend
    recent_avg = sum(prices[-4:-1]) / 3
    if price < recent_avg * 0.97:      trend = "declining"
    elif price > recent_avg * 1.03:    trend = "rising"
    else:                               trend = "stable"

    # Buy recommendation
    if deal_score >= 7:    advice = "Excellent time to buy — price is well below historical average."
    elif deal_score >= 4:  advice = "Good time to buy — price is moderately discounted from launch."
    elif deal_score >= 2:  advice = f"Neutral — near average price. Consider waiting for seasonal sales."
    else:                  advice = f"Price is near launch level. Black Friday or major sales may yield savings."

    labels = []
    import datetime
    today = datetime.date.today()
    for i in range(13):
        m = today.month - (12 - i)
        y = today.year
        while m <= 0: m += 12; y -= 1
        labels.append(f"{datetime.date(y, m, 1).strftime('%b %Y')}")

    return {
        "product_id":   product_id,
        "product_name": product.get("name", ""),
        "current_price": price,
        "avg_historical_price": avg_hist,
        "deal_score":   deal_score,
        "trend":        trend,
        "buy_advice":   advice,
        "labels":       labels,
        "prices":       [round(v, 2) for v in prices],
    }


# ── Market Pulse (per-category aggregated insights) ───────────────────────────
@app.get("/api/market-pulse")
async def market_pulse():
    """Aggregated per-category market insights computed from the full 51-product dataset."""
    products = _load_dataset()
    default_pv = Config.DEFAULT_PRIORITY_VECTOR
    cat_data: dict = {}

    for p in products:
        cat = p.get("category", "other")
        if cat not in cat_data:
            cat_data[cat] = []
        try:
            c_scores, overall = score_product(p, default_pv, category=cat)
            cat_data[cat].append({
                "name": p.get("name", ""),
                "brand": p.get("brand", ""),
                "price": float(p.get("price", 0)),
                "rating": float(p.get("rating", 0)),
                "overall_score": round(overall, 2),
                "criteria_scores": {k: round(v, 2) for k, v in c_scores.items()},
                "value_tier": _infer_value_tier(float(p.get("price", 0)), cat),
                "features": p.get("features", [])[:3],
            })
        except Exception:
            pass

    result = []
    for cat, items in cat_data.items():
        if not items:
            continue
        prices  = [x["price"] for x in items]
        scores  = [x["overall_score"] for x in items]
        ratings = [x["rating"] for x in items]
        best_val = max(items, key=lambda x: x["overall_score"] / max(x["price"], 1))
        top_scored = max(items, key=lambda x: x["overall_score"])
        avg_score = round(sum(scores) / len(scores), 2)
        avg_price = round(sum(prices) / len(prices), 0)
        # Value index: avg_score / (avg_price/100) — normalized
        value_idx = round(avg_score / (avg_price / 100), 3) if avg_price > 0 else 0
        result.append({
            "category":       cat,
            "product_count":  len(items),
            "avg_price":      avg_price,
            "min_price":      min(prices),
            "max_price":      max(prices),
            "avg_score":      avg_score,
            "avg_rating":     round(sum(ratings) / len(ratings), 2),
            "value_index":    value_idx,
            "top_product":    {"name": top_scored["name"], "score": top_scored["overall_score"], "price": top_scored["price"]},
            "best_value":     {"name": best_val["name"], "score": best_val["overall_score"], "price": best_val["price"]},
        })

    return sorted(result, key=lambda x: x["avg_score"], reverse=True)


# ── WebSocket Pipeline ────────────────────────────────────────────────────────
@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        payload    = json.loads(raw)
        query      = payload.get("query", "").strip()
        budget_max = payload.get("budget_max")
        budget_min = payload.get("budget_min")
        pv_override = payload.get("priority_vector")

        if not query:
            await websocket.send_json({"type": "error", "message": "Empty query."})
            return

        await websocket.send_json({"type": "ack", "session_id": session_id, "query": query,
                                   "message": "Pipeline initializing…"})

        initial_pv = Config.DEFAULT_PRIORITY_VECTOR.copy()
        if pv_override and isinstance(pv_override, dict):
            initial_pv.update({k: float(v) for k, v in pv_override.items() if k in initial_pv})

        initial_state: AgentState = {
            "query": query,
            "complexity": "simple",
            "selected_llm": "local",
            "category": "",
            "budget": {
                "min": float(budget_min) if budget_min else None,
                "max": float(budget_max) if budget_max else None,
            },
            "intent_preferences": [],
            "priority_vector": initial_pv,
            "retrieved_products": [],
            "enriched_products": [],
            "scored_products": [],
            "critique_passed": False,
            "critique_feedback": "",
            "revision_count": 0,
            "final_recommendation": "",
            "error": None,
            "history": [],
        }

        workflow    = create_workflow()
        loop        = asyncio.get_event_loop()
        t0          = time.time()
        final_state = None

        def run_workflow():
            return list(workflow.stream(initial_state))

        events = await loop.run_in_executor(executor, run_workflow)
        step   = 0
        for event in events:
            for node, sv in event.items():
                final_state = sv
                meta  = AGENT_META.get(node, {"label": node, "icon": "--", "desc": "Processing…"})
                step += 1  # type: ignore[operator]
                await websocket.send_json({
                    "type":           "agent_update",
                    "agent":          node,
                    "label":          meta["label"],
                    "icon":           meta["icon"],
                    "desc":           meta["desc"],
                    "step":           step,
                    "elapsed":        round(time.time() - t0, 2),
                    "products_count": len(sv.get("scored_products", [])),
                    "category":       sv.get("category", ""),
                    "complexity":     sv.get("complexity", "simple"),
                    "budget_max":     sv.get("budget", {}).get("max"),
                })

        total = round(time.time() - t0, 2)
        if not final_state:
            await websocket.send_json({"type": "error", "message": "Pipeline returned no state."})
            return

        scored = final_state.get("scored_products", [])
        products_out = []
        for p in scored:
            products_out.append({
                "id":               p.get("id", ""),
                "name":             p.get("name", ""),
                "brand":            p.get("brand", ""),
                "category":         p.get("category", ""),
                "price":            p.get("price", 0),
                "rating":           p.get("rating", 0),
                "value_tier":       p.get("value_tier", ""),
                "features":         p.get("features", []),
                "description":      p.get("description", ""),
                "ideal_for":        p.get("ideal_for", []),
                "specs":            p.get("specs", {}),
                "overall_score":    round(p.get("overall_score", 0), 2),
                "criteria_scores":  p.get("criteria_scores", {}),
                "is_pareto_optimal": p.get("is_pareto_optimal", False),
            })

        await websocket.send_json({
            "type":                "result",
            "session_id":          session_id,
            "query":               query,
            "category":            final_state.get("category", ""),
            "complexity":          final_state.get("complexity", "simple"),
            "model_used":          final_state.get("selected_llm", "local"),
            "budget_max":          final_state.get("budget", {}).get("max"),
            "intent_preferences":  final_state.get("intent_preferences", []),
            "priority_vector":     final_state.get("priority_vector", {}),
            "products":            products_out,
            "recommendation_text": final_state.get("final_recommendation", ""),
            "critique_passed":     final_state.get("critique_passed", True),
            "revision_count":      final_state.get("revision_count", 0),
            "total_time":          total,
        })

    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        try:
            await websocket.send_json({"type": "error", "message": "Query timed out."})
        except Exception:
            pass
    except Exception as e:
        err = str(e).encode("ascii", "replace").decode("ascii")
        try:
            await websocket.send_json({"type": "error", "message": f"Pipeline error: {err[:200]}"})
        except Exception:
            pass
