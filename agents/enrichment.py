"""
Agent 4 — Enrichment / Data Enhancement Agent
==============================================
Role: Normalizes and enriches raw product records BEFORE scoring.
      All enrichment is deterministic (rule-based) — no LLM calls needed.
      This design avoids latency spikes and hallucinated specifications.

Responsibilities:
  1. Fill missing spec fields using price-based inference:
       - RAM, storage, battery life estimated from price tier
       - Connectivity type inferred from price and category
  2. Generate fallback descriptions from brand + name + category
  3. Infer ideal_for tags from features, price, and rating
  4. Assign a human-readable value_tier label:
       Budget / Mid-Range / Premium / Ultra-Premium / Flagship / etc.

Value Tier Thresholds (example — laptop):
  < $600    → Budget
  < $1,200  → Mid-Range
  < $2,200  → Premium
  ≥ $2,200  → Ultra-Premium

The enriched product list flows directly into the Comparison Agent,
which requires normalized fields to compute fair 5-axis scores.
"""
from __future__ import annotations
from typing import Dict, Any, List
from state import AgentState

def enrichment_agent(state: AgentState) -> AgentState:
    """
    Data Enhancement Agent: fills missing specs using rule-based inference.
    Avoids expensive per-product LLM calls — uses deterministic enrichment first.
    """
    print("--- [AGENT] Enrichment ---")
    retrieved = state.get("retrieved_products", [])
    if not retrieved:
        print("No products to enrich.")
        state["enriched_products"] = []
        return state

    category = state.get("category", "unknown").lower()
    enriched_list = []

    for prod in retrieved:
        new_prod = prod.copy()
        specs: Dict[str, Any] = dict(new_prod.get("specs") or {})
        features = new_prod.get("features") or []
        features_str = " ".join(features).lower()
        price = float(new_prod.get("price", 0))
        rating = float(new_prod.get("rating", 3.0))

        # --- Fill missing common fields ---
        if not new_prod.get("description"):
            new_prod["description"] = (
                f"{new_prod.get('brand', '')} {new_prod.get('name', '')} "
                f"in the {new_prod.get('category', category)} category."
            )

        if not new_prod.get("ideal_for"):
            new_prod["ideal_for"] = _infer_ideal_for(new_prod, price, rating, features_str)

        # --- Enrich specs based on category ---
        if category in ("laptop", "desktop"):
            if not specs.get("ram"):
                specs["ram"] = _infer_ram(price)
            if not specs.get("storage"):
                specs["storage"] = _infer_storage(price)
            if not specs.get("battery") and category == "laptop":
                specs["battery"] = "8-12 hours (estimated)"

        elif category == "smartphone":
            if not specs.get("battery"):
                specs["battery"] = "4500mAh (estimated)"
            if not specs.get("ram"):
                specs["ram"] = _infer_ram_phone(price)

        elif category == "headphones":
            if not specs.get("connectivity"):
                specs["connectivity"] = "Bluetooth 5.0" if price > 50 else "3.5mm Wired"
            if not specs.get("type"):
                specs["type"] = "Over-ear" if price > 100 else "In-ear"

        elif category in ("smartwatch", "tablet"):
            if not specs.get("battery"):
                specs["battery"] = "All-day battery (estimated)"

        # --- Infer value tier ---
        new_prod["value_tier"] = _infer_value_tier(price, category)
        new_prod["specs"] = specs

        enriched_list.append(new_prod)
        print(f"  Enriched: {new_prod.get('name')}")

    state["enriched_products"] = enriched_list
    return state


def _infer_ram(price: float) -> str:
    if price >= 2500:
        return "32GB+ (inferred)"
    elif price >= 1200:
        return "16GB (inferred)"
    elif price >= 600:
        return "8GB (inferred)"
    return "4-8GB (inferred)"


def _infer_ram_phone(price: float) -> str:
    if price >= 900:
        return "12GB (inferred)"
    elif price >= 500:
        return "8GB (inferred)"
    return "6GB (inferred)"


def _infer_storage(price: float) -> str:
    if price >= 2000:
        return "1TB SSD (inferred)"
    elif price >= 1000:
        return "512GB SSD (inferred)"
    return "256GB SSD (inferred)"


def _infer_value_tier(price: float, category: str) -> str:
    thresholds = {
        "laptop": [(600, "Budget"), (1200, "Mid-Range"), (2200, "Premium"), (float("inf"), "Ultra-Premium")],
        "smartphone": [(350, "Budget"), (700, "Mid-Range"), (1000, "Premium"), (float("inf"), "Flagship")],
        "headphones": [(80, "Budget"), (200, "Mid-Range"), (400, "Premium"), (float("inf"), "Audiophile")],
        "desktop": [(800, "Budget"), (1500, "Mid-Range"), (2500, "Premium"), (float("inf"), "Enthusiast")],
        "tablet": [(250, "Budget"), (600, "Mid-Range"), (1000, "Premium"), (float("inf"), "Pro")],
        "smartwatch": [(200, "Budget"), (400, "Mid-Range"), (700, "Premium"), (float("inf"), "Luxury")],
        "camera": [(500, "Entry"), (1500, "Enthusiast"), (3000, "Professional"), (float("inf"), "Pro-Grade")],
    }
    tiers = thresholds.get(category, [(500, "Budget"), (1500, "Mid-Range"), (float("inf"), "Premium")])
    for threshold, label in tiers:
        if price <= threshold:
            return label
    return "Premium"


def _infer_ideal_for(prod: dict, price: float, rating: float, features_str: str) -> list:
    candidates = []
    if price < 500:
        candidates.append("budget conscious")
        candidates.append("student")
    elif price > 2000:
        candidates.append("premium buyer")
        candidates.append("professional")
    if "gaming" in features_str or "game" in features_str:
        candidates.append("gamer")
    if "business" in features_str or "office" in features_str:
        candidates.append("business user")
    if "travel" in features_str or "lightweight" in features_str:
        candidates.append("traveler")
    if rating >= 4.7:
        candidates.append("enthusiast")
    return candidates if candidates else ["general user"]
