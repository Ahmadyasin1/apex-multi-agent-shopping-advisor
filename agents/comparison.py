"""
Agent 5 — Comparison / Multi-Dimensional Scoring Engine
========================================================
Role: Transforms the enriched product list into a ranked, scored output
      aligned to the user's stated priority vector.

Scoring Formula (for each product):
  overall_score = Σ (dimension_score[d] × priority_vector[d])   for d in 5 dimensions

The 5 Scoring Dimensions:
  1. price        — inverse-price score; budget-relative if budget specified
  2. performance  — rating-based + spec keyword bonuses (RTX 4090, M3 Max, etc.)
  3. brand        — tier lookup in PREMIUM_BRANDS; top-4 brand → 9.0/10
  4. durability   — rating × 1.8 + bonuses for MIL-SPEC, IP68, titanium
  5. innovation   — keyword scan for AI, OLED, LDAC, Neural Engine, etc.

Pareto Optimality Analysis:
  A product is Pareto-optimal if no other product is at least as good
  in ALL dimensions AND strictly better in at least one. Pareto-optimal
  products are "honest" choices — you can't improve one dimension without
  sacrificing another. Flagged as `is_pareto_optimal` in output.

  Algorithm: O(n²) pairwise dominance check — acceptable for n ≤ 20 products.

All scoring is deterministic (no LLM) — prevents hallucination and ensures
reproducible, auditable results.
"""
from state import AgentState

# Premium brand tiers per category — affects brand dimension score.
# Listed roughly in descending brand-prestige order within each category.
PREMIUM_BRANDS = {
    "laptop": ["Apple", "Dell", "Razer", "ASUS", "Sony", "HP", "Lenovo", "Microsoft"],
    "smartphone": ["Apple", "Samsung", "Google", "Sony", "OnePlus"],
    "headphones": ["Sony", "Bose", "Sennheiser", "Apple", "Jabra", "Bang & Olufsen"],
    "desktop": ["Apple", "ASUS", "Dell", "HP"],
    "tablet": ["Apple", "Samsung", "Microsoft"],
    "smartwatch": ["Apple", "Garmin", "Samsung"],
    "camera": ["Sony", "Canon", "Leica", "Nikon", "Fujifilm"],
    "default": ["Apple", "Sony", "Samsung", "Dell", "ASUS", "Bose", "Google", "Microsoft"]
}

def is_pareto_dominant(score1, score2):
    """Returns True if score1 strictly dominates score2 across all metrics."""
    win = False
    for k in score1.keys():
        if score1[k] < score2[k]:
            return False
        if score1[k] > score2[k]:
            win = True
    return win

def calculate_pareto_front(products_with_scores):
    front = []
    for i, p1 in enumerate(products_with_scores):
        dominated = False
        for j, p2 in enumerate(products_with_scores):
            if i != j:
                if is_pareto_dominant(p2['criteria_scores'], p1['criteria_scores']):
                    dominated = True
                    break
        if not dominated:
            front.append(p1.get('id', i))
    return front

def score_product(product, priorities, budget_max=None, category="default"):
    """Multi-dimensional weighted scoring aligned to user priorities."""
    price = float(product.get('price', 1000))
    rating = float(product.get('rating', 3.0))
    brand = product.get('brand', '')
    features = product.get('features', [])
    specs = product.get('specs', {}) or {}

    # --- Price Score (0-10): Cheaper relative to budget = higher score ---
    if budget_max and budget_max > 0:
        # How well does price fit within budget?
        ratio = price / budget_max
        if ratio <= 0.5:
            price_score = 10.0
        elif ratio <= 0.75:
            price_score = 8.0
        elif ratio <= 1.0:
            price_score = 6.0
        else:
            price_score = max(0.0, 10.0 - (ratio * 5))
    else:
        # Without budget: score inversely by price tier
        if price < 300:
            price_score = 9.5
        elif price < 700:
            price_score = 8.5
        elif price < 1200:
            price_score = 7.0
        elif price < 2000:
            price_score = 5.5
        elif price < 3500:
            price_score = 4.0
        else:
            price_score = 2.5

    # --- Performance Score (0-10): Based on rating ---
    perf_score = min(10.0, rating * 2.0)

    # Bonus for high-performance specs
    features_str = " ".join(features).lower()
    specs_str = str(specs).lower()
    combined_str = features_str + " " + specs_str

    if any(kw in combined_str for kw in ["rtx 4090", "m3 max", "m2 ultra", "i9", "ryzen 9", "a18 pro", "snapdragon 8 gen 3"]):
        perf_score = min(10.0, perf_score + 1.5)
    elif any(kw in combined_str for kw in ["rtx 4080", "rtx 4070", "m3", "i7", "ryzen 7", "a17 pro", "tensor g4"]):
        perf_score = min(10.0, perf_score + 0.8)

    # --- Brand Score (0-10) ---
    premium_list = list(PREMIUM_BRANDS.get(category.lower(), PREMIUM_BRANDS["default"]))
    top_tier = [b for i, b in enumerate(premium_list) if i < 4]
    if brand in top_tier:      # Top-tier brands
        brand_score = 9.0
    elif brand in premium_list: # Premium brands
        brand_score = 7.5
    else:
        brand_score = 5.0

    # --- Durability Score (0-10) ---
    durability_score = rating * 1.8  # Base from rating
    if any(kw in combined_str for kw in ["mil-spec", "ip68", "ip67", "titanium", "sapphire", "gorilla glass", "water resistant"]):
        durability_score = min(10.0, durability_score + 1.5)
    elif any(kw in combined_str for kw in ["aluminum", "metal", "magnesium", "rugged"]):
        durability_score = min(10.0, durability_score + 0.8)

    # --- Innovation Score (0-10) ---
    innovation_score = 5.0
    innovation_keywords = ["ai", "neural", "generative", "adaptive", "smart", "copilot", "gemini", "m3", "a18", "tensor g4", "oled", "amoled", "mini-led", "solar"]
    count = sum(1 for kw in innovation_keywords if kw in combined_str)
    innovation_score = min(10.0, 5.0 + count * 0.8)

    scores = {
        "price": round(price_score, 2),
        "performance": round(perf_score, 2),
        "brand": round(brand_score, 2),
        "durability": round(min(10.0, durability_score), 2),
        "innovation": round(innovation_score, 2)
    }

    # Weighted overall score
    overall = sum(scores.get(k, 0) * priorities.get(k, 0.2) for k in scores.keys())
    return scores, round(overall, 3)

def comparison_agent(state: AgentState) -> AgentState:
    print("--- [AGENT] Comparison ---")
    enriched = state.get("enriched_products", [])
    if not enriched:
        print("No products to compare.")
        state["scored_products"] = []
        return state

    priorities = state.get("priority_vector", {
        "price": 0.2, "performance": 0.2, "brand": 0.2, "durability": 0.2, "innovation": 0.2
    })
    budget_max = state.get("budget", {}).get("max")
    category = state.get("category", "default")

    scored_products = []
    for p in enriched:
        new_p = p.copy()
        c_scores, overall = score_product(p, priorities, budget_max=budget_max, category=category)
        new_p['criteria_scores'] = c_scores
        new_p['overall_score'] = overall
        scored_products.append(new_p)

    # Sort by overall score descending
    scored_products = sorted(scored_products, key=lambda x: x['overall_score'], reverse=True)

    # Calculate pareto front
    pareto_front = calculate_pareto_front(scored_products)
    for p in scored_products:
        p['is_pareto_optimal'] = p.get('id') in pareto_front

    state["scored_products"] = scored_products
    return state
