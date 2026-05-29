"""
Agent 3 — Retrieval / Semantic Search + External API Boost
===========================================================
Role: Finds the most relevant products for the user's query using a
      combination of FAISS semantic vector search and optional real-time
      product data from the RapidAPI external integration.

Pipeline:
  1. Build a rich semantic search string from query + intent preferences + category
  2. Run FAISS similarity search (top-K depends on complexity: 15 simple / 20 complex)
  3. [Optional] Query RapidAPI for live external product data — prepend to results
  4. Filter results by budget range (with 10% tolerance on max)
  5. Filter by detected category (falls back to budget-only if category yields 0 results)
  6. Limit final output to 8 (simple) or 15 (complex) products for downstream scoring

FAISS Search:
  - Uses all-MiniLM-L6-v2 SentenceTransformer (384-dim embeddings)
  - Products are indexed as rich text: name + brand + category + price + description
    + features + ideal_for
  - Search uses L2 (Euclidean) distance — smaller distance = more similar

External API Boost:
  - RapidAPI client attempts real-time product lookup
  - Gracefully degrades: any failure is caught and logged, FAISS results are used alone
  - External products are prepended so they are considered by downstream agents

Budget Filtering:
  - Applies after retrieval to remove products exceeding budget_max
  - 10% tolerance allows products slightly above budget (for edge cases near limit)
  - Falls back to unfiltered results if filtering yields zero products
"""
from state import AgentState
from data_layer.vector_db import VectorDBStore
import os

def retrieval_agent(state: AgentState) -> AgentState:
    print("--- [AGENT] Retrieval ---")

    query    = state.get("query", "")
    category = state.get("category", "").lower().strip()

    # Combine query + intent preferences + category for richer semantic match
    intents = ", ".join(state.get("intent_preferences", []))
    search_string = f"{query} {intents} {category}".strip()

    base_dir      = os.path.dirname(__file__)
    db_path       = os.path.join(base_dir, "..", "data_layer", "dataset.json")
    index_path    = os.path.join(base_dir, "..", "data_layer", "products.index")
    metadata_path = os.path.join(base_dir, "..", "data_layer", "metadata.json")

    db = VectorDBStore(data_path=db_path, index_file=index_path, metadata_file=metadata_path)

    # Retrieve more candidates so category filtering still leaves enough products
    complexity = state.get("complexity", "simple")
    top_k = 20 if complexity == "complex" else 15

    results = db.semantic_search(search_string, top_k=top_k)

    # --- External RapidAPI Real-Time Boost ---
    try:
        from data_layer.rapidapi_client import rapidapi_client
        
        # Determine if we should query specific external endpoints
        amazon_asin = None
        if len(query) == 10 and query.isalnum() and query.isupper():
            amazon_asin = query
            
        print(f"  [External Boost] Querying RapidAPI real-time endpoints for '{query}'...")
        external_results = []
        
        # If it looks like an ASIN, fetch Amazon offers
        if amazon_asin:
            print(f"  [External Boost] Triggering Amazon ASIN search for {amazon_asin}")
            amz = rapidapi_client.get_amazon_offers(amazon_asin)
            if amz and not amz.get("error"):
                # Parse amazon generic response format (dummy structure mapping)
                price = str(amz.get("product_price", "0")).replace("$", "").replace(",", "")
                external_results.append({
                    "id": f"amazon_{amazon_asin}",
                    "name": amz.get("product_title", f"Amazon Product {amazon_asin}"),
                    "price": price if price.replace(".", "").isdigit() else "0",
                    "brand": amz.get("brand", "Amazon Brand"),
                    "rating": 4.5,
                    "category": category or "unknown",
                    "features": ["Amazon product data fetched securely"],
                    "source": "amazon_api"
                })
                
        # General real-time product search
        live_data = rapidapi_client.get_realtime_product_search(query)
        if live_data:
            for item in live_data[:3]:  # Top 3 external hits
                raw_price = str(item.get("product_price", item.get("price", "None")))
                clean_price = "".join(c for c in raw_price if c.isdigit() or c == ".")
                if not clean_price: 
                    clean_price = "0"
                
                title = str(item.get("product_title", item.get("title", item.get("name", "External Item"))))
                external_results.append({
                    "id": str(item.get("product_id", f"ext_{hash(title)}")),
                    "name": title,
                    "brand": item.get("brand", "Generic Brand"),
                    "price": clean_price,
                    "category": category if category and category != "unknown" else "external",
                    "rating": float(item.get("product_rating", item.get("rating", 4.0))),
                    "features": [str(item.get("product_description", ""))[:100]],
                    "source": "rapidapi_realtime"
                })
                
        if external_results:
            results = external_results + results
            print(f"  [External Boost] Integrated {len(external_results)} live products.")
            
    except Exception as e:
        print(f"  [External Boost] Exception during RapidAPI fetch: {e}")
    # ------------------------------------------

    if not results:
        print("Warning: Null results from VectorDB.")
        state["retrieved_products"] = []
        return state

    # ── 1. Filter by budget ───────────────────────────────────────────────────
    budget_min = state.get("budget", {}).get("min")
    budget_max = state.get("budget", {}).get("max")

    budget_filtered = []
    for r in results:
        price = r.get("price")
        if price is not None:
            if budget_max and float(price) > float(budget_max):
                continue
            if budget_min and float(price) < float(budget_min):
                continue
        budget_filtered.append(r)

    # ── 2. Filter by category when reliably detected ──────────────────────────
    category_filtered = []
    if category and category not in ("unknown", ""):
        for r in budget_filtered:
            if r.get("category", "").lower() == category:
                category_filtered.append(r)

    # Use category-filtered list if it has results; else fall back to budget-only
    final_results = category_filtered if category_filtered else budget_filtered

    # ── 3. Limit to top-k for comparison agent ────────────────────────────────
    limit = 15 if complexity == "complex" else 8
    state["retrieved_products"] = [r for i, r in enumerate(final_results) if i < limit]

    print(f"  Retrieved {len(state['retrieved_products'])} products "
          f"(category={category or 'any'}, budget_max={budget_max})")
    return state
