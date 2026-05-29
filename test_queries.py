"""
50 Comprehensive Test Queries for the Intelligent Shopping Advisor
Covers: normal cases, edge cases, conflicting preferences, missing budget, multi-constraint
"""
from graph import create_workflow
from state import AgentState
import json, time

TEST_QUERIES = [
    # ── LAPTOPS ──────────────────────────────────────────────────────────────
    {"id": 1,  "query": "Best laptop under $500 for a college student",                      "category": "laptop",      "budget_max": 500},
    {"id": 2,  "query": "Gaming laptop with RTX 4080 or better, budget $2500",               "category": "laptop",      "budget_max": 2500},
    {"id": 3,  "query": "Ultralight laptop for business travel under $1500",                  "category": "laptop",      "budget_max": 1500},
    {"id": 4,  "query": "Best MacBook for video editing",                                    "category": "laptop",      "budget_max": None},
    {"id": 5,  "query": "Laptop for AI and machine learning research",                       "category": "laptop",      "budget_max": None},
    {"id": 6,  "query": "Cheap laptop just for web browsing and email",                      "category": "laptop",      "budget_max": 400},
    {"id": 7,  "query": "Best 2-in-1 convertible laptop for designers",                      "category": "laptop",      "budget_max": 2000},
    {"id": 8,  "query": "Laptop with the best OLED display",                                 "category": "laptop",      "budget_max": None},
    {"id": 9,  "query": "I need a laptop for programming — no gaming, just coding",          "category": "laptop",      "budget_max": 1200},
    {"id": 10, "query": "Best laptop with longest battery life for remote work",             "category": "laptop",      "budget_max": 1800},

    # ── SMARTPHONES ──────────────────────────────────────────────────────────
    {"id": 11, "query": "Best Android phone under $400",                                     "category": "smartphone",  "budget_max": 400},
    {"id": 12, "query": "iPhone with best camera for photography",                           "category": "smartphone",  "budget_max": None},
    {"id": 13, "query": "Gaming phone with highest refresh rate",                            "category": "smartphone",  "budget_max": 1200},
    {"id": 14, "query": "Best smartphone for battery life",                                  "category": "smartphone",  "budget_max": None},
    {"id": 15, "query": "Affordable 5G phone for a first-time smartphone buyer",             "category": "smartphone",  "budget_max": 350},
    {"id": 16, "query": "Best phone for Leica quality photography",                          "category": "smartphone",  "budget_max": 1000},
    {"id": 17, "query": "Phone with best AI features and clean software experience",         "category": "smartphone",  "budget_max": None},
    {"id": 18, "query": "Compact small-form-factor phone",                                   "category": "smartphone",  "budget_max": 500},
    {"id": 19, "query": "Flagship phone with S Pen stylus",                                  "category": "smartphone",  "budget_max": None},
    {"id": 20, "query": "Most unique and distinctive designed smartphone",                   "category": "smartphone",  "budget_max": 700},

    # ── HEADPHONES ───────────────────────────────────────────────────────────
    {"id": 21, "query": "Best noise cancelling headphones for flights",                      "category": "headphones",  "budget_max": None},
    {"id": 22, "query": "Budget wireless earbuds under $80",                                 "category": "headphones",  "budget_max": 80},
    {"id": 23, "query": "Best headphones for work from home video calls",                    "category": "headphones",  "budget_max": 500},
    {"id": 24, "query": "Audiophile open-back headphones for home listening",                "category": "headphones",  "budget_max": 700},
    {"id": 25, "query": "Gaming headset with the best mic quality",                          "category": "headphones",  "budget_max": 400},
    {"id": 26, "query": "Best earbuds for iPhone users",                                     "category": "headphones",  "budget_max": None},
    {"id": 27, "query": "Wireless headphones with 50+ hour battery",                        "category": "headphones",  "budget_max": 150},
    {"id": 28, "query": "Premium luxury headphones — money is no object",                    "category": "headphones",  "budget_max": None},
    {"id": 29, "query": "Best Samsung Galaxy earbuds for Android",                          "category": "headphones",  "budget_max": 250},
    {"id": 30, "query": "Waterproof earbuds for gym and running",                           "category": "headphones",  "budget_max": 200},

    # ── DESKTOPS ─────────────────────────────────────────────────────────────
    {"id": 31, "query": "Best gaming desktop under $3000",                                   "category": "desktop",     "budget_max": 3000},
    {"id": 32, "query": "Home office desktop for a family",                                  "category": "desktop",     "budget_max": 800},
    {"id": 33, "query": "Best iMac for home creative work",                                  "category": "desktop",     "budget_max": 2000},
    {"id": 34, "query": "Workstation for 3D rendering and video production",                 "category": "desktop",     "budget_max": None},
    {"id": 35, "query": "Small form factor compact desktop PC",                              "category": "desktop",     "budget_max": 1500},

    # ── TABLETS ──────────────────────────────────────────────────────────────
    {"id": 36, "query": "Best tablet for digital art and drawing with a stylus",             "category": "tablet",      "budget_max": 1500},
    {"id": 37, "query": "Affordable tablet for kids and Netflix",                            "category": "tablet",      "budget_max": 200},
    {"id": 38, "query": "iPad alternative for Android users",                                "category": "tablet",      "budget_max": 1200},
    {"id": 39, "query": "Windows tablet that runs full desktop software",                    "category": "tablet",      "budget_max": 1800},
    {"id": 40, "query": "Best productivity tablet for note-taking in university",            "category": "tablet",      "budget_max": 800},

    # ── SMARTWATCHES ─────────────────────────────────────────────────────────
    {"id": 41, "query": "Best smartwatch for marathon runners and triathletes",              "category": "smartwatch",  "budget_max": None},
    {"id": 42, "query": "Affordable smartwatch with heart rate and sleep tracking",          "category": "smartwatch",  "budget_max": 200},
    {"id": 43, "query": "Apple Watch for health monitoring",                                 "category": "smartwatch",  "budget_max": None},
    {"id": 44, "query": "Smartwatch with longest battery life",                              "category": "smartwatch",  "budget_max": 1200},
    {"id": 45, "query": "Best Samsung smartwatch for Galaxy phone users",                    "category": "smartwatch",  "budget_max": 400},

    # ── CAMERAS ──────────────────────────────────────────────────────────────
    {"id": 46, "query": "Best mirrorless camera for professional portrait photography",      "category": "camera",      "budget_max": None},
    {"id": 47, "query": "Action camera for extreme sports and underwater use",               "category": "camera",      "budget_max": 500},
    {"id": 48, "query": "Best camera for YouTube vlogging beginners",                        "category": "camera",      "budget_max": 800},
    {"id": 49, "query": "Retro-style compact camera with film simulation",                   "category": "camera",      "budget_max": 2000},
    {"id": 50, "query": "Camera under $2500 for hybrid photo and video work",               "category": "camera",      "budget_max": 2500},

    # ── EDGE CASES ────────────────────────────────────────────────────────────
    {"id": 51, "query": "I need the best device ever created, no budget",                   "category": "unknown",     "budget_max": None},
    {"id": 52, "query": "Something for gaming AND working AND traveling light",             "category": "laptop",      "budget_max": None},
    {"id": 53, "query": "Give me a phone that has incredible camera but is very cheap",     "category": "smartphone",  "budget_max": 200},
    {"id": 54, "query": "Headphones — I need total silence but also great music quality",   "category": "headphones",  "budget_max": None},
    {"id": 55, "query": "asdfjkl; phone",                                                   "category": "smartphone",  "budget_max": None},
]


def run_single_test(query_obj: dict, workflow) -> dict:
    """Run a single test query through the full pipeline."""
    initial_state: AgentState = {
        "query": query_obj["query"],
        "complexity": "simple",
        "selected_llm": "local",
        "category": "",
        "budget": {"min": None, "max": query_obj.get("budget_max")},
        "intent_preferences": [],
        "priority_vector": {"price": 0.2, "performance": 0.2, "brand": 0.2, "durability": 0.2, "innovation": 0.2},
        "retrieved_products": [],
        "enriched_products": [],
        "scored_products": [],
        "critique_passed": False,
        "critique_feedback": "",
        "revision_count": 0,
        "final_recommendation": "",
        "error": None,
        "history": []
    }

    start = time.time()
    final_step = None
    try:
        for event in workflow.stream(initial_state):
            for _, state in event.items():
                final_step = state
        elapsed = round(time.time() - start, 2)

        rec = final_step.get("final_recommendation", "") if final_step else ""
        products = final_step.get("scored_products", []) if final_step else []

        return {
            "id": query_obj["id"],
            "query": query_obj["query"],
            "status": "PASS" if rec and len(rec) > 50 else "FAIL",
            "products_found": len(products),
            "top_product": products[0].get("name") if products else "None",
            "response_time_s": elapsed,
            "critique_passed": final_step.get("critique_passed") if final_step else False,
            "recommendation_length": len(rec),
        }
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        return {
            "id": query_obj["id"],
            "query": query_obj["query"],
            "status": "ERROR",
            "error": str(e),
            "response_time_s": elapsed,
            "products_found": 0,
        }


def run_all_tests(max_queries: int = 10):
    """
    Run evaluation on test queries.
    max_queries: limit for quick testing (use len(TEST_QUERIES) for full run).
    """
    print(f"\n{'='*60}")
    print("  INTELLIGENT SHOPPING ADVISOR — TEST EVALUATION SUITE")
    print(f"{'='*60}\n")

    workflow = create_workflow()
    results = []

    queries_to_run = TEST_QUERIES[:max_queries]
    for q in queries_to_run:
        print(f"[Test {q['id']:02d}] {q['query'][:60]}...")
        result = run_single_test(q, workflow)
        results.append(result)
        status_icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"  {status_icon} Status: {result['status']} | Products: {result.get('products_found', 0)} | Time: {result.get('response_time_s', 0)}s\n")

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    avg_time = sum(r.get("response_time_s", 0) for r in results) / len(results) if results else 0
    avg_products = sum(r.get("products_found", 0) for r in results) / len(results) if results else 0

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY ({len(results)} queries)")
    print(f"{'='*60}")
    print(f"  ✅ PASSED:        {passed}")
    print(f"  ❌ FAILED:        {failed}")
    print(f"  💥 ERRORS:        {errors}")
    print(f"  ⏱  Avg Response:  {avg_time:.2f}s")
    print(f"  📦 Avg Products:  {avg_products:.1f}")
    print(f"  🎯 Accuracy:      {passed/len(results)*100:.1f}%")
    print(f"{'='*60}\n")

    # Save report
    report_path = "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {"passed": passed, "failed": failed, "errors": errors,
                               "avg_time": avg_time}, "results": results}, f, indent=2)
    print(f"  Full report saved to: {report_path}")

    return results


if __name__ == "__main__":
    # Run first 10 queries for quick validation; change to len(TEST_QUERIES) for full suite
    run_all_tests(max_queries=10)
