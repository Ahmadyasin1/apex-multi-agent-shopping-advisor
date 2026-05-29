"""
APEX — Intelligent Shopping Advisor
Formal Evaluation Script (Assignment Requirement: 50+ test queries)

Measures:
  - Accuracy        : product recommendations match expected category
  - Relevance       : budget constraint respected, category correct
  - Response Time   : end-to-end pipeline latency
  - Robustness      : edge cases (missing budget, conflicting constraints)

Usage:
    python evaluate.py              # run all 55 queries
    python evaluate.py --quick      # run first 10 for smoke test
    python evaluate.py --id 1,5,11  # run specific query IDs
    python evaluate.py --report     # generate CSV + JSON report
"""

import sys, os, time, json, argparse, csv
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Test Query Bank ─────────────────────────────────────────────────────────
TEST_QUERIES = [
    # ── LAPTOPS (10) ─────────────────────────────────────────────────────────
    {"id":  1, "query": "Best laptop under $500 for a college student",                   "expected_cat": "laptop",     "budget_max": 500,   "group": "laptop"},
    {"id":  2, "query": "Gaming laptop with RTX 4080 or better, budget $2500",            "expected_cat": "laptop",     "budget_max": 2500,  "group": "laptop"},
    {"id":  3, "query": "Ultralight laptop for business travel under $1500",               "expected_cat": "laptop",     "budget_max": 1500,  "group": "laptop"},
    {"id":  4, "query": "Best MacBook for professional video editing",                    "expected_cat": "laptop",     "budget_max": None,  "group": "laptop"},
    {"id":  5, "query": "Laptop for AI and machine learning research",                    "expected_cat": "laptop",     "budget_max": None,  "group": "laptop"},
    {"id":  6, "query": "Cheap laptop just for web browsing and email under $400",        "expected_cat": "laptop",     "budget_max": 400,   "group": "laptop"},
    {"id":  7, "query": "Best 2-in-1 convertible laptop for graphic designers",           "expected_cat": "laptop",     "budget_max": 2000,  "group": "laptop"},
    {"id":  8, "query": "Laptop with the best OLED display for media consumption",        "expected_cat": "laptop",     "budget_max": None,  "group": "laptop"},
    {"id":  9, "query": "I need a laptop for programming — no gaming, just coding $1200", "expected_cat": "laptop",     "budget_max": 1200,  "group": "laptop"},
    {"id": 10, "query": "Best laptop with longest battery life for remote work $1800",    "expected_cat": "laptop",     "budget_max": 1800,  "group": "laptop"},

    # ── SMARTPHONES (10) ─────────────────────────────────────────────────────
    {"id": 11, "query": "Best Android phone under $400",                                  "expected_cat": "smartphone", "budget_max": 400,   "group": "smartphone"},
    {"id": 12, "query": "iPhone with the best camera for photography",                    "expected_cat": "smartphone", "budget_max": None,  "group": "smartphone"},
    {"id": 13, "query": "Gaming phone with highest refresh rate under $1200",             "expected_cat": "smartphone", "budget_max": 1200,  "group": "smartphone"},
    {"id": 14, "query": "Smartphone with the best battery life",                          "expected_cat": "smartphone", "budget_max": None,  "group": "smartphone"},
    {"id": 15, "query": "Affordable 5G phone for a first-time buyer under $350",          "expected_cat": "smartphone", "budget_max": 350,   "group": "smartphone"},
    {"id": 16, "query": "Best phone for Leica quality photography under $1000",           "expected_cat": "smartphone", "budget_max": 1000,  "group": "smartphone"},
    {"id": 17, "query": "Phone with best AI features and clean software experience",      "expected_cat": "smartphone", "budget_max": None,  "group": "smartphone"},
    {"id": 18, "query": "Compact small-form-factor phone under $500",                     "expected_cat": "smartphone", "budget_max": 500,   "group": "smartphone"},
    {"id": 19, "query": "Flagship phone with S Pen stylus for note-taking",               "expected_cat": "smartphone", "budget_max": None,  "group": "smartphone"},
    {"id": 20, "query": "Most uniquely designed and distinctive smartphone under $700",   "expected_cat": "smartphone", "budget_max": 700,   "group": "smartphone"},

    # ── HEADPHONES (10) ──────────────────────────────────────────────────────
    {"id": 21, "query": "Best noise cancelling headphones for flights",                   "expected_cat": "headphones", "budget_max": None,  "group": "headphones"},
    {"id": 22, "query": "Budget wireless earbuds under $80",                              "expected_cat": "headphones", "budget_max": 80,    "group": "headphones"},
    {"id": 23, "query": "Best headphones for work from home video calls under $500",      "expected_cat": "headphones", "budget_max": 500,   "group": "headphones"},
    {"id": 24, "query": "Audiophile open-back headphones for home listening under $700",  "expected_cat": "headphones", "budget_max": 700,   "group": "headphones"},
    {"id": 25, "query": "Gaming headset with the best microphone quality under $400",     "expected_cat": "headphones", "budget_max": 400,   "group": "headphones"},
    {"id": 26, "query": "Best earbuds for iPhone users",                                  "expected_cat": "headphones", "budget_max": None,  "group": "headphones"},
    {"id": 27, "query": "Wireless headphones with 50+ hour battery under $150",          "expected_cat": "headphones", "budget_max": 150,   "group": "headphones"},
    {"id": 28, "query": "Premium luxury headphones — money is no object",                 "expected_cat": "headphones", "budget_max": None,  "group": "headphones"},
    {"id": 29, "query": "Best Samsung Galaxy earbuds for Android users under $250",      "expected_cat": "headphones", "budget_max": 250,   "group": "headphones"},
    {"id": 30, "query": "Waterproof earbuds for gym and running under $200",             "expected_cat": "headphones", "budget_max": 200,   "group": "headphones"},

    # ── DESKTOPS (5) ─────────────────────────────────────────────────────────
    {"id": 31, "query": "Best gaming desktop under $3000",                                "expected_cat": "desktop",    "budget_max": 3000,  "group": "desktop"},
    {"id": 32, "query": "Home office desktop for a family under $800",                    "expected_cat": "desktop",    "budget_max": 800,   "group": "desktop"},
    {"id": 33, "query": "Best iMac for home creative work under $2000",                   "expected_cat": "desktop",    "budget_max": 2000,  "group": "desktop"},
    {"id": 34, "query": "Workstation desktop for 3D rendering and video production",      "expected_cat": "desktop",    "budget_max": None,  "group": "desktop"},
    {"id": 35, "query": "Small form factor compact desktop PC under $1500",               "expected_cat": "desktop",    "budget_max": 1500,  "group": "desktop"},

    # ── TABLETS (5) ──────────────────────────────────────────────────────────
    {"id": 36, "query": "Best tablet for digital art and drawing with a stylus $1500",   "expected_cat": "tablet",     "budget_max": 1500,  "group": "tablet"},
    {"id": 37, "query": "Affordable tablet for kids and streaming under $200",            "expected_cat": "tablet",     "budget_max": 200,   "group": "tablet"},
    {"id": 38, "query": "iPad alternative for Android users under $1200",                 "expected_cat": "tablet",     "budget_max": 1200,  "group": "tablet"},
    {"id": 39, "query": "Windows tablet that runs full desktop software under $1800",     "expected_cat": "tablet",     "budget_max": 1800,  "group": "tablet"},
    {"id": 40, "query": "Best productivity tablet for note-taking at university $800",    "expected_cat": "tablet",     "budget_max": 800,   "group": "tablet"},

    # ── SMARTWATCHES (5) ─────────────────────────────────────────────────────
    {"id": 41, "query": "Best smartwatch for marathon runners and triathletes",           "expected_cat": "smartwatch", "budget_max": None,  "group": "smartwatch"},
    {"id": 42, "query": "Affordable smartwatch with heart rate and sleep tracking $200",  "expected_cat": "smartwatch", "budget_max": 200,   "group": "smartwatch"},
    {"id": 43, "query": "Apple Watch for comprehensive health monitoring",                "expected_cat": "smartwatch", "budget_max": None,  "group": "smartwatch"},
    {"id": 44, "query": "Smartwatch with the longest battery life",                       "expected_cat": "smartwatch", "budget_max": None,  "group": "smartwatch"},
    {"id": 45, "query": "Best Samsung smartwatch for Galaxy phone users under $400",      "expected_cat": "smartwatch", "budget_max": 400,   "group": "smartwatch"},

    # ── CAMERAS (5) ──────────────────────────────────────────────────────────
    {"id": 46, "query": "Best mirrorless camera for professional portrait photography",   "expected_cat": "camera",     "budget_max": None,  "group": "camera"},
    {"id": 47, "query": "Action camera for extreme sports and underwater use under $500", "expected_cat": "camera",     "budget_max": 500,   "group": "camera"},
    {"id": 48, "query": "Best camera for YouTube vlogging beginners under $800",          "expected_cat": "camera",     "budget_max": 800,   "group": "camera"},
    {"id": 49, "query": "Retro-style compact camera with film simulation under $2000",    "expected_cat": "camera",     "budget_max": 2000,  "group": "camera"},
    {"id": 50, "query": "Camera under $2500 for hybrid photo and video work",             "expected_cat": "camera",     "budget_max": 2500,  "group": "camera"},

    # ── EDGE CASES (10) ──────────────────────────────────────────────────────
    {"id": 51, "query": "I need the absolute best device ever created — no budget",       "expected_cat": None,         "budget_max": None,  "group": "edge"},
    {"id": 52, "query": "Something for gaming AND working AND traveling — all in one",    "expected_cat": "laptop",     "budget_max": None,  "group": "edge"},
    {"id": 53, "query": "Phone with incredible camera but extremely cheap under $200",    "expected_cat": "smartphone", "budget_max": 200,   "group": "edge"},
    {"id": 54, "query": "Headphones for total silence but also incredible music quality", "expected_cat": "headphones", "budget_max": None,  "group": "edge"},
    {"id": 55, "query": "asdfjkl; phone recommendation please",                           "expected_cat": "smartphone", "budget_max": None,  "group": "edge"},
    {"id": 56, "query": "Best product under $50",                                         "expected_cat": None,         "budget_max": 50,    "group": "edge"},
    {"id": 57, "query": "I want a laptop for gaming but also need it ultra light",        "expected_cat": "laptop",     "budget_max": None,  "group": "edge"},
    {"id": 58, "query": "Recommend something without telling me what it is",              "expected_cat": None,         "budget_max": None,  "group": "edge"},
    {"id": 59, "query": "What is the cheapest laptop with the best performance?",         "expected_cat": "laptop",     "budget_max": None,  "group": "edge"},
    {"id": 60, "query": "Noise cancelling headphones and a laptop for remote work",       "expected_cat": None,         "budget_max": None,  "group": "edge"},
]

# ── Evaluation Engine ───────────────────────────────────────────────────────
def evaluate_query(query_spec: dict, workflow) -> dict:
    """Run a single query through the pipeline and evaluate the result."""
    from state import AgentState
    from config import Config

    q           = query_spec["query"]
    budget_max  = query_spec.get("budget_max")
    expected_cat= query_spec.get("expected_cat")

    initial_state: AgentState = {
        "query":              q,
        "complexity":         "simple",
        "selected_llm":       "local",
        "category":           "",
        "budget":             {"min": None, "max": float(budget_max) if budget_max else None},
        "intent_preferences": [],
        "priority_vector":    Config.DEFAULT_PRIORITY_VECTOR.copy(),
        "retrieved_products": [],
        "enriched_products":  [],
        "scored_products":    [],
        "critique_passed":    False,
        "critique_feedback":  "",
        "revision_count":     0,
        "final_recommendation": "",
        "error":              None,
        "history":            [],
    }

    t0 = time.time()
    result = {"id": query_spec["id"], "query": q, "group": query_spec.get("group", "general"),
              "expected_cat": expected_cat, "budget_max": budget_max}

    try:
        events = list(workflow.stream(initial_state))
        elapsed = round(time.time() - t0, 2)

        final_state = None
        for event in events:
            for node, sv in event.items():
                final_state = sv

        if not final_state:
            raise ValueError("Pipeline returned no state")

        products    = final_state.get("scored_products", [])
        detected_cat= final_state.get("category", "")
        rec_text    = final_state.get("final_recommendation", "")
        n_products  = len(products)

        # ── Metrics ──────────────────────────────────────────────────────────
        # 1. Category accuracy: did we detect the right category?
        cat_correct = (
            True if expected_cat is None  # No expectation → can't fail
            else detected_cat.lower() == expected_cat.lower() if detected_cat
            else False
        )

        # 2. Budget compliance: are all returned products within budget?
        budget_violations = 0
        if budget_max and products:
            budget_violations = sum(
                1 for p in products if float(p.get("price", 0)) > budget_max * 1.1  # 10% tolerance
            )
        budget_compliant = budget_violations == 0

        # 3. Relevance: did we return at least 1 product?
        has_results = n_products > 0

        # 4. Recommendation quality: is the recommendation non-empty and structured?
        rec_quality = "good" if len(rec_text) > 200 else "minimal" if len(rec_text) > 50 else "empty"

        # 5. Top product score (primary recommendation quality signal)
        top_score = round(products[0].get("overall_score", 0), 2) if products else 0.0

        result.update({
            "status":              "pass",
            "elapsed_sec":         elapsed,
            "detected_category":   detected_cat,
            "category_correct":    cat_correct,
            "n_products":          n_products,
            "has_results":         has_results,
            "budget_compliant":    budget_compliant,
            "budget_violations":   budget_violations,
            "recommendation_len":  len(rec_text),
            "rec_quality":         rec_quality,
            "top_product":         products[0].get("name", "—") if products else "—",
            "top_price":           products[0].get("price", 0) if products else 0,
            "top_score":           top_score,
            "critique_passed":     final_state.get("critique_passed", False),
            "revision_count":      final_state.get("revision_count", 0),
            "model_used":          final_state.get("selected_llm", "unknown"),
            "error":               None,
        })

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        result.update({
            "status":           "error",
            "elapsed_sec":      elapsed,
            "detected_category": "",
            "category_correct": False,
            "n_products":       0,
            "has_results":      False,
            "budget_compliant": True,
            "budget_violations": 0,
            "recommendation_len": 0,
            "rec_quality":      "empty",
            "top_product":      "—",
            "top_price":        0,
            "top_score":        0.0,
            "critique_passed":  False,
            "revision_count":   0,
            "model_used":       "unknown",
            "error":            str(e)[:200],
        })

    return result


def print_result(r: dict, idx: int, total: int):
    status  = "✅ PASS" if r["status"] == "pass" else "❌ ERROR"
    cat_ok  = "✓" if r["category_correct"] else "✗"
    bgt_ok  = "✓" if r["budget_compliant"] else "✗"
    has_res = "✓" if r["has_results"] else "✗"
    print(f"[{idx:>3}/{total}] Q{r['id']:>3} [{r['group']:<10}] {status} "
          f"| {r['elapsed_sec']:>5.1f}s "
          f"| cat:{cat_ok} bgt:{bgt_ok} res:{has_res} "
          f"| top: {r['top_product'][:35]:<35} ${r['top_price']:.0f} "
          f"score:{r['top_score']:.2f}")
    if r["error"]:
        print(f"         ERROR: {r['error'][:120]}")


def print_summary(results: list):
    total = len(results)
    passed        = sum(1 for r in results if r["status"] == "pass")
    cat_correct   = sum(1 for r in results if r["category_correct"])
    budget_ok     = sum(1 for r in results if r["budget_compliant"])
    has_results   = sum(1 for r in results if r["has_results"])
    avg_time      = sum(r["elapsed_sec"] for r in results) / total if total else 0
    avg_score     = sum(r["top_score"] for r in results if r["top_score"]) / max(1, sum(1 for r in results if r["top_score"]))

    # Per-group breakdown
    groups = {}
    for r in results:
        g = r["group"]
        groups.setdefault(g, {"total": 0, "pass": 0})
        groups[g]["total"] += 1
        if r["status"] == "pass":
            groups[g]["pass"] += 1

    print("\n" + "═" * 80)
    print("APEX EVALUATION SUMMARY")
    print("═" * 80)
    print(f"  Queries Evaluated   : {total}")
    print(f"  Pipeline Success    : {passed}/{total}  ({passed/total*100:.1f}%)")
    print(f"  Category Accuracy   : {cat_correct}/{total}  ({cat_correct/total*100:.1f}%)")
    print(f"  Budget Compliance   : {budget_ok}/{total}  ({budget_ok/total*100:.1f}%)")
    print(f"  Has Results (≥1 rec): {has_results}/{total}  ({has_results/total*100:.1f}%)")
    print(f"  Avg Response Time   : {avg_time:.2f}s")
    print(f"  Avg Top Score       : {avg_score:.2f}/10")
    print()
    print("  Per-Category Breakdown:")
    for g, d in sorted(groups.items()):
        bar = "█" * d["pass"] + "░" * (d["total"] - d["pass"])
        print(f"    {g:<12} {bar}  {d['pass']}/{d['total']}")
    print("═" * 80)


def save_report(results: list, outdir: str = "."):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(outdir) / f"evaluation_results_{ts}.json"
    csv_path  = Path(outdir) / f"evaluation_results_{ts}.csv"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "total": len(results), "results": results}, f, indent=2)

    # CSV
    if results:
        fields = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)

    print(f"\n  Report saved:")
    print(f"    JSON → {json_path}")
    print(f"    CSV  → {csv_path}")
    return str(json_path), str(csv_path)


# ── CLI Entry Point ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="APEX Evaluation Runner")
    parser.add_argument("--quick",  action="store_true", help="Run only first 10 queries")
    parser.add_argument("--id",     type=str, default="",  help="Comma-separated query IDs to run (e.g. 1,5,21)")
    parser.add_argument("--report", action="store_true", help="Save JSON + CSV report")
    parser.add_argument("--group",  type=str, default="",  help="Filter by group: laptop, smartphone, edge, …")
    args = parser.parse_args()

    # Import here so the script itself loads fast
    from graph import create_workflow
    workflow = create_workflow()

    # Select queries
    queries = TEST_QUERIES[:]
    if args.id:
        ids = {int(x.strip()) for x in args.id.split(",") if x.strip().isdigit()}
        queries = [q for q in queries if q["id"] in ids]
    elif args.quick:
        queries = queries[:10]
    if args.group:
        queries = [q for q in queries if q["group"] == args.group]

    print(f"\n{'═'*80}")
    print(f"  APEX Intelligent Shopping Advisor — Evaluation")
    print(f"  Queries: {len(queries)} | Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'═'*80}\n")

    results = []
    for i, q_spec in enumerate(queries, 1):
        r = evaluate_query(q_spec, workflow)
        results.append(r)
        print_result(r, i, len(queries))

    print_summary(results)
    if args.report:
        save_report(results)

    # Exit code 1 if >50% failure rate
    fail_rate = sum(1 for r in results if r["status"] == "error") / max(1, len(results))
    sys.exit(1 if fail_rate > 0.5 else 0)


if __name__ == "__main__":
    main()
