"""
Agent 7 — Recommendation / AI Advisory Report Generator
========================================================
Role: Final analytical agent. Uses the selected LLM (Gemini 2.0 Flash in
      production) to generate a structured, premium-quality markdown advisory
      report from the top-5 scored products.

The prompt provides the LLM with:
  - User's original query
  - Budget constraint (with % of budget calculations)
  - Priority vector (sorted by importance)
  - Per-product: name, brand, price, rating, tier, 5-axis scores, key features,
    ideal_for list, description, Pareto status

Output Format (enforced via system prompt):
  ## APEX Recommendation — [Brief Context]
  *Analyzed N candidates | Priority: ... | Query: ...*
  ---
  ### #1 — [Product Name]
  [Headline: who this product is made for]
  **Why it wins for you:** [LLM reasoning connecting scores to user needs]
  **✅ Strengths:** [3 bullet points with specific spec references]
  **❌ Trade-offs:** [1-2 honest drawbacks]
  **💡 Best for:** [from ideal_for list]
  **💰 Value Verdict:** [price assessment vs budget]
  ---
  [Repeat for #2 and #3]
  ---
  ### ⚡ Strategic Summary
  [Clear winner, best value, sleeper pick, final verdict]

Fallback: If the LLM call fails for any reason, _build_fallback_recommendation()
is called. This deterministic function generates structured markdown using the
real product data from the scored_products list — NOT generic placeholder text.
"""
from state import AgentState
from llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def recommendation_agent(state: AgentState) -> AgentState:
    print("--- [AGENT] Recommendation ---")

    llm = get_llm(state.get("selected_llm", "local"), temperature=0.7)

    top_products = state.get("scored_products", [])[:5]
    budget = state.get("budget", {})
    budget_max = budget.get("max")
    budget_str = (
        f"${budget_max:,.0f}" if budget_max else "No strict budget — seeking best value"
    )
    critique_feedback = state.get("critique_feedback", "")
    critique_note = f"\n\n**Quality Audit Note:** {critique_feedback}" if critique_feedback and critique_feedback != "None" else ""

    # Build a structured product summary for the LLM
    product_summaries = []
    for i, p in enumerate(top_products):
        scores = p.get("criteria_scores", {})
        score_str = " | ".join(f"{k.capitalize()}: {v:.1f}/10" for k, v in scores.items())
        pareto_badge = " ✅ PARETO OPTIMAL" if p.get("is_pareto_optimal") else ""
        pct_of_budget = ""
        if budget_max and p.get("price"):
            pct = (p["price"] / budget_max) * 100
            pct_of_budget = f" ({pct:.0f}% of budget)"

        summary = (
            f"**Rank #{i+1}: {p.get('name')}** — {p.get('brand')}{pareto_badge}\n"
            f"  💰 Price: ${p.get('price', 0):,.0f}{pct_of_budget} | ⭐ Rating: {p.get('rating')}/5 | 📊 Overall: {p.get('overall_score', 0):.2f}/10\n"
            f"  🏷️  Tier: {p.get('value_tier', 'N/A')} | Category: {p.get('category')}\n"
            f"  📐 Scores → {score_str}\n"
            f"  🔧 Key Features: {', '.join(p.get('features', [])[:5])}\n"
            f"  🎯 Ideal For: {', '.join(p.get('ideal_for', [])[:4])}\n"
            f"  📝 {p.get('description', '')[:200]}"
        )
        product_summaries.append(summary)

    products_text = "\n\n".join(product_summaries)

    priority_vector = state.get("priority_vector", {})
    sorted_priorities = sorted(priority_vector.items(), key=lambda x: -x[1]) if priority_vector else []
    priorities_readable = " > ".join(f"{k} ({v*100:.0f}%)" for k, v in sorted_priorities[:3]) or "balanced"
    preferences_str = ", ".join(state.get("intent_preferences", [])[:6]) or "General quality and value"
    query = state.get("query", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are APEX — an elite AI Shopping Advisor, the equivalent of a personal concierge at a world-class technology boutique. You deliver breathtakingly precise, insightful, and empathetic recommendations.

**Strict Output Structure:**

## 🏆 APEX Recommendation — [Brief Context]

*Analyzed [N] candidates | Priority: [top priorities] | Query: [user's query]*

---

### 🥇 #1 — [Product Name]
**[Compelling headline: who this product is MADE for, in 8-12 words]**

**Why it wins for you:** [2-3 sentences connecting product strengths directly to the user's stated priorities and query. Be specific — reference actual specs, scores, and features.]

**✅ Strengths:**
- [Strength 1 — cite a specific spec or score]
- [Strength 2 — cite a specific spec or score]
- [Strength 3 — cite a specific spec or score]

**❌ Trade-offs:**
- [Honest drawback 1]
- [Honest drawback 2 if applicable]

**💡 Best for:** [Specific user types from ideal_for list]
**💰 Value Verdict:** [Price assessment vs budget/category — be specific. e.g., "At $1,299, this is 65% of your $2,000 budget — exceptional room to spare."]

---

[Repeat for #2 and #3 with same structure]

---

### ⚡ Strategic Summary

**Clear Winner:** [Name] — [1-sentence reason]
**Best Value:** [Name] — [1-sentence reason]
**Sleeper Pick:** [Name if applicable] — [1-sentence reason]

> **Final Verdict:** [2-3 sentences of direct, confident guidance. Tell the user exactly what to do.]{critique_note}

**Tone:** Warm, expert, confident — like a trusted advisor who has done this research for you personally. Never robotic. Always honest about trade-offs. Never over-promise."""),
        ("user", """**User Query:** {query}
**Budget:** {budget}
**Priority Axes:** {priorities}
**Extracted Preferences:** {preferences}

**Ranked Candidates (top {count}):**
{products}

Craft the premium APEX recommendation now.""")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    try:
        final_text = chain.invoke({
            "query":       query,
            "budget":      budget_str,
            "priorities":  priorities_readable,
            "preferences": preferences_str,
            "products":    products_text,
            "count":       len(top_products),
            "critique_note": critique_note,
        })
        state["final_recommendation"] = final_text
    except Exception as e:
        print(f"Recommendation Agent error: {e}")
        state["final_recommendation"] = _build_fallback_recommendation(top_products, query, budget_max)

    return state


def _build_fallback_recommendation(products: list, query: str, budget_max=None) -> str:
    """Deterministic fallback producing structured markdown when LLM is unavailable."""
    budget_str = f"${budget_max:,.0f}" if budget_max else "Open budget"
    lines = [
        f"## 🏆 APEX Recommendation",
        f"*Query: {query} | Budget: {budget_str}*\n",
        "---",
    ]

    dims = ["price", "performance", "brand", "durability", "innovation"]

    for i, p in enumerate(products[:3]):
        scores = p.get("criteria_scores", {})
        best_dim = max(scores, key=lambda k: scores.get(k, 0)) if scores else "performance"
        worst_dim = min(scores, key=lambda k: scores.get(k, 0)) if scores else "price"
        pct = ""
        if budget_max and p.get("price"):
            pct = f" ({(p['price']/budget_max*100):.0f}% of budget)"
        pareto = " ✅ Pareto Optimal" if p.get("is_pareto_optimal") else ""

        lines += [
            f"\n### {'🥇' if i==0 else '🥈' if i==1 else '🥉'} #{i+1} — {p.get('name')} ({p.get('brand')}){pareto}",
            f"**${p.get('price', 0):,.0f}{pct} | ⭐ {p.get('rating')}/5 | 📊 {p.get('overall_score', 0):.2f}/10 | {p.get('value_tier', '')}**",
            "",
            f"**Why it ranks #{i+1}:** Strongest in **{best_dim}** ({scores.get(best_dim, 0):.1f}/10). {p.get('description', '')[:180]}",
            "",
            f"**✅ Strengths:** {', '.join(p.get('features', [])[:3])}",
            f"**❌ Trade-off:** Lower {worst_dim} score ({scores.get(worst_dim, 0):.1f}/10) compared to alternatives.",
            f"**💡 Best for:** {', '.join(p.get('ideal_for', [])[:3])}",
            "",
            "---",
        ]

    # Strategic summary
    if products:
        top = products[0]
        lines += [
            "\n### ⚡ Strategic Summary",
            f"**Clear Winner:** {top.get('name')} — highest composite score ({top.get('overall_score', 0):.2f}/10) with {top.get('value_tier', 'strong')} positioning.",
        ]
        if len(products) > 1:
            runner = products[1]
            lines.append(f"**Runner-Up:** {runner.get('name')} — scores {runner.get('overall_score', 0):.2f}/10, ${runner.get('price', 0):,.0f}.")
        lines.append(f"\n> **Final Verdict:** For '{query}', {top.get('name')} is the standout choice based on multi-dimensional scoring. Review trade-offs above to confirm it aligns with your specific workflow.")

    return "\n".join(lines)
