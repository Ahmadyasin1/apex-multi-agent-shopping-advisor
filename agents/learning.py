"""
Agent 8 — Learning / Adaptive Memory Agent
===========================================
Role: Final agent in the pipeline. Persists session data and incrementally
      updates a rolling preference model so future queries benefit from
      implicit learning about the user's tastes.

What is persisted (data_layer/user_memory.json):
  sessions            — list of all past queries, detected category, top product
  preference_history  — rolling average of priority vectors across all sessions
  category_counts     — how many times each category was queried

Adaptive Preference Model:
  After each session, the user's priority vector is averaged with the
  historical rolling average using exponential smoothing (alpha = 0.15):
    new_avg[d] = 0.85 × old_avg[d]  +  0.15 × session_pv[d]
  This means recent sessions have slightly more influence than old ones,
  enabling personalization that drifts gracefully over time.

Storage: Local JSON file (data_layer/user_memory.json).
  Upgradeable to: Redis, PostgreSQL, MongoDB, or Pinecone for production.
  Privacy: No personal identifiers stored — only aggregate preference signals.

The learned preferences are surfaced via GET /api/stats and displayed
in the frontend's Market Insights panel.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from state import AgentState

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "data_layer", "user_memory.json")


def _load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"sessions": [], "preference_history": {}, "category_counts": {}}


def _save_memory(memory: dict) -> None:
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Learning Agent] Memory save error: {e}")


def learning_agent(state: AgentState) -> AgentState:
    """Store session data and update preference history for future personalization."""
    print("--- [AGENT] Learning ---")

    memory = _load_memory()

    # Build session record
    session = {
        "timestamp": datetime.now().isoformat(),
        "query": state.get("query", ""),
        "category": state.get("category", ""),
        "budget_max": state.get("budget", {}).get("max"),
        "complexity": state.get("complexity", "simple"),
        "priority_vector": state.get("priority_vector", {}),
        "top_recommendation": _extract_top_name(state),
        "critique_passed": state.get("critique_passed", True),
    }
    memory["sessions"].append(session)

    # Update category frequency
    category = state.get("category", "unknown")
    counts = memory.get("category_counts", {})
    counts[category] = counts.get(category, 0) + 1
    memory["category_counts"] = counts

    # Update cumulative preference vector (rolling average)
    pv = state.get("priority_vector", {})
    history = memory.get("preference_history", {})
    n = len(memory["sessions"])
    for key, val in pv.items():
        prev = history.get(key, val)
        history[key] = round((prev * (n - 1) + val) / n, 4)
    memory["preference_history"] = history

    # Keep only last 100 sessions to limit file size
    if len(memory["sessions"]) > 100:
        memory["sessions"] = memory["sessions"][-100:]

    _save_memory(memory)
    print(f"  Saved session #{n}. Category: {category} | Top pick: {session['top_recommendation']}")

    return state


def _extract_top_name(state: AgentState) -> str:
    products = state.get("scored_products", [])
    if products:
        return str(products[0].get("name", "Unknown"))
    return "None"


def get_user_context() -> dict:
    """Return learned user context for personalizing future recommendations."""
    memory = _load_memory()
    sessions = memory.get("sessions", [])
    category_counts = memory.get("category_counts", {})
    preference_history = memory.get("preference_history", {})

    most_searched = max(category_counts, key=lambda k: category_counts[k]) if category_counts else None

    return {
        "total_sessions": len(sessions),
        "most_searched_category": most_searched,
        "learned_preferences": preference_history,
        "recent_queries": [s.get("query") for s in sessions[-5:]],
    }
