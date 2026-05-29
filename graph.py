"""
LangGraph Pipeline — 8-Agent StateGraph Definition
===================================================
Builds and compiles the LangGraph StateGraph that orchestrates all 8
agents of the APEX Intelligent Shopping Advisor.

Graph Topology (linear with conditional self-correction):

  supervisor → preference → retrieval → enrichment → comparison → critique
                                                                      │
                       ┌─────────────── FAIL (≤ 2 retries) ──────────┘
                       │
                       ▼
                  preference  (retry loop re-extracts intent with feedback)
                       │
               ── PASS (or revision_count ≥ 2) ──▶ recommendation → learning → END

Key Design Decisions:
  - Linear pipeline for clarity and debuggability
  - Single conditional edge (critique_router) enables self-correction
  - Hard cap at 2 revisions prevents infinite retry loops
  - Graph is compiled once and reused — thread-safe for concurrent WebSocket users
  - Streaming via workflow.stream() sends per-node events for real-time UI updates

Usage:
    from graph import create_workflow
    workflow = create_workflow()

    # Streaming (WebSocket handler in backend/main.py):
    for event in workflow.stream(initial_state):
        for node_name, state in event.items():
            send_to_websocket(node_name, state)

    # Blocking (evaluation script):
    final_state = workflow.invoke(initial_state)
"""
from langgraph.graph import StateGraph, END
from state import AgentState
from agents.supervisor import supervisor_agent
from agents.preference import preference_agent
from agents.retrieval import retrieval_agent
from agents.enrichment import enrichment_agent
from agents.comparison import comparison_agent
from agents.critique import critique_agent
from agents.recommendation import recommendation_agent
from agents.learning import learning_agent


def create_workflow():
    """
    Build and compile the 8-agent LangGraph StateGraph.

    Returns:
        CompiledGraph: A compiled, executable LangGraph pipeline.
                       Call .stream() for real-time or .invoke() for blocking execution.
    """
    workflow = StateGraph(AgentState)

    # ── Register all 8 agent nodes ────────────────────────────────────────────
    workflow.add_node("supervisor",     supervisor_agent)    # Agent 1: routing
    workflow.add_node("preference",     preference_agent)    # Agent 2: intent modeling
    workflow.add_node("retrieval",      retrieval_agent)     # Agent 3: FAISS + API search
    workflow.add_node("enrichment",     enrichment_agent)    # Agent 4: spec normalization
    workflow.add_node("comparison",     comparison_agent)    # Agent 5: Pareto scoring
    workflow.add_node("critique",       critique_agent)      # Agent 6: quality audit
    workflow.add_node("recommendation", recommendation_agent) # Agent 7: LLM report
    workflow.add_node("learning",       learning_agent)      # Agent 8: adaptive memory

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("supervisor")

    # ── Linear pipeline edges ─────────────────────────────────────────────────
    workflow.add_edge("supervisor",  "preference")
    workflow.add_edge("preference",  "retrieval")
    workflow.add_edge("retrieval",   "enrichment")
    workflow.add_edge("enrichment",  "comparison")
    workflow.add_edge("comparison",  "critique")

    # ── Conditional critique routing (self-correction loop) ───────────────────
    def critique_router(state: AgentState) -> str:
        """
        Route after Critique Agent:
          - If critique passed OR max revisions reached → proceed to Recommendation
          - If critique failed AND revisions remaining  → retry from Preference
        """
        passed         = state.get("critique_passed", True)
        revision_count = state.get("revision_count",  0)

        if passed or revision_count >= 2:
            return "recommendation"
        return "retry_loop"   # sends back to Preference agent

    workflow.add_conditional_edges(
        "critique",
        critique_router,
        {
            "recommendation": "recommendation",
            "retry_loop":     "preference",   # Re-extract intent with critique feedback
        }
    )

    # ── Post-recommendation edges ─────────────────────────────────────────────
    workflow.add_edge("recommendation", "learning")
    workflow.add_edge("learning",        END)

    return workflow.compile()
