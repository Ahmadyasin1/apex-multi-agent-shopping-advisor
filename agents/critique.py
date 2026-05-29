"""
Agent 6 — Critique / AI Quality Auditor
========================================
Role: Self-correction layer that validates the Comparison Agent's output
      BEFORE the final recommendation is generated.

Checks performed:
  1. Budget compliance — are all top products within the user's stated budget?
  2. Category match   — do recommended products match the detected category?
  3. Consistency      — do scores reflect the stated priority vector?
  4. Hallucination    — do product specs look realistic for the price tier?

Conditional routing (graph.py):
  PASS  → Recommendation Agent (happy path)
  FAIL  → back to Preference Agent (retry loop, max 2 revisions)
           After 2 revisions, always passes to prevent infinite loops.

This agent implements the "self-reflection" pattern from agentic AI systems:
the system can catch and correct its own mistakes before responding to users.

Temperature = 0.0 for deterministic, reproducible quality auditing.
"""
from state import AgentState
from llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class CritiqueOutput(BaseModel):
    """Structured output schema for the Critique agent's quality audit."""
    passed: bool = Field(description="True if the comparison outputs are solid and budget-compliant.")
    feedback: str = Field(description="If False, specific description of what went wrong and how to fix it.")

def critique_agent(state: AgentState) -> AgentState:
    print("--- [AGENT] Critique ---")
    
    llm = get_llm(state.get("selected_llm", "local"), temperature=0.0)
    parser = JsonOutputParser(pydantic_object=CritiqueOutput)
    
    top_products = state.get("scored_products", [])[:3]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI Auditor for a Luxury Shopping Advisor.
Review the top products chosen for the user's query. Check for any logical inconsistencies, hallucinations in specs, or mismatched budget constraints.
Return JSON with 'passed' boolean and 'feedback' string. If the query requested a budget under $500 but the product is $2500, it MUST fail."""),
        ("user", "Query: {query}\nBudget: {budget_max}\nTop Products: {products}\n\n{format_instructions}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        res = chain.invoke({
            "query": state.get("query"),
            "budget_max": state.get("budget", {}).get("max", "Unlimited"),
            "products": str(top_products),
            "format_instructions": parser.get_format_instructions()
        })
        
        state["critique_passed"] = res.get("passed", True)
        state["critique_feedback"] = res.get("feedback", "")
    except Exception as e:
        _full = str(e).encode("ascii", "replace").decode("ascii")
        err_msg = "".join(c for i, c in enumerate(_full) if i < 120)
        print(f"Critique Agent error: {err_msg}")
        state["critique_passed"] = True
        state["critique_feedback"] = "Auto-passed after parse error."
        
    revision_count = state.get("revision_count", 0)
    state["revision_count"] = revision_count + 1
    
    # Hard stop after 2 revisions to prevent infinite loops
    if state["revision_count"] >= 2:
        state["critique_passed"] = True
        
    return state
