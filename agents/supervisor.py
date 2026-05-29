"""
Agent 1 — Supervisor / Master Orchestrator
==========================================
Role: Entry-point agent that analyses the raw user query and makes two
      critical routing decisions BEFORE any retrieval or scoring happens:

  1. Complexity classification ('simple' vs 'complex')
     - 'simple'  → clear, explicit request (e.g., "cheap laptop under $500")
     - 'complex' → multi-constraint, trade-off-heavy, or ambiguous request

  2. LLM tier selection ('local' vs 'premium')
     - 'local'   → Ollama / MockLLM (fast, free, good for simple tasks)
     - 'premium' → Gemini 2.0 Flash / 1.5 Pro (deep reasoning, trade-off analysis)

  3. Primary category detection (laptop, smartphone, headphones, …)

The supervisor's output sets `complexity`, `selected_llm`, and `category`
in the shared AgentState, which all downstream agents read.

Design decision: deliberately uses the cheapest LLM for routing — the
supervisor itself is a fast O(1) classification task, so a local model is
always used here regardless of downstream LLM selection.
"""
from state import AgentState
from llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class SupervisorOutput(BaseModel):
    """Structured output schema for the Supervisor agent's JSON response."""
    complexity: str = Field(description="Either 'simple' or 'complex'")
    selected_llm: str = Field(description="Either 'local' for budget/simple or 'premium' for complex reasoning")
    category: str = Field(description="The primary category of the product, e.g. 'laptop', 'headphones', 'desktop'")

def supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor Agent — query analysis and pipeline routing.

    Reads:
        state['query'] — the raw user query string

    Writes:
        state['complexity']    — 'simple' | 'complex'
        state['selected_llm']  — 'local'  | 'premium'
        state['category']      — canonical product category

    Fallback: if the LLM call fails, defaults to simple/local/unknown so the
    pipeline always continues without breaking.
    """
    print("--- [AGENT] Supervisor ---")
    query = state.get("query", "")

    # Always use the lightweight local model for this routing step.
    # Temperature=0 → deterministic, reproducible classification.
    llm = get_llm("local", temperature=0.0)
    
    parser = JsonOutputParser(pydantic_object=SupervisorOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Master Orchestrator for an Intelligent Shopping Advisor.
Analyze the user's query and determine the complexity and required LLM capability.
Rules:
- 'simple': clear explicit requests (e.g., "cheap laptop", "noise cancelling headphones under 200"). Route to 'local'.
- 'complex': implicit needs, nuanced trade-offs, multiple conflicting constraints (e.g., "I need a device for heavy 3D rendering but I travel a lot, keeping it under $2000"). Route to 'premium'.

You must also detect the product category from the query (e.g., laptop, desktop, headphones). Return in lowercase.

Output JSON exactly matching the requested format."""),
        ("user", "Query: {query}\n\n{format_instructions}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "query": query,
            "format_instructions": parser.get_format_instructions()
        })
        
        state["complexity"] = result.get("complexity", "simple")
        state["selected_llm"] = result.get("selected_llm", "local")
        state["category"] = result.get("category", "unknown")
    except Exception as e:
        print(f"Supervisor parsing error: {e}")
        state["complexity"] = "simple"
        state["selected_llm"] = "local"
        state["category"] = "unknown"
        
    return state
