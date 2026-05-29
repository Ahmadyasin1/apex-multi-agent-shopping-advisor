"""
Agent 2 — Preference / Cognitive Intent Analyzer
=================================================
Role: Transforms the raw natural-language query into a structured semantic
      model that all downstream agents can act on.

Outputs a 5-dimensional **priority vector** that encodes what the user cares
about most. The vector is used directly by the Comparison Agent to weight
the multi-dimensional scoring formula:

  Priority Vector Dimensions:
    price        — how much the user cares about spending less
    performance  — raw speed, GPU/CPU horsepower, benchmark results
    brand        — manufacturer reputation, support, ecosystem trust
    durability   — build quality, MIL-SPEC, IP ratings, longevity
    innovation   — cutting-edge features, AI integration, new tech

Example transformations:
  "Cheap laptop for email"         → price: 0.50, performance: 0.15, ...
  "Best laptop for 3D rendering"   → price: 0.05, performance: 0.55, ...
  "MIL-SPEC business laptop"       → brand: 0.35, durability: 0.35, ...

The agent also extracts:
  - Budget range (min/max USD)
  - Product category (confirms/refines supervisor's detection)
  - Intent preferences (natural-language feature list for retrieval)

Uses LLM selected by Supervisor (local for simple, premium for complex).
Temperature = 0.1 for consistent but slightly flexible parsing.
"""
from state import AgentState
from llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class PreferenceOutput(BaseModel):
    """Pydantic schema for structured preference extraction output."""
    category: str = Field(description="Product category")
    budget_min: Optional[float] = Field(description="Minimum budget if specified, else null")
    budget_max: Optional[float] = Field(description="Maximum budget if specified, else null")
    intent_preferences: List[str] = Field(description="Extracted feature preferences or implicit needs")
    priority_vector: Dict[str, float] = Field(
        description="Weights (0.0 to 1.0) for 'price', 'performance', 'brand', 'durability', 'innovation'. Should sum to ~1.0."
    )

def preference_agent(state: AgentState) -> AgentState:
    """
    Preference Agent — semantic intent modeling.

    Reads:  state['query'], state['selected_llm']
    Writes: state['category'], state['budget'], state['intent_preferences'],
            state['priority_vector']

    Fallback: on LLM parse failure, a balanced default priority vector is
    applied so the pipeline always produces a meaningful ranking.
    """
    print("--- [AGENT] Preference ---")
    
    llm = get_llm(state.get("selected_llm", "local"), temperature=0.1)
    
    parser = JsonOutputParser(pydantic_object=PreferenceOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Cognitive Intent Analyzer.
Analyze the user query to extract explicit requirements and infer implicit needs.
Assign priority weights (0 to 1) for the priority_vector based on the user's intent. The weights should roughly sum to 1.0 across the keys: price, performance, brand, durability, innovation.

Format the output strictly as JSON matching these instructions."""),
        ("user", "Query: {query}\n\n{format_instructions}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        res = chain.invoke({
            "query": state.get("query"),
            "format_instructions": parser.get_format_instructions()
        })
        
        state["category"] = res.get("category", state.get("category", "unknown"))
        state["budget"] = {"min": res.get("budget_min"), "max": res.get("budget_max")}
        state["intent_preferences"] = res.get("intent_preferences", [])
        state["priority_vector"] = res.get("priority_vector", {"price":0.2, "performance":0.2, "brand":0.2, "durability":0.2, "innovation":0.2})
        
    except Exception as e:
        print(f"Preference Agent error: {e}")
        state["budget"] = {"min": None, "max": None}
        state["intent_preferences"] = []
        state["priority_vector"] = {"price":0.2, "performance":0.2, "brand":0.2, "durability":0.2, "innovation":0.2}
        
    return state
