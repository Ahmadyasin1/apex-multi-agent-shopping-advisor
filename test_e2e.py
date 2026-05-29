from graph import create_workflow
from state import AgentState

workflow = create_workflow()

initial_state: AgentState = {
    "query": "I want a relatively cheap laptop for a student",
    "complexity": "simple",
    "selected_llm": "local",
    "category": "",
    "budget": {"min": None, "max": None},
    "intent_preferences": [],
    "priority_vector": {"price":0.2, "performance":0.2, "brand":0.2, "durability":0.2, "innovation":0.2},
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

print("Running e2e test... Query: 'I want a relatively cheap laptop for a student'")
final_step = None
for event in workflow.stream(initial_state):
    for node_name, state in event.items():
        print(f"--- Finished node: {node_name} ---")
        final_step = state

print("\n\n=============== FINAL RESULT ===============\n")
print(final_step.get("final_recommendation"))
print("\n============================================\n")
