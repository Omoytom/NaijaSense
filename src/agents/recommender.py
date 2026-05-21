import os
import json
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.core.schemas import MultiAgentArbitrationOutput
from src.agents.user_modelling import execute_with_retry

load_dotenv(override=True)
client = genai.Client()

STAGING_PATH = "data/staging_dataset.json"
MODEL_NAME = 'gemini-3.5-flash'

def get_candidate_pool(exclude_user_id: str, target_category: str, pool_size: int = 15) -> str:
    """Retrieves a diverse pool of product candidates from the staging dataset."""
    if not os.path.exists(STAGING_PATH):
        return "[]"
    
    df = pd.read_json(STAGING_PATH)
    category_pool = df[df["domain_category"] == target_category].drop_duplicates(subset=["item_id"])
    
    sample_size = min(pool_size, len(category_pool))
    sampled_items = category_pool.sample(n=sample_size, random_state=42)
    
    pool_data = []
    for _, row in sampled_items.iterrows():
        pool_data.append({
            "item_id": row["item_id"],
            "product_title": row["product_title"],
            "category": target_category
        })
    return json.dumps(pool_data, indent=2)

def run_multi_agent_recommender(user_id: str, context_signal: str, target_category: str) -> MultiAgentArbitrationOutput:
    """
    Executes a Multi-Agent Cognitive Arbitration loop in a single parallel pass.
    Deconstructs recommendations through three competitive structural personas.
    """
    from src.agents.user_modelling import get_user_history
    
    user_history = get_user_history(user_id)
    if not user_history:
        user_history = "Cold-start profile state. No prior cross-domain purchases verified."

    candidate_shelf = get_candidate_pool(user_id, target_category)

    # Systemic multi-agent orchestration instructions
    arbitration_system_prompt = (
        "You are an advanced multi-agent orchestration framework executing a cognitive arbitration loop.\n\n"
        "You must simultaneously simulate three specialized micro-agents to debate the candidate pool items:\n"
        "1. THE INFRASTRUCTURE REALIST: Evaluates items strictly through local operational constraints "
        "(e.g., Nigerian power grid fluctuations, thermal thresholds, component lifespan under heat, mechanical wear).\n"
        "2. THE VALUE/BUDGET HAWK: Focuses entirely on price-to-performance scaling, utility optimization, and practical value.\n"
        "3. THE TECHNICAL VISIONARY: Analyzes high-spec future-proofing, computational headroom, and long-term tech trajectory.\n\n"
        "CORE TASK:\n"
        "- Let each agent write its structural critique of the choices relative to the user situation.\n"
        "- Act as the Master Arbitrator to synthesize their friction points, resolve trade-offs, and output the absolute best items."
    )

    arbitration_prompt = f"""
    [USER PROFILE HISTORY]
    {user_history}

    [REAL-TIME SITUATION SIGNAL]
    {context_signal}

    [CANDIDATE SHELF SPACE]
    {candidate_shelf}

    Trigger your internal agent debate panels and populate the MultiAgentArbitrationOutput schema.
    """

    print(f" Initiating Multi-Agent Arbitration Panel for Category [{target_category}]...")
    
    arbitration_res = execute_with_retry(
        prompt=arbitration_prompt,
        system_instruction=arbitration_system_prompt,
        response_schema=MultiAgentArbitrationOutput
    )
    
    return arbitration_res

if __name__ == "__main__":
    test_user = "AFKZENTNBQ7A7V7UXW5JJI6UGRYQ" 
    test_signal = "User is building a high-throughput remote data workstation in Lagos and needs to protect delicate hardware from voltage drops."
    
    try:
        decision_space = run_multi_agent_recommender(
            user_id=test_user,
            context_signal=test_signal,
            target_category="Electronics"
        )
        
        print("\n [AGENT A] THE INFRASTRUCTURE REALIST CRITIQUE:")
        print(decision_space.infrastructure_realist_critique)
        
        print("\n [AGENT B] THE VALUE/BUDGET HAWK CRITIQUE:")
        print(decision_space.value_budget_hawk_critique)
        
        print("\n [AGENT C] THE TECHNICAL VISIONARY CRITIQUE:")
        print(decision_space.technical_visionary_critique)
        
        print("\n[MASTER ARBITRATOR] FINAL SYNTHESIS RESOLUTION:")
        print(decision_space.arbitrator_synthesis)
        
        print("\n CURATED MULTI-AGENT SELECTIONS:")
        for idx, item in enumerate(decision_space.recommendations, 1):
            print(f"{idx}. {item.product_title}")
            print(f"    Resolution Rationale: {item.curation_justification}\n")
            
    except Exception as e:
        print(f"\n Arbitration panel execution fault: {e}")