import os
import json
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.core.schemas import ReActRecommendationOutput
from src.agents.user_modelling import execute_with_retry

load_dotenv()
client = genai.Client()

STAGING_PATH = "data/staging_dataset.json"
MODEL_NAME = 'gemini-3.5-flash'

def get_candidate_pool(exclude_user_id: str, target_category: str, pool_size: int = 15) -> str:
    """
    Retrieves a diverse pool of product candidates from the staging dataset 
    to provide the LLM agent with a contextual shelf to choose from.
    """
    if not os.path.exists(STAGING_PATH):
        return "[]"
    
    df = pd.read_json(STAGING_PATH)
    # Filter by category and get unique products to prevent redundant lists
    category_pool = df[df["domain_category"] == target_category].drop_duplicates(subset=["item_id"])
    
    # Take a sample slice for the agent's context shelf
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

def run_react_recommender(user_id: str, context_signal: str, target_category: str) -> ReActRecommendationOutput:
    """
    Executes a strict Reason-Before-Recommend cognitive loop using a ReAct structure.
    Maps user profiles cross-domain based on contextual signals.
    """
    from src.agents.user_modelling import get_user_history
    
    # 1. Gather existing cross-domain profile history
    user_history = get_user_history(user_id)
    if not user_history:
        user_history = "Cold-start profile state. No prior cross-domain purchases verified."

    # 2. Dynamically gather available items on the digital storefront shelf
    candidate_shelf = get_candidate_pool(user_id, target_category)

    # 3. Systemic cognitive architecture framing
    recommender_system_prompt = (
        "You are an advanced agentic recommender system running a strict 'Reason-Before-Recommend' loop.\n\n"
        "YOUR CORE WORKFLOW:\n"
        "1. THOUGHT: Deeply analyze the user's historical profile patterns and map them against the real-time context signal "
        "and structural market conditions (e.g., Nigerian infrastructure realities, pricing sensitivity, durability needs).\n"
        "2. ACTION: Formulate a definitive filtering target or strategy (e.g., 'Prioritizing heavy-duty equipment over fragile electronics').\n"
        "3. RECOMMENDATION: Select the best items from the provided Candidate Shelf that perfectly resolve the user's situation.\n\n"
        "CRUCIAL: Your output must strictly match the response schema, explicitly spelling out your inner thoughts and localized rationale."
    )

    recommender_prompt = f"""
    [USER PROFILE HISTORY]
    {user_history}

    [REAL-TIME USER SITUATION / SIGNAL]
    {context_signal}

    [CANDIDATE SHELF (Select only from this JSON list)]
    {candidate_shelf}

    Analyze the inputs. Run your cognitive reasoning chain and output your recommendation array.
    """

    print(f"🎯 Running Task B Engine: Executing Reason-Before-Recommend loop for category [{target_category}]...")
    
    # Execute safely via our backoff retry function to bypass 503 capacity limits
    recommendation_res = execute_with_retry(
        prompt=recommender_prompt,
        system_instruction=recommender_system_prompt,
        response_schema=ReActRecommendationOutput
    )
    
    return recommendation_res

if __name__ == "__main__":
    # Test execution simulating a deep cross-domain verification task
    # Passing a user from the dataset, along with a real-world Nigerian context signal
    test_user = "A100WO06OQR8BQ" 
    test_signal = "User is scaling a remote coding workstation in Lagos and needs to survive unstable grid setups without frying gear."
    
    try:
        recommendations_output = run_react_recommender(
            user_id=test_user,
            context_signal=test_signal,
            target_category="Electronics"
        )
        
        print("\n🧠 --- AGENT INTERNAL THOUGHT PROCESS ---")
        print(recommendations_output.thought_process)
        
        print("\n🛠️ --- SELECTED FILTERING ACTION ---")
        print(recommendations_output.selected_action)
        
        print("\n🇳🇬 --- CURATED PERSONALIZED RECOMMENDATIONS ---")
        for idx, item in enumerate(recommendations_output.recommendations, 1):
            print(f"{idx}. [{item.category}] {item.product_title}")
            print(f"   💡 Rationale: {item.curation_justification}\n")
            
    except Exception as e:
        print(f"\n❌ Recommendation execution stopped: {e}")