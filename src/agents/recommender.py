import os
import re
import json
import pandas as pd
from google import genai
from dotenv import load_dotenv
from src.core.schemas import MultiAgentArbitrationOutput, RecommendationRequest, UserContextInput
from src.agents.user_modelling import execute_with_retry, get_user_history

load_dotenv(override=True)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

STAGING_PATH = "data/staging_dataset.json"

# Open src/agents/recommender.py and ensure the dictionary extraction appends the URL string

def get_candidate_pool(target_category: str, interests: list, browsing_context: str, pool_size: int = 15) -> str:
    if not os.path.exists(STAGING_PATH):
        mock_data = [
            {"item_id": "MOV_01", "product_title": "The Wedding Party", "category": "Movies", "image_url": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=300&q=80"}
        ]
        return json.dumps(mock_data)
        
    df = pd.read_json(STAGING_PATH)
    cat_df = df[df["domain_category"].str.lower() == target_category.lower()].drop_duplicates(subset=["item_id"]).copy()
    
    top_candidates = cat_df.head(pool_size)
    
    pool_data = [
        {
            "item_id": row["item_id"],
            "product_title": row["product_title"],
            "category": target_category,
            "image_url": row.get("image_url", "") # Extracts image URL to feed to the multi-agent panel
        }
        for _, row in top_candidates.iterrows()
    ]
    return json.dumps(pool_data, indent=2)

from src.agents.user_modelling import infer_nigerian_environmental_context

def run_dynamic_multi_agent_recommender(payload: RecommendationRequest) -> MultiAgentArbitrationOutput:
    user = payload.user_context
    
    if user.selected_id == "None":
        # Trigger the inference layer to deduce environmental variables live
        inferred_demo, inferred_infra = infer_nigerian_environmental_context(user.age, user.interests)
        routing_diagnostic = "New user path active. Environmental metrics generated via Context Inference Layer."
    else:
        # Load profile configurations from user_registry.json
        registry_path = "data/user_registry.json"
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                reg = json.load(f)
            cached = reg.get(user.selected_id, {})
            inferred_demo = cached.get("extracted_behavioral_persona", "Historical consumer context footprint")
        else:
            inferred_demo = "Historical consumer context footprint"
        inferred_infra = "Sourced via direct database transaction histories"
        routing_diagnostic = f"Direct verification path active for verified ID: [{user.selected_id}]"

    persona_context = f"""
    - Active Profile Consumer: {user.name} (Age: {user.age})
    - Core Interest Fields: {', '.join(user.interests)}
    - Deduced Demographic Trajectory: {inferred_demo}
    - Deduced Electrical Infrastructure Reality: {inferred_infra}
    \nPipeline Routing Trace: {routing_diagnostic}
    """

    candidate_shelf = get_candidate_pool(
        target_category=payload.target_category,
        interests=user.interests,
        browsing_context=payload.browsing_spec_context,
        pool_size=15
    )

    # Open src/agents/recommender.py and update the arbitration prompt within run_dynamic_multi_agent_recommender

    # Open src/agents/recommender.py and update the agent setup block inside your function:

    arbitration_system_prompt = (
        "You are an advanced multi-agent orchestration framework running a cognitive arbitration loop.\n\n"
    "Simulate three distinct micro-agents debating items on the candidate shelf for this user:\n"
    "1. THE INFRASTRUCTURE REALIST: Critiques items against the user's inferred grid limits and data restrictions.\n"
    "2. THE VALUE HAWK: Assesses price-to-utility returns matching their profile status.\n"
    "3. THE TECHNICAL/NARRATIVE VISIONARY: Evaluates storytelling depth (Movies) or spec longevity (Electronics).\n\n"
    " STRICT ANTI-HALLUCINATION GUARDRAILS:\n"
    "- DO NOT invent specific geographical neighborhoods, specific universities, or academic stages unless explicitly provided in the user context input.\n"
    "- Focus strictly on the broad age bracket and generalized national/regional infrastructure realities.\n\n"
    " CRITICAL DATA CONTRACT:\n"
    "- For the 'category' field, populate it with the matching domain name (Books, Electronics, or Movies)\n"
    "- The items returned in your structured 'recommendations' array MUST be chosen strictly from the candidate shelf space.\n"
    "- You MUST copy the 'product_title' and 'item_id' EXACTLY as they appear in the candidate JSON text.\n"
    "-  COMPLETELY OMIT any image fields, image links, or 'image_url' keys from your thought processes and output structure."
    )

    arbitration_prompt = f"""
    [TARGET USER PERSONA PROFILE]
    {persona_context}

    [CURRENT VIEWING INTENT CONTEXT]
    The user is searching the category [{payload.target_category}] for options matching: {payload.browsing_spec_context}

    [CANDIDATE STOREFRONT SHELF SPACE]
    {candidate_shelf}
    """
    return execute_with_retry(arbitration_prompt, arbitration_system_prompt, MultiAgentArbitrationOutput)

#  LOCAL TERMINAL VERIFICATION TEST

if __name__ == "__main__":
    print(" Executing Intent-Routed Multi-Agent Recommender verification test...")
    
    # Simulate a UNILAG engineering student looking for specific technical hardware
    test_context = UserContextInput(
        name="David",
        age=21,
        selected_id="None",
        interests=["Electronics", "Books"],
        demographic_profile="Systems Engineering Undergraduate student living near campus at UNILAG",
        infrastructure_context="Frequent campus grid collapses, relies heavily on surge protectors"
    )
    
    # Specific signal payload designed to trigger the keyword alignment engine
    test_payload = RecommendationRequest(
        user_context=test_context,
        target_category="Electronics",
        browsing_spec_context="High-efficiency backup inverter setup with surge protection fuses for sensitive computer gear"
    )
    
    try:
        # Step 1: Print candidate pool diagnostic directly to verify keyword hits
        print("\n Generating Intent-Routed Storefront Shelf (Diagnostic View):")
        shelf_json = get_candidate_pool(
            target_category=test_payload.target_category,
            interests=test_context.interests,
            browsing_context=test_payload.browsing_spec_context,
            pool_size=5
        )
        print(shelf_json)
        
        # Step 2: Run full agent arbitration pipeline over the shelf space
        print("\n Passing shelf to Multi-Agent Arbitration Panel...")
        arbitration_verification = run_dynamic_multi_agent_recommender(test_payload)
        
        print(f"\n Recommender Pipeline Execution Successful!")
        print(f" Master Arbitrator Synthesis:\n{arbitration_verification.arbitrator_synthesis}")
        print("\n Top Selections Selected by Panel:")
        for idx, item in enumerate(arbitration_verification.recommendations[:3], 1):
            print(f" {idx}. {item.product_title} -> {item.curation_justification}")
            
    except Exception as error:
        print(f" Verification Pipeline Fault: {error}")