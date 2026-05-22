import os
import time
import random
import json
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from src.core.schemas import CriticOutput, VoiceOutput, UnifiedSimulationResult, ReviewGenerationRequest

# Force override to clear out stale terminal session caches
load_dotenv(override=True)
api_key_token = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key_token)

STAGING_PATH = "data/staging_dataset.json"
REGISTRY_PATH = "data/user_registry.json"
MODEL_NAME = 'gemini-3.5-flash'

def execute_with_retry(prompt: str, system_instruction: str, response_schema: type, max_retries: int = 5):
    """Autonomous exponential backoff middleware with jitter protecting free-tier capacity."""
    base_delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.2
                )
            )
            return response.parsed
        except APIError as api_err:
            if api_err.code == 429 and attempt < max_retries - 1:
                sleep_duration = (base_delay ** attempt) + random.uniform(0.5, 1.5)
                print(f"⚠️ Server busy (429). Retrying in {sleep_duration:.2f} seconds...")
                time.sleep(sleep_duration)
                continue
            raise api_err
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay ** attempt)
                continue
            raise e

def get_user_history(user_id: str) -> str:
    """
    Extracts real historical text reviews from the staging database 
    cache to feed active transaction patterns into the prompt context window.
    """
    if user_id == "None" or not os.path.exists(STAGING_PATH):
        return ""
    try:
        df = pd.read_json(STAGING_PATH)
        user_data = df[df["user_id"] == user_id].sort_values(by="timestamp", ascending=False).head(3)
        if user_data.empty:
            return ""
        
        history_blocks = []
        for _, row in user_data.iterrows():
            history_blocks.append(
                f"-[Category: {row.get('domain_category')}] "
                f"Item: {row.get('product_title')} | "
                f"Rating Given: {row.get('rating')}/5\n"
                f"Review Text: {row.get('review_text')}"
            )
        return "\n\n".join(history_blocks)
    except Exception as e:
        print(f"⚠️ Error reading historical text trace patterns: {e}")
        return ""

def infer_nigerian_environmental_context(age: int, interests: list) -> tuple:
    """
    Zero-Shot Context Generation: Deduces a realistic demographic profile and 
    generalized infrastructure reality based purely on age and interests.
    """
    system_prompt = (
        "You are an expert sociologist and infrastructural analyst tracking contemporary Nigerian life.\n"
        "Your job is to generate highly accurate background contexts for user profiles based on age and interests."
    )
    
    prompt = f"""
    Generate a realistic background profile for a person living in Nigeria with these attributes:
    - Age: {age}
    - Stated Core Interests: {', '.join(interests)}
    
    Deduce two specific items:
    1. DEMOGRAPHIC COHORT SUMMARY: Where do they likely live or work/study? What is their current life path?
    2. OPERATIONAL INFRASTRUCTURE REALITY: What is their daily electrical and digital setup? Do they rely on grid power, fuel generators, solar inverters, or mobile data caps? Keep this grounded in everyday Nigerian context.
    
    Return your response as a raw JSON object matching this structure exactly:
    {{
        "demographic": "Brief description sentence",
        "infrastructure": "Brief infrastructure reality sentence"
    }}
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        parsed_data = json.loads(response.text)
        return parsed_data.get("demographic", ""), parsed_data.get("infrastructure", "")
    except Exception:
        return "Urban professional/student community member", "Standard urban grid connection with alternative generator backup profiles"

def run_dynamic_user_simulation(payload: ReviewGenerationRequest) -> UnifiedSimulationResult:
    """Orchestrates Head 1 and Head 2 concurrently using inferred or cached profile constants."""
    user = payload.user_context
    
    # Gateway routing matrix paths
    if user.selected_id == "None":
        inferred_demo, inferred_infra = infer_nigerian_environmental_context(user.age, user.interests)
        routing_diagnostic = "Cold-Start Workspace Matrix: Background contexts dynamically generated via inference engine."
    else:
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "r") as f:
                reg = json.load(f)
            cached = reg.get(user.selected_id, {})
            inferred_demo = cached.get("extracted_behavioral_persona", "Verified historical user profile patterns")
            inferred_infra = "Sourced via direct dataset row correlation attributes"
        else:
            inferred_demo = "Verified historical user profile patterns"
            inferred_infra = "Standard direct dataset trace parameters"
        routing_diagnostic = f"Direct Identity Link Active. Sourcing actual history for User ID: [{user.selected_id}]"

    # Pull the real historical reviews from text blocks
    historical_footprint = get_user_history(user.selected_id)

    persona_context = f"""
    [ACTIVE PROFILE CORE]
    - Name: {user.name} | Age: {user.age}
    - Domains of Interest: {', '.join(user.interests)}
    - Assumed Socio-Economic Cohort: {inferred_demo}
    - Local Infrastructure Realities: {inferred_infra}
    
    [PIPELINE TRACE DIAGNOSTIC]: {routing_diagnostic}
    """
    if historical_footprint:
        persona_context += f"\n\n[VERIFIED ANCHOR HISTORICAL TRANSACTIONS]\n{historical_footprint}"

    # --- Head 1: The Critic ---
    critic_system_prompt = "Calculate the strict numerical star rating (1.0 to 5.0) this persona profile would assign to the item."
    critic_prompt = f"User Context:\n{persona_context}\n\nEvaluate specifications:\nTitle: {payload.product_title}\nSpecs: {payload.product_specifications}"
    critic_res = execute_with_retry(critic_prompt, critic_system_prompt, CriticOutput)

    # --- Head 2: The Voice ---
    voice_system_prompt = (
        "Write a first-person localized consumer review matching the provided persona profile details.\n"
        "Ensure your linguistic tone heavily matches their age criteria (youth slang vs corporate/formal Nigerian expressions words such as 'abi' or 'biko' or 'No yawa' ).\n"
        "The review context details must completely support an evaluation score of exactly " f"{critic_res.predicted_rating}/5."
    )
    voice_prompt = f"User Profile Matrix:\n{persona_context}\nTarget Title: {payload.product_title}\nEnforce Numerical Rating Constraint: {critic_res.predicted_rating}"
    voice_res = execute_with_retry(voice_prompt, voice_system_prompt, VoiceOutput)

    return UnifiedSimulationResult(
        user_id=user.selected_id if user.selected_id != "None" else "Inferred_Sandbox_User",
        product_title=payload.product_title,
        predicted_rating=critic_res.predicted_rating,
        critic_reasoning=critic_res.rating_justification,
        generated_review=voice_res.generated_review,
        cultural_notes=voice_res.cultural_nuance_notes,
        inferred_demographic=inferred_demo,
        inferred_infrastructure=inferred_infra
    )