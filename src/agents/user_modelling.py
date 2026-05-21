import json
import os
import sys
import time
import random
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.schemas import CriticOutput, VoiceOutput, UnifiedSimulationResult

# Load environment variables
load_dotenv(override=True)
client = genai.Client()

STAGING_PATH = "data/staging_dataset.json"
# Upgrading to the new 3.5 model for better agentic throughput and fresher capacity
MODEL_NAME = 'gemini-3.5-flash' 

def get_user_history(user_id: str) -> str:
    """Extracts and stringifies past reviews of a user to construct context."""
    if not os.path.exists(STAGING_PATH):
        raise FileNotFoundError("Run data loader pipeline first to generate staging_dataset.json")
        
    df = pd.read_json(STAGING_PATH)
    user_records = df[df["user_id"] == user_id]
    
    if user_records.empty:
        return ""
        
    context_lines = []
    for _, row in user_records.iterrows():
        context_lines.append(
            f"- [{row['domain_category']}] Item: {row['product_title']} | "
            f"Given Rating: {row['rating']}/5 | Review: {row['review_text']}"
        )
    return "\n".join(context_lines)

def execute_with_retry(prompt: str, system_instruction: str, response_schema: type, max_retries: int = 5, base_delay: float = 2.0):
    """
    Executes a Gemini API generation request with randomized exponential backoff
    to gracefully handle 503 (Unavailable) and 429 (Rate Limit) server events.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.7 if response_schema == VoiceOutput else 0.1,
                ),
            )
            # Successfully got a response, validate and return
            return response_schema.model_validate_json(response.text)
            
        except APIError as e:
            # Trap 503 (Service Unavailable) or 429 (Rate Limits)
            if e.code in [503, 429] and attempt < max_retries - 1:
                # Calculate sleep time: base_delay * (2^attempt) + random jitter
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.5, 1.5)
                print(f"⚠️ Server busy ({e.code}). Retrying in {sleep_time:.2f} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                # Raise immediately if it's a critical structural error (like 400 Bad Request)
                raise e
    raise APIError("Failed to get response from Gemini API after maximum retries due to server saturation.")

def run_dual_head_simulation(user_id: str, product_title: str, category: str, additional_context: str = "") -> UnifiedSimulationResult:
    """Orchestrates Head 1 (The Critic) and Head 2 (The Voice) safely with retries."""
    
    # 1. Gather historical baseline
    history_context = get_user_history(user_id)
    if not history_context:
        history_context = "No previous history found (Cold-start persona baseline initialization)."

    # ==========================================
    # HEAD 1: THE CRITIC (Rating Calibration)
    # ==========================================
    critic_system_prompt = (
        "You are an analytical data engine tracking human grading consistency. "
        "Your sole task is to predict the numerical rating (1.0 to 5.0) a user would assign "
        "to a new product based on their explicit historical review patterns and tendencies."
    )
    
    critic_prompt = f"""
    [USER HISTORY]
    {history_context}
    
    [NEW TARGET PRODUCT]
    Category: {category}
    Title: {product_title}
    Context: {additional_context}
    
    Analyze the user's grading strictness or leniency from history. Output a calibrated predicted rating.
    """
    
    print(" Running Head 1: Calibrating User Rating Scale...")
    critic_res = execute_with_retry(critic_prompt, critic_system_prompt, CriticOutput)

    
    # HEAD 2: THE VOICE (Nigerian Behavioral Localizer)
    
    voice_system_prompt = (
        "You are a human behavioral simulation agent specializing in localized linguistic modeling.\n\n"
        "CRUCIAL TASK SPECIFICATIONS:\n"
        "1. BEHAVIORAL FIDELITY: You must write a consumer review from the standpoint of a Nigerian buyer.\n"
        "2. VALUE MATRICES: Nigerian consumers assess value-for-money heavily based on physical durability, "
        "cost-effectiveness, and infrastructure constraints (e.g., resilience against erratic power, heat, battery life).\n"
        "3. LINGUISTIC BLENDING: Seamlessly blend formal Nigerian English with natural local phrasing "
        "(e.g., expressions like 'abeg', 'no cap', 'No yawa', 'It makes sense die', 'it gave what it was supposed to give', or sharp structural sarcasm if performance fails).\n"
        "4. CONSTRAINT ENFORCEMENT: The review tone MUST perfectly match the rating provided by the Critic."
    )
    
    voice_prompt = f"""
    [USER HISTORY]
    {history_context}
    
    [TARGET ITEM]
    Product: {product_title} ({category})
    
    [CRITIC MANDATE CONSTRAINT]
    You MUST simulate an experience that warrants exactly a {critic_res.predicted_rating}/5 rating.
    """
    
    print("🇳🇬 Running Head 2: Simulating Localized Behavioral Voice...")
    voice_res = execute_with_retry(voice_prompt, voice_system_prompt, VoiceOutput)

    # Combine both heads into the final unified object
    return UnifiedSimulationResult(
        user_id=user_id,
        product_title=product_title,
        predicted_rating=critic_res.predicted_rating,
        critic_reasoning=critic_res.rating_justification,
        generated_review=voice_res.generated_review,
        cultural_notes=voice_res.cultural_nuance_notes
    )

if __name__ == "__main__":
    # Test execution using a valid user profile from your dataset
    test_user = "AFKZENTNBQ7A7V7UXW5JJI6UGRYQ" 
    test_title = "Heavy Duty 3000W Automatic Voltage Regulator / Stabilizer"
    
    print(" Testing Gemini-Powered Dual-Head Simulation Core...")
    try:
        result = run_dual_head_simulation(
            user_id=test_user,
            product_title=test_title,
            category="Electronics",
            additional_context="Handles extreme electrical spikes, rugged casing."
        )
        print(f"\n Predicted Rating: {result.predicted_rating}/5.0")
        print(f" Critic Reasoning: {result.critic_reasoning}")
        print(f"🇳🇬 Generated Localized Review:\n\"{result.generated_review}\"")
        print(f" Cultural Alignment Insights: {result.cultural_notes}")
    except Exception as e:
        print(f"\n Execution stopped: {e}")