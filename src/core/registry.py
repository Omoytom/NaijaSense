import os
import json
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Force override to clear out stale terminal session caches
load_dotenv(override=True)

REGISTRY_PATH = "data/user_registry.json"
STAGING_PATH = "data/staging_dataset.json"
MODEL_NAME = 'gemini-3.5-flash'

def compile_actual_user_personas(max_users_to_profile: int = 15) -> dict:
    """
    Groups historical data by unique user IDs, aggregates their actual written reviews,
    and extracts their genuine consumer personas using behavioral text parsing.
    """
    if not os.path.exists(STAGING_PATH):
        print(" Dataset cache missing. Run your data loader pipeline first!")
        return {}

    # Initialize the GenAI Client explicitly using the environment key token
    api_key_token = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key_token)

    print(" Reading raw transaction footprints to analyze user behaviors...")
    df = pd.read_json(STAGING_PATH)
    
    # Isolate dense users who have written multiple actual reviews in your dataset
    user_frequencies = df["user_id"].value_counts()
    dense_user_ids = user_frequencies.index.tolist()[:max_users_to_profile]
    
    registry = {}
    print(f" Generating text-driven behavioral profiles for {len(dense_user_ids)} actual Amazon users...")
    
    system_instruction = (
        "You are an advanced behavioral NLP modeling engine.\n"
        "Your task is to read a user's historical product reviews and extract their "
        "implicit consumer identity profile as a single, consolidated data string."
    )
    
    for idx, uid in enumerate(dense_user_ids):
        # Extract all actual reviews written by this specific user
        user_rows = df[df["user_id"] == uid]
        
        compiled_reviews = []
        for _, row in user_rows.iterrows():
            compiled_reviews.append(
                f"Domain: {row.get('domain_category')} | Product: {row.get('product_title')} | "
                f"Rating: {row.get('rating')} Stars\nReview: {row.get('review_text')}"
            )
        
        all_user_text = "\n\n".join(compiled_reviews)
        
        prompt = f"""
        Analyze the following real product reviews written by User ID [{uid}]. 
        Deduce their behavioral persona traits by summarizing:
        1. GRADING TEMPERAMENT: Are they a strict rater, objective, or highly forgiving?
        2. QUALITY FOCUS: Do they look for structural durability, software performance, or aesthetic style?
        3. CORE IRRITANTS: What specific issues make them leave negative or frustrated feedback?
        
        [ACTUAL HISTORICAL REVIEWS]:
        {all_user_text[:3000]}
        
        Output your analysis as a concise summary paragraph (max 3 sentences). Do not invent names or ages.
        """
        
        try:
            print(f"   Profiling User {idx+1}/{len(dense_user_ids)} ({uid[:8]}...)")
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )
            
            # Store the real user data alongside their actual text history
            registry[uid] = {
                "extracted_behavioral_persona": response.text.strip(),
                "total_historical_reviews_count": len(user_rows),
                "preferred_domains": user_rows["domain_category"].unique().tolist()
            }
            
        except Exception as e:
            print(f" Could not profile user {uid}: {e}")
            continue

    # Ensure local persistence directory exists
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)
        
    print(f" Real-Data Identity Registry generated successfully at '{REGISTRY_PATH}'")
    return registry

def get_or_create_user_registry() -> dict:
    """
    Main entrypoint used by app.py. Safely reads the cached user registry JSON,
    or compiles it dynamically from scratch if it is missing.
    """
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                data = json.load(f)
                if data:  # Ensure file is not empty
                    return data
        except Exception:
            print(" Registry file corrupted. Re-compiling behavioral profiles...")
            
    # Trigger generation if file doesn't exist or is invalid
    return compile_actual_user_personas()

if __name__ == "__main__":
    # Allows you to test or manually force-refresh the registry file from your terminal
    print(" Running standalone identity registry compilation pass...")
    compile_actual_user_personas(max_users_to_profile=10)