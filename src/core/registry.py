import os
import sys
import json
import time
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

STAGING_PATH = "data/staging_dataset.json"
REGISTRY_PATH = "data/user_registry.json"

# ==========================================
# STRICT SCHEMAS FOR STRUCTURED PROMPTING
# ==========================================
class UserProfileSchema(BaseModel):
    name: str = Field(description="A unique, realistic name for this profile wrapper (e.g., 'Tunde', 'Chidi', 'Aminu', 'Ronke'). Avoid using 'David' for everyone.")
    age: int = Field(description="A realistic biological age between 19 and 55 deduced from their review tone and context.")
    extracted_behavioral_persona: str = Field(description="A deep paragraph summarizing their purchasing habits, standards, and behavioral persona traits.")
    preferred_domains: list[str] = Field(description="List of categories they frequently engage with based on their history.")

# ==========================================
# REGISTRY CORE ENGINE
# ==========================================
def build_user_registry():
    print("👥 Initiating Persona Profile Extraction Pipeline...")
    
    if not os.path.exists(STAGING_PATH):
        print(f"❌ Error: Cannot build profiles. Staging file '{STAGING_PATH}' is missing!")
        return
        
    df = pd.read_json(STAGING_PATH)
    unique_users = df["user_id"].unique().tolist()
    
    registry_cache = {}
    print(f"🧠 Scanning historical traces for {len(unique_users)} unique user footprints...")
    
    for idx, user_id in enumerate(unique_users):
        user_rows = df[df["user_id"] == user_id]
        total_reviews = len(user_rows)
        
        # Build a coherent interaction footprint history for this user
        history_lines = []
        for _, row in user_rows.head(4).iterrows():
            history_lines.append(
                f"- Category: {row['domain_category']} | Item: {row['product_title']} | Rating: {row['rating']}★\n"
                f"  Review: Text: {row['review_text'][:120]}"
            )
        
        history_payload = "\n".join(history_lines)
        
        prompt = f"""
        You are a psychographic consumer profiling engine. Analyze this user's historical Amazon review history footprints to deduce a distinct identity and persona:
        
        [USER TRANSACTION FOOTPRINTS]:
        {history_payload}
        
        MANDATE:
        - Deduce a realistic commercial name and an age corresponding to their linguistic maturity.
        - Vary names naturally (use common regional names like Tunde, Chioma, Fatima, Emeka, Kunle, Blessing). Do not repeat the name David.
        - Extract a detailed 'behavioral_persona' summarizing what they care about (e.g., price-sensitive, high-spec demanding, fast data-bundle optimizer).
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=UserProfileSchema,
                    temperature=0.3
                )
            )
            
            profile_json = json.loads(response.text)
            
            # Inject meta metrics along with the structured profile attributes
            registry_cache[user_id] = {
                "name": profile_json["name"],
                "age": profile_json["age"],
                "extracted_behavioral_persona": profile_json["extracted_behavioral_persona"],
                "preferred_domains": profile_json["preferred_domains"],
                "total_historical_reviews_count": total_reviews
            }
            
            time.sleep(0.2) # Avoid rate limit bottlenecks
            
        except Exception as e:
            # Safe unique fallback mapping
            registry_cache[user_id] = {
                "name": f"User_{user_id[:5]}",
                "age": 24 + (idx % 7),
                "extracted_behavioral_persona": "Standard consumer seeking high utility and balanced product ecosystems.",
                "preferred_domains": user_rows["domain_category"].unique().tolist(),
                "total_historical_reviews_count": total_reviews
            }
            continue

    # Write out the clean structural dataset registry file
    os.makedirs("data", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry_cache, f, indent=4)
        
    print(f"🏆 Success! Identity registry matrix successfully compiled at: '{REGISTRY_PATH}'")

def get_or_create_user_registry():
    """Exposes the registry to our core application frameworks safely."""
    if not os.path.exists(REGISTRY_PATH):
        build_user_registry()
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    build_user_registry()