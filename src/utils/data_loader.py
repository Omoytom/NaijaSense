import os
import sys
import json
import time
import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

STAGING_PATH = "data/staging_dataset.json"

def lightning_title_cleaner(df: pd.DataFrame, batch_size: int = 15) -> pd.DataFrame:
    """
    Bulletproof Data Cleaner: Generates realistic titles via raw JSON matching 
    and uses an NLP string fallback to ensure zero generic repetitions.
    """
    item_clusters = df.groupby("item_id").agg({
        "domain_category": "first",
        "review_text": lambda x: " | ".join(x.head(2).astype(str))
    }).reset_index()
    
    unique_count = len(item_clusters)
    title_mapping = {}
    
    print(f" Analyzing {unique_count} unique product clusters...")
    
    for i in range(0, unique_count, batch_size):
        batch_rows = item_clusters.iloc[i : i + batch_size]
        manifest_lines = []
        
        for _, row in batch_rows.iterrows():
            clean_text = row["review_text"].replace("\n", " ").replace('"', "'")[:150]
            manifest_lines.append(f"ID: {row['item_id']} | Domain: {row['domain_category']} | Clues: {clean_text}")
        
        #  SYNTAX FIX: Join lines outside the f-string to prevent Python backslash parsing crashes
        manifest_payload = "\n".join(manifest_lines)
            
        prompt = f"""
        You are an e-commerce catalog engineer. Convert these review text clues into real, descriptive retail product names or movie/book titles.
        
        Output strictly as a valid JSON object mapping the alphanumeric IDs directly to your generated titles. Do not include markdown code blocks or formatting.
        
        Example Output Format:
        {{
            "B003VJTGM0": "Classic Family Christmas DVD Collection",
            "B07V5X1FPH": "Indie Psychological Thriller Feature Film"
        }}
        
        [ITEMS MANIFEST TO TRANSLATE]:
        {manifest_payload}
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            
            raw_text = response.text.strip()
            # Strip any accidental markdown wrap filters if they leak through
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
                
            batch_mapping = json.loads(raw_text)
            for k, v in batch_mapping.items():
                title_mapping[k] = str(v).strip()
                
            time.sleep(0.2)
            
        except Exception as e:
            #  ERROR EXPOSAL: Prints the underlying issue to the console if a batch triggers a fallback
            print(f" Batch notice: API parse bypassed for chunk index {i}. Reason: {e}")
            continue

    
    #  DYNAMIC NLP FALLBACK NET
    
    # If any item failed the API call, parse its review text directly to generate a unique name
    for _, row in item_clusters.iterrows():
        uid = row["item_id"]
        if uid not in title_mapping or "Curated" in title_mapping[uid] or "Selection" in title_mapping[uid]:
            domain = row["domain_category"]
            first_review = row["review_text"].split("|")[0].strip()
            
            # Clean punctuation and split into significant words
            clean_clue = re.sub(r'[^a-zA-Z0-9\s]', '', first_review)
            descriptive_words = [
                word.capitalize() for word in clean_clue.split() 
                if len(word) > 3 and word.lower() not in [
                    'this', 'that', 'with', 'from', 'have', 'were', 'they', 'them', 'their', 'good'
                ]
            ]
            
            if descriptive_words:
                seed_phrase = " ".join(descriptive_words[:3])
                if domain == "Movies":
                    title_mapping[uid] = f"{seed_phrase} Cinema Release"
                elif domain == "Electronics":
                    title_mapping[uid] = f"{seed_phrase} System Gear"
                else:
                    title_mapping[uid] = f"The {seed_phrase} Edition"
            else:
                title_mapping[uid] = f"Premium Authentic {domain} Product ({uid[:4]})"

    df["product_title"] = df["item_id"].map(title_mapping)
    return df

def ultra_fast_stream(category: str, process_limit: int = 250) -> pd.DataFrame:
    """Streams a tight data segment slice from the Hugging Face hub."""
    dataset_stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    
    extracted = []
    for row in dataset_stream:
        # Image URL stripped cleanly out at ingestion layer to minimize token memory usage
        extracted.append({
            "user_id": row["user_id"],
            "item_id": row["parent_asin"],
            "rating": float(row["rating"]),
            "review_text": row["text"],
            "timestamp": row["timestamp"],
            "domain_category": category,
            "image_url": "" 
        })
        if len(extracted) >= process_limit:
            break
            
    return pd.DataFrame(extracted)

def run_pipeline():
    print(" Running structural dataset cache compilation...")
    books = ultra_fast_stream("Books", process_limit=150)
    electronics = ultra_fast_stream("Electronics", process_limit=150)
    movies = ultra_fast_stream("Movies_and_TV", process_limit=150)
    movies["domain_category"] = "Movies"
    
    unified_dataset = pd.concat([books, electronics, movies], ignore_index=True)
    
    # Run our updated hybrid text pipeline cleaner
    cleaned_dataset = lightning_title_cleaner(unified_dataset, batch_size=15)
    
    os.makedirs("data", exist_ok=True)
    cleaned_dataset.to_json(STAGING_PATH, orient="records", indent=4)
    print(f"\n Success! Re-compiled clean dataset staging file at: '{STAGING_PATH}'")

if __name__ == "__main__":
    run_pipeline()