import os
import sys
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def stream_and_filter_category(category: str, process_limit: int = 25000) -> pd.DataFrame:
    """
    Streams a category split from the McAuley-Lab Amazon Reviews 2023 dataset,
    gathers raw transaction paths, and isolates dense user histories.
    """
    print(f" Connecting to Hugging Face Stream for Category: [raw_review_{category}]...")
    
    # streaming=True allows execution without downloading multi-gigabyte raw files
    dataset_stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    
    extracted_records = []
    print(f" Pulling batch chunks to find historical context slots...")
    
    for row in tqdm(dataset_stream, total=process_limit):
        extracted_records.append({
            "user_id": row["user_id"],
            "item_id": row["parent_asin"],
            "rating": float(row["rating"]),
            "product_title": row.get("title", "Unknown Product"),
            "review_text": row["text"],
            "timestamp": row["timestamp"],
            "domain_category": category
        })
        if len(extracted_records) >= process_limit:
            break
            
    raw_df = pd.DataFrame(extracted_records)
    
    # Metric Check: Keep only users who have strictly more than 3 reviews (> 3)
    user_frequencies = raw_df["user_id"].value_counts()
    dense_user_ids = user_frequencies[user_frequencies > 3].index
    
    filtered_df = raw_df[raw_df["user_id"].isin(dense_user_ids)].copy()
    print(f" Isolated {len(filtered_df)} dense records from {len(dense_user_ids)} unique profiles in [{category}].\n")
    return filtered_df

def run_pipeline():
    """Compiles the cross-domain shards into a single structured staging file."""
    # Process both target verticals
    books_data = stream_and_filter_category("Books", process_limit=30000)
    electronics_data = stream_and_filter_category("Electronics", process_limit=30000)
    
    # Merge for seamless cross-domain matrix queries
    unified_dataset = pd.concat([books_data, electronics_data], ignore_index=True)
    
    # Ensure local persistence directory exists
    os.makedirs("data", exist_ok=True)
    destination_path = "data/staging_dataset.json"
    
    unified_dataset.to_json(destination_path, orient="records", indent=4)
    print(f" Success! Clean data contract built at: '{destination_path}'")

if __name__ == "__main__":
    run_pipeline()