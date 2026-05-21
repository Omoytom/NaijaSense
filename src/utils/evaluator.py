import os
import math
import time  
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer

# Import your validated AI core modules
from src.agents.user_modelling import run_dual_head_simulation
from src.agents.recommender import run_react_recommender

STAGING_PATH = "data/staging_dataset.json"

def run_evaluation_suite(sample_size: int = 5):
    if not os.path.exists(STAGING_PATH):
        raise FileNotFoundError("Staging data asset missing. Run data loader first.")
        
    df = pd.read_json(STAGING_PATH)
    unique_users = df["user_id"].unique()[:sample_size]
    
    squared_errors = []
    rouge_l_f1_scores = []
    hit_count_at_5 = 0
    total_evals = 0
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print(f"🔬 Starting throttled validation sweep across {len(unique_users)} profiles...")
    
    for idx, user_id in enumerate(unique_users):
        user_records = df[df["user_id"] == user_id].sort_values(by="timestamp")
        
        if len(user_records) < 3:
            continue
            
        ground_truth_row = user_records.iloc[-1]
        true_rating = float(ground_truth_row["rating"])
        true_review = ground_truth_row["review_text"]
        target_item_id = ground_truth_row["item_id"]
        target_title = ground_truth_row["product_title"]
        target_category = ground_truth_row["domain_category"]
        
        try:
            total_evals += 1
            print(f"\nEvaluating Profile {idx+1}/{len(unique_users)} (User: {user_id})...")
            
            # --- Task A Evaluation ---
            sim_result = run_dual_head_simulation(
                user_id=user_id,
                product_title=target_title,
                category=target_category,
                additional_context="Evaluation pass."
            )
            
            error = true_rating - sim_result.predicted_rating
            squared_errors.append(error ** 2)
            
            text_scores = scorer.score(true_review, sim_result.generated_review)
            rouge_l_f1_scores.append(text_scores['rougeL'].fmeasure)
            
            # --- Task B Evaluation ---
            rec_result = run_react_recommender(
                user_id=user_id,
                context_signal=f"User is actively browsing for relevant items in {target_category}.",
                target_category=target_category
            )
            
            predicted_item_ids = [item.item_id for item in rec_result.recommendations]
            if target_item_id in predicted_item_ids:
                hit_count_at_5 += 1
            
            #  THE PACING THROTTLE: Pause for 5 seconds to honor the 15 RPM Free Tier limit
            if idx < len(unique_users) - 1:
                print(" Pacing delay active to protect free-tier limits. Pausing for 5 seconds...")
                time.sleep(5)
                
        except Exception as e:
            print(f" Bypassing profile {user_id} due to rate limits or API event: {e}")
            print("Waiting 20 seconds to reset quota window...")
            time.sleep(20)  # Self-heal pause if a 429 still triggers
            continue

    # --- Print Report ---
    if total_evals == 0:
        print(" Evaluation run terminated: No profiles processed.")
        return

    final_rmse = math.sqrt(sum(squared_errors) / total_evals)
    final_rouge_l = (sum(rouge_l_f1_scores) / total_evals) * 100
    final_hit_rate = (hit_count_at_5 / total_evals) * 100

    print("\n" + "="*50)
    print(" DNPE AGENT PERFORMANCE EVALUATION REPORT")
    print("="*50)
    print(f" Total Validated Profiles:   {total_evals}")
    print(f" Rating Precision (RMSE):   {final_rmse:.4f}")
    print(f" Text Quality (ROUGE-L F1): {final_rouge_l:.2f}%")
    print(f" Ranking Quality (HR@5):    {final_hit_rate:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_evaluation_suite(sample_size=20)