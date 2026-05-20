import os
import math
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer

# Import your validated AI core modules
from src.agents.user_modelling import run_dual_head_simulation
from src.agents.recommender import run_react_recommender

STAGING_PATH = "data/staging_dataset.json"

def run_evaluation_suite(sample_size: int = 10):
    """
    Runs a programmatic benchmark over a slice of users to compute 
    RMSE, ROUGE-L text fidelity, and Recommendation Hit Rate.
    """
    if not os.path.exists(STAGING_PATH):
        raise FileNotFoundError("Staging data asset missing. Run data loader first.")
        
    df = pd.read_json(STAGING_PATH)
    unique_users = df["user_id"].unique()[:sample_size]
    
    # Storage arrays for automated metric calculation
    squared_errors = []
    rouge_l_f1_scores = []
    hit_count_at_5 = 0
    total_evals = 0
    
    # Initialize the standard ROUGE calculation engine
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print(f" Starting local validation sweep across {len(unique_users)} dense user profiles...\n")
    
    for user_id in tqdm(unique_users):
        user_records = df[df["user_id"] == user_id].sort_values(by="timestamp")
        
        # We need at least 3 historical rows to safely hold out 1 for testing
        if len(user_records) < 3:
            continue
            
        # Hold out the final chronological interaction as our Ground Truth target
        ground_truth_row = user_records.iloc[-1]
        
        true_rating = float(ground_truth_row["rating"])
        true_review = ground_truth_row["review_text"]
        target_item_id = ground_truth_row["item_id"]
        target_title = ground_truth_row["product_title"]
        target_category = ground_truth_row["domain_category"]
        
        try:
            total_evals += 1
            
            
            # EVALUATING TASK A: USER MODELING (RMSE & ROUGE)
            
            sim_result = run_dual_head_simulation(
                user_id=user_id,
                product_title=target_title,
                category=target_category,
                additional_context="Standard evaluation holdout pass."
            )
            
            # 1. Compute Rating Squared Error
            error = true_rating - sim_result.predicted_rating
            squared_errors.append(error ** 2)
            
            # 2. Compute Text Overlap Similarity (ROUGE-L)
            text_scores = scorer.score(true_review, sim_result.generated_review)
            rouge_l_f1_scores.append(text_scores['rougeL'].fmeasure)
            
            
            # EVALUATING TASK B: RECOMMENDATION (Hit Rate @ 5)
            
            rec_result = run_react_recommender(
                user_id=user_id,
                context_signal=f"User is actively browsing for relevant items in {target_category}.",
                target_category=target_category
            )
            
            # Extract recommended item IDs from the agent's structured response array
            predicted_item_ids = [item.item_id for item in rec_result.recommendations]
            
            # Check if the hidden item actually made it into the top recommendations list
            if target_item_id in predicted_item_ids:
                hit_count_at_5 += 1
                
        except Exception as e:
            print(f"\n Bypassing evaluation block for user {user_id} due to API/Timeout event: {e}")
            continue

    
    # FINAL METRICS AGGREGATION & REPORT RENDERING
    
    if total_evals == 0:
        print(" Evaluation run terminated: No valid profiles processed successfully.")
        return

    final_rmse = math.sqrt(sum(squared_errors) / total_evals)
    final_rouge_l = (sum(rouge_l_f1_scores) / total_evals) * 100
    final_hit_rate = (hit_count_at_5 / total_evals) * 100

    print("\n" + "="*50)
    print(" DNPE AGENT PERFORMANCE EVALUATION REPORT")
    print("="*50)
    print(f" Total Validated Profiles:   {total_evals}")
    print(f" Rating Precision (RMSE):   {final_rmse:.4f}  (Lower is better)")
    print(f" Text Quality (ROUGE-L F1): {final_rouge_l:.2f}% (Higher is better)")
    print(f" Ranking Quality (HR@5):    {final_hit_rate:.2f}% (Higher is better)")
    print("="*50)
    print("\n Copy this metrics framework matrix directly into Section 4 of your Solution Paper!")

if __name__ == "__main__":
    # Run a quick evaluation pass across 5 profiles to establish a baseline
    run_evaluation_suite(sample_size=5)