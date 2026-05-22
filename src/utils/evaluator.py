import os
import math
import time
import random
import pandas as pd
from rouge_score import rouge_scorer
from src.core.schemas import ReviewGenerationRequest, UserContextInput
from src.agents.user_modelling import run_dynamic_user_simulation

STAGING_PATH = "data/staging_dataset.json"

def run_evaluation_suite(sample_size: int = 3):
    """
    Automated pipeline testing accuracy metrics across historical interaction rows
    using the updated hybrid context parameters.
    """
    if not os.path.exists(STAGING_PATH):
        print(f" Verification Asset Not Found at '{STAGING_PATH}'. Run data loader first.")
        return
        
    # Read our streaming data compile cache
    df = pd.read_json(STAGING_PATH)
    
    # Sample random profiles across all domains including the new Movies entries
    sampled_records = df.sample(n=min(sample_size, len(df)), random_state=101)
    
    squared_errors = []
    rouge_l_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print(f" Launching verification sweep across {len(sampled_records)} diverse historical records...")
    print("--------------------------------------------------")
    
    for idx, (_, row) in enumerate(sampled_records.iterrows()):
        current_domain = row.get("domain_category", "Electronics")
        true_rating = float(row.get("rating", 5.0))
        true_text = row.get("review_text", "")
        
        #  HYBRID ADAPTATION: Map historical row metrics into the new UserContextInput configuration
        mock_context = UserContextInput(
            name=f"Historical Tester #{idx+1}",
            # Synthesize age traits to test grading robustness across generational models
            age=random.choice([21, 24, 35, 48]), 
            selected_id=str(row.get("user_id")),
            interests=[current_domain],
            demographic_profile="Automated validation agent tracing dataset alignments",
            infrastructure_context="Standard baseline testing environment grid profiles"
        )
        
        # Package target evaluation variables matching our schema requirements
        request_payload = ReviewGenerationRequest(
            user_context=mock_context,
            product_category=current_domain,
            product_title=row.get("product_title", "Unknown Target Item"),
            product_specifications=f"Historical Product Specifications ID Reference Code: {row.get('item_id')}.",
            additional_details=f"Ground-truth benchmarking pass for domain: {current_domain}"
        )
        
        try:
            print(f" Processing Trace Profile {idx+1}/{len(sampled_records)} | Domain: [{current_domain}]")
            
            # Execute the core engine pass
            simulation_output = run_dynamic_user_simulation(request_payload)
            
            # 1. Compute Rating Deviation Variance
            error = true_rating - simulation_output.predicted_rating
            squared_errors.append(error ** 2)
            
            # 2. Compute String N-Gram Overlap Variance
            scores = scorer.score(true_text, simulation_output.generated_review)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
            
            #  Safety throttle gap to ensure we stay under the free tier 15 RPM limits
            if idx < len(sampled_records) - 1:
                print(" Pacing interval active. Holding for 4 seconds...")
                time.sleep(4)
                
        except Exception as api_fault:
            print(f" Skipping row due to execution fault: {api_fault}")
            continue

    
    # EMIT METRICS PERFORMANCE REPORT
    
    if squared_errors:
        final_rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
        final_rouge = (sum(rouge_l_scores) / len(rouge_l_scores)) * 100
        
        print("\n" + "="*50)
        print(" DNPE HYBRID RE-ALIGNED EVALUATION SUMMARY")
        print("="*50)
        print(f" Total Profiles Checked:      {len(squared_errors)}")
        print(f" Rating Precision Score (RMSE): {final_rmse:.4f}  (Lower is better)")
        print(f" Linguistic Overlap (ROUGE-L):  {final_rouge:.2f}% (Metric benchmark marker)")
        print("="*50)
    else:
        print(" No successful evaluations were processed.")

if __name__ == "__main__":
    run_evaluation_suite(sample_size=3)