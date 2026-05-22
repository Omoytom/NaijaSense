import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Enforce project root discovery inside ASGI web server application processes
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import schemas and registry utilities explicitly
from src.core.schemas import (
    ReviewGenerationRequest, 
    UnifiedSimulationResult, 
    RecommendationRequest, 
    MultiAgentArbitrationOutput
)
from src.core.registry import get_or_create_user_registry

# Aligned imports tracking the singular spelling convention
from src.agents.user_modelling import run_dynamic_user_simulation
from src.agents.recommender import run_dynamic_multi_agent_recommender

app = FastAPI(
    title="🇳🇬 NaijaSense Cognitive API Engine", 
    description="Production ASGI server routing layer handling dynamic lookalike clustering and zero-shot context inference dashboards.",
    version="3.0.0"
)

# Enable robust cross-origin sharing configurations to protect decoupled deployment sockets
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STAGING_PATH = "data/staging_dataset.json"

@app.get("/api/v1/dataset-users", status_code=status.HTTP_200_OK)
def get_dataset_users():
    """
    Identity Sync Endpoint: Exposes the text-extracted historical reviewer keys 
    to the client dropdown panel, ensuring strict state synchronization.
    """
    try:
        user_registry = get_or_create_user_registry()
        return {"user_ids": list(user_registry.keys())}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract identity registry matrix maps: {str(e)}"
        )

@app.get("/api/v1/random-product/{category}", status_code=status.HTTP_200_OK)
def get_random_product_by_category(category: str):
    """
    Dynamic Segment Sampler: Extracts a single real product row from the targeted 
    domain category slice to instantly populate the interactive sandbox workbench.
    """
    try:
        if not os.path.exists(STAGING_PATH):
            return {
                "product_title": f"Standard Premium {category} Asset Node",
                "specifications": "Standard baseline data sheet profile parameters configuration."
            }
            
        df = pd.read_json(STAGING_PATH)
        # Isolate rows matching the specific category query criteria
        filtered_df = df[df["domain_category"].str.lower() == category.lower()]
        
        if filtered_df.empty:
            filtered_df = df # Graceful fallback tracking arrays if category matches are sparse
            
        sampled_row = filtered_df.sample(n=1).iloc[0]
        return {
            "product_title": sampled_row.get("product_title", "Premium Retail Asset"),
            "specifications": f"Item ASIN: {sampled_row.get('item_id')}. Context footprints: {sampled_row.get('review_text', '')[:200]}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data sampler execution array trace fault: {str(e)}"
        )

@app.post("/api/v1/generate-review", response_model=UnifiedSimulationResult)
async def generate_user_review(payload: ReviewGenerationRequest):
    """
    Task A Execution Node: Intercepts onboarding parameters, infers local background variables 
    autonomously, and executes the Dual-Head constraint simulation loop.
    """
    try:
        return run_dynamic_user_simulation(payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task A cognitive simulation processing matrix failure: {str(e)}"
        )

@app.post("/api/v1/recommend", response_model=MultiAgentArbitrationOutput)
async def recommend_items(payload: RecommendationRequest):
    """
    Task B Execution Node: Builds an intent-routed candidate shelf using pandas feature weights, 
    and passes it directly down to convene the specialized multi-agent debate panels.
    """
    try:
        return run_dynamic_multi_agent_recommender(payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task B multi-agent arbitration execution matrix failure: {str(e)}"
        )