from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Import your strict input/output verification schemas
from src.core.schemas import (
    ReviewGenerationRequest, 
    UnifiedSimulationResult,
    RecommendationRequest, 
    MultiAgentArbitrationOutput,
)

from src.agents.user_modelling import run_dual_head_simulation
from src.agents.recommender import run_multi_agent_recommender

# 1. Initialize the Enterprise API Engine
app = FastAPI(
    title="🇳🇬 NaijaSense Cognitive API Engine",
    description="Production API endpoints providing culturally localized user-modeling and agentic recommendation logic.",
    version="1.0.0"
)

# 2. Configure Cross-Origin Resource Sharing (CORS) Guardrails
# This allows external applications (like Streamlit, React, or the judge's testing tools) to safely bind to the endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits access from any origin infrastructure during the hackathon sprint
    allow_credentials=True,
    allow_methods=["*"],  # Allows all standard HTTP actions (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],
)

@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    """Health check endpoint to verify system uptime and routing integrity."""
    return {
        "status": "healthy",
        "framework": "FastAPI",
        "engine": "NaijaSense (DNPE) v1.0.0",
        "documentation_path": "/docs"
    }


#  ENDPOINT FOR TASK A: USER MODELING & SIMULATION
@app.post(
    "/api/v1/generate-review", 
    response_model=UnifiedSimulationResult,
    status_code=status.HTTP_200_OK,
    summary="Simulate user grading scale and written text reviews"
)
async def generate_user_review(payload: ReviewGenerationRequest):
    """
    Triggers the Dual-Head architecture:
    - Head 1 (The Critic) predicts a mathematically calibrated star rating.
    - Head 2 (The Voice) applies a localized Nigerian market persona to express the critique textually.
    """
    try:
        # Pass request payload fields cleanly into your core modeling script
        result = run_dual_head_simulation(
            user_id=payload.user_id,
            product_title=payload.product_title,
            category=payload.category,
            additional_context=payload.additional_context
        )
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Cognitive Engine simulation fault: {str(e)}"
        )

# ========================================================
#  ENDPOINT FOR TASK B: INTELLIGENT RECOMMENDATION
# ========================================================
@app.post(
    "/api/v1/recommend", 
    response_model=MultiAgentArbitrationOutput,
    status_code=status.HTTP_200_OK,
    summary="Execute Reason-Before-Recommend cross-domain pipeline"
)
async def recommend_items(payload: RecommendationRequest):
    """
    Triggers a stateful ReAct loop that processes cross-domain transaction footprints,
    formulates explicit hypotheses regarding environmental constraints, logs filtering actions,
    and returns a curated list of personalized recommendations.
    """
    try:
        # Pass request payload fields cleanly into your core recommender script
        recommendations = run_multi_agent_recommender(
            user_id=payload.user_id,
            context_signal=payload.context_signal,
            target_category=payload.target_category
        )
        return recommendations
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Agentic Recommender loop fault: {str(e)}"
        )