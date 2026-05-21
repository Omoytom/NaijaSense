from pydantic import BaseModel
from typing import List

class CriticOutput(BaseModel):
    """Head 1: Numerical Rating Calibration"""
    predicted_rating: float
    rating_justification: str

class VoiceOutput(BaseModel):
    """Head 2: Culturally Localized Text Generation"""
    generated_review: str
    cultural_nuance_notes: str

class UnifiedSimulationResult(BaseModel):
    """The complete pipeline object handed off to the frontend"""
    user_id: str
    product_title: str
    predicted_rating: float
    critic_reasoning: str
    generated_review: str
    cultural_notes: str

class RecommendedItem(BaseModel):
    """A single curated recommendation item"""
    item_id: str
    product_title: str
    category: str
    curation_justification: str  # Culturally contextual reason for this pick

class MultiAgentArbitrationOutput(BaseModel):
    """Head 3: Multi-Agent Cognitive Arbitration Output Structure"""
    infrastructure_realist_critique: str  # Agent A: Grid, thermal, and mechanical durability focus
    value_budget_hawk_critique: str       # Agent B: Cost-to-utility scaling and lifespan verification
    technical_visionary_critique: str     # Agent C: Spec future-proofing and ecosystem alignment
    arbitrator_synthesis: str             # The final resolution resolving agent trade-offs
    recommendations: List[RecommendedItem] # The targeted top recommendations

class ReviewGenerationRequest(BaseModel):
    """Input contract for the Task A review simulation endpoint"""
    user_id: str
    product_title: str
    category: str
    additional_context: str = ""

class RecommendationRequest(BaseModel):
    """Input contract for the Task B intelligent recommendation endpoint"""
    user_id: str
    context_signal: str
    target_category: str
