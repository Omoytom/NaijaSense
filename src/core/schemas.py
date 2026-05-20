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

class ReActRecommendationOutput(BaseModel):
    """Head 3: ReAct Reason-Before-Recommend Output Structure"""
    thought_process: str         # The agent's raw internal psychological deduction
    selected_action: str         # The filtering heuristic chosen by the agent
    recommendations: List[RecommendedItem]