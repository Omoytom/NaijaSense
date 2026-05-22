from pydantic import BaseModel, Field
from typing import List, Optional

# REFACTORED HYBRID INPUT MATRIX

class UserContextInput(BaseModel):
    """Unified profile contract allowing dynamic internal context generation"""
    name: str = Field(description="User identification label")
    age: int = Field(description="User age, used to calibrate linguistic generation vectors")
    selected_id: str = Field(description="The dataset row key identifier, or 'None' for new profiles")
    interests: List[str] = Field(description="Declared consumer domain paths (Movies, Books, Electronics)")
    
    # Made optional: Inferred autonomously by Gemini if the user is a fresh sandbox profile
    demographic_profile: Optional[str] = Field(default=None, description="Inferred socio-economic background")
    infrastructure_context: Optional[str] = Field(default=None, description="Inferred power grid / network settings")

class ReviewGenerationRequest(BaseModel):
    user_context: UserContextInput
    product_category: str
    product_title: str
    product_specifications: str
    additional_details: Optional[str] = ""

class RecommendationRequest(BaseModel):
    user_context: UserContextInput
    target_category: str
    browsing_spec_context: str

# OUTPUT SYSTEM SEGMENTS

class CriticOutput(BaseModel):
    predicted_rating: float
    rating_justification: str

class VoiceOutput(BaseModel):
    generated_review: str
    cultural_nuance_notes: str

class UnifiedSimulationResult(BaseModel):
    user_id: str
    predicted_rating: float
    critic_reasoning: str
    generated_review: str
    cultural_notes: str
    inferred_demographic: str     # Diagnostic visibility fields
    inferred_infrastructure: str  

class RecommendedItem(BaseModel):
    item_id: str
    product_title: str
    category: str
    curation_justification: str
    image_url: str

class MultiAgentArbitrationOutput(BaseModel):
    infrastructure_realist_critique: str
    value_budget_hawk_critique: str
    technical_visionary_critique: str
    arbitrator_synthesis: str
    recommendations: List[RecommendedItem]