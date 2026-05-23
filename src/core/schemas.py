from pydantic import BaseModel, Field
from typing import List, Optional

#  REFACTORED HYBRID INPUT MATRICES


class UserContextInput(BaseModel):
    """Unified profile contract allowing dynamic internal context generation"""
    name: str = Field(description="User identification label")
    age: int = Field(description="User age, used to calibrate linguistic generation vectors")
    selected_id: str = Field(description="The dataset row key identifier, or 'None' for new profiles")
    interests: List[str] = Field(description="Declared consumer domain paths (Movies, Books, Electronics)")
    
    # Inferred autonomously by Gemini if the user is a fresh sandbox profile
    demographic_profile: Optional[str] = Field(default=None, description="Inferred socio-economic background")
    infrastructure_context: Optional[str] = Field(default=None, description="Inferred power grid / network settings")

class ReviewGenerationRequest(BaseModel):
    """Input encapsulation for Task A: Decoupled Review Simulation"""
    user_context: UserContextInput
    product_category: str
    product_title: str
    product_specifications: str
    additional_details: Optional[str] = ""

class RecommendationRequest(BaseModel):
    """Input encapsulation for Task B: Multi-Agent Storefront Arbitration"""
    user_context: UserContextInput
    target_category: str
    browsing_spec_context: str


#  TASK A: OUTPUT SYSTEM SEGMENTS (CRITIC & VOICE SIMULATION)


class CriticOutput(BaseModel):
    """Structured containment for the analytical evaluation head"""
    predicted_rating: float = Field(description="Strict categorical score between 1.0 and 5.0")
    rating_justification: str = Field(description="Analytical trace proving alignment with infrastructure bounds")

class VoiceOutput(BaseModel):
    """Structured containment for the age-stratified linguistic expression head"""
    generated_review: str = Field(description="First-person localized consumer review incorporating age-appropriate slang")
    cultural_nuance_notes: str = Field(description="Linguistic explanation highlighting chosen regional terminology")

class UnifiedSimulationResult(BaseModel):
    """The final payload returned to the frontend client for Task A"""
    user_id: str
    predicted_rating: float
    critic_reasoning: str
    generated_review: str
    cultural_notes: str
    inferred_demographic: str      # Diagnostic visibility fields for the UI sandbox
    inferred_infrastructure: str   


class RecommendedItem(BaseModel):
    """Streamlined product representation with absolute data purity and zero image footprint"""
    item_id: str = Field(description="The exact alphanumeric parent_asin ID of the item.")
    product_title: str = Field(description="The formatted product name string from the catalog dataset.")
    category: str = Field(description="The domain market segment path (e.g., Books, Electronics, Movies).")
    curation_justification: str = Field(description="The tailored justification explaining why this item fits the user's inferred context.")

class MultiAgentArbitrationOutput(BaseModel):
    """Internal diagnostic trace of the competitive agent debate loop"""
    infrastructure_realist_critique: str
    value_budget_hawk_critique: str
    technical_visionary_critique: str
    arbitrator_synthesis: str
    recommendations: List[RecommendedItem]

class RecommendationResponse(BaseModel):
    """The clean, final production payload returned to your teammate's frontend client for Task B"""
    user_id: str
    inferred_infrastructure_notes: str = Field(description="The dynamic context layer's structural assessment.")
    internal_debate_transcript: str = Field(description="The raw logs of the multi-agent discussion.")
    master_resolution_summary: str = Field(description="The final decision from the Master Arbitrator.")
    recommendations: List[RecommendedItem] = Field(description="Top curated selections matching the candidate shelf space.")