import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.schemas import UserContextInput, RecommendationRequest
from src.agents.recommender import run_dynamic_multi_agent_recommender
from src.core.registry import get_or_create_user_registry

st.set_page_config(page_title="NaijaSense Task B Panel", layout="wide", page_icon="🏆")
st.title("🇳🇬 NaijaSense: Task B Arbitration Engine")
st.caption("Independent Deployment Target — Multi-Agent Recommendation Track Link")

user_registry = get_or_create_user_registry()
user_id_pool = ["None"] + list(user_registry.keys())


# SIDEBAR PERSONALIZATION ENGINE (IDENTICAL STRUCTURE)

with st.sidebar:
    st.header("👤 Active Persona Registry")
    st.markdown("---")
    id_in = st.selectbox("Link Actual Amazon User ID", options=user_id_pool)
    st.markdown("---")
    
    if id_in != "None":
        profile_data = user_registry[id_in]
        st.info(f" **Dataset Identity Linked**\n- Past Reviews: **{profile_data['total_historical_reviews_count']}**")
        st.success(f" **Behavior Persona:**\n*{profile_data['extracted_behavioral_persona']}*")
        name_in = st.text_input("Name Wrapper", value=profile_data.get("name", "User"), disabled=True)
        age_in = st.number_input("Age Window", value=int(profile_data.get("age", 25)), disabled=True)
        interests_in = profile_data["preferred_domains"]
    else:
        st.caption("🔓 **Sandbox Mode Active**")
        name_in = st.text_input("Profile Name", "David Omoyola")
        age_in = st.number_input("Biological Age", min_value=12, max_value=90, value=21)
        interests_in = st.multiselect("Explicit Declared Interests", options=["Movies", "Electronics", "Books", "Software"], default=["Electronics", "Books", "Movies"])

    active_user_context = UserContextInput(
        name=name_in, age=int(age_in), selected_id=id_in, interests=interests_in,
        demographic_profile="Inferred", infrastructure_context="Inferred"
    )


# WORKBENCH INTERFACE EXECUTIONS

st.subheader("Cross-Domain Reason-Before-Recommend Panel")

with st.form("rec_form_standalone"):
    rec_category = st.selectbox("Target Recommendation Track", ["Movies", "Electronics", "Books"])
    spec_context = st.text_area("Current Browsing Focus Specification Signal", 
                                value=f"Looking for options matching premium utility, tailored specifically to my declared {rec_category} expectations.")
    submit_rec = st.form_submit_button("Convene Parallelized Arbitration Panel")

if submit_rec:
    payload = RecommendationRequest(
        user_context=active_user_context, target_category=rec_category, browsing_spec_context=spec_context
    )
    with st.spinner("Convening technical micro-agent debate panels..."):
        try:
            response = run_dynamic_multi_agent_recommender(payload)
            
            with st.expander("🤖 View Micro-Agent Internal Debate Transcript Logs", expanded=True):
                st.markdown(f"** Agent A (Infrastructure Realist):**\n{response.infrastructure_realist_critique}")
                st.markdown("---")
                st.markdown(f"** Agent B (Value/Budget Hawk):**\n{response.value_budget_hawk_critique}")
                st.markdown("---")
                st.markdown(f"** Agent C (Technical / Narrative Visionary):**\n{response.technical_visionary_critique}")
            
            st.info(f" **Master Arbitrator Synthesis Resolution Summary:**\n{response.arbitrator_synthesis}")
            
            st.subheader(" Curated Selections")
            for idx, item in enumerate(response.recommendations, 1):
                with st.container():
                    img_link = item.image_url.strip() if (hasattr(item, 'image_url') and item.image_url) else ""
                    
                    if img_link and img_link.startswith("http"):
                        col_img, col_txt = st.columns([1, 4])
                        with col_img:
                            st.image(img_link, use_container_width=True)
                        with col_txt:
                            st.markdown(f"### {idx}. {item.product_title}")
                            st.caption(f"ASIN: {item.item_id} | Segment Catalog: {item.category}")
                            st.markdown(f"👉 *Curation Rationale:* {item.curation_justification}")
                    else:
                        st.markdown(f"### {idx}. {item.product_title}")
                        st.caption(f"ASIN: {item.item_id} | Segment Catalog: {item.category}")
                        st.markdown(f"👉 *Curation Rationale:* {item.curation_justification}")
                        
                st.markdown("---")
        except Exception as err:
            st.error(f"Execution Error: {err}")