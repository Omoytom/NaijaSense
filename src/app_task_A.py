import os
import sys
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.schemas import UserContextInput, ReviewGenerationRequest
from src.agents.user_modelling import run_dynamic_user_simulation
from src.core.registry import get_or_create_user_registry

st.set_page_config(page_title="NaijaSense Task A Sandbox", layout="wide", page_icon="📝")
st.title("🇳🇬 NaijaSense: Task A review Calibration Studio")
st.caption("Independent Deployment Target — Review Generation Sandbox Link")

STAGING_PATH = "data/staging_dataset.json"
user_registry = get_or_create_user_registry()
user_id_pool = ["None"] + list(user_registry.keys())

# ==========================================
# SIDEBAR PERSONALIZATION ENGINE
# ==========================================
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
        interests_in = st.multiselect("Explicit Declared Interests", options=["Movies", "Electronics", "Books", "Software"], default=["Electronics", "Books"])

    active_user_context = UserContextInput(
        name=name_in, age=int(age_in), selected_id=id_in, interests=interests_in,
        demographic_profile="Inferred", infrastructure_context="Inferred"
    )


# WORKBENCH INTERFACE EXECUTIONS
st.subheader("Dual-Head Structural Review Simulation Workspace")
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### Configure Input Space")
    target_cat = st.selectbox("Target Product Category", ["Movies", "Electronics", "Books"])
    
    available_titles = []
    if os.path.exists(STAGING_PATH):
        try:
            df_load = pd.read_json(STAGING_PATH)
            cat_match = df_load[df_load["domain_category"].str.lower() == target_cat.lower()]
            if not cat_match.empty:
                available_titles = sorted(cat_match["product_title"].dropna().unique().tolist())
        except Exception:
            pass
            
    if not available_titles:
        available_titles = [f"Sample Premium {target_cat} Choice Option Asset"]

    selected_title = st.selectbox("Select Product / Movie From Dataset", options=available_titles)
    
    default_specs = ""
    if os.path.exists(STAGING_PATH) and selected_title:
        try:
            df_load = pd.read_json(STAGING_PATH)
            matched_row = df_load[df_load["product_title"] == selected_title].iloc[0]
            default_specs = f"ASIN Reference: {matched_row.get('item_id')}. Metadata footprint traces: {matched_row.get('review_text', '')[:250]}"
        except Exception:
            default_specs = "Standard baseline item parameters configuration."

    with st.form("sim_form_standalone"):
        prod_specs = st.text_area("Technical Specifications / Movie Metadata Plot-lines", value=default_specs, height=120)
        add_details = st.text_input("Additional Scenario Details (Optional)", "")
        submit_sim = st.form_submit_button("Launch Simulation Execution")

with col_right:
    st.markdown("### Simulation Engine Outputs")
    if submit_sim:
        payload = ReviewGenerationRequest(
            user_context=active_user_context, product_category=target_cat,
            product_title=selected_title, product_specifications=prod_specs, additional_details=add_details
        )
        with st.spinner("Processing zero-shot cognitive calculations..."):
            try:
                response = run_dynamic_user_simulation(payload)
                st.metric("Predicted Grading Value", f"⭐ {response.predicted_rating} / 5.0")
                st.info(f"**Critic Justification:**\n{response.critic_reasoning}")
                st.success(f"**Generated Localized Review:**\n\n\"{response.generated_review}\"")
                st.warning(f"**Cultural Nuance Notes:**\n{response.cultural_notes}")
                
                if id_in == "None":
                    with st.expander("🔍 View AI Internal Context Inference Diagnostics", expanded=True):
                        st.markdown(f"** Assumed Demographic Cohort:** {response.inferred_demographic}")
                        st.markdown(f"** Assumed Infrastructure Realities:** {response.inferred_infrastructure}")
            except Exception as err:
                st.error(f"Execution Error: {err}")