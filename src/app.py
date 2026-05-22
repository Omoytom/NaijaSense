import os
import sys
import json
import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.schemas import UserContextInput, ReviewGenerationRequest, RecommendationRequest
from src.agents.user_modelling import run_dynamic_user_simulation
from src.agents.recommender import run_dynamic_multi_agent_recommender
from src.core.registry import get_or_create_user_registry

st.set_page_config(page_title="NaijaSense Framework Studio", layout="wide", page_icon="🇳🇬")
st.title("🇳🇬 NaijaSense: Monolithic Diagnostic Studio")
st.caption("Direct Core Pipeline Integration - Local Data Execution Matrix")

STAGING_PATH = "data/staging_dataset.json"
REGISTRY_PATH = "data/user_registry.json"

# Load the real behavioral registry mapping file
user_registry = get_or_create_user_registry()
user_id_pool = ["None"] + list(user_registry.keys())

# ==========================================
# SIDEBAR RECONSTRUCTION CONTROL PANEL
# ==========================================
# Open src/app.py and replace the complete 'with st.sidebar:' section with this block:

with st.sidebar:
    st.header("👤 Personalization Control Matrix")
    st.markdown("---")
    
    id_in = st.selectbox("Link Actual Amazon User ID", options=user_id_pool)
    st.markdown("---")
    
    # 👤 USER METADATA IDENTIFICATION UPDATE
    if id_in != "None":
        profile_data = user_registry[id_in]
        st.info(f"📊 **Actual Dataset Identity Connected**\n"
                f"- User Key Reference: `{id_in[:10]}...`\n"
                f"- Past Reviews Traced: **{profile_data['total_historical_reviews_count']}**")
        st.success(f"🧠 **AI-Extracted Behavior Persona:**\n*{profile_data['extracted_behavioral_persona']}*")
        
        # FIX: Extract the actual identity metrics from the look up registry cache dynamically
        name_in = st.text_input("Name Wrapper", value=profile_data.get("name", f"User_{id_in[:4]}"), disabled=True)
        age_in = st.number_input("Age Window", value=int(profile_data.get("age", 25)), disabled=True)
        interests_in = profile_data["preferred_domains"]
        st.caption("🔒 *Inputs locked to historical trace constants.*")
    else:
        st.subheader("🛠️ Create Custom Profile")
        st.caption("🔓 *Sandbox Mode Active. Context Inference Layer will deduce environmental parameters.*")
        name_in = st.text_input("Profile Name", "David Omoyola")
        age_in = st.number_input("Biological Age", min_value=12, max_value=90, value=21)
        interests_in = st.multiselect("Explicit Declared Interests", 
                                      options=["Movies", "Electronics", "Books", "Software"],
                                      default=["Electronics", "Books", "Movies"])

    # Package clean arguments without manual entries
    active_user_context = UserContextInput(
        name=name_in,
        age=int(age_in),
        selected_id=id_in,
        interests=interests_in,
        demographic_profile="Inferred",
        infrastructure_context="Inferred"
    )

tabs = st.tabs(["📝 Task A: Review Simulation Sandbox", "🏆 Task B: Multi-Agent Recommendation Track"])

# ==========================================
# TAB 1: RUNNING DIRECT SIMULATION TASK A
# ==========================================
with tabs[0]:
    st.subheader("Dual-Head Structural Review Simulation Engine")
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### Configure Input Space")
        target_cat = st.selectbox("Target Product Category", ["Movies", "Electronics", "Books"])
        
        # Gather unique product items directly from our local dataset tracking file
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

        # 🛒 DATASET FIX: Product titles are selected via an explicit dropdown selection box
        selected_title = st.selectbox("Select Product / Movie From Dataset", options=available_titles)
        
        # Automatically pull technical metadata context records when a user shifts selections
        default_specs = ""
        if os.path.exists(STAGING_PATH) and selected_title:
            try:
                df_load = pd.read_json(STAGING_PATH)
                matched_row = df_load[df_load["product_title"] == selected_title].iloc[0]
                default_specs = f"ASIN Reference: {matched_row.get('item_id')}. Metadata footprint traces: {matched_row.get('review_text', '')[:250]}"
            except Exception:
                default_specs = "Standard baseline item parameters configuration."

        with st.form("sim_form_v4"):
            # 💡 DISCRETION FIX: Specifications remain completely free and open for manual user adjustments
            prod_specs = st.text_area("Technical Specifications / Movie Metadata Plot-lines", value=default_specs, height=120)
            add_details = st.text_input("Additional Scenario Details (Optional)", "")
            submit_sim = st.form_submit_button("Launch Simulation Execution")

    with col_right:
        st.markdown("### Direct Process Outputs")
        if submit_sim:
            payload = ReviewGenerationRequest(
                user_context=active_user_context,
                product_category=target_cat,
                product_title=selected_title,
                product_specifications=prod_specs,
                additional_details=add_details
            )
            with st.spinner("Processing zero-shot cognitive calculations..."):
                try:
                    response = run_dynamic_user_simulation(payload)
                    st.metric("Predicted Grading Value", f"⭐ {response.predicted_rating} / 5.0")
                    st.info(f"**Critic Structural Justification:**\n{response.critic_reasoning}")
                    st.success(f"**Generated First-Person Review Text:**\n\n\"{response.generated_review}\"")
                    st.warning(f"**Age-Localized Cultural Notes:**\n{response.cultural_notes}")
                    
                    if id_in == "None":
                        with st.expander("🔍 View AI Internal Context Inference Diagnostics", expanded=True):
                            st.markdown(f"**🤖 Assumed Demographic Cohort:** {response.inferred_demographic}")
                            st.markdown(f"**⚡ Assumed Infrastructure Realities:** {response.inferred_infrastructure}")
                            
                except Exception as err:
                    st.error(f"Execution Error processing local parameters: {err}")

# ==========================================
# TAB 2: RUNNING DIRECT ARCHITECTURE TASK B
# ==========================================
with tabs[1]:
    st.subheader("Cross-Domain Reason-Before-Recommend Panel")
    
    with st.form("rec_form_v4"):
        rec_category = st.selectbox("Target Recommendation Track", ["Movies", "Electronics", "Books"])
        spec_context = st.text_area("Current Browsing Focus Specification Signal", 
                                    value=f"Looking for choices matching premium utility, tailored specifically to my declared {rec_category} expectations.")
        submit_rec = st.form_submit_button("Convene Parallelized Arbitration Panel")

    if submit_rec:
        payload = RecommendationRequest(
            user_context=active_user_context,
            target_category=rec_category,
            browsing_spec_context=spec_context
        )
        with st.spinner("Convening technical micro-agent debate panels..."):
            try:
                response = run_dynamic_multi_agent_recommender(payload)
                
                with st.expander("🤖 View Micro-Agent Internal Debate Transcript Logs", expanded=True):
                    st.markdown(f"**🧱 Agent A (Infrastructure Realist):**\n{response.infrastructure_realist_critique}")
                    st.markdown("---")
                    st.markdown(f"**🦅 Agent B (Value/Budget Hawk):**\n{response.value_budget_hawk_critique}")
                    st.markdown("---")
                    st.markdown(f"**⚡ Agent C (Technical / Narrative Visionary):**\n{response.technical_visionary_critique}")
                
                st.info(f"⚖️ **Master Arbitrator Synthesis Resolution Summary:**\n{response.arbitrator_synthesis}")
                
                # Open src/app.py and locate the Tab 2 output rendering loop block. 
# Replace the old loop with this clean, adaptive version:

                st.subheader("🏆 Curated Selections")
                for idx, item in enumerate(response.recommendations, 1):
                    with st.container():
                        # Read the product image link straight from the dataset item variables
                        img_link = item.image_url.strip() if (hasattr(item, 'image_url') and item.image_url) else ""
                        
                        if img_link and img_link.startswith("http"):
                            # 🖼️ Case A: An authentic image exists. Render using the 1:4 side-by-side grid.
                            col_img, col_txt = st.columns([1, 4])
                            with col_img:
                                st.image(img_link, use_container_width=True)
                            with col_txt:
                                st.markdown(f"### {idx}. {item.product_title}")
                                st.caption(f"ASIN: {item.item_id} | Segment Catalog: {item.category}")
                                st.markdown(f"👉 *Curation Rationale:* {item.curation_justification}")
                        else:
                            # 🚫 Case B: Image is missing. Render text elements across the full width of the screen.
                            st.markdown(f"### {idx}. {item.product_title}")
                            st.caption(f"ASIN: {item.item_id} | Segment Catalog: {item.category}")
                            st.markdown(f"👉 *Curation Rationale:* {item.curation_justification}")
                            
                    st.markdown("---")
            except Exception as err:
                st.error(f"Execution error processing arbitration matrices: {err}")