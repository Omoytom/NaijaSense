import os
import sys
import json
import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path when Streamlit runs this script.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your validated AI/ML engines from Phase 1
from src.agents.user_modelling import run_dual_head_simulation
from src.agents.recommender import run_multi_agent_recommender

# 1. Global Page Layout & Theme Configuration
st.set_page_config(
    page_title="NaijaSense Cognitive Engine", 
    page_icon="🚀", 
    layout="wide"
)

# Custom styling to give it an elite corporate feel
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stTabs [data-baseweb="tab"] {font-size: 1.1rem; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)

st.title(" NaijaSense: Agentic Personalization & Recommendation Engine")
st.caption("A culturally intelligent, context-sensitive reasoning framework tailored for the Bluechip Tech AI Challenge.")
st.markdown("---")

# 2. Optimized Local Data Asset Caching
STAGING_PATH = "data/staging_dataset.json"

@st.cache_data
def load_cached_profiles():
    if not os.path.exists(STAGING_PATH):
        return pd.DataFrame()
    return pd.read_json(STAGING_PATH)

df = load_cached_profiles()

if df.empty:
    st.error(" Local staging data asset missing! Please run 'python -m src.utils.data_loader' first.")
else:
    # 3. Sidebar Profile Navigation
    st.sidebar.header("👤 User Persona Registry")
    unique_user_ids = df["user_id"].unique()
    selected_user_id = st.sidebar.selectbox("Select Target User ID:", unique_user_ids)
    
    # Isolate selected user's cross-domain transaction history
    user_history_df = df[df["user_id"] == selected_user_id]
    
    st.sidebar.markdown("### Transaction Footprint")
    st.sidebar.metric("Historical Review Count", len(user_history_df))
    
    # Collapsible viewport to inspect raw context constraints
    with st.sidebar.expander(" View Profile History Logs", expanded=False):
        st.dataframe(
            user_history_df[["domain_category", "rating", "product_title"]],
            hide_index=True
        )

    # 4. Task Modular Layout Tabs
    tab1, tab2 = st.tabs([
        " Task A: Deep User Modeling & Simulation", 
        " Task B: Reason-Before-Recommend Loop"
    ])

    
    # WORKSPACE FOR TASK A: USER MODELING
    
    with tab1:
        st.header("Simulate Star Ratings and Written Reviews")
        st.info("Simulates an explicit behavioral critique for an unseen product by enforcing strict persona constraints.")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("Target Item Parameters")
            item_category = st.selectbox("Product Category:", ["Electronics", "Books"])
            item_title = st.text_input(
                "Product Title / Specification:", 
                value="Heavy Duty Anti-Surge Multi-Plug Extension Box with Type-C Ports"
            )
            item_context = st.text_area(
                "Additional Product Spec Details:", 
                value="Thick 3-meter copper cable, high thermal resilience, built-in safety fuse circuit breaker."
            )
            
            run_sim = st.button(" Execute Simulation Agent", use_container_width=True)
            
        with col2:
            st.subheader("Agent Execution Feed")
            if run_sim:
                with st.spinner("Calibrating rating scale and generating localized voice..."):
                    try:
                        # Direct execution call to David's verified Dual-Head script
                        sim_res = run_dual_head_simulation(
                            user_id=selected_user_id,
                            product_title=item_title,
                            category=item_category,
                            additional_context=item_context
                        )
                        
                        # Visually separate analytical metrics from creative text outputs
                        st.metric(label="Predicted Numerical Rating", value=f"{sim_res.predicted_rating} / 5.0")
                        
                        st.markdown("####  Critic Rating Justification")
                        st.write(sim_res.critic_reasoning)
                        
                        st.markdown("#### 🇳🇬 Generated Localized Text Review")
                        st.chat_message("user").write(sim_res.generated_review)
                        
                        with st.expander(" View Cultural Alignment Diagnostics"):
                            st.write(sim_res.cultural_notes)
                            
                    except Exception as e:
                        st.error(f"Execution Error: {e}")
            else:
                st.caption("Awaiting parameters. Click 'Execute Simulation Agent' to trigger the cognitive engine.")

    
    # WORKSPACE FOR TASK B: INTELLIGENT RECOMMENDATION
    
    with tab2:
        st.header("Context-Sensitive Cross-Domain Recommendation")
        st.info("Triggers a ReAct sequence to form hypotheses and query cross-domain candidates before suggesting final items.")
        
        col3, col4 = st.columns([1, 1], gap="large")
        
        with col3:
            st.subheader("Real-Time Context Input")
            rec_category = st.selectbox("Target Category for Recommendations:", ["Electronics", "Books"])
            context_signal = st.text_area(
                "Live User Situation / Signal:", 
                value="User is scaling up a remote coding workstation in Lagos and needs to protect high-end tech from severe grid fluctuations."
            )
            
            run_rec = st.button(" Generate Smart Recommendations", use_container_width=True)
            
        with col4:
            st.subheader("Agent Cognitive Output")
            if run_rec:
                with st.spinner("Analyzing profile patterns and testing candidate shelf space..."):
                    try:
                        # Direct execution call to David's verified ReAct recommender
                        rec_res = run_multi_agent_recommender(
                            user_id=selected_user_id,
                            context_signal=context_signal,
                            target_category=rec_category
                        )
                        
                        st.markdown("####  Agent Internal Thought Process")
                        st.info(rec_res.thought_process)
                        
                        st.markdown("####  Selected Filtering Heuristic")
                        st.warning(rec_res.selected_action)
                        
                        st.markdown("#### 🇳🇬 Curated Top Recommendations")
                        for idx, item in enumerate(rec_res.recommendations, 1):
                            with st.container():
                                st.markdown(f"**{idx}. {item.product_title}**")
                                st.caption(f"Category: {item.category} | Product ID: {item.item_id}")
                                st.write(f"*{item.curation_justification}*")
                                st.markdown("---")
                                
                    except Exception as e:
                        st.error(f"Recommendation Error: {e}")
            else:
                st.caption("Awaiting live signals. Click 'Generate Smart Recommendations' to watch the agent reason.")