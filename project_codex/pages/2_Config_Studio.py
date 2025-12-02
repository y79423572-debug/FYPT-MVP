import streamlit as st
import os
from project_codex.core.manager_factory import get_managers

st.set_page_config(layout="wide", page_title="Config Studio")

# Access Managers
config_manager, _, _, _, _, _, _ = get_managers()

st.title("⚙️ Config Studio")

# Tabs
tab1, tab2, tab3 = st.tabs(["Model Router", "Prompt Lab", "Profiles"])

# --- Tab 1: Model Router ---
with tab1:
    st.header("Model Router")
    st.markdown("Configure which LLM powers each step of the pipeline.")

    # Available models (Hardcoded for now as per common LiteLLM supports,
    # but ideally could be fetched if supported)
    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Step Models")

        # We need to bind these to config_manager.models
        # We use a helper to update session state or config directly

        step1 = st.selectbox(
            "Step 1: Extraction",
            options=AVAILABLE_MODELS,
            index=(
                AVAILABLE_MODELS.index(
                    config_manager.models.get("step_1_model", "gpt-4o")
                )
                if config_manager.models.get("step_1_model") in AVAILABLE_MODELS
                else 0
            ),
        )
        step2 = st.selectbox(
            "Step 2: Drafting",
            options=AVAILABLE_MODELS,
            index=(
                AVAILABLE_MODELS.index(
                    config_manager.models.get("step_2_model", "gpt-4o")
                )
                if config_manager.models.get("step_2_model") in AVAILABLE_MODELS
                else 0
            ),
        )
        step3 = st.selectbox(
            "Step 3: Judging",
            options=AVAILABLE_MODELS,
            index=(
                AVAILABLE_MODELS.index(
                    config_manager.models.get("step_3_model", "gpt-4o")
                )
                if config_manager.models.get("step_3_model") in AVAILABLE_MODELS
                else 0
            ),
        )
        step4 = st.selectbox(
            "Step 4: Polishing",
            options=AVAILABLE_MODELS,
            index=(
                AVAILABLE_MODELS.index(
                    config_manager.models.get("step_4_model", "gpt-4o")
                )
                if config_manager.models.get("step_4_model") in AVAILABLE_MODELS
                else 0
            ),
        )

        # Update config object (in memory)
        config_manager.models["step_1_model"] = step1
        config_manager.models["step_2_model"] = step2
        config_manager.models["step_3_model"] = step3
        config_manager.models["step_4_model"] = step4

    with col2:
        st.subheader("API Keys")
        st.caption("Keys are stored in environment variables / session.")

        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=config_manager.api_keys.get("openai", ""),
        )
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=config_manager.api_keys.get("anthropic", ""),
        )
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=config_manager.api_keys.get("gemini", ""),
        )

        if st.button("Update API Keys"):
            if openai_key:
                config_manager.set_api_key("openai", openai_key)
            if anthropic_key:
                config_manager.set_api_key("anthropic", anthropic_key)
            if gemini_key:
                config_manager.set_api_key("gemini", gemini_key)
            st.success("API Keys updated in memory and environment.")

# --- Tab 2: Prompt Lab ---
with tab2:
    st.header("Prompt Lab")
    st.markdown(
        "Customize system prompts for each pipeline step. Use `{placeholders}` for dynamic injection."
    )

    prompts = config_manager.prompts

    new_extractor = st.text_area(
        "Step 1: Extractor System Prompt",
        value=prompts.get("extractor", ""),
        height=150,
        help="Variables: None (Input is raw text)",
    )

    new_drafter = st.text_area(
        "Step 2: Drafter System Prompt",
        value=prompts.get("drafter_p1", ""),
        height=150,
        help="Variables: {glossary_text} (if using glossary)",
    )

    new_judge = st.text_area(
        "Step 3: Judge System Prompt",
        value=prompts.get("judge", ""),
        height=150,
        help="Variables: {project_context}",
    )

    new_polisher = st.text_area(
        "Step 4: Polisher System Prompt",
        value=prompts.get("polisher_p2", ""),
        height=150,
        help="Variables: {critique}, {style_examples}",
    )

    if st.button("Save Prompts"):
        updated = {
            "extractor": new_extractor,
            "drafter_p1": new_drafter,
            "judge": new_judge,
            "polisher_p2": new_polisher,
        }
        config_manager.save_prompts(updated)
        st.success("Prompts saved to system_prompts.json")

# --- Tab 3: Profiles ---
with tab3:
    st.header("Config Profiles")
    st.markdown("Save and load entire configuration sets (Models + Prompts).")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.subheader("Save Current Profile")
        profile_name = st.text_input(
            "Profile Name", placeholder="e.g., SciFi_Translation_v1"
        )
        if st.button("Save Profile"):
            if profile_name:
                config_manager.save_profile(profile_name)
                st.success(f"Profile '{profile_name}' saved!")
            else:
                st.error("Please enter a profile name.")

    with col_p2:
        st.subheader("Load Profile")
        profiles = config_manager.list_profiles()
        if profiles:
            selected_profile = st.selectbox("Select Profile", profiles)
            if st.button("Load Selected Profile"):
                try:
                    config_manager.load_profile(selected_profile)
                    st.success(
                        f"Profile '{selected_profile}' loaded! Please refresh the page to see changes."
                    )
                except Exception as e:
                    st.error(f"Error loading profile: {e}")
        else:
            st.info("No profiles found.")
