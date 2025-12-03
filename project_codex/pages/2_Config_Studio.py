import streamlit as st
from project_codex.core.manager_factory import get_managers
from project_codex.utils.ui import init_page
from project_codex.utils.i18n import I18N

st.set_page_config(layout="wide", page_title="Config Studio")

init_page("sidebar_config")

# Access Managers
config_manager, _, _, _, _, _, _, _ = get_managers()

# Tabs
tab_router, tab_providers, tab_prompts, tab_profiles = st.tabs([
    I18N.get("config_router"),
    I18N.get("config_provider"),
    I18N.get("config_prompts"),
    I18N.get("config_profiles")
])

# --- Tab 1: Model Router ---
with tab_router:
    st.header("Model Router")
    st.markdown("Configure which LLM powers each step. You can enter any model ID supported by LiteLLM.")

    # Helper for quick selection
    COMMON_MODELS = [
        "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo",
        "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
        "gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash",
        "deepseek/deepseek-chat", "deepseek/deepseek-coder",
        "qwen-max", "qwen-turbo"
    ]

    col_r1, col_r2 = st.columns([1, 1])

    steps = [
        ("step_1_model", "Step 1: Extraction"),
        ("step_2_model", "Step 2: Drafting"),
        ("step_3_model", "Step 3: Judging"),
        ("step_4_model", "Step 4: Polishing")
    ]

    for key, label in steps:
        current_val = config_manager.models.get(key, "gpt-4o")
        with st.container():
            c1, c2 = st.columns([3, 1])
            with c1:
                # Text input for flexibility
                new_val = st.text_input(label, value=current_val, key=f"input_{key}")
            with c2:
                # Quick select helper
                # We use a selectbox that updates the text input via state if changed
                # But syncing them bidirectionally is tricky in Streamlit.
                # Simplified: Just text input with help tooltip.
                st.caption("Common: " + ", ".join(COMMON_MODELS[:3]) + "...")

        # Update config immediately
        if new_val != current_val:
            config_manager.models[key] = new_val

# --- Tab 2: Provider Config ---
with tab_providers:
    st.header("Provider Configuration")
    st.markdown("Configure API Keys and Base URLs for each provider.")

    providers = ["openai", "anthropic", "google", "deepseek", "aliyun"]

    for p in providers:
        with st.expander(f"{p.capitalize()} Settings", expanded=False):
            p_conf = config_manager.providers.get(p, {})

            c1, c2 = st.columns(2)
            with c1:
                new_key = st.text_input(
                    f"{p.capitalize()} API Key",
                    value=p_conf.get("api_key", ""),
                    type="password",
                    key=f"key_{p}"
                )
            with c2:
                new_base = st.text_input(
                    f"{p.capitalize()} Base URL",
                    value=p_conf.get("base_url", ""),
                    placeholder="Leave empty for default",
                    key=f"base_{p}"
                )

            # Save button for this provider
            if st.button(f"Update {p.capitalize()}", key=f"btn_{p}"):
                config_manager.set_api_key(p, new_key)
                config_manager.set_base_url(p, new_base)
                st.success(f"{p.capitalize()} settings updated!")

# --- Tab 3: Prompt Lab ---
with tab_prompts:
    st.header("Prompt Lab")
    st.markdown("Customize system prompts.")

    prompts = config_manager.prompts

    new_extractor = st.text_area("Step 1: Extractor", value=prompts.get("extractor", ""), height=150)
    new_drafter = st.text_area("Step 2: Drafter", value=prompts.get("drafter_p1", ""), height=150)
    new_judge = st.text_area("Step 3: Judge", value=prompts.get("judge", ""), height=150)
    new_polisher = st.text_area("Step 4: Polisher", value=prompts.get("polisher_p2", ""), height=150)

    if st.button("Save Prompts"):
        updated = {
            "extractor": new_extractor,
            "drafter_p1": new_drafter,
            "judge": new_judge,
            "polisher_p2": new_polisher
        }
        config_manager.save_prompts(updated)
        st.success("Prompts saved!")

# --- Tab 4: Profiles ---
with tab_profiles:
    st.header("Config Profiles")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Save Profile")
        p_name = st.text_input("Profile Name")
        if st.button("Save Current Profile"):
            if p_name:
                config_manager.save_profile(p_name)
                st.success(f"Profile '{p_name}' saved!")

    with col_p2:
        st.subheader("Load Profile")
        profiles = config_manager.list_profiles()
        if profiles:
            sel_prof = st.selectbox("Select Profile", profiles)
            if st.button("Load Profile"):
                config_manager.load_profile(sel_prof)
                st.success(f"Loaded '{sel_prof}'! Refresh to see changes.")
                st.rerun()
