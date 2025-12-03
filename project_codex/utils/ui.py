import streamlit as st
from project_codex.utils.session_manager import SessionManager
from project_codex.utils.i18n import I18N

def init_page(page_title_key: str = "app_title"):
    """
    Initializes the page with SessionManager, Sidebar settings, and Title.
    Returns the I18N helper for convenience.
    """
    # 1. Init Session
    SessionManager.init_session()

    # 2. Sidebar Global Settings
    with st.sidebar:
        st.header("Global Settings")

        # Language Selector
        # We use session state to persist choice
        current_idx = 0
        if st.session_state.get("language") == "English":
            current_idx = 1

        selected_lang = st.selectbox(
            "Language / 语言",
            ["中文", "English"],
            index=current_idx,
            key="language"
        )
        # Note: 'key="language"' automatically updates st.session_state["language"]

        st.divider()
        st.caption("Project Codex v2.1")

    # 3. Set Page Title (Browser tab and Header)
    # st.set_page_config must be called at top level of script, so we assume page called it.
    # But we can set the header title here.
    st.title(I18N.get(page_title_key))
