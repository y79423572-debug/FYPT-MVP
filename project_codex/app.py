import streamlit as st
from project_codex.utils.ui import init_page
from project_codex.utils.i18n import I18N

st.set_page_config(
    page_title="Project Codex",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/your-repo/project-codex',
        'About': "# Project Codex v2.1"
    }
)

init_page("app_title")

st.markdown(I18N.get("welcome_msg"))

st.markdown("""
### Modules:
*   **""" + I18N.get("sidebar_dashboard") + """**: Upload manuscripts, start batch processing.
*   **""" + I18N.get("sidebar_config") + """**: Configure LLM models & API keys.
*   **""" + I18N.get("sidebar_kb") + """**: Manage glossaries and characters.
*   **""" + I18N.get("sidebar_workbench") + """**: Review and polish.
""")
