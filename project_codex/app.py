import streamlit as st

st.set_page_config(
    page_title="Project Codex",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/your-repo/project-codex",
        "Report a bug": "https://github.com/your-repo/project-codex/issues",
        "About": "# Project Codex v2.0\nIntelligent Manuscript Localization System.",
    },
)

st.title("📖 Project Codex v2.0")

st.markdown(
    """
### Intelligent Manuscript Localization System

Welcome to Project Codex. Please select a module from the sidebar to begin your work.

#### Modules:
*   **🏠 Task Dashboard**: Upload manuscripts, start batch processing, and monitor progress.
*   **⚙️ Config Studio**: Configure LLM models, API keys, and System Prompts.
*   **📚 Knowledge Base**: Manage glossaries and style banks.
*   **📝 Editor Workbench**: Review and polish the localized text.

#### Quick Start:
1.  Go to **Config Studio** to set up your API keys and Models.
2.  Go to **Knowledge Base** to upload relevant glossaries.
3.  Go to **Task Dashboard** to upload your manuscript and start the pipeline.
"""
)

st.info("System initialized. Navigate using the sidebar.")
