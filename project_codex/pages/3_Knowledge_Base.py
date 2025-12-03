import streamlit as st
import pandas as pd
import os
from project_codex.core.manager_factory import get_managers
from project_codex.utils.ui import init_page
from project_codex.utils.i18n import I18N

st.set_page_config(layout="wide", page_title="Knowledge Base")

init_page("sidebar_kb")

# Access Managers
_, _, rag_engine, term_manager, _, _, _, char_mgr = get_managers()

tab1, tab2 = st.tabs([I18N.get("kb_glossary"), I18N.get("kb_chars")])

# --- Tab 1: Glossary ---
with tab1:
    st.header("Glossary Management")
    st.markdown("Upload CSV or edit terms directly.")

    # 1. Upload Logic
    uploaded_file = st.file_uploader("Import CSV", type="csv")
    if uploaded_file:
         # Save to temp
         with open("temp_glossary.csv", "wb") as f:
             f.write(uploaded_file.getbuffer())

         try:
             terms = term_manager.load_terms_from_csv("temp_glossary.csv")
             st.session_state["glossary_data"] = pd.DataFrame(terms)
             st.success("CSV Loaded into Editor! Review and click 'Save & Sync'.")
         except Exception as e:
             st.error(f"Error loading CSV: {e}")
         finally:
             if os.path.exists("temp_glossary.csv"):
                 os.remove("temp_glossary.csv")

    # 2. Data Editor
    if "glossary_data" not in st.session_state:
        # Default empty
        st.session_state["glossary_data"] = pd.DataFrame(columns=["original", "translation", "remark"])

    df = st.session_state["glossary_data"]
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    # 3. Save Action
    if st.button("Save & Sync to RAG"):
        # Save to CSV (temp) then sync
        temp_path = "glossary_sync.csv"
        try:
            edited_df.to_csv(temp_path, index=False)
            term_manager.sync_terms_to_rag(temp_path)
            st.success(f"Glossary Synced! {len(edited_df)} terms indexed.")
            # Update session state
            st.session_state["glossary_data"] = edited_df
        except Exception as e:
            st.error(f"Sync failed: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    st.divider()
    if st.button("Export Current Glossary"):
        export_path = "glossary_export.csv"
        try:
            term_manager.export_terms_to_csv(export_path)
            with open(export_path, "r", encoding="utf-8") as f:
                 st.download_button("Download CSV", f, "glossary_export.csv", "text/csv")
        except Exception as e:
            st.error(f"Export failed: {e}")

# --- Tab 2: Characters ---
with tab2:
    st.header("Character Sheets")
    st.markdown("Define character profiles to maintain consistency in translation.")

    col_list, col_form = st.columns([1, 2])

    with col_list:
        st.subheader("Existing Characters")
        chars = char_mgr.list_characters()
        # Ensure we have a selection state
        if "selected_char" not in st.session_state:
            st.session_state["selected_char"] = "New Character"

        # Radio button sometimes resets if list changes, stick to key
        selection = st.radio(
            "Select Character",
            ["New Character"] + chars,
            key="char_selector"
        )

    with col_form:
        st.subheader("Edit Profile")

        # Load data if selected
        current_data = {"name": "", "bio": "", "tags": "", "quotes": ""}
        if selection != "New Character":
            try:
                current_data = char_mgr.load_character(selection)
            except Exception as e:
                st.error(f"Failed to load character: {e}")

        with st.form("char_form"):
            c_name = st.text_input("Name", value=current_data.get("name", ""))
            c_bio = st.text_area("Biography / Description", value=current_data.get("bio", ""), height=100)
            c_tags = st.text_input("Tags (e.g., Tsundere, Knight)", value=current_data.get("tags", ""))
            c_quotes = st.text_area("Typical Quotes (for Style)", value=current_data.get("quotes", ""), height=100)

            if st.form_submit_button("Save Character"):
                if c_name:
                    new_data = {
                        "name": c_name,
                        "bio": c_bio,
                        "tags": c_tags,
                        "quotes": c_quotes
                    }
                    try:
                        char_mgr.save_character(new_data)
                        st.success(f"Saved {c_name} to RAG!")
                        # Force refresh
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving: {e}")
                else:
                    st.error("Name is required.")
