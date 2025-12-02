import streamlit as st
import os
import json
from project_codex.core.manager_factory import get_managers

st.set_page_config(layout="wide", page_title="Knowledge Base")

_, _, rag_engine, term_manager, _, _, _ = get_managers()

st.title("📚 Knowledge Base")

tab1, tab2, tab3 = st.tabs(["Glossary Terms", "Style Bank", "Role Cards (Future)"])

# --- Tab 1: Glossary ---
with tab1:
    st.header("Glossary Management")
    st.markdown(
        "Upload CSV files to sync terms with the RAG engine. (Headers: `original`, `translation`, `remark`)"
    )

    uploaded_file = st.file_uploader("Upload Glossary (CSV)", type="csv")
    if uploaded_file:
        # Save temp file
        TEMP_PATH = f"temp_{uploaded_file.name}"
        with open(TEMP_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Syncing terms to Vector DB..."):
                term_manager.sync_terms_to_rag(TEMP_PATH)
            st.success("Glossary Synced Successfully!")
        except Exception as e:
            st.error(f"Sync failed: {e}")
        finally:
            if os.path.exists(TEMP_PATH):
                os.remove(TEMP_PATH)

    st.divider()

    st.subheader("Export")
    if st.button("Generate Glossary CSV"):
        try:
            export_path = "glossary_export.csv"
            term_manager.export_terms_to_csv(export_path)
            with open(export_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="Download Glossary",
                    data=f,
                    file_name="glossary_export.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Export failed: {e}")

# --- Tab 2: Style Bank ---
with tab2:
    st.header("Style Bank")
    st.markdown("Upload style references. Can be simple TXT or JSON with metadata.")
    st.info("JSON Format: `[{'text': '...', 'tag': 'scifi'}]`")

    uploaded_style = st.file_uploader(
        "Upload Style Bank (TXT/JSON)", type=["txt", "json"]
    )
    if uploaded_style:
        try:
            content = uploaded_style.read()
            # Decode
            try:
                decoded = content.decode("utf-8")
            except:
                decoded = content.decode("gbk")

            if uploaded_style.type == "application/json":
                data = json.loads(decoded)
                texts = []
                metadatas = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            texts.append(item)
                            metadatas.append({"tag": "general"})
                        elif isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                            metadatas.append({"tag": item.get("tag", "general")})
                if texts:
                    rag_engine.add_texts("style_bank", texts, metadatas=metadatas)
                    st.success(f"Added {len(texts)} styles with metadata to Bank!")
                else:
                    st.warning("No valid text found in JSON.")
            else:
                # TXT
                texts = [t.strip() for t in decoded.split("\n") if t.strip()]
                if texts:
                    rag_engine.add_texts("style_bank", texts)
                    st.success(f"Added {len(texts)} style fragments!")
        except Exception as e:
            st.error(f"Style upload failed: {e}")

# --- Tab 3: Role Cards ---
with tab3:
    st.header("Role Cards")
    st.info(
        "Coming soon: Manage character profiles for consistent personality handling."
    )
