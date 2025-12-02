"""
Main Streamlit Application for Project Codex.
Handles UI, Configuration, and Task execution.
"""
import os
import uuid
import time
import json
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from project_codex.utils.config import ConfigManager
from project_codex.utils.db import DBManager, GlobalCircuitBreakerError
from project_codex.core.rag_engine import RAGEngine
from project_codex.core.term_manager import TermManager
from project_codex.core.llm_client import LLMClient
from project_codex.core.pipeline import WorkflowEngine

# Page Config
st.set_page_config(page_title="Project Codex", layout="wide")


# Initialize Resources (Cached)
@st.cache_resource
def get_managers():
    """Initialize and cache all manager instances."""
    c_mgr = ConfigManager()
    d_mgr = DBManager()
    r_eng = RAGEngine()
    t_mgr = TermManager(r_eng)
    l_cli = LLMClient()
    w_eng = WorkflowEngine(c_mgr, l_cli, r_eng)
    ex = ThreadPoolExecutor(max_workers=3)  # Limit concurrency
    return c_mgr, d_mgr, r_eng, t_mgr, l_cli, w_eng, ex


config_manager, db_manager, rag_engine, term_manager, llm_client, workflow_engine, executor = get_managers()


# --- Worker Function ---
def run_pipeline(t_id: str, text: str, p_context: str = "", s_tag: str = "general"):
    """
    Background worker to run the full pipeline.
    """
    total_tokens = 0
    total_cost = 0.0

    def update_metrics(metadata):
        nonlocal total_tokens, total_cost
        if metadata:
            total_tokens += metadata.get("tokens", 0)
            total_cost += metadata.get("cost", 0.0)

    try:
        # Helper to check cancellation
        def check_cancellation():
            current_task = db_manager.get_task(t_id)
            if current_task and current_task["status"] == "Cancelled":
                raise InterruptedError("Pipeline stopped by user.")

        # Check Circuit Breaker
        db_manager.check_circuit()

        db_manager.update_task_status(t_id, "Processing", step="Extraction")
        check_cancellation()

        # Step 1: Extraction
        _, meta1 = workflow_engine.step1_extract_terms(text)
        update_metrics(meta1)

        db_manager.update_task_status(t_id, "Processing", step="Drafting")
        check_cancellation()

        # Step 2: Drafting
        draft, meta2 = workflow_engine.step2_draft(text)
        update_metrics(meta2)
        db_manager.update_task_status(
            t_id, "Processing", step="Judging", draft_text=draft
        )
        check_cancellation()

        # Step 3: Judging
        critique, meta3 = workflow_engine.step3_judge(draft, project_context=p_context)
        update_metrics(meta3)
        db_manager.update_task_status(
            t_id,
            "Processing",
            step="Polishing",
            critique=json.dumps(critique, ensure_ascii=False),
        )
        check_cancellation()

        # Step 4: Polishing
        final, meta4 = workflow_engine.step4_polish(draft, critique, style_tag=s_tag)
        update_metrics(meta4)

        # Final Update with metrics
        db_manager.update_task_status(
            t_id,
            "Done",
            step="Completed",
            final_text=final,
            token_usage=total_tokens,
            cost_estimate=total_cost,
        )

    except InterruptedError:
        # Task was cancelled, status is already updated or we leave it as Cancelled
        pass
    except GlobalCircuitBreakerError as e:
        db_manager.update_task_status(t_id, "Error", error=str(e))
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Record error in circuit breaker
        try:
            db_manager.record_error()
        except GlobalCircuitBreakerError:
            pass  # Already recorded
        db_manager.update_task_status(t_id, "Error", error=str(e))


# --- UI ---

st.title("Project Codex v1.0")

# Sidebar
with st.sidebar:
    st.header("Global Configuration")

    # API Keys
    with st.expander("API Keys"):
        openai_key = st.text_input(
            "OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", "")
        )
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        gemini_key = st.text_input(
            "Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", "")
        )

        if st.button("Update Keys"):
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
            st.success("Keys updated!")

    # Prompt Configuration (New)
    with st.expander("Prompt Templates"):
        # Load current prompts
        current_prompts = config_manager.prompts.copy()

        new_extractor = st.text_area(
            "Extractor Prompt", value=current_prompts.get("extractor", "")
        )
        new_drafter = st.text_area(
            "Drafter (P1) Prompt", value=current_prompts.get("drafter_p1", "")
        )
        new_judge = st.text_area(
            "Judge Prompt", value=current_prompts.get("judge", "")
        )
        new_polisher = st.text_area(
            "Polisher (P2) Prompt", value=current_prompts.get("polisher_p2", "")
        )

        if st.button("Save Prompts"):
            updated_prompts = {
                "extractor": new_extractor,
                "drafter_p1": new_drafter,
                "judge": new_judge,
                "polisher_p2": new_polisher,
            }
            try:
                config_manager.save_prompts(updated_prompts)
                st.success("Prompts saved!")
            except Exception as e:  # pylint: disable=broad-exception-caught
                st.error(f"Failed to save: {e}")

    # Knowledge Base
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload Glossary (CSV)", type="csv")
    if uploaded_file:
        # Save temp file
        TEMP_PATH = f"temp_{uploaded_file.name}"
        with open(TEMP_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            term_manager.sync_terms_to_rag(TEMP_PATH)
            st.success("Glossary Synced to RAG!")
        except Exception as e:  # pylint: disable=broad-exception-caught
            st.error(f"Sync failed: {e}")
        finally:
            if os.path.exists(TEMP_PATH):
                os.remove(TEMP_PATH)

    # Style Bank Upload
    uploaded_style = st.file_uploader("Upload Style Bank (TXT/JSON)", type=["txt", "json"])
    if uploaded_style:
        try:
            content = uploaded_style.read().decode("utf-8")
            if uploaded_style.type == "application/json":
                # Expect list of strings or objects
                # Format: [{"text": "...", "tag": "action"}]
                data = json.loads(content)
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
                # TXT: Split by paragraphs or lines
                texts = [t.strip() for t in content.split('\n') if t.strip()]
                if texts:
                    rag_engine.add_texts("style_bank", texts)
                    st.success(f"Added {len(texts)} style fragments!")
        except Exception as e: # pylint: disable=broad-exception-caught
            st.error(f"Style upload failed: {e}")

    # Glossary Export
    if st.button("Generate Glossary CSV"):
        try:
            export_path = "glossary_export.csv"
            term_manager.export_terms_to_csv(export_path)
            with open(export_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="Download Glossary",
                    data=f,
                    file_name="glossary_export.csv",
                    mime="text/csv"
                )
        except Exception as e: # pylint: disable=broad-exception-caught
            st.error(f"Export failed: {e}")

# Main Area
st.header("Task Dashboard")

# Task Inputs
col1, col2 = st.columns([2, 1])
with col1:
    source_file = st.file_uploader("Upload Manuscript (TXT)", type="txt")
with col2:
    project_context = st.text_area(
        "Project Context (for Judge)", placeholder="Enter core settings/summary..."
    )
    style_tag_input = st.selectbox(
        "Style Tag (for Polisher)",
        ["general", "action", "dialogue", "description", "scifi", "fantasy"]
    )

col_start, col_stop = st.columns(2)
with col_start:
    if st.button("Start Pipeline"):
        if not source_file:
            st.warning("Please upload a file.")
        else:
            text_content = source_file.read().decode("utf-8")
            task_id = str(uuid.uuid4())
            file_name = source_file.name

            # Create Task
            db_manager.create_task(task_id, file_name)
            st.session_state["current_task_id"] = task_id
            st.session_state["current_source_text"] = text_content

            # Submit to background thread
            executor.submit(run_pipeline, task_id, text_content, project_context, style_tag_input)
            st.success(f"Task {task_id} started!")

with col_stop:
    if st.button("Stop Pipeline"):
        if "current_task_id" in st.session_state:
            t_id = st.session_state["current_task_id"]
            db_manager.update_task_status(t_id, "Cancelled")
            st.warning("Cancellation requested...")
        else:
            st.info("No active task to stop.")

# Real-time Status
if "current_task_id" in st.session_state:
    task_id = st.session_state["current_task_id"]
    task = db_manager.get_task(task_id)

    if task:
        # Display Status & Metrics
        st.info(f"Status: {task['status']} | Step: {task['current_step']}")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Tokens Used", value=task.get("token_usage", 0))
        with m_col2:
            st.metric("Est. Cost ($)", value=f"{task.get('cost_estimate', 0.0):.4f}")

        if task["status"] == "Processing":
            step_progress = {
                "Extraction": 0.25,
                "Drafting": 0.50,
                "Judging": 0.75,
                "Polishing": 0.90
            }
            progress_val = step_progress.get(task['current_step'], 0.1)
            st.progress(progress_val)
            time.sleep(1)  # Simple polling loop for visual effect
            st.rerun()
        elif task["status"] == "Done":
             st.progress(1.0)
             st.balloons()

        if task["status"] == "Error":
            st.error(f"Error: {task['error_message']}")

        # Workspace
        st.header("Result Workspace")

        # Hidden Original
        with st.expander("Show Original Source"):
            if "current_source_text" in st.session_state:
                st.text(st.session_state["current_source_text"])
            else:
                st.info("Source text not available.")

        # Dual Column Layout
        row1 = st.columns(2)

        # Left: Draft / Critique
        with row1[0]:
            st.subheader("Draft / Critique")
            tab1, tab2 = st.tabs(["Draft (P1)", "Judge Critique"])
            with tab1:
                st.text_area(
                    "Initial Draft", value=task.get("draft_text") or "", height=400
                )
            with tab2:
                st.json(task.get("critique") or {})

        # Right: Final
        with row1[1]:
            st.subheader("Final Polish (P2)")
            final_text_val = task.get("final_text") or ""
            st.text_area("Final Text", value=final_text_val, height=600)
            if final_text_val:
                st.download_button(
                    label="Export Final Text",
                    data=final_text_val,
                    file_name=f"localized_{task_id[:8]}.txt",
                    mime="text/plain"
                )
