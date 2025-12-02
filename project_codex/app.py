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
def run_pipeline(t_id: str, text: str, p_context: str = ""):
    """
    Background worker to run the full pipeline.
    """
    try:
        # Check Circuit Breaker
        db_manager.check_circuit()

        db_manager.update_task_status(t_id, "Processing", step="Extraction")

        # Step 1: Extraction (Run but mostly side-effect for now)
        _ = workflow_engine.step1_extract_terms(text)

        db_manager.update_task_status(t_id, "Processing", step="Drafting")

        # Step 2: Drafting
        draft = workflow_engine.step2_draft(text)
        db_manager.update_task_status(
            t_id, "Processing", step="Judging", draft_text=draft
        )

        # Step 3: Judging
        critique = workflow_engine.step3_judge(draft, project_context=p_context)
        db_manager.update_task_status(
            t_id,
            "Processing",
            step="Polishing",
            critique=json.dumps(critique, ensure_ascii=False),
        )

        # Step 4: Polishing
        final = workflow_engine.step4_polish(draft, critique)
        db_manager.update_task_status(
            t_id, "Done", step="Completed", final_text=final
        )

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
        executor.submit(run_pipeline, task_id, text_content, project_context)
        st.success(f"Task {task_id} started!")

# Real-time Status
if "current_task_id" in st.session_state:
    task_id = st.session_state["current_task_id"]
    task = db_manager.get_task(task_id)

    if task:
        st.info(f"Status: {task['status']} | Step: {task['current_step']}")

        if task["status"] == "Processing":
            st.progress(0.5)  # Generic progress, could refine based on step
            time.sleep(1)  # Simple polling loop for visual effect
            st.rerun()

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
            st.text_area("Final Text", value=task.get("final_text") or "", height=600)
