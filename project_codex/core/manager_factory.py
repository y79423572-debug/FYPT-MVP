import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from project_codex.utils.config import ConfigManager
from project_codex.utils.db import DBManager
from project_codex.core.rag_engine import RAGEngine
from project_codex.core.term_manager import TermManager, CharacterManager
from project_codex.core.llm_client import LLMClient
from project_codex.core.pipeline import WorkflowEngine

@st.cache_resource
def get_managers():
    """Initialize and cache all manager instances."""
    print("Initializing managers...")

    c_mgr = ConfigManager()
    d_mgr = DBManager()

    # RAGEngine might take time
    r_eng = RAGEngine()

    t_mgr = TermManager(r_eng)
    char_mgr = CharacterManager(r_eng)
    l_cli = LLMClient()
    w_eng = WorkflowEngine(c_mgr, l_cli, r_eng)
    ex = ThreadPoolExecutor(max_workers=3)

    return c_mgr, d_mgr, r_eng, t_mgr, l_cli, w_eng, ex, char_mgr
