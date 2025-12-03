import streamlit as st
import os
import shutil

class SessionManager:
    """
    Manages Streamlit session state and persistent file storage for sessions.
    """

    # Use absolute path relative to this file
    TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/temp_uploads"))

    @staticmethod
    def init_session():
        """Initialize all necessary session state variables."""
        if "initialized" not in st.session_state:
            # Initialize core state
            st.session_state["user_config"] = {}
            st.session_state["current_task_id"] = None
            st.session_state["uploaded_file_paths"] = []
            st.session_state["initialized"] = True

            # Ensure temp dir exists
            os.makedirs(SessionManager.TEMP_DIR, exist_ok=True)

    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """
        Saves an uploaded file to the temporary directory.
        Returns the absolute file path.
        """
        if not os.path.exists(SessionManager.TEMP_DIR):
             os.makedirs(SessionManager.TEMP_DIR, exist_ok=True)

        file_path = os.path.join(SessionManager.TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    @staticmethod
    def get_file_content(file_path: str) -> str:
        """Reads content of a saved file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        # Try decoding with common encodings
        encodings = ["utf-8", "gbk", "latin-1"]
        with open(file_path, "rb") as f:
            content_bytes = f.read()

        for enc in encodings:
            try:
                return content_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode file {file_path}")

    @staticmethod
    def cleanup_temp_files():
        """Cleans up the temp directory."""
        if os.path.exists(SessionManager.TEMP_DIR):
            shutil.rmtree(SessionManager.TEMP_DIR)
            os.makedirs(SessionManager.TEMP_DIR, exist_ok=True)
