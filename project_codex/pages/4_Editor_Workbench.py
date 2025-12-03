import streamlit as st
import json
from project_codex.core.manager_factory import get_managers
from project_codex.utils.ui import init_page
from project_codex.utils.i18n import I18N

st.set_page_config(layout="wide", page_title="Editor Workbench")

init_page("sidebar_workbench")

_, db_manager, _, _, _, _, _, _ = get_managers()

# --- Task Selection ---
default_index = 0
task_options = {}
sorted_tasks = []

try:
    conn = db_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, updated_at, status FROM tasks ORDER BY updated_at DESC LIMIT 50"
    )
    raw_tasks = cursor.fetchall()
    conn.close()

    if not raw_tasks:
        st.info("No tasks found.")
        st.stop()

    for t in raw_tasks:
        label = f"[{t[3]}] {t[1]} (ID: {t[0][:8]})"
        task_options[label] = t[0]
        sorted_tasks.append(label)

except Exception as e:
    st.error(f"Error fetching tasks: {e}")
    st.stop()

pre_selected_id = st.session_state.get("selected_task_id")
if pre_selected_id:
    for label, tid in task_options.items():
        if tid == pre_selected_id:
            try:
                default_index = sorted_tasks.index(label)
            except ValueError:
                pass
            break

selected_label = st.selectbox(
    "Select Task to Review", options=sorted_tasks, index=default_index
)
task_id = task_options[selected_label]

# Load Task Data
task = db_manager.get_task(task_id)

if not task:
    st.error("Task not found.")
    st.stop()

# --- Layout Configuration ---
st.sidebar.header("View Options")
god_mode = st.sidebar.checkbox(I18N.get("god_mode"), value=False)

# --- Source Text ---
source_text = task.get("source_text") or ""

# --- Render ---

with st.expander("📄 查看原始中文稿件 (Original Source)", expanded=False):
    if source_text:
        st.text(source_text)
    else:
        st.warning("Source text not available.")

# We use a key for the final text editor to capture changes
final_text_key = f"editor_{task_id}"

# Layouts
if god_mode:
    # Three columns: Source | Draft+Judge | Final
    col_src, col_mid, col_fin = st.columns(3)

    with col_src:
        st.subheader("Source")
        st.text_area("Original", value=source_text, height=600, disabled=True)

    with col_mid:
        st.subheader("Draft & Judge")
        st.caption("Draft (P1)")
        st.text_area("Draft", value=task.get("result_p1") or "", height=250)
        st.caption("Judge Report")
        st.json(json.loads(task.get("result_judge") or "{}"))

    with col_fin:
        st.subheader("Final Polish (P2)")
        final_val = task.get("result_final") or ""
        st.text_area("Final Output", value=final_val, height=600, key=final_text_key)

else:
    # Two columns
    col_left, col_right = st.columns(2)

    with col_left:
        # Switcher
        view_mode = st.selectbox("Left View", ["Draft (P1)", "Judge Report"])

        if view_mode == "Draft (P1)":
            st.subheader("Initial Draft")
            st.text_area("P1 Output", value=task.get("result_p1") or "", height=600)
        else:
            st.subheader("Judge's Critique")
            st.json(json.loads(task.get("result_judge") or "{}"))

    with col_right:
        st.subheader("Final Polish (P2)")
        final_val = task.get("result_final") or ""
        st.text_area("Final Output", value=final_val, height=600, key=final_text_key)

# --- Actions ---
st.divider()

col_act1, col_act2 = st.columns([1, 4])

with col_act1:
    if st.button(I18N.get("save_btn")):
        new_content = st.session_state.get(final_text_key)
        if new_content is not None:
            try:
                db_manager.update_task_status(
                    task_id, task["status"], result_final=new_content
                )
                st.success("Changes saved!")
                # Force refresh to update 'task' variable on next run
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

with col_act2:
    if task.get("result_final"):
        st.download_button(
            label=I18N.get("export_btn"),
            data=task.get("result_final") or "",
            file_name=f"localized_{task['filename']}",
            mime="text/plain",
        )
