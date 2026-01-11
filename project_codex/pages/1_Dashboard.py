import streamlit as st
import uuid
import time
import pandas as pd
from project_codex.core.manager_factory import get_managers
from project_codex.core.pipeline import run_pipeline

st.set_page_config(layout="wide", page_title="Dashboard")

(
    config_manager,
    db_manager,
    rag_engine,
    term_manager,
    llm_client,
    workflow_engine,
    executor,
) = get_managers()

st.title("🏠 Task Dashboard")

# --- Batch Upload Section ---
st.header("1. Upload & Configure")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload Manuscripts (TXT)", type="txt", accept_multiple_files=True
    )

with col2:
    project_context = st.text_area(
        "Project Context (Global)",
        placeholder="Enter core settings/summary...",
        help="Passed to Judge step.",
    )
    style_tag_input = st.selectbox(
        "Style Tag",
        ["general", "action", "dialogue", "description", "scifi", "fantasy"],
        help="Used by Polisher to fetch style examples.",
    )

if uploaded_files:
    total_chars = sum(len(f.getvalue()) for f in uploaded_files)
    st.info(
        f"Detected {len(uploaded_files)} chapters/files. Total size: {total_chars} bytes."
    )

if st.button("🚀 Start Production"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        # Check API keys presence (basic check)
        if (
            not config_manager.api_keys.get("openai")
            and not config_manager.api_keys.get("anthropic")
            and not config_manager.api_keys.get("gemini")
        ):
            st.error("No API Keys configured! Please go to Config Studio.")
        else:
            # Snapshot current config
            current_config = config_manager.get_full_config()

            for u_file in uploaded_files:
                try:
                    # simplistic decoding
                    content = u_file.read().decode("utf-8")
                except UnicodeDecodeError:
                    # Try gbk
                    try:
                        u_file.seek(0)
                        content = u_file.read().decode("gbk")
                    except:
                        st.error(f"Failed to decode {u_file.name}")
                        continue

                task_id = str(uuid.uuid4())
                db_manager.create_task(task_id, u_file.name, source_text=content)

                # Submit job
                executor.submit(
                    run_pipeline,
                    task_id,
                    content,
                    current_config,
                    db_manager,
                    workflow_engine,
                    project_context,
                    style_tag_input,
                )

            st.success(f"Started {len(uploaded_files)} tasks!")
            time.sleep(1)
            st.rerun()

# --- Task Monitor ---
st.header("2. Live Monitor")
st.markdown("Real-time progress of active tasks.")

# Fetch tasks from DB
# Since DBManager doesn't have "get_all_tasks" or "get_active_tasks", we might need to add it
# or just query directly if we want to list them.
# For now, let's add `get_recent_tasks` to DBManager or just query here.
# accessing db_manager internal connection is cleaner if we extend DBManager.

# Extending DBManager on the fly or just use direct query here for dashboard?
# It's better to add a method to DBManager.
# But I can't modify DBManager file again easily without rewriting it.
# I'll use private method access or execute directly since I imported db_manager.

conn = db_manager._get_connection()
try:
    conn.row_factory = lambda cursor, row: dict(
        zip([col[0] for col in cursor.description], row)
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20")
    tasks = cursor.fetchall()
finally:
    conn.close()

if not tasks:
    st.info("No recent tasks found.")
else:
    # Auto-refresh
    if any(t["status"] == "Processing" for t in tasks):
        time.sleep(2)
        st.rerun()

    for task in tasks:
        with st.container():
            col_id, col_status, col_prog, col_metrics = st.columns([1, 1, 3, 1])

            with col_id:
                st.write(f"**{task['filename']}**")
                st.caption(f"ID: {task['id'][:8]}...")

            with col_status:
                status_color = "blue"
                if task["status"] == "Done":
                    status_color = "green"
                elif task["status"] == "Error":
                    status_color = "red"
                elif task["status"] == "Cancelled":
                    status_color = "grey"

                st.markdown(f":{status_color}[{task['status']}]")
                if task["status"] == "Error":
                    st.error(task.get("error_log", "Unknown Error"))

            with col_prog:
                # 4 independent progress bars visualization?
                # Or just one main bar with steps?
                # "Must display 4 independent progress bars"

                # Let's visualize 4 steps
                steps = ["Extraction", "Drafting", "Judging", "Polishing"]
                current = task.get("current_step")

                # If Done, all full. If Processing, depend on step.
                cols = st.columns(4)
                for i, step_name in enumerate(steps):
                    is_complete = False
                    is_active = False

                    if task["status"] == "Done":
                        is_complete = True
                    elif task["status"] == "Processing":
                        # Mapping steps to order
                        step_order = {s: idx for idx, s in enumerate(steps)}
                        curr_idx = step_order.get(current, -1)
                        if i < curr_idx:
                            is_complete = True
                        elif i == curr_idx:
                            is_active = True

                    with cols[i]:
                        st.caption(step_name)
                        if is_complete:
                            st.progress(1.0)
                        elif is_active:
                            st.progress(0.5)  # In progress
                        else:
                            st.progress(0.0)

            with col_metrics:
                st.caption(f"Tokens: {task.get('token_usage', 0)}")
                st.caption(f"Cost: ${task.get('cost_estimate', 0):.4f}")

            st.divider()

# --- Stop/Clear Control ---
if st.button("Stop All Processing Tasks"):
    # This requires updating all 'Processing' tasks to 'Cancelled'
    conn = db_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE tasks SET status='Cancelled' WHERE status='Processing'")
    conn.commit()
    conn.close()
    st.warning("Cancellation signal sent to all workers.")
