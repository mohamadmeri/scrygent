"""Reusable UI components for the Scrygent presentation layer."""

import os
from typing import Any

import streamlit as st

from .theme import NODE_ORDER


@st.cache_data(ttl=30, show_spinner=False)
def check_groq_health() -> bool:
    """Pings the Groq models endpoint. Returns False on any failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False
    try:
        import requests

        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=4,
        )
        return resp.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=30, show_spinner=False)
def check_qdrant_health() -> bool:
    """Confirms Qdrant Cloud credentials resolve to a reachable cluster."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        return False
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, api_key=api_key, timeout=4)
        client.get_collections()
        return True
    except Exception:
        return False


def render_topbar() -> None:
    """Renders the global status bar indicating backend service health."""
    groq_ok = check_groq_health()
    qdrant_ok = check_qdrant_health()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("##### ◆ SCRYGENT")
    with col2:
        st.markdown(
            "<div style='text-align:center; color:#A0A0A0; font-size:0.8rem;'>Deterministic Data Compiler</div>",
            unsafe_allow_html=True,
        )
    with col3:
        c1, c2 = st.columns(2)
        c1.metric("Groq", "Online" if groq_ok else "Offline")
        c2.metric("Qdrant", "Online" if qdrant_ok else "Offline")
    st.divider()


def render_pipeline(active_node: str | None, completed: set[str], errored: bool) -> None:
    """Renders the signature pipeline strip indicating execution progress."""
    segments = []
    for i, name in enumerate(NODE_ORDER):
        if errored and name == active_node:
            cls = "sg-node sg-node-error"
        elif name == active_node:
            cls = "sg-node sg-node-active"
        elif name in completed:
            cls = "sg-node sg-node-done"
        else:
            cls = "sg-node"

        segments.append(f'<div class="{cls}"><span class="sg-node-dot"></span>{name}</div>')
        if i < len(NODE_ORDER) - 1:
            segments.append('<div class="sg-node-connector"></div>')

    st.markdown(f'<div class="sg-pipeline">{"".join(segments)}</div>', unsafe_allow_html=True)


def render_control_panel() -> None:
    """Renders the right-hand IDE-style control panel."""
    st.markdown("### Compiler Control Panel")

    # Dataset Telemetry
    with st.container(border=True):
        st.markdown("##### Active Dataset")
        if st.session_state.csv_path:
            st.code(st.session_state.csv_path.name, language=None)
            if st.button("Reset Session", use_container_width=True, type="secondary"):
                _reset_session()
                st.rerun()
        else:
            st.caption("No dataset loaded.")

    # System Health
    with st.container(border=True):
        st.markdown("##### System Telemetry")
        groq_ok = check_groq_health()
        qdrant_ok = check_qdrant_health()

        status_groq = "🟢 Connected" if groq_ok else "🔴 Disconnected"
        status_qdrant = "🟢 Connected" if qdrant_ok else "🔴 Disconnected"

        st.markdown(f"**Groq LLM:** {status_groq}")
        st.markdown(f"**Vector Memory:** {status_qdrant}")

    # Execution Trace
    if st.session_state.final_state:
        _render_execution_trace(st.session_state.final_state)


def _render_execution_trace(state: Any) -> None:
    """Renders the strict IR execution trace using native Streamlit status components."""
    with st.container(border=True):
        st.markdown("##### Compiled IR Trace")

        if state.execution_status == "aborted":
            st.error("Execution Aborted")
            if state.error_log:
                st.caption(state.error_log[-1])
            return

        if not state.execution_trace:
            st.caption("No execution steps recorded.")
            return

        # Use native st.status for a premium, animated trace experience
        with st.status("Compilation & Execution", expanded=True) as status:
            for step in state.execution_trace:
                t_name = getattr(step.tool_name, "value", step.tool_name)
                if step.status == "success":
                    st.write(f"✅ **{t_name}** ({step.duration_ms}ms)")
                else:
                    st.write(f"❌ **{t_name}** failed")
                    if step.error:
                        st.caption(f"Error: {step.error}")

                if step.summary:
                    st.caption(step.summary)

            if state.execution_status == "complete":
                status.update(label="Execution Complete", state="complete", expanded=False)


def _reset_session() -> None:
    """Clears all session state to return the UI to the initial upload screen."""
    st.session_state.csv_path = None
    st.session_state.query = None
    st.session_state.final_state = None
    st.session_state.pipeline_progress = {"active": None, "completed": set(), "errored": False}
