"""Reusable UI components for the Scrygent presentation layer."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .theme import AMBER, BG_ELEVATED, NODE_ORDER, SAGE, TEXT_MUTED, TEXT_PRIMARY, _image_data_uri

# ============================================
# DEMO DATASETS
# ============================================
DEMO_DATASETS = {
    "titanic_lite": {
        "name": "Titanic Passengers (Lite)",
        "source": "Kaggle / DataScienceDojo (Public)",
        "rows": 100,
        "cols": 8,
        "path": "demo_data/titanic_lite.csv",
        "questions": [
            "Compare the survival rate between first class and third class passengers and show me a bar chart.",
            "Who are the top 5 oldest passengers?",
            "What is the average fare for passengers who embarked at Southampton?",
        ],
    },
    "spotify_lite": {
        "name": "Spotify Top Tracks (Lite)",
        "source": "Spotify Charts (Public)",
        "rows": 100,
        "cols": 7,
        "path": "demo_data/spotify_lite.csv",
        "questions": [
            "What is the average danceability by genre and show me which genre has the highest energy?",
            "Show me the top 3 most popular tracks.",
            "Is there a correlation between energy and valence?",
        ],
    },
}


# ============================================
# HEALTH CHECKS
# ============================================
@st.cache_data(ttl=5, show_spinner=False)
def check_llm_health() -> str:
    """Returns 'ok', 'cooldown', or 'offline'."""
    from scrygent.resilience import is_system_cooling_down

    if is_system_cooling_down():
        return "cooldown"

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "offline"
    try:
        import requests

        # Try Groq first
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {groq_key}"},
                timeout=4,
            )
            if resp.status_code == 200:
                return "ok"

        # Fallback to OpenRouter
        or_key = os.getenv("OPENROUTER_API_KEY")
        if or_key:
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {or_key}"},
                timeout=4,
            )
            return "ok" if resp.status_code == 200 else "offline"

        return "offline"
    except Exception:
        return "offline"


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


# ============================================
# TOPBAR
# ============================================
def render_topbar() -> None:
    """Renders the global status bar indicating backend service health."""
    llm_status = check_llm_health()
    qdrant_ok = check_qdrant_health()

    if llm_status == "ok":
        llm_dot, llm_text = "sg-dot-ok", "ONLINE"
    elif llm_status == "cooldown":
        llm_dot, llm_text = "sg-dot-warn", "COOLDOWN"
    else:
        llm_dot, llm_text = "sg-dot-bad", "OFFLINE"

    qdrant_dot = "sg-dot-ok" if qdrant_ok else "sg-dot-bad"
    qdrant_text = "ONLINE" if qdrant_ok else "OFFLINE"

    logo_src = _image_data_uri("assets/logo-icon.png") or ""

    st.markdown(
        f"""
        <div class="sg-topbar">
            <div style="display:flex;align-items:center;gap:12px;">
                <a href="https://scrygent.netlify.app/" class="sg-brand" target="_blank">
                    <img src="{logo_src}" width="32" height="32" alt="" style="vertical-align:middle;margin-right:8px;border-radius:4px;">
                    Scrygent
                </a>
                <span class="sg-divider">|</span>
                <span class="sg-subtitle">Deterministic Data Compiler</span>
            </div>
            <div class="sg-topbar-right">
                <a href="https://scrygent.netlify.app/" class="sg-about-link" target="_blank">← About</a>
                <div class="sg-status-group">
                    <div class="sg-status-pill">
                        <span class="sg-dot {llm_dot}"></span> LLM: {llm_text}
                    </div>
                    <div class="sg-status-pill">
                        <span class="sg-dot {qdrant_dot}"></span> QDRANT: {qdrant_text}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================
# PIPELINE
# ============================================
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


# ============================================
# DEMO DATASET SHOWCASE
# ============================================
def render_demo_showcase(
    on_select_dataset: Callable[[str, dict[str, Any]], None],
    on_select_question: Callable[[str, dict[str, Any], str], None],
) -> None:
    """Renders the curated demo dataset cards on the upload screen."""
    st.markdown("---")
    st.markdown(
        '<p class="sg-mono-label" style="margin-bottom:1.5rem;">Or try a demo dataset</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(DEMO_DATASETS), gap="medium")
    for col, (key, meta) in zip(cols, DEMO_DATASETS.items()):
        with col:
            with st.container(border=True):
                # Header
                st.markdown(f"**{meta['name']}**")
                st.caption(f"{meta['source']} · {meta['rows']:,} rows · {meta['cols']} cols")

                # Try to load and preview
                df_preview = _load_demo_preview(key, meta)
                if df_preview is not None:
                    st.dataframe(
                        df_preview.head(6),
                        height=180,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("Preview unavailable — dataset not found locally.")

                # Load button
                if st.button(
                    f"Load {meta['name'].split()[0]}",  # type: ignore[attr-defined]
                    key=f"demo_load_{key}",
                    use_container_width=True,
                    type="secondary",
                ):
                    on_select_dataset(key, meta)

                # Suggested questions
                st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
                for q in meta["questions"]:  # type: ignore[attr-defined]
                    if st.button(
                        f'"{q[:40]}{"..." if len(q) > 40 else ""}"',
                        key=f"demo_q_{key}_{hash(q) & 0xFFFFFFFF}",
                        use_container_width=True,
                    ):
                        on_select_question(key, meta, q)


def _load_demo_preview(key: str, meta: dict[str, Any]) -> pd.DataFrame | None:
    """Attempts to load a demo dataset for preview."""
    path = Path(meta["path"])
    if path.exists():
        try:
            return pd.read_csv(path, nrows=20)
        except Exception:
            pass

    # Fallback: try to find in upload directory
    upload_dir = Path("/mnt/agents/upload/")
    candidates = list(upload_dir.glob(f"*{key}*.csv"))
    if candidates:
        try:
            return pd.read_csv(candidates[0], nrows=20)
        except Exception:
            pass

    return None


# ============================================
# CONTROL PANEL
# ============================================
def render_control_panel() -> None:
    """Renders the right-hand IDE-style control panel."""
    st.markdown(
        "<h3 style='font-family:Cormorant Garamond,serif;font-size:1.4rem;margin-bottom:1rem;'>Compiler Control Panel</h3>",
        unsafe_allow_html=True,
    )

    # Dataset Telemetry
    with st.container(border=True):
        st.markdown(
            '<p class="sg-mono-label" style="margin-bottom:0.75rem;">Active Dataset</p>', unsafe_allow_html=True
        )
        if st.session_state.csv_path:
            st.code(st.session_state.csv_path.name, language=None)

            if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
                _reset_session()
                st.rerun()
        else:
            st.caption("No dataset loaded.")

    # System Health
    with st.container(border=True):
        st.markdown(
            '<p class="sg-mono-label" style="margin-bottom:0.75rem;">System Telemetry</p>', unsafe_allow_html=True
        )
        llm_ok = check_llm_health()
        qdrant_ok = check_qdrant_health()

        status_llm = (
            "🟢 Connected" if llm_ok == "ok" else ("🟡 Cooldown" if llm_ok == "cooldown" else "🔴 Disconnected")
        )
        status_qdrant = "🟢 Connected" if qdrant_ok else "🔴 Disconnected"

        st.markdown(f"**LLM Backend:** {status_llm}")
        st.markdown(f"**Vector Memory:** {status_qdrant}")

    # Emitted Plan (IR) — Summary Cards
    if st.session_state.get("emitted_plan"):
        with st.container(border=True):
            st.markdown(
                '<p class="sg-mono-label" style="margin-bottom:0.75rem;">Compiled Execution Plan</p>',
                unsafe_allow_html=True,
            )
            plan = st.session_state.emitted_plan

            if isinstance(plan, dict) and "steps" in plan:
                for i, step in enumerate(plan["steps"]):
                    tool_name = step.get("tool_name", "unknown")
                    rationale = step.get("rationale", "")
                    params = step.get("parameters", {})

                    # Format parameters cleanly
                    param_str = ", ".join([f"{k}={v}" for k, v in params.items() if v is not None])
                    if not param_str:
                        param_str = "default parameters"

                    st.markdown(
                        f"<div style='margin-bottom: 0.75rem; padding: 0.75rem; background: {BG_ELEVATED}; border-radius: 6px; border-left: 3px solid {SAGE};'>"
                        f"<div style='font-family: JetBrains Mono, monospace; font-size: 0.8rem; color: {SAGE}; margin-bottom: 0.25rem; font-weight: 500;'>"
                        f"Step {i + 1}: {tool_name}"
                        f"</div>"
                        f"<div style='font-size: 0.85rem; color: {TEXT_PRIMARY}; margin-bottom: 0.25rem; line-height: 1.4;'>"
                        f"{rationale}"
                        f"</div>"
                        f"<div style='font-family: JetBrains Mono, monospace; font-size: 0.75rem; color: {TEXT_MUTED};'>"
                        f"Params: {param_str}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Raw JSON for technical reviewers
                with st.expander("View Raw JSON IR"):
                    st.json(plan, expanded=False)

    # Execution Trace
    if st.session_state.final_state:
        _render_execution_trace(st.session_state.final_state)


# ============================================
# EXECUTION TRACE
# ============================================
def _render_execution_trace(state: Any) -> None:
    """Renders the strict IR execution trace using native Streamlit status components."""
    with st.container(border=True):
        st.markdown(
            '<p class="sg-mono-label" style="margin-bottom:0.75rem;">Compiled IR Trace</p>', unsafe_allow_html=True
        )

        if state.execution_status == "aborted":
            st.error("Execution Aborted")
            if state.error_log:
                st.caption(state.error_log[-1])
            return

        if not state.execution_trace:
            st.caption("No execution steps recorded.")
            return

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

            # Self-healing explanation
            if getattr(state, "correction_count", 0) > 0:
                st.divider()
                st.markdown(
                    f"<p style='color:{AMBER};font-size:0.85rem;'>"
                    f"⚡ Self-healing engaged: {state.correction_count} correction loop(s) resolved validation failures. "
                    f"The LLM repaired its own payload syntax mid-flight using available column context."
                    f"</p>",
                    unsafe_allow_html=True,
                )

            if state.execution_status == "complete":
                status.update(label="Execution Complete", state="complete", expanded=False)


def _reset_session() -> None:
    """Clears all session state to return the UI to the initial upload screen."""
    st.session_state.csv_path = None
    st.session_state.query = None
    st.session_state.final_state = None
    st.session_state.emitted_plan = None
    st.session_state.pipeline_progress = {"active": None, "completed": set(), "errored": False}
