"""Main page assembly and view routing for the Scrygent UI."""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from scrygent.models.state import AgentState

from .components import render_control_panel, render_pipeline, render_topbar
from .orchestration import initialize_session_state, run_graph_with_resilience


def run_app() -> None:
    """Main entry point for the Streamlit application."""
    from .theme import configure_page, inject_css

    configure_page()
    inject_css()
    initialize_session_state()

    render_topbar()

    if not st.session_state.csv_path:
        _render_upload_view()
        return

    _render_main_interface()


def _render_upload_view() -> None:
    """Renders the initial file upload screen with pre-flight validation."""
    st.markdown("### Upload Dataset")
    st.caption("Scrygent requires a structured CSV to begin deterministic compilation.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        # Pre-flight validation: `type=["csv"]` only checks the file extension.
        # A renamed non-CSV file or bad encoding would otherwise fail later
        # inside the Profiler node. Catch it here with a fast, cheap parse.
        try:
            uploaded_file.seek(0)
            pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"**Couldn't read `{uploaded_file.name}` as a CSV.** {type(e).__name__}: {e}")
            st.stop()

        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())

        st.session_state.csv_path = file_path
        st.toast(f"Loaded {uploaded_file.name}", icon="✅")
        st.rerun()


def _render_main_interface() -> None:
    """Renders the IDE-style split-pane interface for chat and telemetry."""
    left_col, right_col = st.columns([2.5, 1], gap="large")

    with left_col:
        _render_chat_interface()

    with right_col:
        render_control_panel()


def _render_chat_interface() -> None:
    """Handles the conversational flow, pipeline invocation, and result rendering."""
    pipeline_placeholder = st.empty()
    cooldown_placeholder = st.empty()

    # Render existing chat history
    if st.session_state.query:
        with st.chat_message("user"):
            st.markdown(st.session_state.query)

    if st.session_state.final_state:
        _render_final_result(st.session_state.final_state)

    # Handle graph invocation if a query is pending
    if st.session_state.query and st.session_state.final_state is None:
        with st.chat_message("assistant"):
            with pipeline_placeholder.container():
                render_pipeline(None, set(), errored=False)

            try:
                initial_state = AgentState(
                    original_csv_path=str(st.session_state.csv_path),  # type: ignore[arg-type]
                    current_csv_path=str(st.session_state.csv_path),  # type: ignore[arg-type]
                    user_query=st.session_state.query,
                )
                payload = initial_state.model_dump(mode="json")

                raw_final_update, exhausted_error = run_graph_with_resilience(
                    payload, pipeline_placeholder, cooldown_placeholder
                )

                if exhausted_error is not None:
                    st.error(f"**Service temporarily unavailable.** {exhausted_error.service} is still rate limited.")
                    st.session_state.query = None
                    st.stop()

                if raw_final_update is None:
                    raise RuntimeError("Graph stream produced no state updates.")

                st.session_state.final_state = AgentState.model_validate(raw_final_update)

                if st.session_state.final_state.execution_status == "complete":
                    st.toast("Compilation complete", icon="✅")

            except Exception:
                st.error("A system error occurred during execution. See logs for details.")
                st.session_state.query = None
                st.stop()

            st.rerun()

    # Chat input at the bottom
    if not st.session_state.query:
        user_input = st.chat_input("Ask a question about your data...")
        if user_input:
            st.session_state.query = user_input
            st.rerun()


def _render_final_result(state: AgentState) -> None:
    """Renders the final synthesized report from the deterministic engine."""
    with st.chat_message("assistant"):
        if state.execution_status == "aborted":
            st.error("Execution aborted")
            if state.error_log:
                st.caption(state.error_log[-1])
            return

        if state.execution_status == "complete" and state.final_report:
            report = state.final_report

            # Frame the output as a compiled result, not a chat response
            st.markdown("### Compiled Analysis")
            st.markdown(getattr(report, "primary_answer", "Analysis Complete"))

            insights = getattr(report, "additional_insights", None)
            if insights:
                st.markdown("##### Secondary Observations")
                for insight in insights:
                    st.markdown(f"- {insight}")

            plots = getattr(report, "plots", None)
            if plots:
                st.markdown("##### Generated Visualizations")
                for plot in plots:
                    st.image(str(plot.file_path), caption=plot.description)
