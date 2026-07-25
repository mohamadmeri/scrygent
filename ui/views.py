"""Main page assembly and view routing for the Scrygent UI."""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from scrygent.models.state import AgentState

from .components import render_control_panel, render_demo_showcase, render_pipeline, render_topbar
from .orchestration import initialize_session_state, run_graph_with_resilience
from .theme import TEXT_PRIMARY, TEXT_SECONDARY


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
    """Renders the initial file upload screen with pre-flight validation and demo datasets."""
    st.markdown(
        "<h2 style='font-family:Cormorant Garamond,serif;font-size:2.2rem;margin-bottom:0.5rem;'>Compile Your Data</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Scrygent requires a structured CSV to begin deterministic compilation. Upload your own or try a curated demo dataset below.")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
        help="Maximum file size: 200 MB. The file is processed locally and never leaves your environment.",
    )

    st.caption("⚠️ **Max upload size:** 200 MB")

    if uploaded_file:
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

    # Demo dataset showcase
    def _on_select_dataset(key: str, meta: dict) -> None:  # type: ignore[type-arg]
        """Callback when user selects a demo dataset."""
        path = Path(meta["path"])
        if path.exists():
            st.session_state.csv_path = path
            st.toast(f"Loaded {meta['name']}", icon="✅")
            st.rerun()
        else:
            # Try to find in upload directory
            upload_dir = Path("/mnt/agents/upload/")
            candidates = list(upload_dir.glob(f"*{key}*.csv"))
            if candidates:
                st.session_state.csv_path = candidates[0]
                st.toast(f"Loaded {meta['name']}", icon="✅")
                st.rerun()
            else:
                st.error(f"Demo dataset `{meta['name']}` not found. Please ensure `{meta['path']}` exists.")

    def _on_select_question(key: str, meta: dict, question: str) -> None:  # type: ignore[type-arg]
        """Callback when user clicks a suggested question."""
        # First ensure dataset is loaded
        path = Path(meta["path"])
        if not path.exists():
            upload_dir = Path("/mnt/agents/upload/")
            candidates = list(upload_dir.glob(f"*{key}*.csv"))
            if candidates:
                path = candidates[0]
            else:
                st.error("Demo dataset not found.")
                return

        st.session_state.csv_path = path
        st.session_state.query = question
        st.rerun()

    render_demo_showcase(_on_select_dataset, _on_select_question)


def _render_main_interface() -> None:
    """Renders the IDE-style split-pane interface for query and telemetry."""
    left_col, right_col = st.columns([2.5, 1], gap="large")

    with left_col:
        _render_query_interface()

    with right_col:
        render_control_panel()


def _render_query_interface() -> None:
    """Handles the query submission, pipeline invocation, and result rendering."""
    pipeline_placeholder = st.empty()
    cooldown_placeholder = st.empty()

    # Render existing query history
    if st.session_state.query:
        with st.chat_message("user"):
            st.markdown(f"**Query:** {st.session_state.query}")

    if st.session_state.final_state:
        _render_final_result(st.session_state.final_state)

    # Handle graph invocation if a query is pending
    if st.session_state.query and st.session_state.final_state is None:
        with st.chat_message("assistant"):
            with pipeline_placeholder.container():
                render_pipeline(None, set(), errored=False)

            try:
                initial_state = AgentState(
                    original_csv_path=Path(st.session_state.csv_path),
                    current_csv_path=Path(st.session_state.csv_path),
                    user_query=st.session_state.query,
                )
                payload = initial_state.model_dump(mode="json")

                raw_final_update, exhausted_error = run_graph_with_resilience(payload, pipeline_placeholder, cooldown_placeholder)

                if exhausted_error is not None:
                    st.error(f"**Service temporarily unavailable.** {exhausted_error.service} is still rate limited.")
                    st.session_state.query = None
                    st.stop()

                if raw_final_update is None:
                    raise RuntimeError("Graph stream produced no state updates.")

                st.session_state.final_state = AgentState.model_validate(raw_final_update)

                # Store emitted plan for display in control panel
                if hasattr(st.session_state.final_state, "plan") and st.session_state.final_state.plan:
                    try:
                        st.session_state.emitted_plan = st.session_state.final_state.plan.model_dump(mode="json")
                    except Exception:
                        pass

                if st.session_state.final_state.execution_status == "complete":
                    st.toast("Compilation complete", icon="✅")

            except Exception:
                st.error("A system error occurred during execution. See logs for details.")
                st.session_state.query = None
                st.stop()

            st.rerun()

    # Query input at the bottom
    if not st.session_state.query:
        user_input = st.chat_input("Compile a query about your data...")
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

            # Compiled result header
            st.markdown(
                "<h3 style='font-family:Cormorant Garamond,serif;font-size:1.6rem;margin-bottom:0.75rem;padding-top:0.25rem;'>Compiled Analysis</h3>",
                unsafe_allow_html=True,
            )

            # Primary answer
            primary = getattr(report, "primary_answer", "Analysis Complete")
            st.markdown(
                f"<div style='font-size:1.15rem;line-height:1.7;color:{TEXT_PRIMARY};margin-bottom:1.5rem;padding:0.25rem 0;'>{primary}</div>",
                unsafe_allow_html=True,
            )

            # Secondary observations
            insights = getattr(report, "additional_insights", None)
            if insights:
                st.markdown(
                    "<p class='sg-mono-label' style='margin-bottom:0.5rem;margin-top:1rem;'>Secondary Observations</p>",
                    unsafe_allow_html=True,
                )
                for insight in insights:
                    st.markdown(
                        f"<div style='color:{TEXT_SECONDARY};margin-bottom:0.75rem;line-height:1.6;padding-right:0.5rem;padding-left:1.25rem;text-indent:-0.75rem;'>"
                        f"— {insight}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Generated visualizations
        plots = getattr(report, "plots", None)
        if plots:
            import json

            st.markdown(
                "<p class='sg-mono-label' style='margin:1rem 0 0.5rem;'>Generated Visualizations</p>",
                unsafe_allow_html=True,
            )
            for i, plot in enumerate(plots):
                # Parse the JSON string back into a dictionary for Streamlit
                fig_dict = json.loads(plot.plotly_json)

                # Render interactive chart
                st.plotly_chart(fig_dict, width="stretch", key=f"plot_{i}_{hash(plot.plotly_json)}")

                # Caption below the chart
                if plot.description:
                    st.caption(plot.description)
