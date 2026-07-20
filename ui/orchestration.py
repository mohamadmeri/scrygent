"""Graph execution orchestration and resilience wiring."""

import time
from typing import Any

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from scrygent.core.resilience import RetryEvent, ServiceExhaustedError, set_retry_handler
from scrygent.graph.builder import build_graph

from .theme import NODE_KEY_MAP


def initialize_session_state() -> None:
    """Ensures all required session state keys exist with default values."""
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "csv_path" not in st.session_state:
        st.session_state.csv_path = None
    if "query" not in st.session_state:
        st.session_state.query = None
    if "final_state" not in st.session_state:
        st.session_state.final_state = None
    if "emitted_plan" not in st.session_state:
        st.session_state.emitted_plan = None
    if "pipeline_progress" not in st.session_state:
        st.session_state.pipeline_progress = {"active": None, "completed": set(), "errored": False}


def run_graph_with_resilience(
    payload: dict[str, Any],
    pipeline_placeholder: DeltaGenerator,
    cooldown_placeholder: DeltaGenerator,
) -> tuple[dict[str, Any] | None, ServiceExhaustedError | None]:
    """Streams the graph node-by-node and wires the backend retry wrapper to a live UI banner.

    This function manually accumulates state updates from LangGraph's stream_mode="updates"
    to replicate the internal state merge without executing the graph twice. This is safe
    because AgentState fields use standard Pydantic Field() without custom reducers,
    meaning LangGraph's default last-write-wins merge is perfectly reproduced by a
    standard dictionary merge.
    """
    from .components import render_pipeline

    reverse_map = {v: k for k, v in NODE_KEY_MAP.items()}

    def on_retry(event: RetryEvent) -> None:
        """Callback invoked by the resilience layer during 429 cooldowns."""
        from .theme import AMBER

        deadline = time.time() + event.wait_seconds
        while True:
            remaining = max(0.0, deadline - time.time())
            frac = 1 - (remaining / event.wait_seconds) if event.wait_seconds > 0 else 1

            with cooldown_placeholder.container():
                st.markdown(
                    f"""
                    <div class="sg-cooldown">
                        <div style="font-weight: 600; color: {AMBER}; margin-bottom: 4px;">
                            ⏳ System Cooldown — {event.service}
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #78716C;">
                            Rate limited (attempt {event.attempt}/{event.max_attempts}). Retrying in {remaining:0.1f}s.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(1.0, max(0.0, frac)))

            if remaining <= 0:
                break
            time.sleep(min(0.5, remaining))
        cooldown_placeholder.empty()

    set_retry_handler(on_retry)

    final_state = None
    try:
        for update in st.session_state.graph.stream(payload, stream_mode="updates"):
            for node_key, partial in update.items():
                display_name = reverse_map.get(node_key)
                aborted = isinstance(partial, dict) and partial.get("execution_status") == "aborted"

                if aborted:
                    st.session_state.pipeline_progress["errored"] = True

                if display_name:
                    st.session_state.pipeline_progress["active"] = display_name
                    with pipeline_placeholder.container():
                        render_pipeline(
                            display_name,
                            st.session_state.pipeline_progress["completed"],
                            errored=st.session_state.pipeline_progress["errored"],
                        )
                    if not aborted:
                        st.session_state.pipeline_progress["completed"].add(display_name)

                new_state = {**payload, **(final_state or {})}
                for key, value in (partial or {}).items():
                    if (
                        key in ["execution_trace", "error_log"]
                        and isinstance(value, list)
                        and isinstance(new_state.get(key), list)
                    ):
                        new_state[key] = new_state[key] + value  # Append to existing list
                    else:
                        new_state[key] = value  # Overwrite/Update scalar fields
                final_state = new_state

        return final_state, None

    except ServiceExhaustedError as e:
        st.session_state.pipeline_progress["errored"] = True
        with pipeline_placeholder.container():
            render_pipeline(
                st.session_state.pipeline_progress["active"],
                st.session_state.pipeline_progress["completed"],
                errored=True,
            )
        return None, e
    finally:
        set_retry_handler(None)
