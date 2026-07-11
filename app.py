import logging
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

# Import the core engine
from scrygent.graph.builder import build_graph
from scrygent.models.state import AgentState
from scrygent.resilience import (
    set_retry_handler,
    RetryEvent,
    ServiceExhaustedError,
)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Scrygent · Deterministic Data Agent",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# DESIGN TOKENS
# ==============================================================================
# Palette: a dark "compiler console" surface rather than a generic dashboard.
# Two accents carry meaning, not decoration: teal = verified/healthy, amber =
# cooldown/attention, red = aborted. Everything else stays quiet on purpose.

BG          = "#0B0E14"
SURFACE     = "#131826"
SURFACE_ALT = "#1A2135"
BORDER      = "#232C42"
TEXT        = "#E7EAF2"
TEXT_MUTED  = "#8891A7"
ACCENT_OK   = "#4FD1C5"   # verified / healthy
ACCENT_WARN = "#F2B84B"   # cooldown / attention
ACCENT_BAD  = "#F2645A"   # aborted / down

NODE_ORDER = ["Profiler", "Planner", "Executor", "Reporter"]
# Matches the node keys registered in graph/builder.py exactly:
# graph.add_node("profiler", ...), ("planner", ...), ("executor", ...), ("reporter", ...)
NODE_KEY_MAP = {
    "Profiler": "profiler",
    "Planner": "planner",
    "Executor": "executor",
    "Reporter": "reporter",
}

# ==============================================================================
# GLOBAL CSS
# ==============================================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.stApp {{
    background: {BG};
    color: {TEXT};
}}

/* Hide Streamlit chrome for a product feel */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem; max-width: 760px; }}

/* ---------- Top status bar ---------- */
.sg-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-bottom: 18px;
}}
.sg-brand {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}}
.sg-brand-mark {{ color: {ACCENT_OK}; }}
.sg-status-group {{ display: flex; gap: 14px; }}
.sg-status-pill {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {TEXT_MUTED};
}}
.sg-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 6px currentColor;
}}
.sg-dot-ok   {{ background: {ACCENT_OK};   color: {ACCENT_OK}; }}
.sg-dot-bad  {{ background: {ACCENT_BAD};  color: {ACCENT_BAD}; }}
.sg-dot-warn {{ background: {ACCENT_WARN}; color: {ACCENT_WARN}; }}
.sg-dot-off  {{ background: #3A4256; color: #3A4256; }}

/* ---------- Pipeline strip (signature element) ---------- */
.sg-pipeline {{
    display: flex;
    align-items: center;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
}}
.sg-node {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {TEXT_MUTED};
}}
.sg-node-active {{ color: {TEXT}; }}
.sg-node-active .sg-node-dot {{
    background: {ACCENT_OK}; box-shadow: 0 0 8px {ACCENT_OK};
}}
.sg-node-done .sg-node-dot {{
    background: {ACCENT_OK}; opacity: 0.55; box-shadow: none;
}}
.sg-node-error .sg-node-dot {{ background: {ACCENT_BAD}; box-shadow: 0 0 8px {ACCENT_BAD}; }}
.sg-node-dot {{
    width: 8px; height: 8px; border-radius: 50%; background: #3A4256;
    transition: all 0.3s ease;
}}
.sg-node-connector {{
    flex: 1; height: 1px; background: {BORDER}; margin: 0 10px;
}}

/* ---------- Cooldown banner ---------- */
.sg-cooldown {{
    background: linear-gradient(90deg, rgba(242,184,75,0.10), rgba(242,184,75,0.03));
    border: 1px solid rgba(242,184,75,0.35);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
}}
.sg-cooldown-title {{
    font-weight: 600; font-size: 0.88rem; color: {ACCENT_WARN};
    display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
}}
.sg-cooldown-sub {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {TEXT_MUTED};
}}

/* ---------- Cards / surfaces ---------- */
.sg-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
}}
.sg-mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: {TEXT_MUTED}; }}

[data-testid="stChatMessage"] {{
    background: transparent;
    padding: 0;
}}

.stButton > button {{
    background: {SURFACE_ALT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    font-size: 0.85rem;
    padding: 6px 14px;
}}
.stButton > button:hover {{ border-color: {ACCENT_OK}; color: {ACCENT_OK}; }}

.stAlert {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HEALTH CHECKS (cached, lightweight, non-blocking to the main flow)
# ==============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def check_groq_health() -> bool:
    """Pings the Groq models endpoint. Returns False on any failure (never raises)."""
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


def render_topbar():
    groq_ok = check_groq_health()
    qdrant_ok = check_qdrant_health()
    groq_dot = "sg-dot-ok" if groq_ok else "sg-dot-bad"
    qdrant_dot = "sg-dot-ok" if qdrant_ok else "sg-dot-bad"
    st.markdown(f"""
    <div class="sg-topbar">
        <div class="sg-brand"><span class="sg-brand-mark">◆</span> SCRYGENT</div>
        <div class="sg-status-group">
            <div class="sg-status-pill"><span class="sg-dot {groq_dot}"></span>GROQ</div>
            <div class="sg-status-pill"><span class="sg-dot {qdrant_dot}"></span>QDRANT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline(active_node: str | None, completed: set[str], errored: bool):
    """Renders the signature pipeline strip. `active_node` is one of NODE_ORDER or None."""
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


# ==============================================================================
# SESSION STATE
# ==============================================================================

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "query" not in st.session_state:
    st.session_state.query = None
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "pipeline_progress" not in st.session_state:
    st.session_state.pipeline_progress = {"active": None, "completed": set(), "errored": False}


def reset_session():
    st.session_state.csv_path = None
    st.session_state.query = None
    st.session_state.final_state = None
    st.session_state.pipeline_progress = {"active": None, "completed": set(), "errored": False}


# ==============================================================================
# RESILIENT INVOCATION — wires the backend retry wrapper to a live UI banner
# ==============================================================================

def run_graph_with_resilience(payload: dict, pipeline_placeholder, cooldown_placeholder):
    """
    Streams the graph node-by-node so the pipeline strip updates live, and
    registers a UI callback with the resilience layer so 429 cooldowns render
    as a countdown banner instead of a frozen spinner.
    """
    reverse_map = {v: k for k, v in NODE_KEY_MAP.items()}

    def on_retry(event: RetryEvent):
        deadline = time.time() + event.wait_seconds
        while True:
            remaining = max(0.0, deadline - time.time())
            frac = 1 - (remaining / event.wait_seconds) if event.wait_seconds > 0 else 1
            with cooldown_placeholder.container():
                st.markdown(f"""
                <div class="sg-cooldown">
                    <div class="sg-cooldown-title">⏳ System Cooldown — {event.service}</div>
                    <div class="sg-cooldown-sub">Rate limited (attempt {event.attempt}/{event.max_attempts}).
                    Retrying in {remaining:0.1f}s.</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(1.0, max(0.0, frac)))
            if remaining <= 0:
                break
            time.sleep(min(0.5, remaining))
        cooldown_placeholder.empty()

    set_retry_handler(on_retry)

    final_state = None
    try:
        # stream_mode="updates" tells us WHICH node just ran (for the pipeline strip);
        # we track the most recently completed node key ourselves and take the FINAL
        # full state via a single terminal invoke-equivalent — but to avoid running the
        # graph twice, we instead accumulate state manually as LangGraph reports it.
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
                # Merge each node's partial state update into a running snapshot.
                # This is the SAME accumulation LangGraph performs internally, so the
                # end result matches what a plain .invoke(payload) would have returned
                # -- without executing the graph a second time. Safe here because
                # AgentState's fields (state.py) use plain pydantic Field(), not
                # Annotated[..., reducer] -- LangGraph's default merge is last-write-
                # wins per key, which is exactly what this dict-merge reproduces.
                final_state = {**payload, **(final_state or {}), **(partial or {})}
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


# ==============================================================================
# UI: HEADER
# ==============================================================================

render_topbar()
st.markdown("### Scrygent")
st.markdown(
    '<p style="color:#8891A7; margin-top:-8px;">Deterministically analyze your data via natural language.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.csv_path:
    with st.chat_message("assistant"):
        st.markdown("Upload a CSV to begin.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            # Pre-flight validation: `type=["csv"]` above only checks the file
            # extension, not the content. A renamed non-CSV file, an empty file,
            # or a bad encoding would otherwise sail through here and only fail
            # later inside the Profiler node -- after a full rerun cycle. Catch
            # it here instead, with a fast, cheap parse of just a few rows.
            try:
                import pandas as pd
                uploaded_file.seek(0)
                pd.read_csv(uploaded_file, nrows=5)
                uploaded_file.seek(0)
            except Exception as e:
                st.error(
                    f"**Couldn't read `{uploaded_file.name}` as a CSV.** {type(e).__name__}: {e}",
                    icon="🛑",
                )
                st.stop()

            temp_dir = Path(tempfile.gettempdir())
            file_path = temp_dir / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            st.session_state.csv_path = file_path
            st.toast(f"Loaded {uploaded_file.name}", icon="✅")
            st.rerun()

if not st.session_state.csv_path:
    st.stop()

# ==============================================================================
# UI: DATASET CARD + CHAT HISTORY
# ==============================================================================

with st.chat_message("assistant"):
    st.markdown(f"""
    <div class="sg-card">
        <div class="sg-mono">DATASET</div>
        <div style="font-weight:600; margin-top:4px;">{st.session_state.csv_path.name}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start over", on_click=reset_session):
        pass

if st.session_state.query:
    with st.chat_message("user"):
        st.markdown(st.session_state.query)

# ==============================================================================
# UI: GRAPH INVOCATION (streamed, resilient)
# ==============================================================================

if st.session_state.query and st.session_state.final_state is None:
    with st.chat_message("assistant"):
        pipeline_placeholder = st.empty()
        cooldown_placeholder = st.empty()
        with pipeline_placeholder.container():
            render_pipeline(None, set(), errored=False)

        try:
            # Constructing AgentState is now INSIDE the try block. Previously it
            # sat above it -- unlikely to throw for a plain str/Path payload, but
            # if it ever did, it'd be a genuinely uncaught exception mid-script,
            # and unlike every other failure path here, st.session_state.query
            # wouldn't get reset -- the next rerun would hit the same line and
            # fail again in a loop.
            initial_state = AgentState(
                original_csv_path=str(st.session_state.csv_path),  # type: ignore
                current_csv_path=str(st.session_state.csv_path),  # type: ignore
                user_query=st.session_state.query,
            )
            payload = initial_state.model_dump(mode="json")
            raw_final_update, exhausted_error = run_graph_with_resilience(
                payload, pipeline_placeholder, cooldown_placeholder
            )

            if exhausted_error is not None:
                st.error(
                    f"**Service temporarily unavailable.** {exhausted_error.service} is still "
                    f"rate limited after {exhausted_error.attempts} attempts. Please try again "
                    f"in a minute.",
                    icon="🛑",
                )
                st.session_state.query = None  # allow retry without a stale query stuck in state
                st.stop()

            if raw_final_update is None:
                raise RuntimeError("Graph stream produced no state updates.")
            st.session_state.final_state = AgentState.model_validate(raw_final_update)

            # model_validate succeeding only means the STATE OBJECT is well-formed
            # -- it says nothing about whether the run actually succeeded. Gate the
            # success toast on execution_status specifically, so an aborted run
            # (bad CSV, tool failure, etc.) doesn't show a cheerful green toast
            # right next to the error banner below.
            if st.session_state.final_state.execution_status == "complete":
                st.toast("Analysis complete", icon="✅")
        except ServiceExhaustedError as e:
            st.error(f"**{e.service} is temporarily unavailable.** Please try again shortly.", icon="🛑")
            st.session_state.query = None
            st.stop()
        except Exception as e:
            logger.error("Graph execution failed: %s", e, exc_info=True)
            st.error("A system error occurred during execution. See logs for details.", icon="🛑")
            st.session_state.query = None
            st.stop()

        st.rerun()

# ==============================================================================
# UI: FINAL RESULT
# ==============================================================================

if st.session_state.final_state:
    state: AgentState = st.session_state.final_state

    with st.chat_message("assistant"):
        if state.execution_status == "aborted":
            st.error("Execution aborted", icon="🛑")
            if state.error_log:
                st.markdown(f"**Reason:** {state.error_log[-1]}")
            with st.expander("View full error log"):
                for err in state.error_log:
                    st.markdown(f"- {err}")

        elif state.execution_status == "complete" and state.final_report:
            report = state.final_report
            st.markdown(f"### {getattr(report, 'primary_answer', 'Analysis Complete')}")

            insights = getattr(report, "additional_insights", None)
            if insights:
                st.markdown("**Additional insights**")
                for insight in insights:
                    st.markdown(f"- {insight}")

            plots = getattr(report, "plots", None)
            if plots:
                st.markdown("**Visualizations**")
                for plot in plots:
                    st.image(str(plot.file_path), caption=plot.description)

            with st.expander("View execution trace"):
                for step in state.execution_trace:
                    icon = "✅" if step.status == "success" else "❌"
                    t_name = getattr(step.tool_name, "value", step.tool_name)
                    st.markdown(f"{icon} **{t_name}** ({step.duration_ms}ms)")
                    if step.summary:
                        st.caption(step.summary)
                    if step.error:
                        st.caption(f"*Error: {step.error}*")

# ==============================================================================
# UI: INPUT BAR
# ==============================================================================

if not st.session_state.query:
    user_input = st.chat_input("Ask a question about your data...")
    if user_input:
        st.session_state.query = user_input
        st.rerun()
