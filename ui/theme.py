"""Design tokens, global CSS, and page configuration for the Scrygent UI."""

import streamlit as st

# Warm dark palette
BG = "#1E1E1E"
SURFACE = "#2A2A2A"
BORDER = "rgba(255, 255, 255, 0.08)"
TEXT = "#F5F5F5"
TEXT_MUTED = "#A0A0A0"

# Semantic accents
ACCENT_OK = "#84A571"  # Sage green for verified/healthy
ACCENT_WARN = "#E5C07B"  # Soft gold for cooldown
ACCENT_BAD = "#D97757"  # Warm clay for aborted/error

NODE_ORDER = ["Profiler", "Planner", "Executor", "Reporter"]
NODE_KEY_MAP = {
    "Profiler": "profiler",
    "Planner": "planner",
    "Executor": "executor",
    "Reporter": "reporter",
}


def configure_page() -> None:
    """Sets the global Streamlit page configuration."""
    st.set_page_config(
        page_title="Scrygent · Deterministic Compiler",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_css() -> None:
    """Injects the global CSS for the warm dark theme and pipeline animations."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; color: {TEXT}; }}
        .stApp {{ background: {BG}; }}
        
        /* Hide default Streamlit chrome for a product feel */
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 2rem; }}
        
        /* Pipeline animation */
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(132, 165, 113, 0.7); }}
            70% {{ box-shadow: 0 0 0 6px rgba(132, 165, 113, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(132, 165, 113, 0); }}
        }}
        .sg-node-active .sg-node-dot {{ animation: pulse 2s infinite; }}
        
        /* Custom pipeline strip styling */
        .sg-pipeline {{
            display: flex; align-items: center; background: {SURFACE};
            border: 1px solid {BORDER}; border-radius: 8px;
            padding: 12px 16px; margin-bottom: 16px;
        }}
        .sg-node {{
            display: flex; align-items: center; gap: 8px;
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {TEXT_MUTED};
        }}
        .sg-node-active {{ color: {TEXT}; font-weight: 500; }}
        .sg-node-dot {{
            width: 8px; height: 8px; border-radius: 50%; background: #444; transition: all 0.3s ease;
        }}
        .sg-node-active .sg-node-dot {{ background: {ACCENT_OK}; }}
        .sg-node-done .sg-node-dot {{ background: {ACCENT_OK}; opacity: 0.6; }}
        .sg-node-error .sg-node-dot {{ background: {ACCENT_BAD}; }}
        .sg-node-connector {{ flex: 1; height: 1px; background: {BORDER}; margin: 0 12px; }}
        
        /* Cooldown banner */
        .sg-cooldown {{
            background: linear-gradient(90deg, rgba(229, 192, 123, 0.10), rgba(229, 192, 123, 0.03));
            border: 1px solid rgba(229, 192, 123, 0.35); border-radius: 8px;
            padding: 12px 16px; margin-bottom: 16px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
