"""Design tokens, global CSS, and page configuration for the Scrygent UI."""

import streamlit as st

# ============================================
# WARM DARK PALETTE (cohesive with landing page)
# ============================================
BG_BASE = "#0F0E0D"  # Deepest background
BG_SURFACE = "#1C1A18"  # Cards, panels, topbar
BG_ELEVATED = "#2A2724"  # Hover states, active containers
BORDER_SUBTLE = "#44403C"  # 1px borders, dividers

TEXT_PRIMARY = "#F5F0EB"  # Warm white headlines
TEXT_SECONDARY = "#A8A29A"  # Body copy
TEXT_MUTED = "#78716C"  # Captions, metadata

# Accents
SAGE = "#7FB069"
TEAL = "#5EEAD4"
AMBER = "#F59E0B"
RED = "#EF4444"

NODE_ORDER = ["Profiler", "Planner", "Executor", "Reporter"]
NODE_KEY_MAP = {
    "Profiler": "profiler",
    "Planner": "planner",
    "Executor": "executor",
    "Reporter": "reporter",
}

# Google Fonts URL for injection
FONTS_URL = (
    "https://fonts.googleapis.com/css2?family="
    "Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&"
    "family=Inter:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)


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
        @import url('{FONTS_URL}');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
            color: {TEXT_PRIMARY};
        }}
        .stApp {{
            background: {BG_BASE};
        }}

        /* Hide default Streamlit chrome */
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 1.5rem; }}

        /* =========================================
           TYPOGRAPHY
           ========================================= */
        h1, h2, h3 {{
            font-family: 'Cormorant Garamond', serif;
            font-weight: 500;
            letter-spacing: -0.02em;
            color: {TEXT_PRIMARY};
        }}
        h1 {{ font-size: 2.2rem; }}
        h2 {{ font-size: 1.8rem; }}
        h3 {{ font-size: 1.4rem; }}

        /* Mono tags / labels */
        .sg-mono-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {TEXT_MUTED};
        }}

        /* =========================================
           PIPELINE ANIMATION
           ========================================= */
        @keyframes pulse-sage {{
            0% {{ box-shadow: 0 0 0 0 rgba(127, 176, 105, 0.6); }}
            70% {{ box-shadow: 0 0 0 6px rgba(127, 176, 105, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(127, 176, 105, 0); }}
        }}
        .sg-node-active .sg-node-dot {{
            animation: pulse-sage 2s infinite;
        }}

        .sg-pipeline {{
            display: flex;
            align-items: center;
            background: {BG_SURFACE};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
        }}
        .sg-node {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: {TEXT_MUTED};
            white-space: nowrap;
        }}
        .sg-node-active {{
            color: {TEXT_PRIMARY};
            font-weight: 500;
        }}
        .sg-node-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {BORDER_SUBTLE};
            transition: all 0.3s ease;
        }}
        .sg-node-active .sg-node-dot {{
            background: {SAGE};
            box-shadow: 0 0 8px {SAGE};
        }}
        .sg-node-done .sg-node-dot {{
            background: {SAGE};
            opacity: 0.6;
        }}
        .sg-node-error .sg-node-dot {{
            background: {RED};
            box-shadow: 0 0 8px {RED};
        }}
        .sg-node-connector {{
            flex: 1;
            height: 1px;
            background: {BORDER_SUBTLE};
            margin: 0 10px;
            transition: background 0.3s ease;
        }}
        .sg-node-connector.active {{
            background: {SAGE};
            opacity: 0.4;
        }}

        /* =========================================
           TOPBAR
           ========================================= */
        .sg-topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            background: {BG_SURFACE};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 8px;
            margin-bottom: 20px;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            color: {TEXT_PRIMARY};
            flex-wrap: wrap;
            gap: 10px;
        }}
        .sg-brand {{
            font-family: 'Cormorant Garamond', serif;
            font-weight: 600;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            color: {TEXT_PRIMARY};
        }}
        .sg-brand-mark {{
            color: {SAGE};
            font-size: 1.1rem;
        }}
        .sg-divider {{
            color: {TEXT_MUTED};
            margin: 0 6px;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
        }}
        .sg-subtitle {{
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            color: {TEXT_MUTED};
            font-size: 0.82rem;
        }}
        .sg-topbar-right {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .sg-about-link {{
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: {TEXT_MUTED};
            text-decoration: none;
            transition: color 0.2s;
        }}
        .sg-about-link:hover {{
            color: {SAGE};
        }}
        .sg-status-group {{
            display: flex;
            gap: 16px;
        }}
        .sg-status-pill {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: {TEXT_MUTED};
        }}
        .sg-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}
        .sg-dot-ok {{
            background: {SAGE};
            box-shadow: 0 0 6px {SAGE};
        }}
        .sg-dot-warn {{
            background: {AMBER};
            box-shadow: 0 0 6px {AMBER};
        }}
        .sg-dot-bad {{
            background: {RED};
            box-shadow: 0 0 6px {RED};
        }}

        /* =========================================
           COOLDOWN BANNER
           ========================================= */
        .sg-cooldown {{
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.08), rgba(245, 158, 11, 0.02));
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
        }}

        /* =========================================
           CHAT & MESSAGES
           ========================================= */
        .stChatMessage {{
            background: transparent !important;
        }}
        .stChatMessage [data-testid="stChatMessageContent"] {{
            background: {BG_SURFACE};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 10px;
            padding: 1.25rem 1.5rem !important;
        }}
        .stChatMessage [data-testid="stChatMessageContent"] p {{
            margin-bottom: 0.75rem;
        }}
        .stChatMessage [data-testid="stChatMessageContent"] p:last-child {{
            margin-bottom: 0;
        }}

        /* =========================================
           BUTTONS & INPUTS
           ========================================= */
        .stButton > button {{
            background: {BG_ELEVATED} !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 6px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
        }}
        .stButton > button:hover {{
            border-color: {SAGE} !important;
            box-shadow: 0 0 12px rgba(127, 176, 105, 0.1) !important;
        }}

        .stChatInputContainer textarea {{
            background: {BG_SURFACE} !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 8px !important;
            font-family: 'Inter', sans-serif !important;
        }}
        .stChatInputContainer textarea:focus {{
            border-color: {SAGE} !important;
            box-shadow: 0 0 0 1px rgba(127, 176, 105, 0.2) !important;
        }}

        /* =========================================
           CONTAINERS & CARDS
           ========================================= */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > div > div {{
            /* Streamlit container border override */
        }}
        .sg-card {{
            background: {BG_SURFACE};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 10px;
            padding: 1.25rem;
        }}

        /* =========================================
           CODE BLOCKS
           ========================================= */
        .stCodeBlock pre {{
            background: {BG_BASE} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 8px !important;
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* =========================================
           SCROLLBARS
           ========================================= */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {BG_BASE};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {BORDER_SUBTLE};
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {TEXT_MUTED};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
