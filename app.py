"""Streamlit entry point for the Scrygent deterministic compiler."""

import logging
import os

import streamlit as st

from ui import run_app

# STREAMLIT CLOUD SECRET BRIDGE

# The core engine (src/scrygent/) remains strictly UI-agnostic and reads
# credentials via os.getenv(). Streamlit Cloud injects credentials via
# st.secrets. This bridge maps the Streamlit secrets to environment variables
# at runtime, preserving the architectural boundary.
for key, value in st.secrets.items():
    if key not in os.environ:
        os.environ[key] = str(value)

# Configure root logger so INFO messages from the core engine appear in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)

if __name__ == "__main__":
    run_app()
