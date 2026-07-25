"""Streamlit entry point for the Scrygent deterministic compiler."""

import logging
import os

import streamlit as st

# INJECT SECRETS FIRST
if hasattr(st, "secrets"):
    for key, value in st.secrets.items():
        if key not in os.environ:
            os.environ[key] = str(value)

# IMPORT UI LATER
from ui import run_app

# Configure root logger so INFO messages from the core engine appear in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)

if __name__ == "__main__":
    run_app()
