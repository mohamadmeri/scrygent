"""Streamlit entry point for the Scrygent deterministic compiler."""

import logging

from ui import run_app

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)

if __name__ == "__main__":
    run_app()
