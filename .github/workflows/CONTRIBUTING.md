# Contributing to Scrygent

Thank you for your interest in Scrygent! 

## Architecture First
Scrygent is a **deterministic compiler**, not a ReAct code-generation agent. Any contribution that introduces arbitrary Python execution (e.g., `eval()`, `exec()`, or sandboxed REPLs) will be rejected to maintain the zero-trust security boundary.

## Development Setup
1. Clone the repo and install dependencies using `uv`:
   ```bash
   git clone https://github.com/mohamadmeri/scrygent.git
   cd scrygent
   uv sync
   ```
2. Copy the environment templates:
   ```bash
   cp .env.example .env
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

## Pull Request Process
1. Ensure all code passes `ruff check .` and `mypy .`.
2. Add unit tests for any new tools or core infrastructure changes.
3. Update the `docs/ARCHITECTURE.md` if your PR changes the graph routing or dependency hierarchy.