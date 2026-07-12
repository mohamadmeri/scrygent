"""Public API for the LangGraph state machine orchestration.

Exposes the graph builder function used to compile and invoke
the Scrygent execution pipeline.
"""

from .builder import build_graph

__all__ = ["build_graph"]
