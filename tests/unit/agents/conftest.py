"""Shared fixtures for the LangGraph agent node test suite."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_analyze_data(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the analyze_data tool to isolate Executor routing logic."""
    mock = MagicMock(return_value={"result": "success"})
    monkeypatch.setattr("scrygent.agents.executor_node.analyze_data", mock)
    return mock


@pytest.fixture
def mock_filter_dataset(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the filter_dataset tool to isolate Executor routing logic."""
    mock = MagicMock(return_value={"current_csv_path": "/tmp/fake.csv", "row_count": 1})
    monkeypatch.setattr("scrygent.agents.executor_node.filter_dataset", mock)
    return mock
