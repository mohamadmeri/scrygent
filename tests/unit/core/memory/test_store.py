"""Destructive test suite for the semantic memory engine.

This module aggressively tests the Qdrant and Hugging Face integration. It
ensures that missing credentials fail safely, malformed API responses do not
crash the compiler, and vector payloads are constructed and filtered with
strict adherence to the relevance threshold.
"""

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from scrygent.core.memory import store


class DummyPlan(BaseModel):
    """Mock plan to satisfy the commit_experience payload contract."""

    step: str = "analyze"


@pytest.fixture
def mock_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject dummy Hugging Face and Qdrant credentials."""
    monkeypatch.setenv("HF_API_TOKEN", "test-hf-token")
    monkeypatch.setenv("QDRANT_URL", "http://mock-qdrant")
    monkeypatch.setenv("QDRANT_API_KEY", "test-qdrant-key")


@pytest.fixture
def mock_hf_flat_list_response(monkeypatch: pytest.MonkeyPatch, mock_hf_env: None) -> None:
    """Mock HF API to return a flat list of floats."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [0.1, 0.2, 0.3]
    monkeypatch.setattr(store.requests, "post", lambda *args, **kwargs: mock_resp)


@pytest.fixture
def mock_hf_nested_list_response(monkeypatch: pytest.MonkeyPatch, mock_hf_env: None) -> None:
    """Mock HF API to return a nested list of floats (common HF format)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [[0.4, 0.5, 0.6]]
    monkeypatch.setattr(store.requests, "post", lambda *args, **kwargs: mock_resp)


class TestMemoryClientInitialization:
    """Tests validating strict credential enforcement and safe fallbacks."""

    def test_get_client_returns_none_on_missing_qdrant_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete the `QDRANT_URL` environment variable.

        The factory must return None and log a warning, disabling memory gracefully.
        """
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.setenv("QDRANT_API_KEY", "test")

        assert store._get_client() is None

    def test_get_client_returns_none_on_missing_qdrant_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete the `QDRANT_API_KEY` environment variable.

        The factory must return None and disable memory gracefully.
        """
        monkeypatch.setenv("QDRANT_URL", "http://test")
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        assert store._get_client() is None


class TestHuggingFaceEmbedding:
    """Tests validating the embedding API interaction and response parsing."""

    def test_parses_flat_list_response_correctly(self, mock_hf_flat_list_response: None) -> None:
        """Inject a mock HF API response returning a flat list of floats.

        Asserts the wrapper correctly extracts the flat list as the vector.
        """
        vec = store._embed_text_huggingface("test query")
        assert vec == [0.1, 0.2, 0.3]

    def test_parses_nested_list_response_correctly(self, mock_hf_nested_list_response: None) -> None:
        """Inject a mock HF API response returning a nested list.

        Asserts the wrapper correctly extracts the inner list as the vector.
        """
        vec = store._embed_text_huggingface("test query")
        assert vec == [0.4, 0.5, 0.6]

    def test_returns_none_on_empty_hf_response(self, monkeypatch: pytest.MonkeyPatch, mock_hf_env: None) -> None:
        """Inject an empty list response from the HF API.

        The wrapper must catch the unexpected shape and return None.
        """
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = []
        monkeypatch.setattr(store.requests, "post", lambda *args, **kwargs: mock_resp)

        assert store._embed_text_huggingface("test query") is None

    def test_returns_none_on_missing_hf_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete the `HF_API_TOKEN` environment variable.

        The wrapper must return None immediately without attempting the API call.
        """
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        assert store._embed_text_huggingface("test query") is None

    def test_returns_none_on_api_network_failure(self, monkeypatch: pytest.MonkeyPatch, mock_hf_env: None) -> None:
        """Inject a `requests.exceptions.RequestException` during the API call.

        The wrapper must catch the exception, log it, and return None to prevent
        crashing the main execution graph.
        """

        def raise_error(*args: Any, **kwargs: Any) -> None:
            raise store.requests.exceptions.RequestException("Network Error")

        monkeypatch.setattr(store.requests, "post", raise_error)

        assert store._embed_text_huggingface("test query") is None


class TestRetrieveExperience:
    """Tests validating the Qdrant query construction and relevance filtering."""

    def test_returns_no_experience_when_client_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force `_get_client` to return None (missing creds).

        The function must return the exact fallback string without attempting
        to embed the query.
        """
        monkeypatch.setattr(store, "_get_client", lambda: None)
        monkeypatch.setattr(store, "_embed_text_huggingface", lambda x: pytest.fail("Embedding should not be called"))

        result = store.retrieve_experience("query")
        assert result == "No past experience available."

    def test_returns_no_experience_when_embedding_fails(
        self, monkeypatch: pytest.MonkeyPatch, mock_hf_env: None
    ) -> None:
        """Force `_embed_text_huggingface` to return None.

        The function must return the exact fallback string without querying Qdrant.
        """
        monkeypatch.setattr(store, "_get_client", lambda: MagicMock())
        monkeypatch.setattr(store, "_embed_text_huggingface", lambda x: None)

        result = store.retrieve_experience("query")
        assert result == "No past experience available."

    def test_returns_formatted_string_on_relevant_hit(
        self, monkeypatch: pytest.MonkeyPatch, mock_hf_flat_list_response: None
    ) -> None:
        """Inject a mock Qdrant client returning a point with score > 0.75.

        Asserts the function extracts the payload and formats the exact string
        containing the past query and plan.
        """
        mock_client = MagicMock()
        mock_point = MagicMock(score=0.85, payload={"query": "past query", "plan_json": "{...}"})
        mock_client.query_points.return_value = MagicMock(points=[mock_point])
        monkeypatch.setattr(store, "_get_client", lambda: mock_client)

        result = store.retrieve_experience("test query")

        assert "PAST QUERY: past query" in result
        assert "SUCCESSFUL PLAN:\n{...}" in result

    def test_returns_no_relevant_experience_on_low_score(
        self, monkeypatch: pytest.MonkeyPatch, mock_hf_flat_list_response: None
    ) -> None:
        """Inject a mock Qdrant client returning a point with score < 0.75.

        The function must filter it out and return the exact fallback string.
        """
        mock_client = MagicMock()
        mock_point = MagicMock(score=0.50, payload={"query": "past query", "plan_json": "{...}"})
        mock_client.query_points.return_value = MagicMock(points=[mock_point])
        monkeypatch.setattr(store, "_get_client", lambda: mock_client)

        result = store.retrieve_experience("test query")
        assert result == "No highly relevant experience found."

    def test_returns_fallback_on_qdrant_exception(
        self, monkeypatch: pytest.MonkeyPatch, mock_hf_flat_list_response: None
    ) -> None:
        """Inject a mock Qdrant client that raises an Exception during query.

        The function must catch the exception and return the exact fallback string.
        """
        mock_client = MagicMock()
        mock_client.query_points.side_effect = Exception("Qdrant exploded")
        monkeypatch.setattr(store, "_get_client", lambda: mock_client)

        result = store.retrieve_experience("test query")
        assert result == "No past experience available."


class TestCommitExperience:
    """Tests validating the Qdrant upsert payload and error handling."""

    def test_silently_returns_when_client_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force `_get_client` to return None.

        The function must return None without attempting to embed or upsert.
        """
        monkeypatch.setattr(store, "_get_client", lambda: None)
        monkeypatch.setattr(store, "_embed_text_huggingface", lambda x: pytest.fail("Embedding should not be called"))

        assert store.commit_experience("query", DummyPlan()) is None

    def test_silently_returns_when_embedding_fails(self, monkeypatch: pytest.MonkeyPatch, mock_hf_env: None) -> None:
        """Force `_embed_text_huggingface` to return None.

        The function must return None without attempting to upsert.
        """
        monkeypatch.setattr(store, "_get_client", lambda: MagicMock())
        monkeypatch.setattr(store, "_embed_text_huggingface", lambda x: None)

        assert store.commit_experience("query", DummyPlan()) is None

    def test_generates_correct_point_struct_and_upserts(
        self, monkeypatch: pytest.MonkeyPatch, mock_hf_flat_list_response: None
    ) -> None:
        """Inject a valid mock client and embedding.

        Asserts the function computes the exact MD5 vector ID, dumps the plan
        to JSON, and calls `upsert` with a correctly structured PointStruct.
        """
        mock_client = MagicMock()
        monkeypatch.setattr(store, "_get_client", lambda: mock_client)

        query = "my test query"
        plan = DummyPlan(step="execute")
        expected_id = hashlib.md5(query.encode("utf-8")).hexdigest()

        store.commit_experience(query, plan)

        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args.kwargs

        assert call_args["collection_name"] == store.COLLECTION_NAME
        points = call_args["points"]
        assert len(points) == 1
        assert points[0].id == expected_id
        assert points[0].vector == {store.VECTOR_NAME_TARGET: [0.1, 0.2, 0.3]}
        assert points[0].payload["query"] == query
        assert points[0].payload["plan_json"] == plan.model_dump_json(indent=2)

    def test_silently_catches_qdrant_upsert_exception(
        self, monkeypatch: pytest.MonkeyPatch, mock_hf_flat_list_response: None
    ) -> None:
        """Inject a mock Qdrant client that raises an Exception during upsert.

        The function must catch the exception and return None to prevent
        crashing the Reporter node after a successful execution.
        """
        mock_client = MagicMock()
        mock_client.upsert.side_effect = Exception("Qdrant write failed")
        monkeypatch.setattr(store, "_get_client", lambda: mock_client)

        assert store.commit_experience("query", DummyPlan()) is None
