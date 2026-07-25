"""Serverless semantic memory engine for experience replay.

This module provides long-term memory capabilities by embedding successful
execution plans and storing them in a Qdrant vector database. It utilizes
Hugging Face's serverless inference API for zero-infrastructure embeddings.
"""

import hashlib
import logging
from typing import Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "scrygent_experience"
RELEVANCE_THRESHOLD = 0.75
VECTOR_DIMENSION_SIZE = 384
VECTOR_NAME_TARGET = "fast-bge-small-en"


def _get_client() -> QdrantClient | None:
    """Initializes the Qdrant client using typed settings."""
    if not settings.memory_enabled:
        return None

    if not settings.qdrant_url or not settings.qdrant_api_key:
        logger.warning("Memory disabled: Qdrant credentials missing in configuration.")
        return None

    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key.get_secret_value())


def _ensure_collection(client: QdrantClient) -> None:
    """Creates the Qdrant collection with a named vector configuration."""
    if not client.collection_exists(COLLECTION_NAME):
        # Use a named vector configuration to align with query_points and PointStruct
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={VECTOR_NAME_TARGET: VectorParams(size=VECTOR_DIMENSION_SIZE, distance=Distance.COSINE)},
        )


def _embed_text_huggingface(text: str) -> list[float] | None:
    """Embeds text using the Hugging Face Inference API.

    Args:
        text: The natural language query to embed.

    Returns:
        A list of floats representing the embedding vector, or None on failure.
    """
    if not settings.hf_api_token:
        logger.warning("HF_API_TOKEN configuration missing; skipping memory embedding.")
        return None

    headers = {"Authorization": f"Bearer {settings.hf_api_token.get_secret_value()}"}

    try:
        response = requests.post(
            settings.hf_embedding_api_url,
            json={"inputs": text, "options": {"wait_for_model": True}},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()

        # Handle structural variations from Hugging Face returns
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result[0]
            return result

        logger.warning("Unexpected array output shape from Hugging Face: %s", result)
        return None
    except Exception as e:
        logger.error("Hugging Face API call failed: %s", e)
        return None


def retrieve_experience(query: str, top_k: int = 2) -> str:
    """Fetches past successful plans from Qdrant using semantic similarity.

    Args:
        query: The current user query to match against past experiences.
        top_k: The maximum number of past experiences to retrieve.

    Returns:
        A formatted string containing past queries and their successful plans,
        or a fallback message if no relevant experience is found.
    """
    client = _get_client()
    if not client:
        return "No past experience available."

    vec = _embed_text_huggingface(query)
    if vec is None:
        logger.info("Embedding processing unavailable; bypassing cache retrieval.")
        return "No past experience available."

    try:
        _ensure_collection(client)
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            using=VECTOR_NAME_TARGET,
            limit=top_k,
        )

        if not response.points:
            return "No past experience available."

        experiences = []
        for hit in response.points:
            score = getattr(hit, "score", None)
            payload = getattr(hit, "payload", None) or {}
            if score is not None and score > RELEVANCE_THRESHOLD:
                past_query = payload.get("query", "")
                past_plan = payload.get("plan_json", "")
                experiences.append(f"PAST QUERY: {past_query}\nSUCCESSFUL PLAN:\n{past_plan}")

        return "\n\n".join(experiences) if experiences else "No highly relevant experience found."

    except Exception as e:
        logger.error("Memory retrieval trace failed: %s", e, exc_info=True)
        return "No past experience available."


def commit_experience(query: str, plan: Any) -> None:
    """Embeds a query and commits the validated execution plan to long-term memory.

    Args:
        query: The original natural language query.
        plan: The validated Pydantic Plan object to store.
    """
    client = _get_client()
    if not client:
        return

    vec = _embed_text_huggingface(query)
    if vec is None:
        logger.info("Embedding processing unavailable; bypassing cache commit.")
        return

    try:
        _ensure_collection(client)
        vector_id = hashlib.md5(query.encode("utf-8")).hexdigest()
        plan_json = plan.model_dump_json(indent=2)

        point = PointStruct(id=vector_id, vector={VECTOR_NAME_TARGET: vec}, payload={"query": query, "plan_json": plan_json})

        client.upsert(collection_name=COLLECTION_NAME, points=[point])
        logger.info("Successfully committed execution to long-term memory.")
    except Exception as e:
        logger.error("Memory commit failed: %s", e, exc_info=True)
