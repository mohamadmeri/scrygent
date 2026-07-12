import hashlib
import logging
import os
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Load .env file at module import time
load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "scrygent_experience"
RELEVANCE_THRESHOLD = 0.75
VECTOR_DIMENSION_SIZE = 384
VECTOR_NAME_TARGET = "fast-bge-small-en"


def _get_client() -> QdrantClient | None:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        logger.warning("Memory disabled: Qdrant credentials missing.")
        return None
    return QdrantClient(url=url, api_key=api_key)


def _ensure_collection(client: QdrantClient) -> None:
    """Creates the collection matching the 384-dimension schema definition."""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            # Configured as an unnamed default vector block for client.search / client.upsert alignment
            vectors_config=VectorParams(size=VECTOR_DIMENSION_SIZE, distance=Distance.COSINE),
        )


def _embed_text_huggingface(text: str) -> list[float] | None:
    """Embed text using the Hugging Face Inference API Feature Extraction Pipeline."""
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        logger.warning("HF_API_TOKEN environment variable not set; skipping memory.")
        return None

    try:
        import requests
    except ImportError:
        logger.error("requests library missing. Install via pip install requests")
        return None

    # Fixed: Targeting the explicit serverless feature-extraction router path
    api_url = os.getenv(
        "HF_EMBEDDING_API_URL",
        "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
    )
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        response = requests.post(
            api_url,
            json={"inputs": text, "options": {"wait_for_model": True}},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()

        # Guard clause: Handle structural variations from Hugging Face returns
        if isinstance(result, list) and len(result) > 0:
            # If the response is a 2D array, squeeze out the inner layer
            if isinstance(result[0], list):
                return result[0]
            return result
        else:
            logger.warning("Unexpected array output shape from Hugging Face: %s", result)
            return None
    except Exception as e:
        logger.error("Hugging Face API call failed: %s", e)
        return None


def retrieve_experience(query: str, top_k: int = 2) -> str:
    """Fetches past successful plans from Qdrant using Hugging Face serverless embeddings."""
    client = _get_client()
    if not client:
        return "No past experience available."

    info = client.get_collection(COLLECTION_NAME)
    print(f"DEBUG: Collection vector config: {info.config.params.vectors}")

    vec = _embed_text_huggingface(query)
    if vec is None:
        logger.info("Embedding processing unavailable; bypassing cache retrieval.")
        return "No past experience available."

    try:
        _ensure_collection(client)

        # Query via standard structural vector matching
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            using=VECTOR_NAME_TARGET,
            limit=top_k,
        )

        if not response.points:
            return "No past experience available."

        experiences = []
        # Iterate specifically over response.points
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
    """Embeds query via Hugging Face and commits the records to Qdrant Cloud."""
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

        point = PointStruct(
            id=vector_id, vector={VECTOR_NAME_TARGET: vec}, payload={"query": query, "plan_json": plan_json}
        )

        # Uses stable transaction payloads
        client.upsert(collection_name=COLLECTION_NAME, points=[point])
        logger.info("Successfully committed execution to long-term memory.")
    except Exception as e:
        logger.error("Memory commit failed: %s", e, exc_info=True)
