"""Idempotent dataset downloader for Scrygent benchmarks."""

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATASET_TARGETS = {
    "infiagent": {
        "repo_id": "infiagent/DABench",
        "allow_patterns": ["da-dev-questions.jsonl", "da-dev-labels.jsonl", "da-dev-tables/*.csv"],
    },
    "databench_lite": {
        "repo_id": "SemanticEval/databench_lite",
        "allow_patterns": ["*"],
    },
}


def download_dataset(dataset_name: str, base_dir: str) -> None:
    """Downloads the specified dataset files idempotently using snapshot_download."""
    if dataset_name not in DATASET_TARGETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_TARGETS.keys())}")

    target = DATASET_TARGETS[dataset_name]
    output_dir = Path(base_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching dataset '%s' from Hugging Face...", dataset_name)

    # snapshot_download handles caching and skips if files already exist
    snapshot_download(
        repo_id=target["repo_id"],
        local_dir=str(output_dir),
        repo_type="dataset",
        allow_patterns=target["allow_patterns"],
        tqdm_class=tqdm,  # type: ignore
    )

    logger.info("✅ Dataset '%s' downloaded successfully to %s", dataset_name, output_dir)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_TARGETS.keys())
    parser.add_argument("--base_dir", type=str, default="benchmarks/datasets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_dataset(args.dataset, args.base_dir)


if __name__ == "__main__":
    main()
