"""Idempotent dataset downloader for Scrygent benchmarks."""

import argparse
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATASET_TARGETS: dict[str, dict[str, Any]] = {
    "infiagent": {
        "repo_id": "InfiAgent/InfiAgent-DABench",
        "subset": "da-dev",
        "default_split": "dev",
    },
    "databench_lite": {
        "repo_id": "SemanticEval/databench_lite",
        "subset": "default",
        "default_split": "test",
    },
}


def download_dataset(dataset_name: str, base_dir: Path, split: str) -> None:
    """Downloads the specified dataset and extracts CSVs idempotently."""
    if dataset_name not in DATASET_TARGETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_TARGETS.keys())}")

    target = DATASET_TARGETS[dataset_name]
    output_dir = base_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching dataset '%s' (split='%s') from HuggingFace...", dataset_name, split)
    ds = load_dataset(target["repo_id"], target["subset"], split=split)

    # Save metadata
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        ds.to_json(metadata_path)
        logger.info("Saved metadata to %s", metadata_path)
    else:
        logger.info("Metadata already exists. Skipping.")

    # Save CSVs
    csv_dir = output_dir / "csvs"
    csv_dir.mkdir(exist_ok=True)

    logger.info("Extracting CSV files...")
    for row in tqdm(ds, desc="Extracting CSVs"):
        csv_name = row.get("filename") or row.get("csv") or f"{row.get('id', 'unknown')}.csv"
        if isinstance(csv_name, str) and not csv_name.endswith(".csv"):
            csv_name = f"{csv_name}.csv"

        csv_path = csv_dir / csv_name
        if csv_path.exists():
            continue

        csv_content = row.get("data") or row.get("csv_content")
        if csv_content:
            csv_path.write_text(str(csv_content), encoding="utf-8")

    logger.info("✅ Dataset '%s' downloaded successfully to %s", dataset_name, output_dir)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_TARGETS.keys())
    parser.add_argument("--base_dir", type=str, default="benchmarks/datasets")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (e.g., 'dev', 'test')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Fallback to default split if not provided
    split = args.split or DATASET_TARGETS[args.dataset]["default_split"]
    download_dataset(args.dataset, Path(args.base_dir), split)


if __name__ == "__main__":
    main()
