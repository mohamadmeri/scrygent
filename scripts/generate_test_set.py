"""Generates an unbiased, randomized test set of benchmark questions.

Downloads the necessary datasets and outputs a JSONL file ready for the Scrygent Evaluator.
"""

import json
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from datasets import Dataset, load_dataset  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_set(target_count: int = 10, output_file: str = "data/sample_10_queries.jsonl") -> Any:
    """Create a sample set for evaluate_system.py.

    Parameters
    ----------
    target_count : int, optional
        _description_, by default 10
    output_file : str, optional
        _description_, by default "data/sample_10_queries.jsonl"
    """
    output_path = Path(output_file)
    csv_dir = Path("data/databench_lite")
    csv_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DataBench SemEval dataset from HuggingFace...")
    # Load the dataset
    hf_dataset = load_dataset("cardiffnlp/databench", "semeval", split="train")

    # Shuffle with a fixed seed so we get a consistent, reproducible "stress test" set
    shuffled_dataset: Dataset = hf_dataset.shuffle(seed=42)

    saved_queries = []  # type: ignore
    downloaded_datasets = set()

    logger.info(f"Extracting {target_count} random queries...")

    for row in shuffled_dataset:
        if len(saved_queries) >= target_count:
            break

        row_data = cast(dict[str, Any], row)

        dataset_id = row_data.get("dataset")
        question = row_data.get("question")
        answer = row_data.get("answer")

        if not dataset_id or not question or not answer:
            continue

        csv_path = csv_dir / f"{dataset_id}.csv"

        # Download the CSV if we haven't already grabbed it for a previous question
        if dataset_id not in downloaded_datasets and not csv_path.exists():
            try:
                parquet_url = f"https://huggingface.co/datasets/cardiffnlp/databench/resolve/main/data/{dataset_id}/sample.parquet"
                df = pd.read_parquet(parquet_url)
                df.to_csv(csv_path, index=False)
                logger.info(f"Downloaded supporting dataset: {dataset_id}.csv")
                downloaded_datasets.add(dataset_id)
            except Exception as e:
                logger.warning(f"Failed to download {dataset_id}. Skipping question. Error: {e}")
                continue
        elif csv_path.exists():
            downloaded_datasets.add(dataset_id)

        # Append to our test batch
        saved_queries.append({
            "id": f"DB-{dataset_id}-{len(saved_queries) + 1}",
            "query": question,
            "csv_path": str(csv_path),
            "gold_answer": str(answer),
        })

    # Write the JSONL file for the Evaluator
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for q in saved_queries:
            f.write(json.dumps(q) + "\n")

    logger.info(f"Successfully generated {len(saved_queries)} test queries at {output_path}")


if __name__ == "__main__":
    generate_sample_set(target_count=10)
