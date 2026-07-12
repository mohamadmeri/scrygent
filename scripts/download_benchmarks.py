"""Downloads a subset of the DataBench Lite dataset for local benchmarking.

DataBench Lite (SemEval 2025 Task 8) is used to evaluate Scrygent's
end-to-end accuracy against the GPT-4 baseline.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_databench_lite(target_dir: Path, target_count: int) -> None:
    """Donwload function for Databench (not the full dataset) dataset.

    Parameters
    ----------
    target_dir : Path
    target_count : int
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DataBench SemEval split to gather dataset IDs...")
    hf_dataset = load_dataset("cardiffnlp/databench", "semeval", split="train")

    # Gather unique dataset IDs, preserving order of first appearance
    dataset_ids = list(
        dict.fromkeys(row["dataset"] for row in hf_dataset if isinstance(row, dict) and row.get("dataset"))
    )

    saved = 0
    for dataset_id in dataset_ids:
        if saved >= target_count:
            logger.info("Reached target of %d datasets. Stopping.", target_count)
            break

        try:
            parquet_url = (
                f"https://huggingface.co/datasets/cardiffnlp/databench/resolve/main/data/{dataset_id}/sample.parquet"
            )
            df = pd.read_parquet(parquet_url)

            csv_path = target_dir / f"{dataset_id}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Saved dataset: %s.csv (Rows: %d)", dataset_id, len(df))
            saved += 1
        except Exception as e:
            logger.warning("Skipping %s: %s", dataset_id, e)


def main() -> None:
    """Entry point of the script."""
    parser = argparse.ArgumentParser(description="Download DataBench Lite datasets.")
    parser.add_argument("--dir", type=Path, default=Path("data/databench_lite"), help="Target directory for CSVs.")
    parser.add_argument("--count", type=int, default=5, help="Number of datasets to download.")
    args = parser.parse_args()

    download_databench_lite(args.dir, args.count)


if __name__ == "__main__":
    main()
