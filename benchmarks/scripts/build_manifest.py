"""Builds a standardized manifest from raw benchmark metadata."""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_infiagent_manifest(raw_dir: Path, output_path: Path) -> None:
    """Parses InfiAgent metadata into the standard manifest format."""
    metadata_path = raw_dir / "metadata.jsonl"
    csv_dir = raw_dir / "csvs"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Building manifest"):
            raw_item = json.loads(line)

            query = raw_item.get("query", "")
            gold_answer = raw_item.get("answer", "")
            item_id = raw_item.get("id", "")

            csv_filename = raw_item.get("filename", f"{item_id}.csv")
            if not csv_filename.endswith(".csv"):
                csv_filename += ".csv"

            csv_path = (csv_dir / csv_filename).resolve()

            manifest_entry = {"id": item_id, "query": query, "gold_answer": str(gold_answer), "csv_path": str(csv_path)}
            outfile.write(json.dumps(manifest_entry) + "\n")

    logger.info("✅ Manifest successfully written to %s", output_path)


def main() -> None:
    """Entry point.

    Raises:
    ------
    NotImplementedError
    """
    parser = argparse.ArgumentParser(description="Build standardized benchmark manifest.")
    parser.add_argument("--dataset", type=str, required=True, choices=["infiagent", "databench_lite"])
    parser.add_argument("--raw_dir", type=str, default="benchmarks/datasets")
    parser.add_argument("--output_dir", type=str, default="benchmarks/manifests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw_dir = Path(args.raw_dir) / args.dataset
    output_path = Path(args.output_dir) / f"{args.dataset}.jsonl"

    if args.dataset == "infiagent":
        build_infiagent_manifest(raw_dir, output_path)
    else:
        raise NotImplementedError(f"Manifest builder for {args.dataset} not implemented yet.")


if __name__ == "__main__":
    main()
