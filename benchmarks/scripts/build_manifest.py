"""Builds a standardized manifest from raw benchmark metadata."""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_infiagent_manifest(raw_dir: Path, output_path: Path) -> None:
    """Parses InfiAgent metadata into the standard manifest format."""
    questions_path = raw_dir / "da-dev-questions.jsonl"
    labels_path = raw_dir / "da-dev-labels.jsonl"
    csv_dir = raw_dir / "da-dev-tables"

    if not questions_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Metadata files not found in {raw_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load labels into a dictionary for O(1) lookup, using string keys for safety
    labels: dict[str, str] = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Handle potential variations in the answer key name and list serialization
            ans = item.get("common_answers", item.get("answer", ""))
            labels[str(item["id"])] = json.dumps(ans) if isinstance(ans, list) else str(ans)

    with open(questions_path, encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Building manifest"):
            q_item = json.loads(line)
            q_id = str(q_item.get("id", ""))
            query = q_item.get("question", "")
            file_name = q_item.get("file_name", "")

            # Map to the local CSV path
            csv_path = (csv_dir / file_name).resolve()

            # Get the gold answer from the labels dictionary
            gold_answer = labels.get(q_id, "")

            manifest_entry = {"id": f"IA-{q_id}", "query": query, "gold_answer": gold_answer, "csv_path": str(csv_path)}
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
