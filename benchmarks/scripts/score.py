"""Computes and prints a rich summary of benchmark results."""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

try:
    from scrygent import __version__ as scrygent_version
except ImportError:
    scrygent_version = "0.1.0"  # Fallback

logger = logging.getLogger(__name__)


def compute_scores(results_dir: Path) -> None:
    """Reads predictions.csv and generates summary.json."""
    predictions_path = results_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {results_dir}")

    df = pd.read_csv(predictions_path)

    if df.empty:
        logger.warning("Predictions file is empty. No metrics to compute.")
        return

    total_items = len(df)
    exact_matches = df["score"].sum()
    accuracy = (exact_matches / total_items) * 100

    failure_mask = df["score"] < 1.0
    failure_rate = (failure_mask.sum() / total_items) * 100

    latency = df["latency_s"]
    avg_latency = latency.mean()
    median_latency = latency.median()
    p95_latency = latency.quantile(0.95)

    avg_retries = df["retries"].mean()

    # Breakdown of failure categories
    failures_df = df[failure_mask]
    failure_categories = failures_df["failure_category"].value_counts().to_dict()

    # Breakdown of match types (New Telemetry)
    match_types = df["match_type"].value_counts().to_dict()

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "scrygent_version": scrygent_version,
        "total_items": int(total_items),
        "accuracy_percent": round(accuracy, 2),
        "failure_rate_percent": round(failure_rate, 2),
        "latency": {"average_seconds": round(avg_latency, 4), "median_seconds": round(median_latency, 4), "p95_seconds": round(p95_latency, 4)},
        "average_retries": round(avg_retries, 2),
        "match_type_breakdown": {str(k): int(v) for k, v in match_types.items()},
        "failure_categories": {str(k): int(v) for k, v in failure_categories.items()},
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"📊 Scrygent Benchmark Summary (v{summary['scrygent_version']})")
    print("=" * 60)
    print(f"Total Items      : {summary['total_items']}")
    print(f"Accuracy         : {summary['accuracy_percent']}%")
    print(f"Failure Rate     : {summary['failure_rate_percent']}%")
    print("-" * 60)
    print("Latency (s)      :")
    print(f"  Average        : {summary['latency']['average_seconds']}")
    print(f"  Median         : {summary['latency']['median_seconds']}")
    print(f"  P95            : {summary['latency']['p95_seconds']}")
    print("-" * 60)
    print(f"Avg Retries      : {summary['average_retries']}")
    print("-" * 60)
    print("Match Types      :")
    for m_type, count in summary["match_type_breakdown"].items():
        print(f"  {m_type:<15}: {count}")
    print("-" * 60)
    print("Failure Categories:")
    for cat, count in summary["failure_categories"].items():
        print(f"  {cat or 'Unknown':<15}: {count}")
    print("=" * 60 + "\n")

    logger.info("Summary saved to %s", summary_path)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Score Scrygent benchmark results.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing predictions.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    compute_scores(Path(args.results_dir))


if __name__ == "__main__":
    main()
