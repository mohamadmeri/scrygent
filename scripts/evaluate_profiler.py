"""Manual profiler output inspector.

Generates a comprehensive text report of the Profiler Node's output
for a given CSV and query, useful for debugging context window sizing.
"""

import argparse
import json
import logging
from pathlib import Path

from scrygent.tools.io import load_csv
from scrygent.tools.profiler import profile_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def inspect_profile(csv_path: Path, user_query: str, out_file: Path) -> None:
    """Profile inspection utility: it simulates the Profiler's output.

    Parameters
    ----------
    csv_path : Path
    user_query : str
    out_file : Path

    Raises:
    ------
    FileNotFoundError
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = load_csv(csv_path)
    profile = profile_dataframe(df, user_query)

    lines = []

    def write(text: str = "") -> None:
        lines.append(text + "\n")

    def section(title: str) -> None:
        write()
        write("=" * 80)
        write(f"  {title}")
        write("=" * 80)

    section("FILE & QUERY")
    write(f"File: {csv_path.name}  |  Rows: {profile['row_count']}")
    write(f"Query: {user_query}")

    section("GLOBAL SCHEMA (all columns)")
    for col, dtype in profile["global_schema"].items():
        write(f"  {col:<25} {dtype}")

    section("DETAILED STATS (query + high-density columns)")
    if profile["detailed_stats"]:
        for col, stats in profile["detailed_stats"].items():
            write(f"\n  [{col}]")
            for k, v in stats.items():
                if isinstance(v, float):
                    write(f"    {k:<20} {v:.4f}")
                else:
                    write(f"    {k:<20} {v}")
    else:
        write("  (empty)")

    section("ROW SAMPLE (first 3 rows, NaNs -> null)")
    for i, row in enumerate(profile["row_sample"], 1):
        write(f"  Row {i}: {row}")

    section("TRUNCATION INFO")
    write(f"  Truncated: {profile['truncated']}")
    if profile["truncated"]:
        write(f"  Columns in global_schema: {len(profile['global_schema'])}")
        write(f"  Columns with detailed stats: {len(profile['detailed_stats'])}")
        missing = profile.get("missing_detailed_stats", [])
        if missing:
            write(f"  Columns missing from detailed_stats ({len(missing)}):")
            for col in missing:
                write(f"    - {col}")

    section("RAW JSON (as Planner receives)")
    write(json.dumps(profile, indent=2, default=str))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(lines), encoding="utf-8")
    print(f"Profiler output written to {out_file}")


def main() -> None:
    """Entry point of the script."""
    parser = argparse.ArgumentParser(description="Inspect Scrygent Profiler output.")
    parser.add_argument("csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument("query", type=str, help="The natural language query to simulate.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("scripts/profiler_output.txt"), help="Output text file."
    )
    args = parser.parse_args()

    inspect_profile(args.csv, args.query, args.output)


if __name__ == "__main__":
    main()
