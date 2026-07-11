"""Manual profiler output inspector – writes full profile to a file.

Run from the project root:
    python scripts/evaluate_profiler.py [path-to-csv] ["user query"] [output-file]

Defaults: titanic.csv, a sample query, and 'scripts/profiler_output.txt'.
"""

import json
import logging
import sys
from pathlib import Path

from scrygent.tools.io import load_csv
from scrygent.tools.profiler import profile_dataframe

logging.basicConfig(level=logging.INFO)


def inspect_profile(csv_path: Path, user_query: str, out_file: Path) -> None:
    if not csv_path.exists():
        print(f"Error: file not found – {csv_path}")
        return

    df = load_csv(csv_path)
    profile = profile_dataframe(df, user_query)

    # Build output as a list of strings
    lines = []

    def write(text: str = ""):
        lines.append(text + "\n")

    def section(title: str):
        write()
        write("=" * 80)
        write(f"  {title}")
        write("=" * 80)

    # ── Basic stats ──
    section("FILE & QUERY")
    write(f"File: {csv_path.name}  |  Rows: {profile['row_count']}")
    write(f"Query: {user_query}")

    # ── Global schema ──
    section("GLOBAL SCHEMA (all columns)")
    for col, dtype in profile["global_schema"].items():
        write(f"  {col:<25} {dtype}")

    # ── Detailed stats ──
    section("DETAILED STATS (query + high‑density columns)")
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

    # ── Row sample ──
    section("ROW SAMPLE (first 3 rows, NaNs → null)")
    for i, row in enumerate(profile["row_sample"], 1):
        write(f"  Row {i}: {row}")

    # ── Truncation ──
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
    else:
        write("  (All columns have detailed stats)")

    # ── Raw JSON ──
    section("RAW JSON (as Planner receives)")
    write(json.dumps(profile, indent=2, default=str))

    # Write everything to the output file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(lines), encoding="utf-8")
    print(f"Profiler output written to {out_file}")
    # Also print a short summary to console
    print(
        f"Rows: {profile['row_count']}, Columns: {len(profile['global_schema'])}"
        f", Detailed: {len(profile['detailed_stats'])}, Truncated: {profile['truncated']}"
    )


if __name__ == "__main__":
    default_csv = "data/InfiAgent/data/da-dev-tables/titanic.csv"
    default_query = "What is the survival rate of female passengers who paid a fare greater than $50?"
    default_out = "scripts/profiler_output.txt"

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_csv)
    user_query = sys.argv[2] if len(sys.argv) > 2 else default_query
    out_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(default_out)

    inspect_profile(csv_path, user_query, out_file)
