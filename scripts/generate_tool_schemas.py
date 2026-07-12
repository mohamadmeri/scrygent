"""Generates and exports the strict JSON schemas for all registered tools.

This utility is used for debugging the Intermediate Representation (IR)
layer and can be piped into documentation generators.
"""

import argparse
import json
from pathlib import Path

from scrygent.models.registry import TOOL_PARAM_MODELS


def main() -> None:
    """Entry point of the script."""
    parser = argparse.ArgumentParser(description="Generate JSON schemas for Scrygent tools.")
    parser.add_argument("-o", "--output", type=Path, help="Optional file path to save the JSON output.")
    args = parser.parse_args()

    schemas = {}
    for tool_name, model in TOOL_PARAM_MODELS.items():
        schemas[tool_name.value] = model.model_json_schema()

    output_str = json.dumps(schemas, indent=2)

    if args.output:
        args.output.write_text(output_str, encoding="utf-8")
        print(f"Successfully wrote schemas for {len(schemas)} tools to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
