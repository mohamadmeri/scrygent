"""Architectural layer isolation test: tools must not import models."""
import ast
import os
from pathlib import Path

import pytest

from scrygent import tools


def _find_tool_source_files() -> list[Path]:
    """Collect all .py files in the tools/ directory, excluding __init__.py."""
    tools_dir = Path(tools.__file__).parent
    source_files = []
    for root, _, files in os.walk(tools_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                source_files.append(Path(root) / file)
    return source_files


def _file_imports_models(file_path: Path) -> list[str]:
    """
    Parse the file and return a list of import statements that reference
    `scrygent.models` (either directly or via relative imports that
    resolve into models).
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "scrygent.models" in alias.name:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # Check module name for models references
            if node.module and "scrygent.models" in node.module:
                for alias in node.names:
                    violations.append(f"from {node.module} import {alias.name}")
            # Also catch relative imports that might reach models, but those
            # would normally be like `..models` which should be caught by
            # checking if the parent package is models. We'll simply scan for
            # any `..models` relative module.
            if node.module and node.module.endswith("models"):
                for alias in node.names:
                    violations.append(f"from {node.module} import {alias.name}")
    return violations


class TestLayerIsolation:
    """Tools must never import from models to preserve dependency direction."""

    @pytest.fixture(scope="module")
    def tool_files(self):
        return _find_tool_source_files()

    def test_no_model_imports_in_tools(self, tool_files):
        violations = {}
        for file_path in tool_files:
            bad_imports = _file_imports_models(file_path)
            if bad_imports:
                violations[file_path.name] = bad_imports
        assert not violations, (
            f"Tools imported models in the following files:\n{violations}\n"
            "Tools may only import from contracts/ and tools/_shared/."
        )
