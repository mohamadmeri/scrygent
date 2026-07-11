"""Module-wide contract integrity: every StrEnum in contracts/ must be used by at least one consumer module
(tools, models, ...). This prevents dead contract code from accumulating."""
import importlib
import inspect
import os
from pathlib import Path
from enum import StrEnum

import pytest

from src.scrygent import contracts, tools, models


def _discover_contract_enums() -> dict[str, type[StrEnum]]:
    """
    Find all StrEnum subclasses defined in any module within the contracts package.
    Returns { 'QualifiedName': enum_class }.
    """
    enums = {}
    contracts_dir = Path(contracts.__file__).parent
    for path in contracts_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        module_name = f"src.scrygent.contracts.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, StrEnum) and obj is not StrEnum and obj.__module__ == mod.__name__:
                enums[f"{module_name}.{name}"] = obj
    return enums


def _find_consumer_source_files() -> list[Path]:
    """
    Collect all .py files in directories that are allowed to import contracts:
    tools/, models/. Exclude __init__.py.
    """
    allowed_dirs = []
    for pkg in (tools, models):
        allowed_dirs.append(Path(pkg.__file__).parent) # type: ignore

    source_files = []
    for directory in allowed_dirs:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    source_files.append(Path(root) / file)
    return source_files


def _enum_name_used_in_file(enum_class: type[StrEnum], file_path: Path) -> bool:
    """
    Check if the enum class name appears anywhere in the file's source code.
    This is a simple heuristic; it catches direct imports and usage.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return enum_class.__name__ in source


class TestContractsPackageIntegrity:
    """All enums in contracts must be actively used by at least one consumer module."""

    @pytest.fixture(scope="module")
    def contract_enums(self):
        return _discover_contract_enums()

    @pytest.fixture(scope="module")
    def consumer_sources(self):
        return _find_consumer_source_files()

    def test_every_enum_used_in_some_consumer(self, contract_enums, consumer_sources):
        """No orphaned contract enums: each must be referenced in tools/, models/, or agents/."""
        orphaned = []
        for qualname, enum_cls in contract_enums.items():
            if not any(_enum_name_used_in_file(enum_cls, f) for f in consumer_sources):
                orphaned.append(qualname)
        assert not orphaned, (
            f"Orphaned contract enums (not used in any consumer module): {orphaned}. "
            "Either remove the unused enum or integrate it into the appropriate tool/model."
        )
