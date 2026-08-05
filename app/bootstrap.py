"""Deployment bootstrap helpers that do not import the ROOT-FPT package."""

from __future__ import annotations

import re
import sys
from collections.abc import MutableMapping
from pathlib import Path
from types import ModuleType

_VERSION_PATTERN = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def source_version(repository_root: Path) -> str | None:
    """Read the package version without importing potentially stale modules."""
    package_file = repository_root / "src" / "rootfpt" / "__init__.py"
    if not package_file.is_file():
        return None
    match = _VERSION_PATTERN.search(package_file.read_text(encoding="utf-8"))
    return None if match is None else match.group(1)


def discard_stale_rootfpt_modules(
    repository_root: Path,
    modules: MutableMapping[str, ModuleType] | None = None,
) -> bool:
    """Discard cached package modules when deployed source has a new version."""
    loaded_modules = sys.modules if modules is None else modules
    package = loaded_modules.get("rootfpt")
    if package is None:
        return False
    expected = source_version(repository_root)
    loaded = getattr(package, "__version__", None)
    if expected is None or loaded == expected:
        return False
    for name in tuple(loaded_modules):
        if name == "rootfpt" or name.startswith("rootfpt."):
            loaded_modules.pop(name, None)
    return True
