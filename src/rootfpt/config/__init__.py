"""Immutable configuration loading and hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def load_yaml(path: Path | str) -> Mapping[str, Any]:
    """Load a YAML mapping and recursively make it immutable."""
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    return _freeze(data)


def canonical_config(config: Mapping[str, Any]) -> str:
    """Return a platform-stable canonical JSON representation."""
    return json.dumps(
        _thaw(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(config: Mapping[str, Any], length: int = 12) -> str:
    """Return the leading hexadecimal characters of a SHA-256 config digest."""
    if length < 8 or length > 64:
        raise ValueError("hash length must be between 8 and 64")
    return hashlib.sha256(canonical_config(config).encode("utf-8")).hexdigest()[:length]


def mutable_copy(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create a JSON-compatible mutable copy."""
    return _thaw(config)


__all__ = ["canonical_config", "config_hash", "load_yaml", "mutable_copy"]

