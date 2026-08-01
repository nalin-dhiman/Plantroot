from __future__ import annotations

from rootfpt.config import config_hash
from rootfpt.random import RandomStreamManager


def test_config_hash_is_order_invariant() -> None:
    left = {"b": [2, 3], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 3]}
    assert config_hash(left) == config_hash(right)


def test_named_random_streams_reproduce_and_differ() -> None:
    first = RandomStreamManager(42)
    second = RandomStreamManager(42)
    env_a = first.generator("environment").normal(size=8)
    env_b = second.generator("environment").normal(size=8)
    dev = first.generator("development").normal(size=8)
    assert (env_a == env_b).all()
    assert not (env_a == dev).all()
    assert first.manifest()["streams"]["environment"]["spawn_key"] == [0]

