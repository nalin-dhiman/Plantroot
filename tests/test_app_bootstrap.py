from pathlib import Path
from types import ModuleType

from app.bootstrap import discard_stale_rootfpt_modules, source_version


def test_stale_rootfpt_modules_are_discarded_without_touching_other_modules() -> None:
    repository = Path(__file__).resolve().parents[1]
    package = ModuleType("rootfpt")
    package.__version__ = "0.0.0"
    modules = {
        "rootfpt": package,
        "rootfpt.explorer": ModuleType("rootfpt.explorer"),
        "unrelated": ModuleType("unrelated"),
    }

    assert source_version(repository) == "1.1.1"
    assert discard_stale_rootfpt_modules(repository, modules)
    assert set(modules) == {"unrelated"}


def test_current_rootfpt_modules_are_retained() -> None:
    repository = Path(__file__).resolve().parents[1]
    package = ModuleType("rootfpt")
    package.__version__ = source_version(repository)
    modules = {"rootfpt": package}

    assert not discard_stale_rootfpt_modules(repository, modules)
    assert modules["rootfpt"] is package
