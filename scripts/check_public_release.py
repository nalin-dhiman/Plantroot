"""Fail CI if private research-writing artifacts enter the public repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_COMPONENTS = {
    "manuscript",
    "manuscripts",
    "submission",
    "submissions",
    "reviewer_response",
}
FORBIDDEN_SUFFIXES = {".aux", ".bbl", ".bcf", ".bib", ".blg", ".latex", ".tex"}
FORBIDDEN_TEXT = ("\\documentclass", "\\begin{document}", "cover_letter.md")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    violations: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if relative == Path("scripts/check_public_release.py"):
            continue
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & FORBIDDEN_COMPONENTS:
            violations.append(f"forbidden path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden source type: {relative}")
        if path.is_file() and path.suffix.lower() in {".md", ".py", ".txt", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in FORBIDDEN_TEXT:
                if marker in text:
                    violations.append(f"forbidden content marker {marker!r}: {relative}")
    if violations:
        print("Public-release boundary check failed:")
        print("\n".join(f"- {violation}" for violation in sorted(set(violations))))
        return 1
    print(f"Public-release boundary passed for {len(tracked_files())} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
