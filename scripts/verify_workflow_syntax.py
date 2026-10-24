"""Validate GitHub Actions workflow syntax and action pinning.

    python scripts/verify_workflow_syntax.py

Structural checks only - this is NOT actionlint and does not replace it. It
catches the failure modes that silently produce a green job:

* an action referenced by tag or branch instead of a full commit SHA
* a step that swallows a non-zero exit (``continue-on-error``, ``|| true``)
* ``hashFiles()`` used in a job-level ``if:``, which GitHub does not support
  (a real past bug in this repository)
* a workflow running ``python`` without pinning the interpreter
* a missing ``if-no-files-found`` on an upload step
* a self-hosted runner label that ``.github/actionlint.yaml`` does not declare

actionlint 1.7.12 was run manually and passed, but that is a point-in-time
observation. This checker runs on every workflow change, so the file cannot
silently go stale.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOWS = ROOT / ".github" / "workflows"
ACTIONLINT_CONFIG = ROOT / ".github" / "actionlint.yaml"

SHA = re.compile(r"^[0-9a-f]{40}$")
USES = re.compile(r"^\s*(?:-\s+)?uses:\s*['\"]?([^'\"@\s]+)(?:@([^'\"\s]+))?['\"]?", re.MULTILINE)
CONTINUE_ON_ERROR = re.compile(r"continue-on-error:\s*true", re.IGNORECASE)
ALWAYS = re.compile(r"if:\s*.*\balways\(\)", re.IGNORECASE)
HASH_FILES = re.compile(r"hashFiles\(")
UPLOAD_USES = re.compile(r"uses:\s*\S*upload-art\S*", re.IGNORECASE)

BASE_LABELS = {"self-hosted", "linux", "windows", "macos", "x64", "x86", "arm", "arm64"}


def declared_labels(config: Path) -> set[str]:
    labels = set(BASE_LABELS)
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                labels.add(stripped[2:].strip().strip("\"'"))
    return labels


def validate(workflows: Path, actionlint_config: Path) -> tuple[list[str], list[str]]:
    """Return (errors, notes) for every workflow under `workflows`."""
    errors: list[str] = []
    notes: list[str] = []

    if not workflows.is_dir():
        return ["No workflow directory"], notes

    files = sorted(workflows.glob("*.yml")) + sorted(workflows.glob("*.yaml"))
    if not files:
        return ["No workflow files found"], notes

    labels = declared_labels(actionlint_config)
    notes.append(f"declared runner labels: {len(labels)}")
    if not actionlint_config.is_file():
        errors.append(
            f"{actionlint_config.name} is missing; custom self-hosted runner labels would be unverifiable"
        )

    for path in files:
        text = path.read_text(encoding="utf-8")

        for action, ref in USES.findall(text):
            if not ref:
                errors.append(f"{path.name}: 'uses: {action}' has no ref")
            elif not SHA.match(ref):
                errors.append(f"{path.name}: 'uses: {action}@{ref}' is not a full 40-character commit SHA")

        if re.search(r"^steps\s*:", text, re.MULTILINE):
            errors.append(f"{path.name}: top-level 'steps:' key must be indented under a job (P0-5)")
        if CONTINUE_ON_ERROR.search(text):
            errors.append(f"{path.name}: continue-on-error: true can mask a real failure")
        if re.search(r"\|\|\s*true", text):
            errors.append(f"{path.name}: '|| true' can mask a real failure")
        if ALWAYS.search(text):
            notes.append(f"{path.name}: uses if: always() - confirm it is not hiding a failure")

        # Job-level `if:` sits at four spaces under jobs: <name>:. A step-level
        # `if:` is deeper and hashFiles() IS supported there.
        for match in re.finditer(r"^ {4}if:\s*(.+)$", text, re.MULTILINE):
            if HASH_FILES.search(match.group(1)):
                errors.append(
                    f"{path.name}: job-level if: uses hashFiles(), which GitHub only "
                    "supports in step-level conditions"
                )

        for match in re.finditer(
            r"^\s*-\s*(?:name:\s*\S+\s*)?uses:\s*\S*upload-art\S*(.*?)(?=^\s*-\s|\Z)",
            text, re.MULTILINE | re.DOTALL,
        ):
            if "if-no-files-found" not in match.group(1):
                errors.append(f"{path.name}: upload-artifact step is missing if-no-files-found")

        if re.search(r"run:.*\bpython\b", text) and "setup-python" not in text:
            errors.append(
                f"{path.name}: runs python without actions/setup-python; "
                "the interpreter version is whatever the runner image ships"
            )

        for match in re.finditer(r"runs-on:\s*\[([^\]]+)\]", text):
            for label in match.group(1).split(","):
                label = label.strip().strip("\"'")
                if label and label not in labels:
                    errors.append(f"{path.name}: runner label {label!r} is not declared in the actionlint config")

        try:
            import yaml
        except ImportError:
            notes.append(f"{path.name}: PyYAML unavailable; structural checks only")
        else:
            try:
                yaml.safe_load(text)
            except yaml.YAMLError as exc:
                errors.append(f"{path.name}: YAML parse error: {exc}")

        notes.append(f"{path.name}: checked, {len(USES.findall(text))} actions")

    return errors, notes


def main() -> int:
    workflows = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKFLOWS
    errors, notes = validate(workflows, ACTIONLINT_CONFIG)
    print(json.dumps({
        "workflows": [p.name for p in sorted(workflows.glob("*.yml"))] if workflows.is_dir() else [],
        "errors": errors,
        "notes": notes,
    }, indent=2))
    if errors:
        print("\n=== ERRORS ===", file=sys.stderr)
        for message in errors:
            print(f"E: {message}", file=sys.stderr)
        return 1
    print("\nWorkflow syntax and action pinning verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())