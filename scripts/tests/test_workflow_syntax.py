"""Regression tests for scripts/verify_workflow_syntax.py.

The workflows are the only place where a broken step can look green. These
tests feed the checker deliberately malformed workflows and assert it refuses
them, so the checker itself cannot silently degrade into a no-op.
"""
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_workflow_syntax as vws


def run_checker(directory: Path):
    """Validate an isolated workflow directory; returns the error list."""
    errors, _ = vws.validate(directory, vws.ACTIONLINT_CONFIG)
    return errors


BASE_JOB = """\
name: {name}
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - name: Run checks
        run: python scripts/check_android_consistency.py
"""


def write_workflow(directory: Path, text: str, name: str = "wf.yml") -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class WorkflowSyntaxCheckerTest(unittest.TestCase):

    def test_current_workflows_pass(self):
        errors = run_checker(ROOT / ".github" / "workflows")
        self.assertEqual(errors, [], f"repository workflows must pass: {errors}")

    def test_unpinned_action_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            write_workflow(directory, BASE_JOB.format(name="unpinned").replace(
                "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
                "actions/checkout@v4"))
            errors = run_checker(directory)
            self.assertTrue(any("commit SHA" in e for e in errors), errors)

    def test_root_level_steps_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            text = BASE_JOB.format(name="badindent").replace("    steps:\n", "steps:\n")
            write_workflow(directory, text)
            errors = run_checker(directory)
            self.assertTrue(any("top-level 'steps:'" in e for e in errors), errors)

    def test_continue_on_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            text = BASE_JOB.format(name="coe").replace(
                "      - name: Run checks",
                "      - name: Run checks\n        continue-on-error: true")
            write_workflow(directory, text)
            errors = run_checker(directory)
            self.assertTrue(any("continue-on-error" in e for e in errors), errors)

    def test_shell_true_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            write_workflow(directory, BASE_JOB.format(name="shelltrue").replace(
                "run: python scripts/check_android_consistency.py",
                "run: python scripts/check_android_consistency.py || true"))
            errors = run_checker(directory)
            self.assertTrue(any("|| true" in e for e in errors), errors)

    def test_job_level_hashfiles_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            text = BASE_JOB.format(name="hash").replace(
                "  build:\n",
                "  build:\n    if: hashFiles('a') != ''\n")
            write_workflow(directory, text)
            errors = run_checker(directory)
            self.assertTrue(any("hashFiles" in e for e in errors), errors)

    def test_python_without_setup_python_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            write_workflow(directory, BASE_JOB.format(name="nopysetup"))
            errors = run_checker(directory)
            self.assertTrue(any("setup-python" in e for e in errors), errors)

    def test_undeclared_runner_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            text = BASE_JOB.format(name="runner").replace(
                "runs-on: ubuntu-latest", "runs-on: [self-hosted, linux, x64, mystery-box]")
            write_workflow(directory, text)
            errors = run_checker(directory)
            self.assertTrue(any("mystery-box" in e for e in errors), errors)

    def test_upload_without_if_no_files_found_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            text = BASE_JOB.format(name="upload") + (
                "      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a\n"
                "        with:\n"
                "          name: thing\n"
                "          path: out/\n")
            write_workflow(directory, text)
            errors = run_checker(directory)
            self.assertTrue(any("if-no-files-found" in e for e in errors), errors)

    def test_valid_workflow_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            write_workflow(directory, (
                BASE_JOB.format(name="valid").replace(
                    "      - name: Run checks",
                    "      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97\n"
                    "        with:\n"
                    "          python-version: '3.11'\n"
                    "      - name: Run checks")))
            errors = run_checker(directory)
            self.assertEqual(errors, [], f"a well-formed workflow must pass: {errors}")

    def test_actionlint_config_is_tracked_in_the_repository(self):
        """The custom runner label declaration must exist on disk and be valid."""
        self.assertTrue(
            vws.ACTIONLINT_CONFIG.exists(),
            ".github/actionlint.yaml is required for the self-hosted runner label",
        )

    def test_yaml_11_boolean_yes_no_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            text = BASE_JOB.format(name="yesboolean") + (
                "      - uses: android-actions/setup-android@be39fa834029ff78f1a44aa3bb0819b8fc2bd8fd\n"
                "        with:\n"
                "          log-accepted-android-sdk-licenses: yes\n"
            )
            write_workflow(directory, text)
            errors = run_checker(directory)
            self.assertTrue(any("YAML 1.1 boolean 'yes'" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()