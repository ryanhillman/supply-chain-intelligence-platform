"""
The project's packages import through `pip install -e .` (pyproject.toml), with
no sys.path edits: checked in a fresh interpreter started outside the repo.
"""
import subprocess
import sys

PACKAGES = "ml_training, model_serving, simulation"


def test_packages_import_from_the_install(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", f"import {PACKAGES}"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"run `pip install -e .` first\n{result.stderr}"
