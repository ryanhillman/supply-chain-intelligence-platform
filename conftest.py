"""Repo-root pytest conftest.

Ensures the repo root is on sys.path so tests can import the existing
ml_training.* / model_serving.* packages with their normal absolute
import paths, regardless of the directory pytest is invoked from.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
