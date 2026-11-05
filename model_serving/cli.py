"""
Shared pieces of the model_serving command-line tools: the arguments they have
in common, logging setup, service-account credentials and JSON files.

Standard library only at import time, so every module in the package can use it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SA_KEY_HELP = (
    "Path to service account JSON key (GOOGLE_APPLICATION_CREDENTIALS env var). "
    "Omit for application default credentials."
)


# ── Arguments ─────────────────────────────────────────────────────────────────

def add_project_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", "")),
        help="GCP project ID (GCP_PROJECT_ID env var)",
    )


def add_region_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--region",
        default=os.environ.get("GCP_REGION", "us-central1"),
        help="GCP region (default: us-central1)",
    )


def add_sa_key_arg(p: argparse.ArgumentParser, help_text: str = _SA_KEY_HELP) -> None:
    p.add_argument(
        "--sa-key",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help=help_text,
    )


def add_endpoint_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--endpoint",
        default=os.environ.get("VERTEX_AI_ENDPOINT_RESOURCE_NAME", ""),
        help=(
            "Full Vertex AI endpoint resource name "
            "(projects/.../locations/.../endpoints/...). "
            "Use the 'vertex_ai_endpoint_resource_name' Terraform output."
        ),
    )


def add_bigtable_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bigtable-instance",
        default=os.environ.get("BIGTABLE_INSTANCE_ID", "scip-feature-store"),
        help="Bigtable instance ID (default: scip-feature-store)",
    )
    p.add_argument(
        "--bigtable-table",
        default="inventory_features",
        help="Bigtable table ID (default: inventory_features)",
    )


def require(args: argparse.Namespace, flag: str, env_var: str, exit_code: int = 1) -> None:
    """Exit with exit_code unless --flag or its environment variable gave a value."""
    if not getattr(args, flag.replace("-", "_"), None):
        logger.error("Missing required flag --%s or env var %s", flag, env_var)
        sys.exit(exit_code)


# ── Logging, credentials, JSON ────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


def load_credentials(key_path: str | None):
    """Credentials from a service account JSON key; None means application default credentials."""
    if not key_path:
        return None
    from google.oauth2 import service_account
    return service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def read_json(path: str | Path, missing_hint: str = "") -> dict:
    """Parse a JSON file. A missing file raises FileNotFoundError that ends with missing_hint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {missing_hint}".rstrip())
    with open(path) as f:
        return json.load(f)
