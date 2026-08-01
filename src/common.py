"""Small shared helpers used by the ingest/preprocess CLIs.

Kept dependency-light and separate from any single pipeline stage so both
``src/ingest`` and ``src/preprocess`` (and, later, ``src/eval``) can reuse
the same config-loading / path-resolution logic without hardcoding paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Resolve the repo root.

    Prefers the ``PROJECT_ROOT`` env var (set in the Dockerfile) so behavior
    is identical in and out of a container; falls back to walking up from
    this file's location, which lives at ``<root>/src/common.py``.
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving relative paths against the CWD."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-declared, repo-root-relative path to an absolute one."""
    return project_root() / relative_path
