"""Marker module for Jupyter: lets notebooks discover the repo root via ``__file__``."""

from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    """Parent of ``src/`` (repository root)."""
    return Path(__file__).resolve().parent.parent
