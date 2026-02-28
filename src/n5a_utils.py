"""
n5a_utils.py
============
Core utilities for Notebook 05: Fraud Detection & Anomaly Analysis.

Mirrors the utility pattern from n1a_utils / n4a_utils for consistency
across the analytics pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_run_id: str = ""
_logger: logging.Logger | None = None


# ---------------------------------------------------------------------------
# Project structure
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root directory (parent of 'notebooks/' or cwd)."""
    cwd = Path.cwd()
    if cwd.name == "notebooks":
        return cwd.parent
    # Walk up to find config.yaml or src/
    for ancestor in [cwd, *cwd.parents]:
        if (ancestor / "config.yaml").exists() or (ancestor / "src").exists():
            return ancestor
    return cwd


def verify_project_structure(root: Path) -> None:
    """Raise RuntimeError if expected directories are missing."""
    required = [root / "src", root / "data", root / "config.yaml"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Project structure incomplete. Missing: {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Generate a short, unique run identifier."""
    return uuid.uuid4().hex[:8].upper()


def set_run_id(run_id: str | None = None) -> str:
    """Set and return the global run ID."""
    global _run_id
    _run_id = run_id or generate_run_id()
    return _run_id


def get_run_id() -> str:
    """Return the current global run ID."""
    return _run_id


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(
    name: str,
    level: int = logging.INFO,
    stream: Any = None,
) -> logging.Logger:
    """
    Configure and return a named logger.

    Parameters
    ----------
    name:
        Logger name (typically __name__).
    level:
        Logging level (default INFO).
    stream:
        Output stream; defaults to sys.stdout for notebook compatibility.

    Returns
    -------
    logging.Logger
    """
    global _logger

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False

    _logger = logger
    return logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load project configuration from config.yaml.

    Parameters
    ----------
    config_path:
        Explicit path to config.yaml. If None, searches from project root.

    Returns
    -------
    dict containing full configuration.
    """
    if config_path is None:
        root = get_project_root()
        config_path = root / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    return config


def get_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    """
    Resolve and create output directories defined in config.

    Parameters
    ----------
    config:
        Full project configuration dict.

    Returns
    -------
    dict mapping path keys to resolved Path objects.
    """
    root = get_project_root()
    paths_cfg = config.get("paths", {})

    paths: dict[str, Path] = {}
    for key, rel_path in paths_cfg.items():
        if rel_path:
            resolved = root / rel_path
            paths[key] = resolved

    # Ensure output directories exist
    for key in ["notebook5_figures", "processed_data", "models_dir"]:
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)

    # Convenience aliases
    paths["figures"] = paths.get(
        "notebook5_figures",
        root / "outputs" / "figures" / "notebook5_fig",
    )
    paths["figures"].mkdir(parents=True, exist_ok=True)

    return paths


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_section_header(title: str, width: int = 80) -> None:
    """Print a standardised section header matching NB01-NB04 style."""
    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)
    print()


def print_subsection(title: str, width: int = 80) -> None:
    """Print a subsection separator."""
    print()
    print("-" * width)
    print(f"  {title}")
    print("-" * width)


def compute_dataframe_hash(df: "pd.DataFrame", n_bytes: int = 8) -> str:  # noqa: F821
    """
    Compute a short MD5 hash of a DataFrame for integrity verification.

    Parameters
    ----------
    df:
        DataFrame to hash.
    n_bytes:
        Length of returned hex string.

    Returns
    -------
    str
    """
    import pandas as pd  # local import to avoid hard dependency at module level

    buf = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(buf).hexdigest()[:n_bytes]


def validate_config(config: dict[str, Any]) -> bool:
    """
    Check that notebook5-specific keys exist in config.

    Parameters
    ----------
    config:
        Full project configuration dict.

    Returns
    -------
    bool — True if valid, raises ValueError otherwise.
    """
    required_keys = [
        ("notebook5",),
        ("notebook5", "fraud"),
        ("notebook5", "fraud", "rule_thresholds"),
        ("notebook5", "models", "isolation_forest"),
        ("notebook5", "models", "lof"),
        ("notebook5", "risk_tiers"),
    ]

    for key_path in required_keys:
        node = config
        for k in key_path:
            if not isinstance(node, dict) or k not in node:
                raise ValueError(
                    f"Missing config key: {' -> '.join(key_path)}"
                )
            node = node[k]

    return True
