"""
n6a_utils.py
------------
Shared utilities for Notebook 06: Cohort Retention & Lifecycle Analysis.

Mirrors the utility contract established in n1a_utils / n5a_utils:
  - get_project_root()
  - load_config()
  - setup_logger()
  - get_output_paths()
  - print_section_header()
  - set_run_id() / generate_run_id()
  - verify_project_structure()
  - validate_config()
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Module-level run ID store
# ---------------------------------------------------------------------------
_CURRENT_RUN_ID: str | None = None


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root directory.

    Walks up from the src directory until it finds config.yaml or the
    repository root marker (.git). Falls back to two levels above this file.
    """
    here = Path(__file__).resolve().parent
    for candidate in [here.parent, here.parent.parent]:
        if (candidate / "config.yaml").exists():
            return candidate
    return here.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and return the project YAML configuration.

    Parameters
    ----------
    config_path:
        Explicit path to config.yaml. When None, the file is located
        automatically via :func:`get_project_root`.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config.yaml cannot be found at the resolved path.
    """
    if config_path is None:
        config_path = get_project_root() / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_config(config: dict[str, Any]) -> None:
    """Assert that all keys required by NB06 are present in config.

    Parameters
    ----------
    config:
        Loaded configuration dictionary.

    Raises
    ------
    KeyError
        If a required top-level key is missing.
    """
    required_top = ["paths", "visualization", "time_series", "validation"]
    missing = [k for k in required_top if k not in config]
    if missing:
        raise KeyError(
            f"config.yaml is missing required top-level keys: {missing}"
        )

    # notebook6 block is optional — defaults are handled in the notebook setup cell
    nb6 = config.get("notebook6", {})
    _ = nb6.get("business_rules", {})


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------

def get_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve and return output directory paths for NB06.

    Directories are created if they do not already exist.

    Parameters
    ----------
    config:
        Loaded configuration dictionary.

    Returns
    -------
    dict
        Keys: ``figures``, ``processed``, ``interim``.
    """
    root    = get_project_root()
    paths_c = config.get("paths", {})

    figures_rel  = paths_c.get("notebook6_figures", "outputs/figures/notebook6_fig")
    processed_rel = paths_c.get("processed_data", "data/processed/")
    interim_rel   = paths_c.get("interim_data", "data/interim/")

    paths: dict[str, Path] = {
        "figures":   root / figures_rel,
        "processed": root / processed_rel,
        "interim":   root / interim_rel,
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a stdout StreamHandler.

    Idempotent — does not add duplicate handlers if called multiple times
    with the same name.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    level:
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Run ID management
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Generate a unique run ID in the form ``NB06-YYYYMMDD-HHMMSS-<4hex>``.

    Returns
    -------
    str
    """
    ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid  = uuid.uuid4().hex[:4].upper()
    return f"NB06-{ts}-{uid}"


def set_run_id(run_id: str | None = None) -> str:
    """Set (and return) the module-level run ID.

    Parameters
    ----------
    run_id:
        Explicit run ID to store. When None, a new ID is generated.

    Returns
    -------
    str
        The active run ID.
    """
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = run_id if run_id is not None else generate_run_id()
    return _CURRENT_RUN_ID


def get_run_id() -> str:
    """Return the current run ID, generating one if none has been set.

    Returns
    -------
    str
    """
    global _CURRENT_RUN_ID
    if _CURRENT_RUN_ID is None:
        _CURRENT_RUN_ID = generate_run_id()
    return _CURRENT_RUN_ID


# ---------------------------------------------------------------------------
# Project structure verification
# ---------------------------------------------------------------------------

def verify_project_structure(project_root: Path) -> None:
    """Verify that expected project directories and files exist.

    Logs warnings for missing optional paths; raises RuntimeError for
    missing critical inputs.

    Parameters
    ----------
    project_root:
        Resolved project root directory.

    Raises
    ------
    RuntimeError
        If the src directory or config.yaml cannot be found.
    """
    logger = setup_logger(__name__)

    critical = {
        "src directory":  project_root / "src",
        "config.yaml":    project_root / "config.yaml",
    }
    optional = {
        "enhanced_df":    project_root / "data" / "processed" / "enhanced_df.parquet",
        "rfm_df":         project_root / "data" / "processed" / "rfm_df.parquet",
        "outputs/figures": project_root / "outputs" / "figures",
    }

    for label, path in critical.items():
        if not path.exists():
            raise RuntimeError(
                f"Critical path missing: {label} at {path}"
            )
        logger.debug("Verified: %s → %s", label, path)

    for label, path in optional.items():
        if not path.exists():
            logger.warning("Optional path not found: %s → %s", label, path)
        else:
            logger.debug("Verified: %s → %s", label, path)


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------

def print_section_header(title: str, width: int = 80) -> None:
    """Print a formatted section header to stdout.

    Parameters
    ----------
    title:
        Section title string.
    width:
        Total line width (default 80).
    """
    print(f"\n{'=' * width}")
    print(title.center(width))
    print("=" * width)
