# src/n1a_utils.py

"""
n1a_utils.py - Core Utilities for Notebook 01

Key Features:
- Centralised logging with run_id correlation
- Project root detection
- Configuration management
- Output path management
- Session tracking

Design Principles:
- Defensive programming with validation
- Clear, actionable error messages
- Comprehensive logging

Error Handling Strategy:
- get_project_root        : raises RuntimeError with diagnostic info (intentional – callers must know)
- generate_run_id         : falls back to timestamp-based ID if uuid fails
- set_run_id              : validates type; generates fresh ID on bad input instead of crashing
- get_run_id              : never raises; returns None on any failure
- RunIDFilter.filter      : swallows all exceptions; always returns True so logging is never blocked
- SanitizeFilter.filter   : swallows regex failures per-pattern; partial redaction beats a crash
- setup_logger            : validates every argument; returns a minimal safe logger on failure
- load_config             : clear FileNotFoundError / ValueError with path context
- get_config_value        : never raises when default is provided; raises KeyError with full path
- verify_project_structure: returns False instead of raising when called with strict=False
"""

import logging
import uuid
import re
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Context variable – stores run_id across function calls within a task
# ---------------------------------------------------------------------------
run_id_var: ContextVar[Optional[str]] = ContextVar("run_id", default=None)


# ===========================================================================
# 1. get_project_root
# ===========================================================================

def get_project_root() -> Path:
    """
    Find the project root reliably whether running in a notebook or a script.
    Assumes standard structure: notebooks/ and src/ are siblings under root.

    Error handling:
    - Validates that cwd is resolvable (symlink issues, deleted dirs)
    - Logs each search strategy attempted before giving up
    - Raises RuntimeError with a clear diagnostic message if root cannot be found
      (intentional – callers MUST know the project structure is broken)

    Returns:
        Resolved Path to project root

    Raises:
        RuntimeError: If root cannot be determined by any strategy
    """
    try:
        cwd = Path.cwd().resolve()
    except OSError as e:
        raise RuntimeError(
            f"get_project_root: cannot resolve current working directory: {e}\n"
            "Check that the process has read access to its own working directory."
        )

    # Strategy 1: running from notebooks/
    if cwd.name == "notebooks":
        candidate = cwd.parent
        if candidate.exists():
            return candidate

    # Strategy 2: already at project root
    if (cwd / "src").exists() and (cwd / "notebooks").exists():
        return cwd

    # Strategy 3: walk upward looking for config.yaml or data/raw/
    searched: list[Path] = []
    for parent in [cwd, *cwd.parents]:
        searched.append(parent)
        try:
            if (parent / "config.yaml").exists():
                return parent
            if (parent / "data" / "raw").exists():
                return parent
        except OSError:
            # Inaccessible directory – keep walking
            continue

    raise RuntimeError(
        "get_project_root: could not locate project root.\n"
        f"  Started from : {cwd}\n"
        f"  Searched     : {[str(p) for p in searched]}\n"
        "  Expected markers: config.yaml OR data/raw/ in an ancestor directory.\n"
        "  Ensure you are running from inside the project tree."
    )


PROJECT_ROOT = get_project_root()


# ===========================================================================
# 2. generate_run_id
# ===========================================================================

def generate_run_id() -> str:
    """
    Generate a short, unique run identifier (8 hex characters).

    Error handling:
    - Falls back to a timestamp-based ID if uuid generation fails
      (extremely unlikely but guards against restricted environments)

    Returns:
        8-character hex string (uuid-based or timestamp-based fallback)
    """
    try:
        return uuid.uuid4().hex[:8]
    except Exception:
        # Fallback: yyyymmddHHMMSSff (14 chars) – truncate to 8
        fallback = datetime.now().strftime("%Y%m%d%H%M%S%f")[:8]
        return fallback


# ===========================================================================
# 3. set_run_id
# ===========================================================================

def set_run_id(run_id: Optional[str] = None) -> str:
    """
    Set the correlation ID for the current execution context.

    Error handling:
    - If run_id is not a str, logs a warning and generates a fresh ID
      instead of raising, so pipeline steps are never blocked by a bad caller
    - If run_id is an empty string, treats it the same as None (generate new)
    - If ContextVar.set() fails, stores the ID in a module-level fallback dict
      and returns the ID anyway

    Args:
        run_id: Optional custom run_id. If None or invalid, generates a new one.

    Returns:
        The run_id that was set
    """
    # ---- Validate / coerce -------------------------------------------------
    if run_id is not None and not isinstance(run_id, str):
        _warn_once(
            f"set_run_id: run_id must be a str or None, got {type(run_id).__name__} "
            "– generating a new ID"
        )
        run_id = None

    if isinstance(run_id, str) and not run_id.strip():
        _warn_once("set_run_id: run_id is an empty string – generating a new ID")
        run_id = None

    if run_id is None:
        run_id = generate_run_id()

    # ---- Set in ContextVar -------------------------------------------------
    try:
        run_id_var.set(run_id)
    except Exception as e:
        # ContextVar.set() should never fail in normal Python, but guard anyway
        _warn_once(f"set_run_id: ContextVar.set() failed ({e}) – ID stored in fallback")
        _FALLBACK_RUN_ID["current"] = run_id

    return run_id


# Module-level fallback for environments where ContextVar is broken
_FALLBACK_RUN_ID: dict[str, Optional[str]] = {"current": None}

# Simple once-only warning cache so we don't flood logs
_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    """Print a warning exactly once per unique message."""
    if msg not in _WARNED:
        _WARNED.add(msg)
        # Use print because the logger may not be set up yet
        print(f"WARNING [n1a_utils]: {msg}")


# ===========================================================================
# 4. get_run_id
# ===========================================================================

def get_run_id() -> Optional[str]:
    """
    Get the current run ID from context.

    Error handling:
    - Returns the fallback ID if ContextVar.get() fails
    - Returns None (never raises) so log formatters are never broken

    Returns:
        Current run_id string, or None if not set
    """
    try:
        value = run_id_var.get()
        if value is not None:
            return value
        # ContextVar returned None – check module-level fallback
        return _FALLBACK_RUN_ID.get("current")
    except Exception:
        return _FALLBACK_RUN_ID.get("current")


# ===========================================================================
# 5a. RunIDFilter  (filter method)
# ===========================================================================

class RunIDFilter(logging.Filter):
    """Inject run_id into log records for correlation."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Error handling:
        - Catches ALL exceptions so a run_id lookup failure never silences a
          log record (always returns True)
        - Falls back to the placeholder 'XXXXXXXX' on any error
        """
        try:
            record.run_id = get_run_id() or "XXXXXXXX"
        except Exception:
            # Last-resort: set a visible placeholder rather than crashing
            try:
                record.run_id = "ERR_RUNID"
            except Exception:
                pass  # If even attribute assignment fails, move on
        return True  # ALWAYS allow the record through


# ===========================================================================
# 5b. SanitizeFilter  (filter method)
# ===========================================================================

# Pre-compiled patterns for performance and safer error isolation
_SANITIZE_PATTERNS: list[Tuple[re.Pattern, str]] = []

def _build_sanitize_patterns() -> list[Tuple[re.Pattern, str]]:
    """
    Compile redaction patterns once at import time.
    Each pattern is compiled independently so one bad regex does not
    prevent the others from loading.
    """
    raw_patterns = [
        (r"C\d{5}",                                             "[REDACTED_CUSTOMER_ID]"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b","[REDACTED_EMAIL]"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",                     "[REDACTED_PHONE]"),
        (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",       "[REDACTED_CC]"),
    ]
    compiled: list[Tuple[re.Pattern, str]] = []
    for pattern, replacement in raw_patterns:
        try:
            compiled.append((re.compile(pattern), replacement))
        except re.error as e:
            # Don't let a bad pattern prevent the filter from loading
            print(f"WARNING [n1a_utils]: SanitizeFilter could not compile pattern {pattern!r}: {e}")
    return compiled

_SANITIZE_PATTERNS = _build_sanitize_patterns()


class SanitizeFilter(logging.Filter):
    """Redact sensitive data from log messages (security best practice)."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Error handling:
        - Each regex substitution runs in its own try/except; one failing
          pattern does not prevent others from running (partial redaction
          is better than no redaction or a crash)
        - If message coercion to str fails, the original record is left
          untouched and the record is still allowed through
        - Always returns True so logging is never blocked
        """
        try:
            msg = str(record.msg)
        except Exception:
            return True  # Cannot coerce message – pass through unmodified

        for pattern, replacement in _SANITIZE_PATTERNS:
            try:
                msg = pattern.sub(replacement, msg)
            except Exception as e:
                # Log to stderr via print so we don't recurse into logging
                print(f"WARNING [SanitizeFilter]: redaction failed for pattern "
                      f"{pattern.pattern!r}: {e}")

        try:
            record.msg = msg
        except Exception:
            pass  # If we can't write back, leave original

        return True  # ALWAYS allow the record through


# ===========================================================================
# setup_logger   (not in the 7 but lives in the same file – hardened too)
# ===========================================================================

def setup_logger(
    name: str,
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    include_run_id: bool = True,
    include_sanitization: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Error handling:
    - Validates name is a non-empty string; falls back to 'unnamed_logger'
    - Validates level is a recognised logging level; falls back to INFO
    - File handler failures are caught and logged; console handler still works
    - Returns a minimally functional logger even if configuration partially fails

    Args:
        name: Logger name (usually __name__)
        log_file: Optional file path for logging
        level: Logging level (default: INFO)
        include_run_id: Whether to include run_id in logs
        include_sanitization: Whether to sanitise sensitive data

    Returns:
        Configured logger instance
    """
    # ---- Validate name -----------------------------------------------------
    if not isinstance(name, str) or not name.strip():
        _warn_once(f"setup_logger: 'name' must be a non-empty str, got {name!r} – using 'unnamed_logger'")
        name = "unnamed_logger"

    # ---- Validate level ----------------------------------------------------
    valid_levels = {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
    if not isinstance(level, int) or level not in valid_levels:
        _warn_once(f"setup_logger: invalid level {level!r} – defaulting to INFO")
        level = logging.INFO

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ---- Attach filters ----------------------------------------------------
    try:
        if include_sanitization:
            logger.addFilter(SanitizeFilter())
    except Exception as e:
        _warn_once(f"setup_logger: could not attach SanitizeFilter: {e}")

    try:
        if include_run_id:
            logger.addFilter(RunIDFilter())
    except Exception as e:
        _warn_once(f"setup_logger: could not attach RunIDFilter: {e}")

    # ---- Build formatter ---------------------------------------------------
    try:
        if include_run_id:
            fmt = "[%(run_id)s] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    except Exception as e:
        _warn_once(f"setup_logger: formatter creation failed ({e}) – using basicConfig fallback")
        formatter = logging.Formatter("%(levelname)s - %(message)s")

    # ---- Console handler ---------------------------------------------------
    try:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    except Exception as e:
        # If even StreamHandler fails, Python's last-resort handler will output warnings
        _warn_once(f"setup_logger: could not create console handler: {e}")

    # ---- Optional file handler ---------------------------------------------
    if log_file is not None:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info(f"Logging to file: {log_path}")
        except (OSError, ValueError, TypeError) as e:
            logger.warning(f"setup_logger: could not create log file '{log_file}': {e}")

    return logger


# ===========================================================================
# load_config   (hardened)
# ===========================================================================

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Error handling:
    - Validates config_path is a Path-like object before calling .exists()
    - Raises FileNotFoundError with the resolved absolute path for clarity
    - Raises ValueError (not yaml.YAMLError) so callers only need to catch
      one exception type for malformed files; original error is chained

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary (empty dict if YAML file is blank)

    Raises:
        TypeError: If config_path is not Path-like
        FileNotFoundError: If the file does not exist
        ValueError: If the file contains invalid YAML
    """
    # ---- Type check --------------------------------------------------------
    if not isinstance(config_path, (str, Path)):
        raise TypeError(
            f"load_config: 'config_path' must be a Path or str, "
            f"got {type(config_path).__name__}"
        )
    config_path = Path(config_path).resolve()

    # ---- Existence check ---------------------------------------------------
    if not config_path.exists():
        raise FileNotFoundError(
            f"load_config: config file not found: {config_path}\n"
            "Check that config.yaml exists at the project root."
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"load_config: path exists but is not a file: {config_path}"
        )

    # ---- Load YAML ---------------------------------------------------------
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"load_config: invalid YAML in '{config_path}': {e}"
        ) from e
    except OSError as e:
        raise OSError(
            f"load_config: cannot read '{config_path}': {e}"
        ) from e

    # yaml.safe_load returns None for an empty file
    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            f"load_config: expected a YAML mapping (dict) at root, "
            f"got {type(config).__name__} in '{config_path}'"
        )

    return config


# ===========================================================================
# get_config_value   (hardened)
# ===========================================================================

def get_config_value(
    config: dict,
    *keys: str,
    default: Any = None,
    environment: Optional[str] = None,
) -> Any:
    """
    Safe config value retrieval with environment override support.

    Error handling:
    - Validates config is a dict
    - Validates every key in *keys is a non-empty string
    - Validates environment is a str or None
    - Returns default instead of raising KeyError when default is provided
    - Raises KeyError with the full dotted key path when default is not provided,
      so the caller knows exactly what was missing

    Args:
        config: Configuration dictionary
        *keys: Path to config value (e.g., 'rfm', 'churn_threshold_days')
        default: Value returned when key path is not found (default: None means raise)
        environment: Environment name for overrides (default: None → uses 'development')

    Returns:
        Config value, or *default* if key not found and default is not None

    Raises:
        TypeError: If config is not a dict or any key is not a string
        KeyError: If key path not found and no default is provided
    """
    # ---- Validate config ---------------------------------------------------
    if not isinstance(config, dict):
        raise TypeError(
            f"get_config_value: 'config' must be a dict, got {type(config).__name__}"
        )

    # ---- Validate keys -----------------------------------------------------
    if not keys:
        raise TypeError("get_config_value: at least one key must be provided")

    for i, key in enumerate(keys):
        if not isinstance(key, str) or not key.strip():
            raise TypeError(
                f"get_config_value: key at position {i} must be a non-empty str, "
                f"got {key!r}"
            )

    # ---- Validate environment ----------------------------------------------
    if environment is not None and not isinstance(environment, str):
        _warn_once(
            f"get_config_value: 'environment' must be a str or None, "
            f"got {type(environment).__name__} – ignored"
        )
        environment = None

    dotted_path = ".".join(keys)

    # ---- Resolve environment -----------------------------------------------
    if environment is None:
        environment = config.get("environment", "development")
        if not isinstance(environment, str):
            environment = "development"

    # ---- Strategy 1: environment-specific override -------------------------
    try:
        env_section = config.get(environment, {})
        if isinstance(env_section, dict):
            value = env_section
            for key in keys:
                value = value[key]
            return value
    except (KeyError, TypeError):
        pass  # Fall through to global config

    # ---- Strategy 2: global config -----------------------------------------
    try:
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        pass

    # ---- Neither found -----------------------------------------------------
    if default is not None:
        return default

    raise KeyError(
        f"get_config_value: key path '{dotted_path}' not found.\n"
        f"  Searched in  : environment '{environment}' and global config\n"
        f"  Available top-level keys: {list(config.keys())}"
    )


# ===========================================================================
# 6. verify_project_structure
# ===========================================================================

def verify_project_structure(
    project_root: Path,
    strict: bool = True,
) -> bool:
    """
    Verify essential directories and files exist under project_root.

    Error handling:
    - Validates project_root is a Path-like object and actually exists
    - Catches OSError when checking individual paths (permissions, broken symlinks)
    - strict=True  → raises RuntimeError listing all missing items (original behaviour)
    - strict=False → returns False with a warning instead of raising, so callers
      can decide whether to abort or continue in degraded mode

    Args:
        project_root: Path to verify
        strict: If True, raise on failure (default). If False, return False.

    Returns:
        True if the structure is valid

    Raises:
        TypeError: If project_root is not Path-like
        RuntimeError: If structure is invalid and strict=True
    """
    # ---- Type validation ---------------------------------------------------
    if not isinstance(project_root, (str, Path)):
        raise TypeError(
            f"verify_project_structure: 'project_root' must be Path or str, "
            f"got {type(project_root).__name__}"
        )
    project_root = Path(project_root).resolve()

    # ---- Existence check ---------------------------------------------------
    if not project_root.exists():
        msg = (
            f"verify_project_structure: project_root does not exist: {project_root}"
        )
        if strict:
            raise RuntimeError(msg)
        _warn_once(msg)
        return False

    if not project_root.is_dir():
        msg = (
            f"verify_project_structure: project_root is not a directory: {project_root}"
        )
        if strict:
            raise RuntimeError(msg)
        _warn_once(msg)
        return False

    # ---- Required items ----------------------------------------------------
    required_dirs = [
        "src",
        "data",
        "data/raw",
        "outputs",
        "outputs/figures",
    ]
    required_files = [
        "config.yaml",
        "schema.yaml",
    ]

    missing: list[str] = []

    for dir_name in required_dirs:
        try:
            dir_path = project_root / dir_name
            if not dir_path.exists() or not dir_path.is_dir():
                missing.append(f"Directory : {dir_name}")
        except OSError as e:
            missing.append(f"Directory : {dir_name}  (OS error: {e})")

    for file_name in required_files:
        try:
            file_path = project_root / file_name
            if not file_path.exists() or not file_path.is_file():
                missing.append(f"File      : {file_name}")
        except OSError as e:
            missing.append(f"File      : {file_name}  (OS error: {e})")

    # ---- Report ------------------------------------------------------------
    if missing:
        item_list = "\n".join(f"  - {item}" for item in missing)
        msg = (
            f"verify_project_structure: validation failed under '{project_root}'.\n"
            f"Missing items:\n{item_list}"
        )
        if strict:
            raise RuntimeError(msg)
        _warn_once(msg)
        return False

    return True