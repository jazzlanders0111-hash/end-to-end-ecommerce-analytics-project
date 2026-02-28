# src/n2a_utils.py
"""
n2a_utils.py - Shared utilities for Notebook 02 (Sales Analysis)

This module provides common functionality used across all sales analysis modules:
- Logging setup with run ID tracking
- Figure saving utilities
- Project path management
- Color schemes and styling constants (loaded from config)

Usage:
    from n2a_utils import setup_logger, save_plotly_figure, COLORS, get_config
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
import plotly.graph_objects as go
import yaml

# ============================================================================
# Configuration Loading
# ============================================================================

def get_project_root() -> Path:
    """
    Locate the project root directory by searching for config.yaml.

    Returns:
        Path: The project root directory

    Raises:
        FileNotFoundError: If project root cannot be found
    """
    try:
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / 'config.yaml').exists():
                return parent
        raise FileNotFoundError(
            "Could not find project root (no config.yaml found in parent directories)"
        )
    except Exception as e:
        raise FileNotFoundError(f"Error while searching for project root: {e}") from e


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml (auto-detected if None)

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If config file cannot be found
        yaml.YAMLError: If config file is malformed
    """
    try:
        if config_path is None:
            config_path = get_project_root() / 'config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config
    except FileNotFoundError:
        raise
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Malformed config.yaml: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}") from e


# Load configuration at module import
try:
    _CONFIG = load_config()
    _VIZ_CONFIG = _CONFIG.get('visualization', {})
    _NB2_CONFIG = _CONFIG.get('notebook2', {})
except Exception as e:
    # Use module-level basic logger before setup_logger is available
    logging.getLogger(__name__).warning(f"Could not load config.yaml: {e}")
    _CONFIG = {}
    _VIZ_CONFIG = {}
    _NB2_CONFIG = {}

# ============================================================================
# Module-level logger (basic — does not depend on run ID)
# n2a_utils defines setup_logger itself, so it uses getLogger directly
# rather than calling setup_logger(__name__) recursively.
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# Global Constants (loaded from config)
# ============================================================================

# Color scheme for consistent visualizations
COLORS = _VIZ_CONFIG.get('colors', {
    'primary': '#1B1F5E',
    'secondary': '#EA731D',
    'success': '#2E7D32',
    'warning': '#F57C00',
    'danger': '#C62828',
    'neutral': '#757575',
    'gradient': ['#1B1F5E', '#4A5FA8', '#EA731D', '#F59D5A'],
})

# Figure layout defaults
FIGURE_DEFAULTS = _VIZ_CONFIG.get('figure_defaults', {
    'height': 500,
    'template': 'plotly_white',
    'hovermode': 'x unified',
    'font': {'size': 12, 'family': 'Arial, sans-serif'},
})

# Export settings
_EXPORT_CONFIG = _VIZ_CONFIG.get('export', {})
PNG_WIDTH = _EXPORT_CONFIG.get('png', {}).get('width', 1200)
PNG_HEIGHT = _EXPORT_CONFIG.get('png', {}).get('height', 600)
DEFAULT_FORMATS = _EXPORT_CONFIG.get('formats', ['html'])
INCLUDE_TIMESTAMP = _EXPORT_CONFIG.get('include_timestamp', True)

# ============================================================================
# Run ID Management
# ============================================================================

_RUN_ID = None


def set_run_id(run_id: str) -> None:
    """
    Set the global run ID for this analysis session.

    Args:
        run_id: Unique identifier string for this run
    """
    global _RUN_ID
    _RUN_ID = run_id


def get_run_id() -> str:
    """
    Get the current run ID.

    Returns:
        str: The current run ID

    Raises:
        RuntimeError: If run ID has not been set
    """
    if _RUN_ID is None:
        raise RuntimeError("Run ID not set. Call set_run_id() first.")
    return _RUN_ID


def generate_run_id() -> str:
    """
    Generate a unique run ID based on timestamp.

    Returns:
        str: Timestamp-based run ID in format YYMMDDHHMM
    """
    return datetime.now().strftime("%y%m%d%H%M")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with run ID in the format.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    try:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        logger.handlers = []

        # Create console handler
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        # Create formatter with run ID
        try:
            run_id = get_run_id()
            formatter = logging.Formatter(
                f'[{run_id}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        except RuntimeError:
            # Run ID not set yet, use simpler format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
    except Exception as e:
        # Absolute fallback — return a basic logger so callers are never blocked
        fallback = logging.getLogger(name)
        fallback.warning(f"setup_logger failed, using fallback: {e}")
        return fallback


# ============================================================================
# Figure Saving Utilities
# ============================================================================

def save_plotly_figure(
    fig: go.Figure,
    filename: str,
    output_dir: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    include_timestamp: Optional[bool] = None,
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    Save a Plotly figure in multiple formats.

    Args:
        fig: Plotly figure object
        filename: Base filename (without extension)
        output_dir: Output directory (defaults to notebook2_figures from config)
        formats: List of formats to save (defaults from config)
        include_timestamp: Whether to include timestamp (defaults from config)
        logger: Logger instance for logging (optional)

    Returns:
        dict: Paths to saved files {format: filepath}

    Example:
        >>> fig = go.Figure(...)
        >>> paths = save_plotly_figure(fig, "revenue_trends", formats=['html', 'png'])
    """
    _log = logger or globals().get('logger', logging.getLogger(__name__))
    try:
        output_dir_path: Path
        if output_dir is None:
            project_root = get_project_root()
            nb2_figures_path = _CONFIG.get('paths', {}).get(
                'notebook2_figures', 'outputs/figures/notebook2_fig'
            )
            output_dir_path = project_root / nb2_figures_path
        else:
            output_dir_path = Path(output_dir)

        output_dir_path.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = DEFAULT_FORMATS
        if include_timestamp is None:
            include_timestamp = INCLUDE_TIMESTAMP

        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{filename}_{timestamp}"
        else:
            base_name = filename

        saved_files = {}

        for fmt in formats:
            filepath = output_dir_path / f"{base_name}.{fmt}"
            try:
                if fmt == 'html':
                    fig.write_html(str(filepath))
                elif fmt == 'png':
                    fig.write_image(str(filepath), width=PNG_WIDTH, height=PNG_HEIGHT)
                elif fmt == 'svg':
                    fig.write_image(str(filepath))
                else:
                    _log.warning(f"Unknown format '{fmt}', skipping")
                    continue

                saved_files[fmt] = filepath
                _log.info(f"Saved {fmt.upper()}: {filepath.name}")

            except Exception as e:
                _log.error(f"Failed to save {fmt.upper()}: {e}")

        return saved_files

    except Exception as e:
        _log.error(f"save_plotly_figure failed for '{filename}': {e}")
        return {}


def get_figures_dir(subfolder: Optional[str] = None) -> Path:
    """
    Get the figures output directory.

    Args:
        subfolder: Subfolder within figures/ (default from config)

    Returns:
        Path: Figures directory path

    Raises:
        RuntimeError: If directory cannot be created
    """
    try:
        project_root = get_project_root()

        if subfolder is None:
            figures_path = _CONFIG.get('paths', {}).get(
                'notebook2_figures', 'outputs/figures/notebook2_fig'
            )
            figures_dir = project_root / figures_path
        else:
            figures_base = _CONFIG.get('paths', {}).get('figures_dir', 'outputs/figures')
            figures_dir = project_root / figures_base / subfolder

        figures_dir.mkdir(parents=True, exist_ok=True)
        return figures_dir
    except Exception as e:
        raise RuntimeError(f"Failed to get or create figures directory: {e}") from e


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_dataframe_columns(
    df: Any,
    required_columns: list,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate that a DataFrame has all required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        logger: Logger instance (optional)

    Returns:
        bool: True if all columns present, False otherwise
    """
    try:
        _log = logger or logging.getLogger(__name__)
        missing = set(required_columns) - set(df.columns)

        if missing:
            msg = f"Missing required columns: {sorted(missing)}"
            _log.error(msg)
            return False

        _log.info(f"All {len(required_columns)} required columns present")
        return True
    except Exception as e:
        _log = logger or logging.getLogger(__name__)
        _log.error(f"validate_dataframe_columns failed: {e}")
        return False


# ============================================================================
# Display Utilities
# ============================================================================

def print_section_header(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.

    Args:
        title: Section title text
        width: Width of the header line in characters
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str) -> None:
    """
    Print a formatted subsection header.

    Args:
        title: Subsection title text
    """
    print(f"\n{title}")


# ============================================================================
# Configuration Access Functions
# ============================================================================

def get_config(key_path: Optional[str] = None) -> Any:
    """
    Get configuration value by key path.

    Args:
        key_path: Dot-separated path to config value
                  (e.g., 'notebook2.forecasting.train_ratio').
                  If None, returns entire config dict.

    Returns:
        Configuration value or entire config dict

    Example:
        >>> train_ratio = get_config('notebook2.forecasting.train_ratio')
        >>> colors = get_config('visualization.colors')
    """
    try:
        if key_path is None:
            return _CONFIG

        keys = key_path.split('.')
        value = _CONFIG

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None

        return value
    except Exception as e:
        logger.error(f"get_config failed for key_path='{key_path}': {e}")
        return None


# ============================================================================
# Module Information
# ============================================================================

__all__ = [
    # Constants
    'COLORS',
    'FIGURE_DEFAULTS',
    'PNG_WIDTH',
    'PNG_HEIGHT',
    'DEFAULT_FORMATS',
    'INCLUDE_TIMESTAMP',
    # Logging functions
    'setup_logger',
    'set_run_id',
    'get_run_id',
    'generate_run_id',
    # Path functions
    'get_project_root',
    'get_figures_dir',
    # Figure functions
    'save_plotly_figure',
    # Validation functions
    'validate_dataframe_columns',
    # Display functions
    'print_section_header',
    'print_subsection',
    # Configuration functions
    'load_config',
    'get_config',
]

if __name__ == '__main__':
    print("n2a_utils.py - Notebook 02 Utilities Module")
    print("This module should be imported, not run directly.")
    print("\nConfiguration loaded:")
    print(f"  - Notebook2 config: {bool(_NB2_CONFIG)}")
    print(f"  - Visualization config: {bool(_VIZ_CONFIG)}")
    print(f"  - Primary color: {COLORS.get('primary', 'Not set')}")
