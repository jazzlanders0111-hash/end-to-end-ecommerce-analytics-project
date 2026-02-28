# src/n4a_utils.py
"""
n4a_utils.py - Core Utilities for Notebook 04 (Churn Prediction)

Provides essential utility functions for churn prediction analysis,
including logging, configuration, project management, and run correlation.

Key Features:
- Centralized logging with run_id correlation
- Project root detection
- Configuration management
- Output path management
- Session tracking

Design Principles:
- Consistent with n1a_utils.py, n2a_utils.py, n3a_utils.py patterns
- Defensive programming with validation
- Clear error messages
- Comprehensive logging
- NO EMOJIS (professional output only)
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from contextvars import ContextVar
import yaml

logger = logging.getLogger(__name__)

# Context variable to store run_id across function calls
run_id_var: ContextVar[str | None] = ContextVar('run_id', default=None)


def get_project_root() -> Path:
    """
    Detect the project root directory by searching for config.yaml.
    
    Searches current directory and up to 3 parent directories for:
    - config.yaml file
    - notebooks/ and src/ directories
    
    Returns:
        Path: The project root directory
        
    Raises:
        FileNotFoundError: If config.yaml cannot be found
        
    Example:
        >>> root = get_project_root()
        >>> print(root / 'config.yaml')
    """
    try:
        current_path = Path.cwd()
    
        # Try current directory and up to 3 parent directories
        for _ in range(4):
            if (current_path / 'config.yaml').exists():
                return current_path
            if (current_path / 'notebooks').exists() and (current_path / 'src').exists():
                return current_path
            current_path = current_path.parent
    
        # If not found, try one level up from current working directory
        if (Path.cwd().parent / 'config.yaml').exists():
            return Path.cwd().parent
    
        raise FileNotFoundError(
            "Could not find project root. Make sure config.yaml exists in project root."
        )
    except Exception as e:
        logger.error(f"get_project_root failed: {e}")
        raise


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, searches for config.yaml in project root
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
        
    Example:
        >>> config = load_config()
        >>> test_size = config['notebook4']['test_size']
    """
    if config_path is None:
        config_path = get_project_root() / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")


def generate_run_id() -> str:
    """
    Generate a short, unique run identifier (8 hex characters).
    
    Returns:
        String run ID (8 characters)
        
    Example:
        >>> run_id = generate_run_id()
        >>> print(run_id)
        a3f7b2c1
    """
    return uuid.uuid4().hex[:8]


def set_run_id(run_id: str | None = None) -> str:
    """
    Set correlation ID for current execution context.
    
    Args:
        run_id: Optional custom run_id. If None, generates a new one.
        
    Returns:
        The run_id that was set
        
    Example:
        >>> run_id = set_run_id()
        >>> print(f"Pipeline Run ID: {run_id}")
    """
    if run_id is None:
        run_id = generate_run_id()
    run_id_var.set(run_id)
    return run_id


def get_run_id() -> str | None:
    """
    Get current run ID from context.
    
    Returns:
        Current run ID or None if not set
        
    Example:
        >>> current_id = get_run_id()
        >>> print(f"Current run: {current_id}")
    """
    return run_id_var.get()


class RunIDFilter(logging.Filter):
    """Inject run_id into log records for correlation."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Inject run_id into each log record."""
        record.run_id = get_run_id() or "XXXXXXXX"
        return True


def setup_logger(
    name: str,
    level: int = logging.INFO,
    include_run_id: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting and run ID prefix.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        include_run_id: Whether to include run_id in logs (default: True)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        [a3f7b2c1] 2026-02-15 10:15:23 - mymodule - INFO - Processing started
    """
    try:
        logger = logging.getLogger(name)
    
        # Prevent duplicate handlers if logger already configured
        if logger.handlers:
            return logger
    
        logger.setLevel(level)
    
        # Add RunID filter if requested
        if include_run_id:
            logger.addFilter(RunIDFilter())
    
        # Create console handler
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
    
        # Create formatter with or without run_id
        if include_run_id:
            formatter = logging.Formatter(
                '[%(run_id)s] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
        return logger
    except Exception as e:
        logger.error(f"setup_logger failed: {e}")
        raise


def print_section_header(title: str, width: int = 80) -> None:
    """
    Print a formatted section header above all logger output.

    Uses flush=True so the header always prints before any logger lines
    in Jupyter notebook output, regardless of stdout/stderr interleaving.

    Args:
        title: Section title
        width: Width of the header (default: 80)

    Example:
        >>> print_section_header("Data Loading")
        ================================================================================
                                     Data Loading
        ================================================================================
    """
    import sys
    print(f"\n{'=' * width}", flush=True)
    print(f"{title:^{width}}", flush=True)
    print(f"{'=' * width}\n", flush=True)
    sys.stdout.flush()


def get_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get standardized output paths for notebook 04.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of output paths with keys:
        - 'figures': Path to notebook4 figure directory
        - 'processed': Path to processed data directory
        - 'models': Path to trained models directory
        
    Example:
        >>> config = load_config()
        >>> paths = get_output_paths(config)
        >>> print(paths['figures'])
    """
    try:
        project_root = get_project_root()
    
        paths = {
            'figures': project_root / config['paths']['figures_dir'] / 'notebook4_fig',
            'processed': project_root / config['paths']['processed_data'],
            'models': project_root / config['paths']['models_dir']
        }
    
        # Create directories if they don't exist
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
        return paths
    except Exception as e:
        logger.error(f"get_output_paths failed: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that required configuration parameters exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    try:
        logger = setup_logger(__name__)
        logger.info("Validating configuration...")
    
        # Check required top-level keys
        required_keys = ['paths', 'notebook4', 'visualization']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")
    
        # Check notebook4 parameters
        nb4_config = config['notebook4']
        required_nb4 = ['observation_window_days', 'test_size', 'random_state', 'training']
        for key in required_nb4:
            if key not in nb4_config:
                raise ValueError(f"Missing required notebook4 parameter: {key}")
    
        # Validate ranges
        if not (0 < nb4_config['test_size'] < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {nb4_config['test_size']}")
    
        if nb4_config['observation_window_days'] <= 0:
            raise ValueError("observation_window_days must be positive")
    
        logger.info("Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"validate_config failed: {e}")
        raise


__all__ = [
    'run_id_var',
    'get_project_root',
    'load_config',
    'generate_run_id',
    'set_run_id',
    'get_run_id',
    'RunIDFilter',
    'setup_logger',
    'print_section_header',
    'get_output_paths',
    'validate_config',
    'logger',
]

if __name__ == "__main__":
    # Test the module
    print("Testing n4a_utils module...")
    
    # Test run ID generation
    run_id = set_run_id()
    print(f"Generated RUN_ID: {run_id}")
    
    # Test project root detection
    try:
        root = get_project_root()
        print(f"Project root: {root}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Test logger setup
    logger = setup_logger(__name__)
    logger.info("Logger test successful")
    
    # Test section header
    print_section_header("Test Section")
    
    # Test config loading
    try:
        config = load_config()
        print(f"Config loaded: {len(config)} top-level keys")
        validate_config(config)
        print(f"Test size parameter: {config['notebook4']['test_size']}")
    except Exception as e:
        print(f"Config load error: {e}")
    
    print("\nAll tests passed!")
