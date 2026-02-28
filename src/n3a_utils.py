"""
n3a_utils.py - Core Utilities for Notebook 03

Provides essential utility functions for customer segmentation analysis,
including logging, configuration, project management, and run correlation.

Key Features:
- Centralized logging with run_id correlation
- Project root detection
- Configuration management
- Output path management
- Session tracking

Design Principles:
- Consistent with n1a_utils.py and n2a_utils.py patterns
- Defensive programming with validation
- Clear error messages
- Comprehensive logging
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextvars import ContextVar
import yaml

logger = logging.getLogger(__name__)

# Context variable to store run_id across function calls
run_id_var: ContextVar[Optional[str]] = ContextVar('run_id', default=None)


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
    """
    return uuid.uuid4().hex[:8]


def set_run_id(run_id: Optional[str] = None) -> str:
    """
    Set correlation ID for current execution context.
    
    Args:
        run_id: Optional custom run_id. If None, generates a new one.
        
    Returns:
        The run_id that was set
    """
    if run_id is None:
        run_id = generate_run_id()
    run_id_var.set(run_id)
    return run_id


def get_run_id() -> Optional[str]:
    """
    Get current run ID from context.
    
    Returns:
        Current run ID or None if not set
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


def print_section_header(title: str, width: int = 80, char: str = '=') -> None:
    """
    Print a formatted section header for better readability.
    
    Args:
        title: Section title
        width: Width of the header (default: 80)
        char: Character to use for border (default: '=')
    """
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def verify_project_structure(project_root: Path) -> bool:
    """
    Verify that required project directories exist.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        True if all required directories exist
        
    Raises:
        FileNotFoundError: If required directories are missing
    """
    try:
        required_dirs = [
            'src',
            'data',
            'data/raw',
            'data/processed',
            'outputs',
            'outputs/figures',
            'notebooks'
        ]
    
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
    
        if missing_dirs:
            raise FileNotFoundError(
                f"Missing required directories: {', '.join(missing_dirs)}\n"
                f"Project root: {project_root}"
            )
    
        return True
    except Exception as e:
        logger.error(f"verify_project_structure failed: {e}")
        raise


def get_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get standardized output paths for notebook 03.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of output paths
    """
    try:
        project_root = get_project_root()
    
        paths = {
            'figures': project_root / config['paths']['notebook3_figures'],
            'processed': project_root / config['paths']['processed_data'],
            'interim': project_root / config['paths']['interim_data']
        }
    
        # Create directories if they don't exist
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
        return paths
    except Exception as e:
        logger.error(f"get_output_paths failed: {e}")
        raise


def get_colors(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract color scheme from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of color values
    """
    try:
        return config.get('visualization', {}).get('colors', {
            'primary': '#1B1F5E',
            'secondary': '#EA731D',
            'success': '#2E7D32',
            'warning': '#F57C00',
            'danger': '#C62828',
            'neutral': '#757575'
        })
    except Exception as e:
        logger.error(f"get_colors failed: {e}")
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
    'verify_project_structure',
    'get_output_paths',
    'get_colors',
    'logger',
]

if __name__ == "__main__":
    # Test the module
    print("Testing n3a_utils module...")
    
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
    except Exception as e:
        print(f"Config load error: {e}")
    
    print("All tests passed!")
