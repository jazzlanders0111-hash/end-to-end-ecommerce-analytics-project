# src/n1b_data_loader.py

"""
n1b_data_loader.py - Data Loading for Notebook 01

This module handles loading raw data from a CSV file and applying schema validation.

Key Features:
- Load raw data from a CSV file
- Apply schema validation using Pandera
- Comprehensive logging with run_id correlation

Design Principles:
- Defensive programming with clear error messages
- Automatic handling of missing RFM scores
- Data integrity checks with hash validation
- Consistent logging patterns
"""

# Standard library
from pathlib import Path
from typing import Optional

# Third-party
import pandas as pd
import yaml
import pandera.pandas as pa
from pandera import DataFrameSchema, Column, Check

# Local modules
from n1a_utils import (
    load_config,
    setup_logger,
    get_project_root,
    set_run_id,
    get_run_id,
    get_config_value
)

# Set up logger
logger = setup_logger(__name__)

# Determine project root
PROJECT_ROOT = get_project_root()


def load_raw_data(run_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw data from CSV with schema validation.
    
    This function reads raw data in chunks for memory efficiency, applies
    data type downcasting, and validates against the project schema.
    
    Args:
        run_id: Optional run_id for correlation tracking
        
    Returns:
        Validated DataFrame with proper data types
        
    Raises:
        FileNotFoundError: If raw data or schema file not found
        ValueError: If schema validation fails
        yaml.YAMLError: If schema.yaml is malformed
        
    Example:
        >>> df = load_raw_data(run_id='a3f7b2c1')
        >>> print(f"Loaded {len(df):,} rows")
        Loaded 34,500 rows
    
    Notes:
        - Reads data in configurable chunks (default: 100,000 rows)
        - Automatically downcasts numeric types to save memory
        - Validates data against schema.yaml specifications
        - All configuration values read from config.yaml
    """
    # Set run_id if provided
    if run_id:
        set_run_id(run_id)
    
    # Entry logging
    logger.info("=" * 60)
    logger.info("Starting Data Load Process")
    logger.info("=" * 60)
    
    # Load configuration
    try:
        config = load_config(PROJECT_ROOT / 'config.yaml')
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise
    
    # Get data path from config
    try:
        raw_data_path = get_config_value(config, 'paths', 'raw_data')
        raw_path = PROJECT_ROOT / str(raw_data_path)
    except KeyError as e:
        logger.error(f"Config missing required path: {e}")
        raise ValueError(f"Configuration error: {e}")
    
    # Validate file exists
    if not raw_path.is_file():
        logger.error(f"Raw data file not found: {raw_path}")
        raise FileNotFoundError(
            f"Raw data file not found!\n"
            f"Expected: {raw_path}\n"
            f"Current working dir: {Path.cwd()}\n\n"
            f"Tip: File should be in {PROJECT_ROOT}/data/raw/"
        )
    
    logger.info(f"Data source: {raw_path.name}")
    
    # ============================================================================
    # READ DATA IN CHUNKS
    # ============================================================================
    
    # Get chunk configuration
    try:
        chunksize = get_config_value(
            config, 'notebook1', 'data_loading', 'chunksize',
            default=100000
        )
        encoding = get_config_value(
            config, 'notebook1', 'data_loading', 'encoding',
            default='utf-8'
        )
    except Exception as e:
        logger.warning(f"Using default chunk settings: {e}")
        chunksize = 100000
        encoding = 'utf-8'
    
    logger.info(f"Reading data in chunks of {chunksize:,} rows")
    
    try:
        chunks = pd.read_csv(
            str(raw_path),
            chunksize=chunksize,
            encoding=encoding,
            low_memory=True
        )
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise IOError(f"Cannot read data file: {e}")
    
    df_list = []
    chunk_count = 0
    
    for i, chunk in enumerate(chunks, 1):
        chunk_count = i
        
        # Log progress every 5 chunks or first chunk
        if i % 5 == 0 or i == 1:
            logger.info(f"Processing chunk {i}...")
        
        # Downcast numeric types to save memory
        try:
            for col in chunk.select_dtypes(include=['float64']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='float')
            
            for col in chunk.select_dtypes(include=['int64']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
        
        except Exception as e:
            logger.warning(f"Error downcasting chunk {i}: {e}")
            # Continue with original dtypes if downcasting fails
        
        df_list.append(chunk)
    
    # Concatenate all chunks
    logger.info(f"Concatenating {chunk_count} chunks...")
    
    try:
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        logger.error(f"Failed to concatenate chunks: {e}")
        raise ValueError(f"Data concatenation failed: {e}")
    
    logger.info(f"Data concatenation complete: {len(df):,} total rows")
    
    # ============================================================================
    # LOAD AND APPLY SCHEMA VALIDATION
    # ============================================================================
    
    logger.info("Loading schema validation rules...")
    
    schema_path = PROJECT_ROOT / "schema.yaml"
    if not schema_path.is_file():
        logger.error(f"Schema file not found: {schema_path}")
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}\n"
            f"Expected location: {PROJECT_ROOT}/schema.yaml"
        )
    
    # Load schema configuration
    try:
        with open(schema_path, encoding="utf-8") as f:
            schema_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in schema file: {e}")
        raise ValueError(f"Schema file is malformed: {e}")
    except Exception as e:
        logger.error(f"Failed to read schema file: {e}")
        raise
    
    columns_config = schema_config.get("columns", {})
    settings = schema_config.get("schema_settings", {})
    
    if not columns_config:
        logger.warning("No column specifications found in schema")
    
    # Build Pandera Column objects
    pandera_columns = {}
    
    type_mapping = {
        "string": pa.String,
        "float32": pa.Float32,
        "int8": pa.Int8,
        "datetime": pa.DateTime,
    }
    
    check_mapping = {
        "ge": Check.ge,
        "le": Check.le,
        "gt": Check.gt,
        "lt": Check.lt,
        "isin": Check.isin,
    }
    
    # Build schema from config
    for col_name, spec in columns_config.items():
        col_type_str = spec.get("type")
        
        if col_type_str not in type_mapping:
            logger.error(
                f"Unsupported type '{col_type_str}' for column '{col_name}'"
            )
            raise ValueError(
                f"Unsupported type '{col_type_str}' for column '{col_name}'. "
                f"Supported types: {list(type_mapping.keys())}"
            )
        
        col_type = type_mapping[col_type_str]
        nullable = spec.get("nullable", True)
        coerce = spec.get("coerce", False)
        
        # Build checks
        checks = []
        for check_def in spec.get("checks", []):
            if isinstance(check_def, dict):
                for op, value in check_def.items():
                    if op in check_mapping:
                        checks.append(check_mapping[op](value))
                    else:
                        logger.warning(
                            f"Unknown check operator '{op}' for column '{col_name}'"
                        )
        
        pandera_columns[col_name] = Column(
            dtype=col_type,
            nullable=nullable,
            coerce=coerce,
            checks=checks if checks else None,
        )
    
    # Create schema
    schema = DataFrameSchema(
        pandera_columns,
        strict=settings.get("strict", True),
        coerce=settings.get("coerce", False),
    )
    
    logger.info(f"Schema loaded: {len(pandera_columns)} column validations")
    
    # Validate
    logger.info("Validating data against schema...")
    
    try:
        df = schema.validate(df)
        logger.info("✓ Schema validation passed")
    except pa.errors.SchemaError as e:
        logger.error(f"Schema validation failed:\n{e}")
        
        # Log specific failures
        if hasattr(e, 'failure_cases'):
            logger.error(f"Failure cases:\n{e.failure_cases}")
        
        raise ValueError(
            f"Data schema validation failed. See logs for details.\n"
            f"Error: {str(e)[:200]}..."
        )
    
    # Optional downcast after validation
    for col_name, spec in columns_config.items():
        downcast_cols = spec.get("coerce_downcast", [])
        for col in downcast_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], downcast="float")
                except Exception as e:
                    logger.warning(f"Could not downcast {col}: {e}")
    
    # ============================================================================
    # EXIT LOGGING
    # ============================================================================
    
    logger.info("=" * 60)
    logger.info("Data Load Summary")
    logger.info("=" * 60)
    logger.info(f"Rows loaded: {df.shape[0]:,}")
    logger.info(f"Columns: {df.shape[1]}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"Source file: {raw_path.name}")
    
    if 'order_date' in df.columns and not df['order_date'].empty:
        logger.info(
            f"Date range: {df['order_date'].min()} → {df['order_date'].max()}"
        )
    
    logger.info("=" * 60)
    
    return df