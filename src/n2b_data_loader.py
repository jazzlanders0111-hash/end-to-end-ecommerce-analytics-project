# src/n2b_data_loader.py
"""
n2b_data_loader.py - Data Loading & Validation for Notebook 02

This module handles loading the processed data from Notebook 01 and performs
validation checks to ensure data integrity.

Functions:
    load_enhanced_data() - Load the enhanced dataset from parquet
    validate_data_integrity() - Validate data hasn't changed using hash check
    validate_required_columns() - Check all required columns exist

Usage:
    from n2b_data_loader import load_enhanced_data
    df = load_enhanced_data()
"""

import pandas as pd
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
import logging
from n2a_utils import (
    get_project_root,
    setup_logger,
    validate_dataframe_columns,
    print_section_header,
    get_config
)

# Setup logger
logger = setup_logger(__name__)

# Load configuration
_CONFIG = get_config()
_NB2_CONFIG: Dict[str, Any] = _CONFIG.get('notebook2', {})
_DATA_VALIDATION: Dict[str, Any] = _NB2_CONFIG.get('data_validation', {})

# ============================================================================
# Required Columns (from config)
# ============================================================================

REQUIRED_COLUMNS: List[str] = _DATA_VALIDATION.get('required_columns', [
    'order_id', 'customer_id', 'product_id', 'category', 'price',
    'discount', 'quantity', 'payment_method', 'order_date',
    'delivery_time_days', 'region', 'returned',
    'total_amount', 'profit_margin',
])

EXPECTED_HASH: Optional[str] = _DATA_VALIDATION.get('expected_hash', None)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_enhanced_data(
    filepath: Optional[Path] = None,
    validate: bool = True,
    expected_hash: Optional[str] = None
) -> pd.DataFrame:
    """
    Load the enhanced dataset from Notebook 01 with validation.

    Args:
        filepath: Path to enhanced_df.parquet (auto-detected if None)
        validate: Whether to perform validation checks
        expected_hash: Expected data hash for integrity check (uses config if None)

    Returns:
        pd.DataFrame: Loaded and validated dataset

    Raises:
        FileNotFoundError: If data file not found
        ValueError: If validation fails

    Example:
        >>> df = load_enhanced_data()
        >>> df = load_enhanced_data(expected_hash='819fa403')
    """
    logger.info("=" * 60)
    logger.info("Loading Enhanced Dataset from Notebook 01")
    logger.info("=" * 60)

    try:
        filepath_path: Path
        if filepath is None:
            project_root = get_project_root()
            enhanced_path = _CONFIG.get('paths', {}).get(
                'enhanced_df', 'data/processed/enhanced_df.parquet'
            )
            filepath_path = project_root / enhanced_path
        else:
            filepath_path = Path(filepath)

        if not filepath_path.exists():
            error_msg = (
                f"Enhanced data file not found: {filepath_path}\n"
                "Please run Notebook 01 (Data Wrangling) first to generate the processed data."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading from: {filepath_path}")
        try:
            df = pd.read_parquet(filepath_path)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        logger.info(f"Date range: {df['order_date'].min().date()} → {df['order_date'].max().date()}")
        logger.info(f"Unique customers: {df['customer_id'].nunique():,}")
        logger.info(f"Total revenue: ${df['total_amount'].sum():,.2f}")

        if validate:
            logger.info("Running validation checks:")

            if not validate_required_columns(df):
                raise ValueError("Required columns missing from dataset")

            hash_to_check = expected_hash if expected_hash is not None else EXPECTED_HASH
            if hash_to_check:
                actual_hash = validate_data_integrity(df, expected_hash=hash_to_check)
                if actual_hash != hash_to_check:
                    logger.warning("⚠ Data hash mismatch - data may have changed")
            else:
                validate_data_integrity(df)

        logger.info("=" * 60)
        logger.info("Data Loading Complete")
        logger.info("=" * 60)

        return df

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error in load_enhanced_data: {e}")
        raise


def validate_required_columns(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains all required columns.

    Args:
        df: DataFrame to validate

    Returns:
        bool: True if all required columns present
    """
    try:
        return validate_dataframe_columns(df, REQUIRED_COLUMNS, logger)
    except Exception as e:
        logger.error(f"validate_required_columns failed: {e}")
        return False


def validate_data_integrity(
    df: pd.DataFrame,
    expected_hash: Optional[str] = None
) -> str:
    """
    Validate data integrity using a hash of key characteristics.

    Creates a fingerprint of the data based on row count, total revenue,
    and date range.

    Args:
        df: DataFrame to validate
        expected_hash: Expected hash value (8 chars)

    Returns:
        str: Computed hash of the data

    Example:
        >>> hash1 = validate_data_integrity(df)
        >>> hash2 = validate_data_integrity(df, expected_hash=hash1)
    """
    try:
        data_signature = (
            f"{len(df)}_"
            f"{df['total_amount'].sum():.2f}_"
            f"{df['order_date'].min()}_"
            f"{df['order_date'].max()}"
        )

        data_hash = hashlib.md5(data_signature.encode()).hexdigest()[:8]

        logger.info("Data Integrity Check:")
        logger.info(f"Signature hash: {data_hash}")
        logger.info(f"Rows: {len(df):,}")
        logger.info(f"Total revenue: ${df['total_amount'].sum():,.2f}")
        logger.info(
            f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}"
        )

        if expected_hash:
            if data_hash == expected_hash:
                logger.info("✓ Data integrity verified")
            else:
                logger.warning("⚠ Hash mismatch!")
                logger.warning(f"Expected: {expected_hash}")
                logger.warning(f"Got: {data_hash}")
                logger.warning("Data may have changed since last run")
        else:
            logger.info("First run - use this hash for future validation:")
            logger.info(f"expected_hash = '{data_hash}'")

        return data_hash

    except KeyError as e:
        logger.error(f"Missing column for integrity check: {e}")
        raise
    except Exception as e:
        logger.error(f"validate_data_integrity failed: {e}")
        raise


# ============================================================================
# Data Summary Functions
# ============================================================================

def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a formatted summary of the dataset.

    Args:
        df: DataFrame to summarize
    """
    try:
        print_section_header("DATASET SUMMARY")

        print(f"\n📊 Dimensions:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")

        print(f"\n📅 Date Coverage:")
        print(f"  Start: {df['order_date'].min().date()}")
        print(f"  End: {df['order_date'].max().date()}")
        print(f"  Duration: {(df['order_date'].max() - df['order_date'].min()).days} days")

        print(f"\n👥 Customers:")
        print(f"  Unique customers: {df['customer_id'].nunique():,}")
        print(f"  Avg orders/customer: {len(df) / df['customer_id'].nunique():.2f}")

        print(f"\n💰 Revenue:")
        print(f"  Total revenue: ${df['total_amount'].sum():,.2f}")
        print(f"  Average order value: ${df['total_amount'].mean():.2f}")
        print(f"  Median order value: ${df['total_amount'].median():.2f}")

        print(f"\n📦 Products:")
        print(f"  Unique products: {df['product_id'].nunique():,}")
        print(f"  Categories: {df['category'].nunique()}")
        print(f"  Category breakdown: {', '.join(df['category'].unique())}")

        print(f"\n↩️ Returns:")
        print(f"  Return rate: {df['returned'].mean():.2%}")
        print(f"  Returned orders: {df['returned'].sum():,}")

        print(f"\n🗺️ Regions:")
        regions = df['region'].value_counts()
        for region, count in regions.items():
            pct = count / len(df) * 100
            print(f"  {region}: {count:,} ({pct:.1f}%)")

    except Exception as e:
        logger.error(f"print_data_summary failed: {e}")
        raise


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a data quality report.

    Args:
        df: DataFrame to analyze

    Returns:
        dict: Quality metrics including total rows, missing values,
              duplicates, return rate, and a quality score out of 100
    """
    try:
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'date_range_days': (df['order_date'].max() - df['order_date'].min()).days,
            'unique_customers': df['customer_id'].nunique(),
            'unique_products': df['product_id'].nunique(),
            'total_revenue': df['total_amount'].sum(),
            'return_rate': df['returned'].mean(),
        }

        issues = []

        if report['missing_values'] > 0:
            issues.append(f"{report['missing_values']} missing values found")

        if report['duplicate_rows'] > 0:
            issues.append(f"{report['duplicate_rows']} duplicate rows found")

        if report['return_rate'] > 0.15:
            issues.append(f"High return rate: {report['return_rate']:.2%}")

        report['quality_issues'] = issues
        report['quality_score'] = 100 - len(issues) * 10

        return report

    except Exception as e:
        logger.error(f"get_data_quality_report failed: {e}")
        raise


# ============================================================================
# Main Execution (for testing)
# ============================================================================

__all__ = [
    'REQUIRED_COLUMNS',
    'EXPECTED_HASH',
    'load_enhanced_data',
    'validate_required_columns',
    'validate_data_integrity',
    'print_data_summary',
    'get_data_quality_report',
]

if __name__ == '__main__':
    print("n2b_data_loader.py - Testing Data Loading")
    print("=" * 60)

    try:
        df = load_enhanced_data(validate=True)
        print_data_summary(df)

        print_section_header("DATA QUALITY REPORT")
        report = get_data_quality_report(df)
        for key, value in report.items():
            if key != 'quality_issues':
                print(f"{key}: {value}")

        if report['quality_issues']:
            print("\n⚠️ Quality Issues:")
            for issue in report['quality_issues']:
                print(f"  - {issue}")
        else:
            print("\n✓ No quality issues found")

        print(f"\nQuality Score: {report['quality_score']}/100")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
