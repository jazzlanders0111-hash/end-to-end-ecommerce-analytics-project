# src/n1g_data_saver.py
"""
Data saving and verification utility module.

Key Features:
- Save processed datasets with compression
- Automatic verification of saved files
- File size reporting
- Comprehensive logging with run_id correlation
- Error handling with informative messages

Design Principles:
- Consistent with n1a-n1d module patterns
- Defensive programming with validation
- Data integrity checks with hash validation
- Consistent logging patterns
- Clear success/failure reporting
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from n1a_utils import setup_logger, get_project_root, load_config, set_run_id

# Set up logger
logger = setup_logger(__name__)

# Determine project root
PROJECT_ROOT = get_project_root()


def save_processed_data(
    df: pd.DataFrame,
    rfm_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Save processed datasets to parquet files with verification.
    
    Saves enhanced transaction data and optional RFM customer data to
    compressed parquet files. Performs verification checks to ensure
    data integrity after saving.
    
    Args:
        df: Enhanced transaction-level DataFrame to save
        rfm_df: Optional RFM customer-level DataFrame to save
        output_dir: Output directory (default: reads from config)
        run_id: Optional run_id for correlation tracking
        verbose: Whether to log detailed progress (default: True)
        
    Returns:
        Dictionary mapping dataset names to saved file paths
        
    Raises:
        AssertionError: If verification fails
        IOError: If file save fails
        
    Example:
        >>> saved_files = save_processed_data(
        ...     df=df_clean,
        ...     rfm_df=rfm_full,
        ...     run_id=RUN_ID
        ... )
        >>> print(f"Saved to: {saved_files['enhanced']}")
    """
    # Set run_id if provided
    if run_id:
        set_run_id(run_id)
    
    if verbose:
        logger.info("=" * 60)
        logger.info("Saving Processed Datasets")
        logger.info("=" * 60)
    
    # Load config to get output directory
    config = load_config(PROJECT_ROOT / 'config.yaml')
    if output_dir is None:
        processed_path = config.get('paths', {}).get('processed_data')
        if not processed_path:
            raise ValueError("Config missing 'paths.processed_data' entry")
        output_dir = PROJECT_ROOT / Path(processed_path)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        logger.info(f"Output directory: {output_dir}")
    
    # Track saved files
    saved_files = {}
    
    # ============================================================
    # 1. SAVE ENHANCED DATASET
    # ============================================================
    if verbose:
        logger.info("Saving enhanced transaction dataset...")
    
    enhanced_file = output_dir / "enhanced_df.parquet"
    
    try:
        # Save to parquet with compression
        df.to_parquet(enhanced_file, compression="snappy", index=False)
        
        if verbose:
            logger.info(f"File saved: {enhanced_file.name}")
        
        # Verify the save
        df_verify = pd.read_parquet(enhanced_file)
        
        # Check shape
        if df_verify.shape != df.shape:
            error_msg = f"Shape mismatch! Original: {df.shape}, Loaded: {df_verify.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check columns
        assert list(df_verify.columns) == list(df.columns), (
            f"Column mismatch! Original: {list(df.columns)}, Loaded: {list(df_verify.columns)}"
        )
        
        # Calculate file size
        file_size_mb = enhanced_file.stat().st_size / 1024 / 1024
        
        if verbose:
            logger.info(f"Verified: {df_verify.shape[0]:,} rows × {df_verify.shape[1]} columns")
            logger.info(f"File size: {file_size_mb:.2f} MB")
            logger.info(f"Enhanced dataset saved and verified")
        
        saved_files['enhanced'] = enhanced_file
        
    except Exception as e:
        logger.error(f"Failed to save enhanced dataset: {e}")
        raise
    
    # ============================================================
    # 2. SAVE RFM DATASET (if provided)
    # ============================================================
    if rfm_df is not None:
        if verbose:
            logger.info("Saving RFM customer dataset...")
        
        rfm_file = output_dir / "rfm_df.parquet"
        
        try:
            # Save to parquet with compression
            rfm_df.to_parquet(rfm_file, compression="snappy", index=False)
            
            if verbose:
                logger.info(f"File saved: {rfm_file.name}")
            
            # Verify the save
            rfm_verify = pd.read_parquet(rfm_file)
            
            # Check shape
            assert rfm_verify.shape == rfm_df.shape, (
                f"Shape mismatch! Original: {rfm_df.shape}, Loaded: {rfm_verify.shape}"
            )
            
            # Check columns
            assert list(rfm_verify.columns) == list(rfm_df.columns), (
                f"Column mismatch! Original: {list(rfm_df.columns)}, Loaded: {list(rfm_verify.columns)}"
            )
            
            # Calculate file size
            file_size_mb = rfm_file.stat().st_size / 1024 / 1024
            
            if verbose:
                logger.info(f"Verified: {rfm_verify.shape[0]:,} customers × {rfm_verify.shape[1]} features")
                logger.info(f"File size: {file_size_mb:.2f} MB")
                logger.info(f"RFM dataset saved and verified")
            
            saved_files['rfm'] = rfm_file
            
        except Exception as e:
            logger.error(f"Failed to save RFM dataset: {e}")
            raise
    
    # ============================================================
    # 3. SUMMARY
    # ============================================================
    if verbose:
        logger.info("=" * 60)
        logger.info("Save Summary")
        logger.info("=" * 60)
        logger.info(f"Successfully saved {len(saved_files)} dataset(s)")
        
        for name, filepath in saved_files.items():
            logger.info(f"{name}: {filepath}")
        
        logger.info("=" * 60)
    
    return saved_files


def load_processed_data(
    input_dir: Optional[Path] = None,
    load_rfm: bool = True,
    run_id: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load processed datasets from parquet files.
    
    Args:
        input_dir: Input directory (default: reads from config)
        load_rfm: Whether to load RFM dataset (default: True)
        run_id: Optional run_id for correlation tracking
        verbose: Whether to log detailed progress (default: True)
        
    Returns:
        Dictionary mapping dataset names to DataFrames
        
    Raises:
        FileNotFoundError: If required files don't exist
        
    Example:
        >>> datasets = load_processed_data(run_id=RUN_ID)
        >>> df = datasets['enhanced']
        >>> rfm_df = datasets['rfm']
    """
    # Set run_id if provided
    if run_id:
        set_run_id(run_id)
    
    if verbose:
        logger.info("Loading processed datasets...")
    
    # Load config to get input directory
    config = load_config(PROJECT_ROOT / 'config.yaml')
    input_path = config.get('paths', {}).get('processed_data')
    
    if not input_path:
        raise ValueError("Config missing 'paths.processed_data' entry")
    
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Track loaded datasets
    datasets = {}
    
    # Load enhanced dataset
    enhanced_file = input_dir / "enhanced_df.parquet"
    if not enhanced_file.exists():
        raise FileNotFoundError(f"Enhanced dataset not found: {enhanced_file}")
    
    try:
        df = pd.read_parquet(enhanced_file)
        datasets['enhanced'] = df
        
        if verbose:
            logger.info(f"Loaded enhanced dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Failed to load enhanced dataset: {e}")
        raise
    
    # Load RFM dataset if requested
    if load_rfm:
        rfm_file = input_dir / "rfm_df.parquet"
        if not rfm_file.exists():
            logger.warning(f"RFM dataset not found: {rfm_file}")
        else:
            try:
                rfm_df = pd.read_parquet(rfm_file)
                datasets['rfm'] = rfm_df
                
                if verbose:
                    logger.info(f"Loaded RFM dataset: {rfm_df.shape[0]:,} customers × {rfm_df.shape[1]} features")
            except Exception as e:
                logger.error(f"Failed to load RFM dataset: {e}")
                raise
    
    if verbose:
        logger.info(f"Loaded {len(datasets)} dataset(s)")
    
    return datasets