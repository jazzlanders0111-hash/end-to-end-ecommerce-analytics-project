# src/main.py

"""
Main entry point for the E-Commerce Analytics Project.

This module initializes the project configuration and logging system.
It serves as the starting point for running the data pipeline.

Usage:
    python src/main.py
    
Design Principles:
- Load configuration before initializing logger
- Set run_id for correlation tracking
- Comprehensive error handling with exit codes
- Clear logging of initialization steps
"""

# Standard library
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import hashlib

# Local modules
from n1a_utils import load_config, setup_logger, set_run_id, get_run_id


def initialize_project() -> tuple[Dict[str, Any], Any, str]:
    """
    Initialize project configuration and logging.
    
    Returns:
        Tuple of (config_dict, logger_instance, run_id)
        
    Raises:
        SystemExit: If initialization fails
    """
    # ============================================================================
    # 1. GENERATE RUN ID (before any logging)
    # ============================================================================
    run_id = set_run_id()
    
    # ============================================================================
    # 2. LOAD CONFIGURATION (before logger setup)
    # ============================================================================
    config_path = Path(__file__).parent.parent / 'config.yaml'
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        # Cannot use logger yet, print to stderr
        print(f"FATAL ERROR: Config file not found: {config_path}", file=sys.stderr)
        print(f"Current working directory: {Path.cwd()}", file=sys.stderr)
        print(f"Expected config at: {config_path.absolute()}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # ============================================================================
    # 3. SETUP LOGGER (after config loaded)
    # ============================================================================
    try:
        # Get logging config values
        log_level_str = config.get('logging', {}).get('level', 'INFO')
        log_file = config.get('logging', {}).get('file', 'project.log')
        
        # Convert log level string to logging constant
        import logging
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        # Create logger
        logger = setup_logger(
            __name__,
            log_file=log_file,
            level=log_level
        )
        
    except Exception as e:
        print(f"FATAL ERROR: Failed to setup logger: {e}", file=sys.stderr)
        sys.exit(1)
    
    # ============================================================================
    # 4. LOG INITIALIZATION SUCCESS
    # ============================================================================
    logger.info("=" * 60)
    logger.info("E-Commerce Analytics Project - Initialization")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Logging level: {log_level_str}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    return config, logger, run_id


def main():
    """
    Main entry point for the application.
    
    Runs the full NB01 data pipeline:
    1. Load raw data
    2. Analyze missing data patterns
    3. Clean data
    4. Build RFM features
    5. Run sanity checks
    6. Save processed data
    """
    # Initialize project
    try:
        config, logger, run_id = initialize_project()
    except SystemExit:
        raise
    except Exception as e:
        print(f"FATAL ERROR during initialization: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        logger.info("Application initialized successfully")
        logger.info("Starting NB01 data pipeline...")

        # ── STEP 1: LOAD RAW DATA ────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1: Loading raw data")
        logger.info("=" * 60)

        from n1b_data_loader import load_raw_data
        df_raw = load_raw_data(run_id=run_id)

        logger.info(f"Loaded {len(df_raw):,} rows x {df_raw.shape[1]} columns")

        # ── STEP 2: MISSING DATA ANALYSIS ───────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 2: Analyzing missing data patterns")
        logger.info("=" * 60)

        from n1b_missing_data_analysis import analyze_missing_patterns
        missing_results = analyze_missing_patterns(
            df=df_raw,
            config=config,
            plot=False,     # No charts — nobody watching in main.py
            verbose=True
        )

        missing_pct = missing_results.get('summary', {}).get('missing_percentage', 0)
        if missing_pct > 0:
            logger.warning(f"Missing data detected: {missing_pct:.2f}%")
        else:
            logger.info("Missing data check passed: 0% missing")

        # ── STEP 3: CLEAN DATA ───────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3: Cleaning data")
        logger.info("=" * 60)

        from n1c_preprocessing import clean_data
        df_clean = clean_data(
            df=df_raw,
            verbose=True,
            run_id=run_id
        )

        rows_removed = len(df_raw) - len(df_clean)
        removal_pct = rows_removed / len(df_raw) * 100
        logger.info(f"Cleaning complete: {len(df_clean):,} rows retained "
                    f"({rows_removed:,} removed, {removal_pct:.2f}%)")

        # Check data loss is within acceptable threshold
        max_loss = config.get('notebook1', {}).get(
            'business_rules', {}
        ).get('max_acceptable_data_loss_pct', 5.0)

        if removal_pct > max_loss:
            logger.warning(
                f"Data loss {removal_pct:.2f}% exceeds threshold of {max_loss:.0f}%. "
                f"Investigate cleaning logic."
            )

        # ── STEP 4: BUILD RFM FEATURES ───────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 4: Building RFM features")
        logger.info("=" * 60)

        from n1d_rfm_features import build_rfm_features, clear_old_cache

        # Clear stale cache before building
        clear_old_cache(ttl_hours=config.get('rfm', {}).get('cache_ttl_hours', 24))

        # Generate config hash for cache key (same method as notebook)
        rfm_config_str = str(config.get('rfm', {}))
        config_hash = hashlib.md5(rfm_config_str.encode()).hexdigest()[:8]

        rfm_df = build_rfm_features(
            df=df_clean,
            config_hash=config_hash,
            verbose=True,
            run_id=run_id,
            use_cache=True
        )

        logger.info(f"RFM complete: {len(rfm_df):,} customers x {rfm_df.shape[1]} features")

        # ── STEP 5: SANITY CHECKS ────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 5: Running sanity checks")
        logger.info("=" * 60)

        from n1f_sanity_check_wrapper import comprehensive_data_check
        check_results = comprehensive_data_check(
            df_clean=df_clean,
            rfm_df=rfm_df,
            config=config,
            run_id=run_id,
            verbose=True
        )

        logger.info(
            f"Sanity checks: {check_results['passed_checks']}/{check_results['total_checks']} passed"
        )

        if check_results['failed_checks'] > 0:
            logger.warning(
                f"{check_results['failed_checks']} check(s) failed — "
                f"review logs before using output files"
            )
        if check_results['warning_checks'] > 0:
            logger.warning(f"{check_results['warning_checks']} warning(s) raised")

        # ── STEP 6: SAVE PROCESSED DATA ──────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 6: Saving processed data")
        logger.info("=" * 60)

        from n1g_data_saver import save_processed_data
        saved_files = save_processed_data(
            df=df_clean,
            rfm_df=rfm_df,
            run_id=run_id,
            verbose=True
        )

        # ── PIPELINE COMPLETE ─────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Run ID   : {run_id}")
        logger.info(f"Rows     : {len(df_clean):,} transactions")
        logger.info(f"Customers: {len(rfm_df):,}")
        for name, filepath in saved_files.items():
            size_mb = filepath.stat().st_size / 1024 / 1024
            logger.info(f"Saved    : {filepath.name}  ({size_mb:.2f} MB)")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()