"""
n3j_statistical_tests.py - Statistical Validation of Segments

Comprehensive statistical validation to ensure segments are significantly
different and business-meaningful.

Key Features:
- ANOVA tests for continuous features
- Post-hoc pairwise comparisons (Tukey HSD)
- Effect size calculations (Cohen's d, eta-squared)
- Chi-square tests for categorical features
- Non-parametric alternatives (Kruskal-Wallis)
- Homogeneity of variance tests
- Comprehensive validation reports
"""

from scipy import stats
from scipy.stats import f_oneway, kruskal, chi2_contingency, levene
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import scikit_posthocs as sp

from n3a_utils import setup_logger, load_config

logger = setup_logger(__name__)
_config = load_config()


def test_segment_differences_anova(
    segment_df: pd.DataFrame,
    feature: str,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    ANOVA test: Are segments significantly different on this feature?
    
    H0: All segment means are equal
    H1: At least one segment mean differs
    
    Args:
        segment_df: DataFrame with cluster assignments and features
        feature: Feature to test
        alpha: Significance level
        
    Returns:
        Tuple of (f_statistic, p_value)
    """
    groups = [
        group[feature].dropna().values
        for name, group in segment_df.groupby('cluster')
    ]
    
    # Check if we have enough data
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        logger.warning(f"Insufficient data for ANOVA on {feature}")
        return np.nan, np.nan
    
    try:
        f_stat, p_value = f_oneway(*groups)
    except Exception as e:
        logger.error(f"ANOVA failed for {feature}: {e}")
        return np.nan, np.nan
    
    logger.info(f"\nANOVA Test for {feature}:")
    logger.info(f"  F-statistic: {f_stat:.2f}")
    logger.info(f"  P-value: {p_value:.4f}")
    
    if p_value < alpha:
        logger.info(f"  [PASS] Segments are significantly different (p < {alpha})")
    else:
        logger.info(f"  [FAIL] Segments are NOT significantly different (p >= {alpha})")
    
    return f_stat, p_value


def test_segment_differences_kruskal(
    segment_df: pd.DataFrame,
    feature: str,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Kruskal-Wallis H-test (non-parametric alternative to ANOVA).
    
    Use when data is not normally distributed or has unequal variances.
    
    H0: All segment distributions are equal
    H1: At least one segment distribution differs
    
    Args:
        segment_df: DataFrame with cluster assignments and features
        feature: Feature to test
        alpha: Significance level
        
    Returns:
        Tuple of (h_statistic, p_value)
    """
    groups = [
        group[feature].dropna().values
        for name, group in segment_df.groupby('cluster')
    ]
    
    # Check if we have enough data
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        logger.warning(f"Insufficient data for Kruskal-Wallis on {feature}")
        return np.nan, np.nan
    
    try:
        h_stat, p_value = kruskal(*groups)
    except Exception as e:
        logger.error(f"Kruskal-Wallis failed for {feature}: {e}")
        return np.nan, np.nan
    
    logger.info(f"Kruskal-Wallis Test for {feature}:")
    logger.info(f"H-statistic: {h_stat:.2f}")
    logger.info(f"P-value: {p_value:.4f}")
    
    if p_value < alpha:
        logger.info(f"[PASS] Segments have significantly different distributions (p < {alpha})")
    else:
        logger.info(f"[FAIL] Segments do NOT have significantly different distributions")
    
    return h_stat, p_value


def calculate_effect_size_cohens_d(
    group1: pd.Series,
    group2: pd.Series
) -> float:
    """
    Calculate Cohen's d effect size between two segments.
    
    Interpretation:
    - |d| < 0.2: Small effect
    - |d| < 0.5: Medium effect
    - |d| >= 0.8: Large effect
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        Cohen's d effect size
    """
    try:
        mean1, mean2 = group1.mean(), group2.mean()
        std1, std2 = group1.std(), group2.std()
        n1, n2 = len(group1), len(group2)
    
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
    
        if pooled_std == 0:
            return 0.0
    
        cohens_d = (mean1 - mean2) / pooled_std
    
        return cohens_d
    except Exception as e:
        logger.error(f"calculate_effect_size_cohens_d failed: {e}")
        raise


def calculate_eta_squared(segment_df: pd.DataFrame, feature: str) -> float:
    """
    Calculate eta-squared (η²) effect size for ANOVA.
    
    Interpretation:
    - η² < 0.01: Small effect
    - η² < 0.06: Medium effect
    - η² >= 0.14: Large effect
    
    Args:
        segment_df: DataFrame with cluster assignments and features
        feature: Feature to test
        
    Returns:
        Eta-squared value
    """
    try:
        groups = segment_df.groupby('cluster')[feature]
    
        # Between-group variance
        grand_mean = segment_df[feature].mean()
        ss_between = sum(
            len(group) * (group.mean() - grand_mean)**2
            for _, group in groups
        )
    
        # Total variance
        ss_total = sum((segment_df[feature] - grand_mean)**2)
    
        if ss_total == 0:
            return 0.0
    
        eta_squared = ss_between / ss_total
    
        return eta_squared
    except Exception as e:
        logger.error(f"calculate_eta_squared failed: {e}")
        raise


def posthoc_dunn(
    segment_df: pd.DataFrame,
    feature: str,
    alpha: float = 0.05,
    p_adjust: str = 'bonferroni'
) -> Optional[pd.DataFrame]:
    """
    Perform Dunn's post-hoc test for pairwise comparisons after Kruskal-Wallis.

    Dunn's test is the methodologically correct non-parametric follow-up
    to Kruskal-Wallis. It operates on ranks, consistent with K-W assumptions,
    unlike Tukey HSD which assumes normality and is designed for ANOVA.

    Args:
        segment_df: DataFrame with cluster assignments and features
        feature: Feature to test
        alpha: Significance level
        p_adjust: Multiple comparison correction method (default: 'bonferroni')
                  Options: 'bonferroni', 'holm', 'fdr_bh', 'sidak', None

    Returns:
        DataFrame with pairwise p-values (symmetric matrix) or None if test fails
    """
    try:
        import scikit_posthocs as sp

        result = sp.posthoc_dunn(
            segment_df,
            val_col=feature,
            group_col='cluster',
            p_adjust=p_adjust
        )

        # Count and log significant pairs
        n_clusters = len(result)
        sig_pairs = 0
        total_pairs = 0
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                total_pairs += 1
                if result.iloc[i, j] < alpha:
                    sig_pairs += 1

        logger.info(f"Dunn's Post-hoc Results ({p_adjust} correction):")
        logger.info(f"Significant pairs: {sig_pairs}/{total_pairs}")

        return result

    except Exception as e:
        logger.error(f"Dunn's test failed for {feature}: {e}")
        return None


def test_variance_homogeneity(
    segment_df: pd.DataFrame,
    feature: str,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Levene's test for homogeneity of variance across segments.
    
    H0: All segments have equal variance
    H1: At least one segment has different variance
    
    Args:
        segment_df: DataFrame with cluster assignments and features
        feature: Feature to test
        alpha: Significance level
        
    Returns:
        Tuple of (w_statistic, p_value)
    """
    groups = [
        group[feature].dropna().values
        for name, group in segment_df.groupby('cluster')
    ]
    
    # Check if we have enough data
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        logger.warning(f"Insufficient data for Levene's test on {feature}")
        return np.nan, np.nan
    
    try:
        w_stat, p_value = levene(*groups)
    except Exception as e:
        logger.error(f"Levene's test failed for {feature}: {e}")
        return np.nan, np.nan
    
    logger.info(f"Levene's Test for {feature}:")
    logger.info(f"W-statistic: {w_stat:.2f}")
    logger.info(f"P-value: {p_value:.4f}")
    
    if p_value >= alpha:
        logger.info(f"[PASS] Variances are homogeneous (p >= {alpha})")
    else:
        logger.info(f"[WARN] Variances are NOT homogeneous - consider non-parametric tests")
    
    return w_stat, p_value


def test_categorical_independence(
    segment_df: pd.DataFrame,
    categorical_feature: str,
    alpha: float = 0.05
) -> Tuple[float, float, int]:
    """
    Chi-square test for independence between segments and categorical feature.
    
    H0: Segment and categorical feature are independent
    H1: Segment and categorical feature are associated
    
    Args:
        segment_df: DataFrame with cluster assignments and categorical feature
        categorical_feature: Categorical feature to test
        alpha: Significance level
        
    Returns:
        Tuple of (chi2_statistic, p_value, degrees_of_freedom)
    """
    contingency_table = pd.crosstab(
        segment_df['cluster'],
        segment_df[categorical_feature]
    )
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    except Exception as e:
        logger.error(f"Chi-square test failed for {categorical_feature}: {e}")
        return np.nan, np.nan, 0
    
    logger.info(f"Chi-Square Test for {categorical_feature}:")
    logger.info(f"Chi2-statistic: {chi2:.2f}")
    logger.info(f"P-value: {p_value:.4f}")
    logger.info(f"Degrees of freedom: {dof}")
    
    if p_value < alpha:
        logger.info(f"[PASS] Significant association (p < {alpha})")
    else:
        logger.info(f"[FAIL] No significant association")
    
    return chi2, p_value, dof


def validate_segment_quality(
    segment_df: pd.DataFrame,
    continuous_features: List[str],
    categorical_features: Optional[List[str]] = None,
    alpha: float = 0.05,
    use_nonparametric: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive statistical validation of customer segments.
    
    Use the clust_df DataFrame from prepare_clustering_features() merged with
    cluster labels, NOT the original rfm_df which may have NaN values.
    
    Performs multiple tests to ensure segments are:
    1. Significantly different on key metrics (ANOVA/Kruskal-Wallis)
    2. Have meaningful effect sizes (Cohen's d, eta-squared)
    3. Show categorical differences (Chi-square)
    4. Meet statistical assumptions (Levene's test)
    
    Args:
        segment_df: DataFrame with cluster assignments and clean features
        continuous_features: List of continuous features to test
        categorical_features: List of categorical features to test
        alpha: Significance level
        use_nonparametric: Use Kruskal-Wallis instead of ANOVA
        
    Returns:
        Dictionary with comprehensive validation results
    """
    
    # Verify we have cluster column
    try:
        if 'cluster' not in segment_df.columns:
            raise ValueError("segment_df must contain 'cluster' column")
    
        # Verify features exist
        missing_features = [f for f in continuous_features if f not in segment_df.columns]
        if missing_features:
            raise ValueError(f"Missing features in segment_df: {missing_features}")
    
        # Check for NaN values in features
        for feature in continuous_features:
            nan_count = segment_df[feature].isna().sum()
            if nan_count > 0:
                logger.warning(
                    f"Feature '{feature}' has {nan_count} NaN values. "
                    f"Statistical tests may be affected."
                )
    
        results: Dict[str, Any] = {
            'continuous_tests': {},
            'categorical_tests': {},
            'effect_sizes': {},
            'variance_tests': {},
            'posthoc_tests': {},
            'summary': {},
            'verdict': None
        }
    
        # Test continuous features
        for feature in continuous_features:
            # Logger moved to after test to avoid interrupting section header
        
            # 1. Test for variance homogeneity
            w_stat, p_levene = test_variance_homogeneity(segment_df, feature, alpha)
        
            # Log after first output to avoid interrupting section header
            logger.info(f"Testing feature: {feature}")
        
            results['variance_tests'][feature] = {
                'w_statistic': w_stat,
                'p_value': p_levene,
                'homogeneous': p_levene >= alpha if not np.isnan(p_levene) else False
            }
        
            # 2. Choose appropriate test
            if use_nonparametric or (not np.isnan(p_levene) and p_levene < alpha):
                # Use non-parametric test
                h_stat, p_value = test_segment_differences_kruskal(segment_df, feature, alpha)
                results['continuous_tests'][feature] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha if not np.isnan(p_value) else False
                }
            else:
                # Use parametric test
                f_stat, p_value = test_segment_differences_anova(segment_df, feature, alpha)
                results['continuous_tests'][feature] = {
                    'test': 'ANOVA',
                    'statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha if not np.isnan(p_value) else False
                }
        
            # 3. Calculate effect size
            eta_sq = calculate_eta_squared(segment_df, feature)
            results['effect_sizes'][feature] = {
                'eta_squared': eta_sq,
                'interpretation': (
                    'Large' if eta_sq >= 0.14 else
                    'Medium' if eta_sq >= 0.06 else
                    'Small'
                )
            }
        
            # 4. Post-hoc tests if significant (Dunn's — correct pairing for K-W)
            if results['continuous_tests'][feature]['significant']:
                dunn_df = posthoc_dunn(segment_df, feature, alpha)
                if dunn_df is not None:
                    results['posthoc_tests'][feature] = dunn_df

                    # Count significant pairs from the p-value matrix
                    n = len(dunn_df)
                    sig_pairs = sum(
                        1 for i in range(n) for j in range(i + 1, n)
                        if dunn_df.iloc[i, j] < alpha
                    )
                    total_pairs = n * (n - 1) // 2

                    logger.info(f"Post-hoc Results (Dunn, Bonferroni):")
                    logger.info(f"Significant pairs: {sig_pairs}/{total_pairs}")
    
        # Test categorical features
        if categorical_features:
            for feature in categorical_features:
                if feature in segment_df.columns:
                    logger.info(f"Testing categorical feature: {feature}")
                    chi2, p_value, dof = test_categorical_independence(segment_df, feature, alpha)
                
                    results['categorical_tests'][feature] = {
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'dof': dof,
                        'significant': p_value < alpha if not np.isnan(p_value) else False
                    }
    
        # Generate summary
        continuous_passed = sum(
            1 for test in results['continuous_tests'].values()
            if test['significant']
        )
        total_continuous = len(results['continuous_tests'])
    
        categorical_passed = sum(
            1 for test in results['categorical_tests'].values()
            if test['significant']
        )
        total_categorical = len(results['categorical_tests'])
    
        results['summary'] = {
            'continuous_tests_passed': continuous_passed,
            'continuous_tests_total': total_continuous,
            'categorical_tests_passed': categorical_passed,
            'categorical_tests_total': total_categorical,
            'pass_rate': continuous_passed / total_continuous if total_continuous > 0 else 0
        }
    
        # Determine verdict
        pass_rate = results['summary']['pass_rate']
    
        if pass_rate >= 0.8:
            results['verdict'] = 'PASSED'
            verdict_text = "[PASS] Segments are statistically valid"
        elif pass_rate >= 0.6:
            results['verdict'] = 'CONDITIONAL_PASS'
            verdict_text = "[WARN] Most tests passed but some concerns"
        else:
            results['verdict'] = 'FAILED'
            verdict_text = "[FAIL] Segments may not be sufficiently different"
    
        # Print summary
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Continuous Features: {continuous_passed}/{total_continuous} passed")
        if total_categorical > 0:
            logger.info(f"Categorical Features: {categorical_passed}/{total_categorical} passed")
        logger.info(f"Overall Pass Rate: {pass_rate:.1%}")
        logger.info(f"{verdict_text}")
        logger.info("=" * 60)
    
        logger.info(f"Validation complete: {results['verdict']}")
    
        return results
    except Exception as e:
        logger.error(f"validate_segment_quality failed: {e}")
        raise


def print_validation_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted validation report.
    
    Args:
        results: Results dictionary from validate_segment_quality()
    """
    try:
        print("=" * 80)
        print("STATISTICAL VALIDATION REPORT".center(80))
        print("=" * 80)
    
        # Continuous tests
        print("\n1. CONTINUOUS FEATURES")
        print("-" * 80)
    
        for feature, test_result in results['continuous_tests'].items():
            status = "[PASS]" if test_result['significant'] else "[FAIL]"
            test_name = test_result['test']
            p_val = test_result['p_value']
        
            print(f"\n{feature}:")
            print(f"  {status} {test_name}: p = {p_val:.4f}")
        
            # Effect size
            if feature in results['effect_sizes']:
                eta_sq = results['effect_sizes'][feature]['eta_squared']
                interp = results['effect_sizes'][feature]['interpretation']
                print(f"  Effect size: eta-squared = {eta_sq:.4f} ({interp})")
        
            # Variance homogeneity
            if feature in results['variance_tests']:
                var_test = results['variance_tests'][feature]
                var_status = "[PASS]" if var_test['homogeneous'] else "[WARN]"
                print(f"  {var_status} Variance homogeneity: p = {var_test['p_value']:.4f}")
    
        # Categorical tests
        if results['categorical_tests']:
            print("\n2. CATEGORICAL FEATURES")
            print("-" * 80)
        
            for feature, test_result in results['categorical_tests'].items():
                status = "[PASS]" if test_result['significant'] else "[FAIL]"
                chi2 = test_result['chi2_statistic']
                p_val = test_result['p_value']
            
                print(f"\n{feature}:")
                print(f"  {status} Chi-square: chi2 = {chi2:.2f}, p = {p_val:.4f}")
    
        # Summary
        print("\n3. SUMMARY")
        print("-" * 80)
        summary = results['summary']
        print(f"\nTests passed: {summary['continuous_tests_passed']}/{summary['continuous_tests_total']} continuous")
        if summary['categorical_tests_total'] > 0:
            print(f"              {summary['categorical_tests_passed']}/{summary['categorical_tests_total']} categorical")
        print(f"Pass rate: {summary['pass_rate']:.1%}")
        print(f"\nVerdict: {results['verdict']}")
    
        print("\n" + "=" * 80)
    except Exception as e:
        logger.error(f"print_validation_report failed: {e}")
        raise


__all__ = [
    'test_segment_differences_anova',
    'test_segment_differences_kruskal',
    'calculate_effect_size_cohens_d',
    'calculate_eta_squared',
    'posthoc_dunn',
    'test_variance_homogeneity',
    'test_categorical_independence',
    'validate_segment_quality',
    'print_validation_report',
]

if __name__ == "__main__":
    # Test the module
    print("Testing n3j_statistical_tests module...")
    
    from n3b_data_loader import load_data_for_segmentation
    from n3f_clustering import perform_kmeans_clustering
    from n3d_feature_prep import prepare_clustering_features
    
    try:
        # Load data
        df, rfm_df = load_data_for_segmentation()
        
        # Prepare features
        feature_list = ['recency_days', 'frequency', 'monetary', 'avg_order_value', 'loyalty_score']
        X_scaled, feature_names, clust_df = prepare_clustering_features(rfm_df, feature_list)
        
        # Perform clustering
        kmeans, labels = perform_kmeans_clustering(X_scaled, 4)
        clust_df['cluster'] = labels
        
        # Run comprehensive validation (using clust_df, NOT rfm_df)
        validation_results = validate_segment_quality(
            segment_df=clust_df,  # Use clean features from clust_df
            continuous_features=['recency_days', 'frequency', 'monetary', 'loyalty_score'],
            categorical_features=None,
            alpha=0.05
        )
        
        # Print report
        print_validation_report(validation_results)
        
        print("\n[OK] Statistical validation test successful!")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
