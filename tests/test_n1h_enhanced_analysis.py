"""
Test suite for n1h_enhanced_analysis.py

Tests enhanced business analysis functions:
- Data quality scoring
- Temporal distribution analysis
- Churn and retention analytics
- Business summary generation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from n1h_enhanced_analysis import (
    calculate_data_quality_score,
    analyze_temporal_distribution,
    analyze_churn_and_retention,
    generate_business_summary
)


class TestDataQualityScore:
    """Test calculate_data_quality_score() function."""
    
    @pytest.fixture
    def clean_data(self):
        """Create clean test data."""
        return pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'customer_id': ['C001', 'C002', 'C003'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'price': [100.0, 200.0, 150.0],
            'discount': [0.1, 0.2, 0.0],
            'quantity': [1, 2, 1],
            'total_amount': [110.0, 240.0, 150.0]
        })
    
    def test_perfect_data_quality(self, clean_data):
        """Test quality score with perfect data."""
        result = calculate_data_quality_score(
            df_clean=clean_data,
            initial_row_count=3,
            verbose=False
        )
        
        assert result['quality_score'] == 100.0
        assert result['completeness'] == 100.0
        assert result['compliance_rate'] == 100.0
        assert result['retention_rate'] == 100.0
        assert result['violations'] == 0
    
    def test_data_with_removals(self, clean_data):
        """Test quality score when data was removed."""
        result = calculate_data_quality_score(
            df_clean=clean_data,
            initial_row_count=10,
            verbose=False
        )
        
        # Should have lower retention rate
        assert result['retention_rate'] == 30.0
        assert result['quality_score'] < 100.0
        assert result['rows_removed'] == 7
    
    def test_data_with_violations(self):
        """Test detection of business rule violations."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'price': [100.0, -50.0, 150.0],  # Negative price
            'discount': [0.1, 1.5, 0.0],  # Invalid discount > 1
            'quantity': [1, 0, 1]  # Zero quantity
        })
        
        result = calculate_data_quality_score(
            df_clean=df,
            initial_row_count=3,
            verbose=False
        )
        
        assert result['violations'] == 3  # One of each type
        assert result['compliance_rate'] < 100.0
    
    def test_returns_correct_keys(self, clean_data):
        """Test that all expected keys are returned."""
        result = calculate_data_quality_score(clean_data, verbose=False)
        
        expected_keys = [
            'quality_score', 'completeness', 'compliance_rate', 
            'retention_rate', 'initial_rows', 'final_rows', 
            'rows_removed', 'violations'
        ]
        
        for key in expected_keys:
            assert key in result


class TestTemporalDistribution:
    """Test analyze_temporal_distribution() function."""
    
    @pytest.fixture
    def temporal_data(self):
        """Create temporal test data."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        return pd.DataFrame({
            'order_date': np.random.choice(dates, size=1000),
            'total_amount': np.random.uniform(50, 500, 1000),
            'customer_id': [f'C{i%100:03d}' for i in range(1000)]
        })
    
    def test_basic_temporal_analysis(self, temporal_data):
        """Test basic temporal metrics calculation."""
        result = analyze_temporal_distribution(
            df=temporal_data,
            verbose=False
        )
        
        assert 'dataset_span_days' in result
        assert 'active_months' in result
        assert 'avg_monthly_revenue' in result
        assert 'revenue_volatility' in result
        assert result['active_months'] > 0
    
    def test_identifies_peak_and_trough(self, temporal_data):
        """Test identification of peak and trough periods."""
        result = analyze_temporal_distribution(
            df=temporal_data,
            verbose=False
        )
        
        assert 'peak_month' in result
        assert 'trough_month' in result
        assert 'peak_revenue' in result
        assert 'trough_revenue' in result
        assert result['peak_revenue'] > result['trough_revenue']
    
    def test_detects_data_gaps(self):
        """Test detection of missing dates."""
        # Create data with gaps
        dates = list(pd.date_range(start='2024-01-01', end='2024-01-10', freq='D'))
        dates.extend(pd.date_range(start='2024-01-20', end='2024-01-31', freq='D'))
        
        df = pd.DataFrame({
            'order_date': dates,
            'total_amount': [100.0] * len(dates),
            'customer_id': [f'C{i:03d}' for i in range(len(dates))]
        })
        
        result = analyze_temporal_distribution(df, verbose=False)
        
        # Should detect gap from Jan 11-19 (9 days)
        assert result['missing_dates_count'] >= 9
    
    def test_calculates_growth_rate(self, temporal_data):
        """Test growth rate calculation."""
        result = analyze_temporal_distribution(
            df=temporal_data,
            verbose=False
        )
        
        if result['active_months'] >= 3:
            assert 'growth_rate' in result
            assert result['growth_rate'] is not None


class TestChurnRetention:
    """Test analyze_churn_and_retention() function."""
    
    @pytest.fixture
    def rfm_data(self):
        """Create RFM test data."""
        return pd.DataFrame({
            'customer_id': [f'C{i:03d}' for i in range(100)],
            'recency_days': np.random.randint(1, 200, 100),
            'frequency': np.random.randint(1, 20, 100),
            'monetary': np.random.uniform(100, 5000, 100),
            'churn': np.random.choice([0, 1], 100, p=[0.6, 0.4])
        })
    
    def test_basic_churn_analysis(self, rfm_data):
        """Test basic churn metrics calculation."""
        result = analyze_churn_and_retention(
            rfm_df=rfm_data,
            churn_threshold_days=120,
            verbose=False
        )
        
        assert result['total_customers'] == 100
        assert 'one_time_buyers' in result
        assert 'churned_count' in result
        assert 'at_risk_count' in result
    
    def test_identifies_one_time_buyers(self):
        """Test identification of one-time buyers."""
        df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004'],
            'frequency': [1, 1, 5, 10],
            'monetary': [100.0, 150.0, 500.0, 1000.0],
            'recency_days': [30, 60, 45, 20],
            'churn': [0, 0, 0, 0]
        })
        
        result = analyze_churn_and_retention(df, verbose=False)
        
        assert result['one_time_buyers'] == 2
        assert result['one_time_buyer_pct'] == 50.0
    
    def test_calculates_at_risk_customers(self):
        """Test at-risk customer identification."""
        df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004'],
            'frequency': [5, 3, 2, 8],
            'monetary': [500.0, 300.0, 200.0, 800.0],
            'recency_days': [100, 50, 30, 110],  # C001 and C004 at risk (>90 days)
            'churn': [0, 0, 0, 0]
        })
        
        result = analyze_churn_and_retention(
            rfm_df=df,
            churn_threshold_days=120,
            verbose=False
        )
        
        # At-risk threshold is 75% of 120 = 90 days
        assert result['at_risk_count'] == 2  # C001 and C004
        assert result['revenue_at_risk'] == 1300.0
    
    def test_segments_by_loyalty(self):
        """Test customer segmentation by loyalty levels."""
        df = pd.DataFrame({
            'customer_id': [f'C{i:03d}' for i in range(20)],
            'frequency': [1]*10 + [5]*5 + [10]*5,  # 10 one-timers, 5 loyal, 5 super loyal
            'monetary': [100.0] * 20,
            'recency_days': [30] * 20,
            'churn': [0] * 20
        })
        
        result = analyze_churn_and_retention(df, verbose=False)
        
        assert result['one_time_buyers'] == 10
        assert result['loyal_customers'] == 10  # 5+ orders
        assert result['super_loyal'] == 5  # 10+ orders


class TestBusinessSummary:
    """Test generate_business_summary() function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for summary."""
        df_clean = pd.DataFrame({
            'order_date': pd.to_datetime(['2024-01-01'] * 100),
            'total_amount': [100.0] * 100,
            'returned': [0] * 95 + [1] * 5
        })
        
        rfm_df = pd.DataFrame({
            'customer_id': [f'C{i:03d}' for i in range(50)],
            'frequency': [2] * 50,
            'monetary': [200.0] * 50,
            'loyalty_score': [0.5] * 50,
            'churn': [0] * 30 + [1] * 20
        })
        
        quality_metrics = {
            'quality_score': 95.0,
            'completeness': 100.0
        }
        
        churn_metrics = {
            'one_time_buyer_pct': 30.0,
            'revenue_at_risk': 5000.0,
            'at_risk_count': 10
        }
        
        return df_clean, rfm_df, quality_metrics, churn_metrics
    
    def test_generates_summary(self, sample_data):
        """Test summary generation."""
        df_clean, rfm_df, quality_metrics, churn_metrics = sample_data
        
        result = generate_business_summary(
            df_clean=df_clean,
            rfm_df=rfm_df,
            quality_metrics=quality_metrics,
            churn_metrics=churn_metrics,
            run_id='test123',
            verbose=False
        )
        
        assert result['run_id'] == 'test123'
        assert result['total_transactions'] == 100
        assert result['total_customers'] == 50
        assert 'churn_rate' in result
        assert 'avg_customer_value' in result
    
    def test_calculates_business_metrics(self, sample_data):
        """Test business metric calculations."""
        df_clean, rfm_df, quality_metrics, churn_metrics = sample_data
        
        result = generate_business_summary(
            df_clean=df_clean,
            rfm_df=rfm_df,
            quality_metrics=quality_metrics,
            churn_metrics=churn_metrics,
            run_id='test123',
            verbose=False
        )
        
        assert result['avg_orders_per_customer'] == 2.0  # 100 orders / 50 customers
        assert result['return_rate'] == 5.0  # 5 out of 100
        assert result['data_quality_score'] == 95.0


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
