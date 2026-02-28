"""
Test suite for n1c_preprocessing.py

Tests data cleaning and validation functions:
- clean_data()
- validate_data()
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from n1c_preprocessing import clean_data, validate_data


class TestCleanData:
    """Test clean_data() function."""
    
    def test_clean_data_basic(self):
        """Test basic data cleaning functionality."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'order_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'category': ['electronics', 'CLOTHING', '  Books  '],
            'region': ['north', 'SOUTH', 'east'],
            'customer_gender': ['male', 'FEMALE', 'other'],
            'payment_method': ['credit', 'DEBIT', 'cash'],
            'total_amount': [100.0, 200.0, 150.0],
            'price': [80.0, 150.0, 120.0],
            'shipping_cost': [20.0, 50.0, 30.0],
            'returned': ['yes', 'no', 'NO']
        })
        
        result = clean_data(df, verbose=False)
        
        # Check that categorical columns are standardized
        assert result['category'].iloc[0] == 'Electronics'
        assert result['category'].iloc[1] == 'Clothing'
        assert result['category'].iloc[2] == 'Books'
        
        # Check returned flag conversion
        assert result['returned'].iloc[0] == 1
        assert result['returned'].iloc[1] == 0
        assert result['returned'].iloc[2] == 0
        
        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(result['order_date'])
    
    def test_clean_data_removes_duplicates(self):
        """Test that duplicate order_ids are removed."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O001', 'O002'],
            'order_date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'category': ['Electronics', 'Electronics', 'Clothing'],
            'region': ['North', 'North', 'South'],
            'customer_gender': ['Male', 'Male', 'Female'],
            'payment_method': ['Credit', 'Credit', 'Debit'],
            'total_amount': [100.0, 100.0, 200.0],
            'price': [80.0, 80.0, 150.0],
            'shipping_cost': [20.0, 20.0, 50.0],
            'returned': [0, 0, 0]
        })
        
        result = clean_data(df, verbose=False)
        
        # Should keep first occurrence of duplicate
        assert len(result) == 2
        assert result['order_id'].nunique() == 2
    
    def test_clean_data_handles_invalid_dates(self):
        """Test that invalid dates are removed."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'order_date': ['2024-01-01', 'invalid-date', '2024-01-03'],
            'category': ['Electronics', 'Clothing', 'Books'],
            'region': ['North', 'South', 'East'],
            'customer_gender': ['Male', 'Female', 'Other'],
            'payment_method': ['Credit', 'Debit', 'Cash'],
            'total_amount': [100.0, 200.0, 150.0],
            'price': [80.0, 150.0, 120.0],
            'shipping_cost': [20.0, 50.0, 30.0],
            'returned': [0, 0, 0]
        })
        
        result = clean_data(df, verbose=False)
        
        # Row with invalid date should be removed
        assert len(result) == 2
        assert 'O002' not in result['order_id'].values
    
    def test_clean_data_handles_missing_values(self):
        """Test that missing values are imputed."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'order_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'category': ['Electronics', None, 'Books'],
            'region': ['North', 'South', 'East'],
            'customer_gender': ['Male', 'Female', None],
            'payment_method': ['Credit', 'Debit', 'Cash'],
            'total_amount': [100.0, np.nan, 150.0],
            'price': [80.0, 150.0, np.nan],
            'shipping_cost': [20.0, 50.0, 30.0],
            'returned': [0, 0, 0]
        })
        
        result = clean_data(df, verbose=False)
        
        # Check that missing values are filled
        assert result['category'].notna().all()
        assert result['customer_gender'].notna().all()
        assert result['total_amount'].notna().all()
        assert result['price'].notna().all()
    
    def test_clean_data_standardizes_categorical(self):
        """Test categorical column standardization."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003', 'O004'],
            'order_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'category': ['electronics', 'ELECTRONICS', '  electronics  ', 'Electronics'],
            'region': ['NORTH', 'north', 'North', '  NORTH  '],
            'customer_gender': ['male', 'MALE', 'Male', '  male  '],
            'payment_method': ['credit', 'CREDIT', 'Credit', '  credit  '],
            'total_amount': [100.0, 100.0, 100.0, 100.0],
            'price': [80.0, 80.0, 80.0, 80.0],
            'shipping_cost': [20.0, 20.0, 20.0, 20.0],
            'returned': [0, 0, 0, 0]
        })
        
        result = clean_data(df, verbose=False)
        
        # All should be standardized to title case
        assert (result['category'] == 'Electronics').all()
        assert (result['region'] == 'North').all()
        assert (result['customer_gender'] == 'Male').all()
        assert (result['payment_method'] == 'Credit').all()
    
    def test_clean_data_handles_returned_flag_variations(self):
        """Test various returned flag representations."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003', 'O004', 'O005', 'O006'],
            'order_date': ['2024-01-01'] * 6,
            'category': ['Electronics'] * 6,
            'region': ['North'] * 6,
            'customer_gender': ['Male'] * 6,
            'payment_method': ['Credit'] * 6,
            'total_amount': [100.0] * 6,
            'price': [80.0] * 6,
            'shipping_cost': [20.0] * 6,
            'returned': ['yes', 'YES', 'y', 'no', 'NO', 'n']
        })
        
        result = clean_data(df, verbose=False)
        
        # First three should be 1, last three should be 0
        assert result['returned'].iloc[0] == 1
        assert result['returned'].iloc[1] == 1
        assert result['returned'].iloc[2] == 1
        assert result['returned'].iloc[3] == 0
        assert result['returned'].iloc[4] == 0
        assert result['returned'].iloc[5] == 0
    
    def test_clean_data_preserves_dtypes(self):
        """Test that appropriate data types are preserved."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002'],
            'order_date': ['2024-01-01', '2024-01-02'],
            'category': ['Electronics', 'Clothing'],
            'region': ['North', 'South'],
            'customer_gender': ['Male', 'Female'],
            'payment_method': ['Credit', 'Debit'],
            'total_amount': [100.0, 200.0],
            'price': [80.0, 150.0],
            'shipping_cost': [20.0, 50.0],
            'returned': [0, 1]
        })
        
        result = clean_data(df, verbose=False)
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result['order_date'])
        assert pd.api.types.is_numeric_dtype(result['total_amount'])
        assert pd.api.types.is_numeric_dtype(result['price'])
        assert result['returned'].dtype in [np.int8, np.int16, np.int32, np.int64]
    
    def test_clean_data_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=[
            'order_id', 'order_date', 'category', 'region', 
            'customer_gender', 'payment_method', 'total_amount',
            'price', 'shipping_cost', 'returned'
        ])
        
        result = clean_data(df, verbose=False)
        
        # Should return empty DataFrame without errors
        assert len(result) == 0
        assert list(result.columns) == list(df.columns)


class TestValidateData:
    """Test validate_data() function."""
    
    def test_validate_data_all_valid(self):
        """Test validation passes with valid data."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'category': ['Electronics', 'Clothing'],
            'region': ['North', 'South'],
            'customer_gender': ['Male', 'Female'],
            'payment_method': ['Credit', 'Debit'],
            'total_amount': [100.0, 200.0],
            'price': [80.0, 150.0],
            'shipping_cost': [20.0, 50.0],
            'returned': [0, 1]
        })
        
        # Should not raise any errors
        result = validate_data(df, verbose=False, auto_fix=False)
        assert len(result) == 2
    
    def test_validate_data_negative_prices_auto_fix(self):
        """Test that negative prices are fixed when auto_fix=True."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'category': ['Electronics', 'Clothing'],
            'region': ['North', 'South'],
            'customer_gender': ['Male', 'Female'],
            'payment_method': ['Credit', 'Debit'],
            'total_amount': [100.0, 200.0],
            'price': [-80.0, 150.0],  # Negative price
            'shipping_cost': [20.0, 50.0],
            'returned': [0, 1]
        })
        
        result = validate_data(df, verbose=False, auto_fix=True)
        
        # Negative price should be set to 0
        assert result['price'].min() >= 0
        assert result['price'].iloc[0] == 0
    
    def test_validate_data_negative_prices_raises_error(self):
        """Test that negative prices raise error when auto_fix=False."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'category': ['Electronics', 'Clothing'],
            'region': ['North', 'South'],
            'customer_gender': ['Male', 'Female'],
            'payment_method': ['Credit', 'Debit'],
            'total_amount': [100.0, 200.0],
            'price': [-80.0, 150.0],  # Negative price
            'shipping_cost': [20.0, 50.0],
            'returned': [0, 1]
        })
        
        with pytest.raises(ValueError, match="negative values"):
            validate_data(df, verbose=False, auto_fix=False)
    
    def test_validate_data_detects_null_values(self):
        """Test that null values are detected."""
        df = pd.DataFrame({
            'order_id': ['O001', None],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'category': ['Electronics', 'Clothing'],
            'region': ['North', 'South'],
            'customer_gender': ['Male', 'Female'],
            'payment_method': ['Credit', 'Debit'],
            'total_amount': [100.0, 200.0],
            'price': [80.0, 150.0],
            'shipping_cost': [20.0, 50.0],
            'returned': [0, 1]
        })
        
        # Should detect null value (as warning, not error)
        result = validate_data(df, verbose=False, auto_fix=True)
        assert result is not None
    
    def test_validate_data_checks_multiple_monetary_columns(self):
        """Test validation checks all monetary columns."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'category': ['Electronics', 'Clothing', 'Books'],
            'region': ['North', 'South', 'East'],
            'customer_gender': ['Male', 'Female', 'Other'],
            'payment_method': ['Credit', 'Debit', 'Cash'],
            'total_amount': [100.0, -200.0, 150.0],  # Negative total
            'price': [80.0, 150.0, -120.0],  # Negative price
            'shipping_cost': [20.0, -50.0, 30.0],  # Negative shipping
            'returned': [0, 1, 0]
        })
        
        result = validate_data(df, verbose=False, auto_fix=True)
        
        # All negative values should be fixed to 0
        assert result['total_amount'].min() >= 0
        assert result['price'].min() >= 0
        assert result['shipping_cost'].min() >= 0


class TestCleanDataIntegration:
    """Integration tests for the full cleaning pipeline."""
    
    def test_clean_then_validate_workflow(self):
        """Test complete clean -> validate workflow."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O001', 'O002'],  # Duplicate
            'order_date': ['2024-01-01', '2024-01-01', 'invalid'],  # Invalid date
            'category': ['electronics', 'electronics', 'CLOTHING'],
            'region': ['north', 'north', 'SOUTH'],
            'customer_gender': ['male', 'male', None],  # Missing value
            'payment_method': ['credit', 'credit', 'DEBIT'],
            'total_amount': [100.0, 100.0, np.nan],  # Missing value
            'price': [80.0, 80.0, 150.0],
            'shipping_cost': [20.0, 20.0, 50.0],
            'returned': ['yes', 'yes', 'no']
        })
        
        # Clean the data
        cleaned = clean_data(df, verbose=False)
        
        # Validate should pass after cleaning
        validated = validate_data(cleaned, verbose=False, auto_fix=False)
        
        assert len(validated) == 1  # Only one valid row after removing duplicate and invalid date
        assert validated['order_id'].iloc[0] == 'O001'


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
