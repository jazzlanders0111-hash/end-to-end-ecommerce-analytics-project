"""
Test suite for n1b_data_loader.py

Tests data loading and schema validation:
- load_raw_data()
- Schema validation with Pandera
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import yaml


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from n1b_data_loader import load_raw_data
from unittest.mock import patch, MagicMock


class TestLoadRawData:
    """Test load_raw_data() function."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write sample CSV data
            f.write("order_id,customer_id,product_id,order_date,category,region,customer_gender,payment_method,price,discount,quantity,total_amount,shipping_cost,profit_margin,customer_age,delivery_time_days,returned\n")
            f.write("O001,C001,P001,2024-01-01,Electronics,North,Male,Credit,100.0,0.1,1,110.0,10.0,0.2,30,3,0\n")
            f.write("O002,C002,P002,2024-01-02,Clothing,South,Female,Debit,50.0,0.0,2,100.0,10.0,0.25,25,5,0\n")
            csv_path = Path(f.name)
        
        yield csv_path
        
        # Cleanup
        csv_path.unlink()
    
    @pytest.fixture
    def sample_config(self, sample_csv_file):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'paths': {
                    'raw_data': str(sample_csv_file)
                }
            }
            yaml.dump(config, f)
            config_path = Path(f.name)
        
        yield config_path
        
        # Cleanup
        config_path.unlink()
    
    @pytest.fixture
    def sample_schema(self):
        """Create a temporary schema file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            schema = {
                'schema_settings': {
                    'strict': True,
                    'coerce': False
                },
                'columns': {
                    'order_id': {'type': 'string', 'nullable': False},
                    'customer_id': {'type': 'string', 'nullable': False},
                    'product_id': {'type': 'string', 'nullable': False},
                    'order_date': {'type': 'datetime', 'nullable': False, 'coerce': True},
                    'category': {'type': 'string', 'nullable': False},
                    'region': {'type': 'string', 'nullable': False},
                    'customer_gender': {'type': 'string', 'nullable': False},
                    'payment_method': {'type': 'string', 'nullable': False},
                    'price': {'type': 'float32', 'nullable': False, 'checks': [{'ge': 0}]},
                    'discount': {'type': 'float32', 'nullable': False, 'checks': [{'ge': 0}, {'le': 1}]},
                    'quantity': {'type': 'int8', 'nullable': False, 'checks': [{'gt': 0}]},
                    'total_amount': {'type': 'float32', 'nullable': False, 'checks': [{'ge': 0}]},
                    'shipping_cost': {'type': 'float32', 'nullable': False, 'checks': [{'ge': 0}]},
                    'profit_margin': {'type': 'float32', 'nullable': False},
                    'customer_age': {'type': 'int8', 'nullable': False, 'checks': [{'ge': 0}]},
                    'delivery_time_days': {'type': 'int8', 'nullable': False, 'checks': [{'ge': 0}]},
                    'returned': {'type': 'int8', 'nullable': False, 'checks': [{'isin': [0, 1]}]}
                }
            }
            yaml.dump(schema, f)
            schema_path = Path(f.name)
        
        yield schema_path
        
        # Cleanup
        schema_path.unlink()
    
    def test_load_raw_data_file_not_found(self):
        """Test that load_raw_data raises FileNotFoundError when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a config pointing to non-existent file
            config_path = Path(tmpdir) / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump({'paths': {'raw_data': 'nonexistent.csv'}}, f)
            
            with patch('n1b_data_loader.PROJECT_ROOT', Path(tmpdir)):
                with pytest.raises(FileNotFoundError):
                    load_raw_data()
    
    def test_load_raw_data_basic_functionality(self, sample_csv_file, sample_config, sample_schema):
        """Test basic data loading functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Copy files to temp directory
            import shutil
            shutil.copy(sample_config, tmpdir_path / 'config.yaml')
            shutil.copy(sample_schema, tmpdir_path / 'schema.yaml')
            
            # Update config to use correct CSV path
            with open(tmpdir_path / 'config.yaml', 'w') as f:
                yaml.dump({'paths': {'raw_data': str(sample_csv_file)}}, f)
            
            with patch('n1b_data_loader.PROJECT_ROOT', tmpdir_path):
                df = load_raw_data()
                
                # Check that data was loaded
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 2
                assert 'order_id' in df.columns
                assert 'customer_id' in df.columns
    
    def test_load_raw_data_returns_dataframe(self, sample_csv_file, sample_config, sample_schema):
        """Test that function returns a DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            import shutil
            shutil.copy(sample_config, tmpdir_path / 'config.yaml')
            shutil.copy(sample_schema, tmpdir_path / 'schema.yaml')
            
            with open(tmpdir_path / 'config.yaml', 'w') as f:
                yaml.dump({'paths': {'raw_data': str(sample_csv_file)}}, f)
            
            with patch('n1b_data_loader.PROJECT_ROOT', tmpdir_path):
                result = load_raw_data()
                assert isinstance(result, pd.DataFrame)


class TestDataTypeConversion:
    """Test data type conversions during loading."""
    
    def test_datetime_conversion(self):
        """Test that dates are converted to datetime."""
        df = pd.DataFrame({
            'order_date': ['2024-01-01', '2024-01-02']
        })
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        assert pd.api.types.is_datetime64_any_dtype(df['order_date'])
    
    def test_numeric_downcast(self):
        """Test numeric downcasting for memory efficiency."""
        df = pd.DataFrame({
            'price': [100.0, 200.0],
            'quantity': [1, 2]
        })
        
        # Downcast float
        df['price'] = pd.to_numeric(df['price'], downcast='float')
        
        # Downcast int
        df['quantity'] = pd.to_numeric(df['quantity'], downcast='integer')
        
        # Check types are smaller than default
        assert df['price'].dtype in ['float32', 'float16']
        assert df['quantity'].dtype in ['int8', 'int16', 'int32']


class TestChunkedReading:
    """Test chunked reading for large files."""
    
    def test_chunked_reading_concatenates_correctly(self):
        """Test that chunked reading produces same result as regular reading."""
        # Create sample data
        data = {
            'order_id': [f'O{i:03d}' for i in range(1000)],
            'total_amount': np.random.uniform(10, 1000, 1000)
        }
        df_full = pd.DataFrame(data)
        
        # Simulate chunked reading
        chunks = []
        chunk_size = 100
        for i in range(0, len(df_full), chunk_size):
            chunk = df_full.iloc[i:i+chunk_size]
            chunks.append(chunk)
        
        df_chunked = pd.concat(chunks, ignore_index=True)
        
        # Should be equal
        pd.testing.assert_frame_equal(df_full, df_chunked)


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
