"""
Test suite for n1g_data_saver.py

Tests data saving and loading functions:
- save_processed_data()
- load_processed_data()
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from n1g_data_saver import save_processed_data, load_processed_data


class TestSaveProcessedData:
    """Test save_processed_data() function."""
    
    @pytest.fixture
    def sample_enhanced_df(self):
        """Create sample enhanced transaction DataFrame."""
        return pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'customer_id': ['C001', 'C001', 'C002'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01']),
            'category': ['Electronics', 'Clothing', 'Books'],
            'region': ['North', 'North', 'South'],
            'total_amount': [100.0, 150.0, 200.0],
            'price': [80.0, 120.0, 180.0],
            'shipping_cost': [20.0, 30.0, 20.0],
            'returned': [0, 0, 1]
        })
    
    @pytest.fixture
    def sample_rfm_df(self):
        """Create sample RFM DataFrame."""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002'],
            'recency_days': [30, 15],
            'frequency': [2, 1],
            'monetary': [250.0, 200.0],
            'avg_order_value': [125.0, 200.0],
            'loyalty_score': [0.75, 0.60],
            'churn': [0, 0]
        })
    
    def test_save_enhanced_data_only(self, sample_enhanced_df):
        """Test saving enhanced dataset only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            result = save_processed_data(
                df=sample_enhanced_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Check that file was saved
            assert 'enhanced' in result
            assert result['enhanced'].exists()
            
            # Verify file can be loaded
            loaded = pd.read_parquet(result['enhanced'])
            assert len(loaded) == len(sample_enhanced_df)
            assert list(loaded.columns) == list(sample_enhanced_df.columns)
    
    def test_save_both_datasets(self, sample_enhanced_df, sample_rfm_df):
        """Test saving both enhanced and RFM datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            result = save_processed_data(
                df=sample_enhanced_df,
                rfm_df=sample_rfm_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Check that both files were saved
            assert 'enhanced' in result
            assert 'rfm' in result
            assert result['enhanced'].exists()
            assert result['rfm'].exists()
    
    def test_save_creates_output_directory(self, sample_enhanced_df):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "directory"
            
            result = save_processed_data(
                df=sample_enhanced_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Directory should be created
            assert output_dir.exists()
            assert result['enhanced'].exists()
    
    def test_save_verifies_shape(self, sample_enhanced_df):
        """Test that save function verifies DataFrame shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Should not raise any errors
            result = save_processed_data(
                df=sample_enhanced_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Verify by loading
            loaded = pd.read_parquet(result['enhanced'])
            assert loaded.shape == sample_enhanced_df.shape
    
    def test_save_verifies_columns(self, sample_enhanced_df):
        """Test that save function verifies column names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            result = save_processed_data(
                df=sample_enhanced_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Verify by loading
            loaded = pd.read_parquet(result['enhanced'])
            assert list(loaded.columns) == list(sample_enhanced_df.columns)
    
    def test_save_empty_dataframe(self):
        """Test saving empty DataFrame."""
        df = pd.DataFrame(columns=['order_id', 'customer_id', 'total_amount'])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            result = save_processed_data(
                df=df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Should save without errors
            assert result['enhanced'].exists()
            
            # Verify by loading
            loaded = pd.read_parquet(result['enhanced'])
            assert len(loaded) == 0
            assert list(loaded.columns) == list(df.columns)
    
    def test_save_with_compression(self, sample_enhanced_df):
        """Test that files are saved with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            result = save_processed_data(
                df=sample_enhanced_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # File should exist and be smaller than uncompressed
            assert result['enhanced'].exists()
            file_size = result['enhanced'].stat().st_size
            assert file_size > 0
    
    def test_save_returns_file_paths(self, sample_enhanced_df, sample_rfm_df):
        """Test that function returns dictionary of file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            result = save_processed_data(
                df=sample_enhanced_df,
                rfm_df=sample_rfm_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Should return dict with Path objects
            assert isinstance(result, dict)
            assert isinstance(result['enhanced'], Path)
            assert isinstance(result['rfm'], Path)
    
    def test_save_overwrites_existing_files(self, sample_enhanced_df):
        """Test that existing files are overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Save first time
            result1 = save_processed_data(
                df=sample_enhanced_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Modify DataFrame and save again
            modified_df = sample_enhanced_df.copy()
            modified_df.loc[0, 'total_amount'] = 999.0
            
            result2 = save_processed_data(
                df=modified_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Load and verify it's the modified version
            loaded = pd.read_parquet(result2['enhanced'])
            assert loaded.loc[0, 'total_amount'] == 999.0


class TestLoadProcessedData:
    """Test load_processed_data() function."""
    
    @pytest.fixture
    def saved_datasets(self, sample_enhanced_df, sample_rfm_df):
        """Create and save sample datasets."""
        tmpdir = tempfile.mkdtemp()
        output_dir = Path(tmpdir)
        
        # Save both datasets
        save_processed_data(
            df=sample_enhanced_df,
            rfm_df=sample_rfm_df,
            output_dir=output_dir,
            verbose=False
        )
        
        yield output_dir
        
        # Cleanup
        shutil.rmtree(tmpdir)
    
    @pytest.fixture
    def sample_enhanced_df(self):
        """Create sample enhanced transaction DataFrame."""
        return pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'customer_id': ['C001', 'C001', 'C002'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01']),
            'total_amount': [100.0, 150.0, 200.0]
        })
    
    @pytest.fixture
    def sample_rfm_df(self):
        """Create sample RFM DataFrame."""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002'],
            'recency_days': [30, 15],
            'frequency': [2, 1],
            'monetary': [250.0, 200.0]
        })
    
    def test_load_both_datasets(self, saved_datasets):
        """Test loading both enhanced and RFM datasets."""
        result = load_processed_data(
            input_dir=saved_datasets,
            load_rfm=True,
            verbose=False
        )
        
        # Should return dict with both datasets
        assert 'enhanced' in result
        assert 'rfm' in result
        assert isinstance(result['enhanced'], pd.DataFrame)
        assert isinstance(result['rfm'], pd.DataFrame)
    
    def test_load_enhanced_only(self, saved_datasets):
        """Test loading enhanced dataset only."""
        result = load_processed_data(
            input_dir=saved_datasets,
            load_rfm=False,
            verbose=False
        )
        
        # Should return dict with only enhanced
        assert 'enhanced' in result
        assert 'rfm' not in result
    
    def test_load_verifies_data_integrity(self, saved_datasets, sample_enhanced_df):
        """Test that loaded data matches original."""
        result = load_processed_data(
            input_dir=saved_datasets,
            load_rfm=False,
            verbose=False
        )
        
        loaded = result['enhanced']
        
        # Compare key columns
        assert len(loaded) == len(sample_enhanced_df)
        assert list(loaded.columns) == list(sample_enhanced_df.columns)
    
    def test_load_missing_enhanced_file(self):
        """Test error handling when enhanced file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            with pytest.raises(FileNotFoundError):
                load_processed_data(
                    input_dir=output_dir,
                    verbose=False
                )
    
    def test_load_missing_rfm_file_warning(self, saved_datasets):
        """Test that missing RFM file generates warning but doesn't fail."""
        # Remove RFM file
        rfm_file = saved_datasets / "rfm_df.parquet"
        if rfm_file.exists():
            rfm_file.unlink()
        
        # Should load enhanced but warn about missing RFM
        result = load_processed_data(
            input_dir=saved_datasets,
            load_rfm=True,
            verbose=False
        )
        
        assert 'enhanced' in result
        assert 'rfm' not in result


class TestSaveLoadIntegration:
    """Integration tests for save and load workflow."""
    
    def test_save_and_load_roundtrip(self):
        """Test complete save and load roundtrip."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'customer_id': ['C001', 'C001', 'C002'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01']),
            'total_amount': [100.0, 150.0, 200.0],
            'price': [80.0, 120.0, 180.0]
        })
        
        rfm_df = pd.DataFrame({
            'customer_id': ['C001', 'C002'],
            'recency_days': [30, 15],
            'frequency': [2, 1],
            'monetary': [250.0, 200.0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Save
            save_result = save_processed_data(
                df=df,
                rfm_df=rfm_df,
                output_dir=output_dir,
                verbose=False
            )
            
            # Load
            load_result = load_processed_data(
                input_dir=output_dir,
                load_rfm=True,
                verbose=False
            )
            
            # Verify enhanced data
            loaded_df = load_result['enhanced']
            pd.testing.assert_frame_equal(df, loaded_df)
            
            # Verify RFM data
            loaded_rfm = load_result['rfm']
            pd.testing.assert_frame_equal(rfm_df, loaded_rfm)
    
    def test_save_and_load_preserves_dtypes(self):
        """Test that data types are preserved through save/load."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002'],
            'customer_id': ['C001', 'C002'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'total_amount': np.array([100.0, 200.0], dtype=np.float32),
            'quantity': np.array([1, 2], dtype=np.int32),
            'returned': np.array([0, 1], dtype=np.int8)
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Save
            save_processed_data(df=df, output_dir=output_dir, verbose=False)
            
            # Load
            result = load_processed_data(input_dir=output_dir, load_rfm=False, verbose=False)
            loaded = result['enhanced']
            
            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(loaded['order_date'])
            assert pd.api.types.is_numeric_dtype(loaded['total_amount'])
            assert pd.api.types.is_integer_dtype(loaded['quantity'])


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
