"""
Test suite for n1a_utils.py

Tests core utility functions including:
- Project root detection
- Run ID generation and context management
- Logger setup with filters
- Configuration loading
- Project structure verification
"""

import pytest
import logging
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from n1a_utils import (
    get_project_root,
    generate_run_id,
    set_run_id,
    get_run_id,
    RunIDFilter,
    SanitizeFilter,
    setup_logger,
    load_config,
    verify_project_structure,
    run_id_var
)


class TestProjectRoot:
    """Test project root detection functionality."""
    
    def test_generate_run_id_format(self):
        """Test that run_id is 8-character hex string."""
        run_id = generate_run_id()
        assert len(run_id) == 8
        assert all(c in '0123456789abcdef' for c in run_id)
    
    def test_generate_run_id_uniqueness(self):
        """Test that multiple run_ids are unique."""
        run_ids = [generate_run_id() for _ in range(100)]
        assert len(set(run_ids)) == 100
    
    def test_set_and_get_run_id(self):
        """Test run_id context variable management."""
        test_id = "abc12345"
        result = set_run_id(test_id)
        assert result == test_id
        assert get_run_id() == test_id
    
    def test_set_run_id_generates_if_none(self):
        """Test that set_run_id generates new ID if None provided."""
        result = set_run_id(None)
        assert result is not None
        assert len(result) == 8
        assert get_run_id() == result


class TestRunIDFilter:
    """Test RunIDFilter for log record injection."""
    
    def test_filter_adds_run_id(self):
        """Test that filter adds run_id to log record."""
        set_run_id("test1234")
        filter_obj = RunIDFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        assert result is True
        assert hasattr(record, 'run_id')
        assert record.run_id == "test1234"
    
    def test_filter_adds_placeholder_when_no_run_id(self):
        """Test that filter adds placeholder when run_id not set."""
        # Reset the context variable to None
        run_id_var.set(None)
        
        filter_obj = RunIDFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        assert result is True
        assert record.run_id == "XXXXXXXX"


class TestSanitizeFilter:
    """Test SanitizeFilter for PII redaction."""
    
    def test_redact_customer_id(self):
        """Test that customer IDs are redacted."""
        filter_obj = SanitizeFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Processing customer C12345",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        assert "[REDACTED_CUSTOMER_ID]" in str(record.msg)
        assert "C12345" not in str(record.msg)
    
    def test_redact_email(self):
        """Test that email addresses are redacted."""
        filter_obj = SanitizeFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Email sent to user@example.com",
            args=(),
            exc_info=None
        )
        
        filter_obj.filter(record)
        assert "[REDACTED_EMAIL]" in str(record.msg)
        assert "user@example.com" not in str(record.msg)
    
    def test_redact_phone_number(self):
        """Test that phone numbers are redacted."""
        filter_obj = SanitizeFilter()
        test_cases = [
            "Call 555-123-4567",
            "Phone: 555.123.4567",
            "Contact: 5551234567"
        ]
        
        for test_msg in test_cases:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=test_msg,
                args=(),
                exc_info=None
            )
            filter_obj.filter(record)
            assert "[REDACTED_PHONE]" in str(record.msg)
    
    def test_redact_credit_card(self):
        """Test that credit card numbers are redacted."""
        filter_obj = SanitizeFilter()
        test_cases = [
            "Card 4532-1234-5678-9010",
            "CC: 4532 1234 5678 9010",
            "Payment: 4532123456789010"
        ]
        
        for test_msg in test_cases:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=test_msg,
                args=(),
                exc_info=None
            )
            filter_obj.filter(record)
            assert "[REDACTED_CC]" in str(record.msg)


class TestLoggerSetup:
    """Test logger configuration."""
    
    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger_basic")
        assert logger.name == "test_logger_basic"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logger_with_run_id_filter(self):
        """Test that logger includes run_id filter."""
        logger = setup_logger("test_logger_runid", include_run_id=True)
        filters = [f for f in logger.filters if isinstance(f, RunIDFilter)]
        assert len(filters) > 0
    
    def test_setup_logger_with_sanitization(self):
        """Test that logger includes sanitization filter."""
        logger = setup_logger("test_logger_sanitize", include_sanitization=True)
        filters = [f for f in logger.filters if isinstance(f, SanitizeFilter)]
        assert len(filters) > 0
    
    def test_setup_logger_prevents_duplicates(self):
        """Test that calling setup_logger twice doesn't duplicate handlers."""
        logger1 = setup_logger("test_logger_dup")
        handler_count1 = len(logger1.handlers)
        
        logger2 = setup_logger("test_logger_dup")
        handler_count2 = len(logger2.handlers)
        
        assert handler_count1 == handler_count2


class TestConfigLoader:
    """Test configuration loading."""
    
    def test_load_config_valid_yaml(self):
        """Test loading valid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'paths': {'raw_data': 'data/raw/file.csv'},
                'rfm': {'churn_threshold_days': 120}
            }, f)
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            assert 'paths' in config
            assert 'rfm' in config
            assert config['rfm']['churn_threshold_days'] == 120
        finally:
            config_path.unlink()
    
    def test_load_config_file_not_found(self):
        """Test that load_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))
    
    def test_load_config_invalid_yaml(self):
        """Test that load_config raises ValueError for malformed YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            config_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError):
                load_config(config_path)
        finally:
            config_path.unlink()


class TestProjectStructureVerification:
    """Test project structure verification."""
    
    def test_verify_project_structure_valid(self):
        """Test verification with valid project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            # Create required directories
            (project_root / 'src').mkdir()
            (project_root / 'data' / 'raw').mkdir(parents=True)
            (project_root / 'outputs' / 'figures').mkdir(parents=True)
            
            # Create required files
            (project_root / 'config.yaml').write_text('test: value')
            (project_root / 'schema.yaml').write_text('test: value')
            
            # Should not raise
            result = verify_project_structure(project_root)
            assert result is True
    
    def test_verify_project_structure_missing_directory(self):
        """Test verification fails with missing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            # Create only some directories
            (project_root / 'src').mkdir()
            (project_root / 'config.yaml').write_text('test: value')
            (project_root / 'schema.yaml').write_text('test: value')
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Missing"):
                verify_project_structure(project_root)
    
    def test_verify_project_structure_missing_file(self):
        """Test verification fails with missing required file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            # Create directories but not files
            (project_root / 'src').mkdir()
            (project_root / 'data' / 'raw').mkdir(parents=True)
            (project_root / 'outputs' / 'figures').mkdir(parents=True)
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="config.yaml"):
                verify_project_structure(project_root)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
