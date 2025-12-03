import pytest
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

def test_imports():
    """Test that all modules can be imported."""
    import deepseek.model.attention
    import deepseek.model.mla
    import deepseek.model.moe
    import deepseek.training.training
    import deepseek.utils.logging
    
    assert True

def test_logging_config():
    """Test logging configuration."""
    from deepseek.utils.logging import configure_logging, get_logger
    configure_logging()
    logger = get_logger("test")
    logger.info("Test log message")
    assert True
