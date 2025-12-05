import pytest
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

def test_imports():
    """Test that all modules can be imported."""
    import deepseek.torch.model.attention
    import deepseek.torch.model.mla
    import deepseek.torch.model.moe
    import deepseek.torch.training.training
    import deepseek.torch.utils.logging
    
    assert True

def test_logging_config():
    """Test logging configuration."""
    from deepseek.torch.utils.logging import configure_logging, get_logger
    configure_logging()
    logger = get_logger("test")
    logger.info("Test log message")
    assert True
