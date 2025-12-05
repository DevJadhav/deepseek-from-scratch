"""
Tests for Modal GPU logging utilities.

Tests cover:
- Structured logging configuration
- Correlation ID management
- TrainingLogger functionality
- Log output formatting
"""

import pytest
import logging
from unittest.mock import patch, MagicMock


class TestLoggingConfiguration:
    """Tests for logging configuration."""
    
    def test_configure_logging_development(self):
        """Test development logging configuration."""
        from deepseek.cloud.modal.logging_utils import configure_logging
        
        # Should not raise
        configure_logging.cache_clear()  # Clear LRU cache
        configure_logging(env="development")
    
    def test_configure_logging_production(self):
        """Test production logging configuration."""
        from deepseek.cloud.modal.logging_utils import configure_logging
        
        configure_logging.cache_clear()
        configure_logging(env="production", json_logs=True)
    
    def test_configure_logging_custom_level(self):
        """Test custom log level configuration."""
        from deepseek.cloud.modal.logging_utils import configure_logging
        
        configure_logging.cache_clear()
        configure_logging(env="testing", log_level="WARNING")


class TestCorrelationID:
    """Tests for correlation ID management."""
    
    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        from deepseek.cloud.modal.logging_utils import generate_correlation_id
        
        cid = generate_correlation_id()
        assert cid.startswith("trace-")
        assert len(cid) > 6  # "trace-" + hex chars
    
    def test_generate_unique_ids(self):
        """Test that generated IDs are unique."""
        from deepseek.cloud.modal.logging_utils import generate_correlation_id
        
        ids = [generate_correlation_id() for _ in range(100)]
        assert len(set(ids)) == 100
    
    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        from deepseek.cloud.modal.logging_utils import set_correlation_id, get_correlation_id
        
        test_id = "trace-test123"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id
    
    def test_get_correlation_id_generates_new(self):
        """Test that get generates new ID if none set."""
        from deepseek.cloud.modal.logging_utils import _correlation_id, get_correlation_id
        
        # Clear context var
        _correlation_id.set("")
        
        cid = get_correlation_id()
        assert cid.startswith("trace-")


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_logger_returns_bound_logger(self):
        """Test that get_logger returns a structlog BoundLogger."""
        from deepseek.cloud.modal.logging_utils import get_logger
        import structlog
        
        logger = get_logger("test.module")
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")
    
    def test_get_logger_with_name(self):
        """Test logger with specific name."""
        from deepseek.cloud.modal.logging_utils import get_logger
        
        logger = get_logger("my.custom.logger")
        assert logger is not None
    
    def test_logger_bind(self):
        """Test binding context to logger."""
        from deepseek.cloud.modal.logging_utils import get_logger
        
        logger = get_logger("test")
        bound_logger = logger.bind(rank=0, stage="pretrain")
        assert bound_logger is not None


class TestTrainingLogger:
    """Tests for TrainingLogger class."""
    
    @pytest.fixture
    def training_logger(self):
        """Create a training logger for testing."""
        from deepseek.cloud.modal.logging_utils import TrainingLogger
        return TrainingLogger(stage="test", rank=0, world_size=1)
    
    def test_training_logger_creation(self, training_logger):
        """Test TrainingLogger can be created."""
        assert training_logger.stage == "test"
        assert training_logger.rank == 0
        assert training_logger.world_size == 1
    
    def test_log_step(self, training_logger):
        """Test logging a training step."""
        # Should not raise
        training_logger.log_step(step=100, loss=2.5, lr=1e-4)
    
    def test_log_step_with_extras(self, training_logger):
        """Test logging step with extra metrics."""
        training_logger.log_step(
            step=100,
            loss=2.5,
            lr=1e-4,
            throughput=1000.0,
            memory_gb=40.0,
            custom_metric=42,
        )
    
    def test_log_checkpoint_saved(self, training_logger):
        """Test logging checkpoint save event."""
        training_logger.log_checkpoint_saved("/checkpoints/step_100", step=100)
    
    def test_log_checkpoint_loaded(self, training_logger):
        """Test logging checkpoint load event."""
        training_logger.log_checkpoint_loaded("/checkpoints/step_100", step=100)
    
    def test_log_epoch_complete(self, training_logger):
        """Test logging epoch completion."""
        training_logger.log_epoch_complete(epoch=1, avg_loss=2.3)
    
    def test_log_training_started(self, training_logger):
        """Test logging training start."""
        training_logger.log_training_started(
            model_params=1_000_000,
            max_steps=10000,
            batch_size=32,
        )
    
    def test_log_training_complete(self, training_logger):
        """Test logging training completion."""
        training_logger.log_training_complete(
            final_step=10000,
            total_time_seconds=3600.0,
            final_loss=1.5,
        )
    
    def test_log_error(self, training_logger):
        """Test logging an error."""
        training_logger.log_error("checkpoint_failed", ValueError("disk full"))
    
    def test_log_warning(self, training_logger):
        """Test logging a warning."""
        training_logger.log_warning("low_memory", "GPU memory at 90%")
    
    def test_aggregated_metrics_empty(self, training_logger):
        """Test aggregated metrics when no steps logged."""
        metrics = training_logger.get_aggregated_metrics()
        assert metrics == {}
    
    def test_aggregated_metrics(self, training_logger):
        """Test aggregated metrics after logging steps."""
        # Log several steps
        for i in range(10):
            training_logger.log_step(step=i, loss=2.0 - i * 0.1)
        
        metrics = training_logger.get_aggregated_metrics()
        assert metrics["total_steps"] == 10
        assert "avg_loss" in metrics
        assert "min_loss" in metrics
        assert "max_loss" in metrics
        assert "final_loss" in metrics


class TestLogProcessors:
    """Tests for structlog processors."""
    
    def test_add_correlation_id_processor(self):
        """Test correlation ID processor."""
        from deepseek.cloud.modal.logging_utils import add_correlation_id, set_correlation_id
        
        set_correlation_id("trace-test-processor")
        event_dict = {}
        result = add_correlation_id(None, "info", event_dict)
        
        assert "correlation_id" in result
        assert result["correlation_id"] == "trace-test-processor"
    
    def test_add_timestamp_processor(self):
        """Test timestamp processor."""
        from deepseek.cloud.modal.logging_utils import add_timestamp
        
        event_dict = {}
        result = add_timestamp(None, "info", event_dict)
        
        assert "timestamp" in result
        assert "T" in result["timestamp"]  # ISO format includes T
    
    def test_add_service_info_processor(self):
        """Test service info processor."""
        from deepseek.cloud.modal.logging_utils import add_service_info
        
        event_dict = {}
        result = add_service_info(None, "info", event_dict)
        
        assert "service" in result
        assert result["service"] == "deepseek-training"
        assert "version" in result


class TestLoggingIntegration:
    """Integration tests for logging system."""
    
    def test_full_logging_flow(self):
        """Test complete logging flow."""
        from deepseek.cloud.modal.logging_utils import (
            configure_logging,
            get_logger,
            set_correlation_id,
            generate_correlation_id,
        )
        
        configure_logging.cache_clear()
        configure_logging(env="testing", log_level="DEBUG")
        
        cid = generate_correlation_id()
        set_correlation_id(cid)
        
        logger = get_logger("integration.test")
        logger = logger.bind(test_id="integration")
        
        # These should not raise
        logger.info("test_event", data="test")
        logger.debug("debug_event", value=42)
        logger.warning("warning_event", message="test warning")
    
    def test_logger_with_training_context(self):
        """Test logger with training-specific context."""
        from deepseek.cloud.modal.logging_utils import (
            configure_logging,
            TrainingLogger,
            set_correlation_id,
            generate_correlation_id,
        )
        
        configure_logging.cache_clear()
        configure_logging(env="testing")
        
        set_correlation_id(generate_correlation_id())
        
        logger = TrainingLogger(
            stage="pretrain",
            rank=0,
            world_size=4,
        )
        
        # Simulate training loop
        logger.log_training_started(
            model_params=125_000_000,
            max_steps=100,
            batch_size=32,
        )
        
        for step in range(1, 11):
            logger.log_step(
                step=step,
                loss=2.0 - step * 0.1,
                lr=1e-4,
            )
        
        logger.log_checkpoint_saved("/tmp/test_ckpt", step=10)
        
        metrics = logger.get_aggregated_metrics()
        assert metrics["total_steps"] == 10
        assert metrics["final_loss"] < 2.0
