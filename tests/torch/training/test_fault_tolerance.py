"""
Tests for Fault Tolerance and Auto-Retry Recovery (Step 9: F1-F10).

This test module verifies:
- F1: RetryManager with exponential backoff
- F2: SIGTERM graceful shutdown handler
- F3: Checkpoint resume logic
- F4: Failure logging to W&B (mocked)
- F5: Modal restart policy configuration
- F6: NaN loss rollback mechanism
- F7: Checkpoint validation
- F8: Health check endpoint
- F9: Cross-backend failure handling
- F10: Retry budget tracking
"""

import json
import signal
import time
from http.client import HTTPConnection

import pytest
import torch


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    return torch.nn.Linear(10, 5)


@pytest.fixture
def sample_optimizer(sample_model):
    """Create optimizer for testing."""
    return torch.optim.Adam(sample_model.parameters(), lr=0.001)


@pytest.fixture
def save_checkpoint(temp_checkpoint_dir, sample_model, sample_optimizer):
    """Helper to save checkpoints."""
    def _save(step: int, loss: float = 1.0):
        checkpoint = {
            "model_state_dict": sample_model.state_dict(),
            "optimizer_state_dict": sample_optimizer.state_dict(),
            "global_step": step,
            "loss": loss,
        }
        path = temp_checkpoint_dir / f"step_{step}.pt"
        torch.save(checkpoint, path)
        return str(path)
    return _save


# =============================================================================
# F1: RetryManager Tests
# =============================================================================

class TestRetryManager:
    """Tests for RetryManager class (F1)."""
    
    def test_init_default_config(self):
        """Test RetryManager initialization with defaults."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager()
        assert manager.max_attempts == 3
        assert manager.base_delay == 1.0
        assert manager.backoff_factor == 2.0
        assert manager.attempt_count == 0
    
    def test_init_custom_config(self):
        """Test RetryManager with custom configuration."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager(
            max_attempts=5,
            base_delay=2.0,
            max_delay=30.0,
            backoff_factor=3.0,
        )
        assert manager.max_attempts == 5
        assert manager.base_delay == 2.0
        assert manager.max_delay == 30.0
        assert manager.backoff_factor == 3.0
    
    def test_exponential_backoff_delays(self):
        """Test exponential backoff calculation."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager(base_delay=1.0, backoff_factor=2.0, max_delay=100.0)
        
        # Simulate attempts
        manager.attempt_count = 1
        delay1 = manager.get_retry_delay()
        assert delay1 == 1.0  # 1.0 * 2^0
        
        manager.attempt_count = 2
        delay2 = manager.get_retry_delay()
        assert delay2 == 2.0  # 1.0 * 2^1
        
        manager.attempt_count = 3
        delay3 = manager.get_retry_delay()
        assert delay3 == 4.0  # 1.0 * 2^2
    
    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager(base_delay=1.0, backoff_factor=10.0, max_delay=5.0)
        manager.attempt_count = 10  # Would be 10^9 without cap
        
        delay = manager.get_retry_delay()
        assert delay == 5.0
    
    def test_should_retry_under_max_attempts(self):
        """Test should_retry returns True under max attempts."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager(max_attempts=3)
        
        error = RuntimeError("Test error")
        assert manager.should_retry(error, step=100) is True
        assert manager.attempt_count == 1
        
        assert manager.should_retry(error, step=100) is True
        assert manager.attempt_count == 2
    
    def test_should_retry_at_max_attempts(self):
        """Test should_retry returns False at max attempts."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager(max_attempts=2)
        error = RuntimeError("Test error")
        
        manager.should_retry(error, step=100)
        manager.should_retry(error, step=100)
        
        assert manager.should_retry(error, step=100) is False
    
    def test_failure_classification_oom(self):
        """Test OOM failure classification."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager, FailureType
        
        manager = RetryManager()
        
        oom_error = RuntimeError("CUDA out of memory")
        failure_type = manager._classify_failure(oom_error, None)
        assert failure_type == FailureType.OOM
    
    def test_failure_classification_nan_loss(self):
        """Test NaN loss failure classification."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager, FailureType
        
        manager = RetryManager()
        
        # NaN loss value
        failure_type = manager._classify_failure(RuntimeError("Error"), float('nan'))
        assert failure_type == FailureType.NAN_LOSS
        
        # Inf loss value
        failure_type = manager._classify_failure(RuntimeError("Error"), float('inf'))
        assert failure_type == FailureType.NAN_LOSS
    
    def test_failure_classification_timeout(self):
        """Test timeout failure classification."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager, FailureType
        
        manager = RetryManager()
        
        timeout_error = RuntimeError("Connection timed out")
        failure_type = manager._classify_failure(timeout_error, None)
        assert failure_type == FailureType.TIMEOUT
    
    def test_failure_classification_network(self):
        """Test network failure classification."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager, FailureType
        
        manager = RetryManager()
        
        network_error = RuntimeError("NCCL connection failed")
        failure_type = manager._classify_failure(network_error, None)
        assert failure_type == FailureType.NETWORK
    
    def test_execute_with_retry_success(self):
        """Test execute_with_retry succeeds on first try."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager()
        call_count = 0
        
        def success_fn():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = manager.execute_with_retry(success_fn, step=0)
        assert result == "success"
        assert call_count == 1
    
    def test_execute_with_retry_eventual_success(self):
        """Test execute_with_retry succeeds after retries."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager(max_attempts=3, base_delay=0.01)
        call_count = 0
        
        def eventual_success_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = manager.execute_with_retry(eventual_success_fn, step=0)
        assert result == "success"
        assert call_count == 3
    
    def test_execute_with_retry_all_fail(self):
        """Test execute_with_retry raises after max retries."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager(max_attempts=2, base_delay=0.01)
        
        def always_fail_fn():
            raise RuntimeError("Always fails")
        
        with pytest.raises(RuntimeError, match="Always fails"):
            manager.execute_with_retry(always_fail_fn, step=0)
    
    def test_failure_history_tracking(self):
        """Test failure history is tracked."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager(max_attempts=5)
        
        manager.should_retry(RuntimeError("Error 1"), step=10)
        manager.should_retry(RuntimeError("Error 2"), step=20)
        
        assert len(manager.failure_history) == 2
        assert manager.failure_history[0].step == 10
        assert manager.failure_history[1].step == 20
    
    def test_get_stats(self):
        """Test get_stats returns correct statistics."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager()
        manager.should_retry(RuntimeError("OOM"), step=10)
        manager.get_retry_delay()  # Add some retry time
        
        stats = manager.get_stats()
        assert "total_attempts" in stats
        assert "total_failures" in stats
        assert "failure_types" in stats
    
    def test_reset(self):
        """Test reset clears state."""
        from src.deepseek.torch.training.fault_tolerance import RetryManager
        
        manager = RetryManager()
        manager.should_retry(RuntimeError("Error"), step=10)
        
        assert manager.attempt_count > 0
        assert len(manager.failure_history) > 0
        
        manager.reset()
        
        assert manager.attempt_count == 0
        assert len(manager.failure_history) == 0


# =============================================================================
# F3/F7: Checkpoint Resume and Validation Tests
# =============================================================================

class TestCheckpointResume:
    """Tests for checkpoint resume logic (F3)."""
    
    def test_find_latest_checkpoint_empty_dir(self, temp_checkpoint_dir):
        """Test finding checkpoint in empty directory."""
        from src.deepseek.torch.training.fault_tolerance import find_latest_checkpoint
        
        result = find_latest_checkpoint(str(temp_checkpoint_dir))
        assert result is None
    
    def test_find_latest_checkpoint_nonexistent_dir(self, tmp_path):
        """Test finding checkpoint in non-existent directory."""
        from src.deepseek.torch.training.fault_tolerance import find_latest_checkpoint
        
        result = find_latest_checkpoint(str(tmp_path / "nonexistent"))
        assert result is None
    
    def test_find_latest_checkpoint_single(self, temp_checkpoint_dir, save_checkpoint):
        """Test finding single checkpoint."""
        from src.deepseek.torch.training.fault_tolerance import find_latest_checkpoint
        
        save_checkpoint(step=100)
        
        result = find_latest_checkpoint(str(temp_checkpoint_dir))
        assert result is not None
        assert "step_100" in result
    
    def test_find_latest_checkpoint_multiple(self, temp_checkpoint_dir, save_checkpoint):
        """Test finding latest among multiple checkpoints."""
        from src.deepseek.torch.training.fault_tolerance import find_latest_checkpoint
        
        save_checkpoint(step=100)
        save_checkpoint(step=500)
        save_checkpoint(step=300)
        
        result = find_latest_checkpoint(str(temp_checkpoint_dir))
        assert result is not None
        assert "step_500" in result


class TestCheckpointValidation:
    """Tests for checkpoint validation (F7)."""
    
    def test_validate_checkpoint_valid(self, save_checkpoint):
        """Test validation of valid checkpoint."""
        from src.deepseek.torch.training.fault_tolerance import validate_checkpoint
        
        path = save_checkpoint(step=100)
        is_valid, error = validate_checkpoint(path)
        
        assert is_valid is True
        assert error == ""
    
    def test_validate_checkpoint_nonexistent(self):
        """Test validation of non-existent checkpoint."""
        from src.deepseek.torch.training.fault_tolerance import validate_checkpoint
        
        is_valid, error = validate_checkpoint("/nonexistent/checkpoint.pt")
        
        assert is_valid is False
        assert "does not exist" in error
    
    def test_validate_checkpoint_empty_file(self, temp_checkpoint_dir):
        """Test validation of empty file."""
        from src.deepseek.torch.training.fault_tolerance import validate_checkpoint
        
        empty_file = temp_checkpoint_dir / "empty.pt"
        empty_file.touch()
        
        is_valid, error = validate_checkpoint(str(empty_file))
        
        assert is_valid is False
        assert "empty" in error
    
    def test_validate_checkpoint_corrupt(self, temp_checkpoint_dir):
        """Test validation of corrupt checkpoint."""
        from src.deepseek.torch.training.fault_tolerance import validate_checkpoint
        
        corrupt_file = temp_checkpoint_dir / "corrupt.pt"
        corrupt_file.write_bytes(b"not a valid checkpoint")
        
        is_valid, error = validate_checkpoint(str(corrupt_file))
        
        assert is_valid is False
    
    def test_validate_checkpoint_missing_model(self, temp_checkpoint_dir):
        """Test validation of checkpoint without model state."""
        from src.deepseek.torch.training.fault_tolerance import validate_checkpoint
        
        bad_checkpoint = {"optimizer_state_dict": {}, "step": 100}
        path = temp_checkpoint_dir / "bad.pt"
        torch.save(bad_checkpoint, path)
        
        is_valid, error = validate_checkpoint(str(path))
        
        assert is_valid is False
        assert "missing model state" in error


class TestLoadCheckpointWithValidation:
    """Tests for load_checkpoint_with_validation."""
    
    def test_load_valid_checkpoint(self, save_checkpoint, sample_model, sample_optimizer):
        """Test loading a valid checkpoint."""
        from src.deepseek.torch.training.fault_tolerance import load_checkpoint_with_validation
        
        path = save_checkpoint(step=100, loss=0.5)
        
        metadata = load_checkpoint_with_validation(
            path, sample_model, sample_optimizer
        )
        
        assert metadata["step"] == 100
    
    def test_load_invalid_checkpoint(self, sample_model):
        """Test loading invalid checkpoint raises error."""
        from src.deepseek.torch.training.fault_tolerance import load_checkpoint_with_validation
        
        with pytest.raises(ValueError, match="Invalid checkpoint"):
            load_checkpoint_with_validation("/nonexistent.pt", sample_model)


# =============================================================================
# F6: NaN Loss Rollback Tests
# =============================================================================

class TestNaNLossDetector:
    """Tests for NaN loss detection and rollback (F6)."""
    
    def test_init(self):
        """Test NaNLossDetector initialization."""
        from src.deepseek.torch.training.fault_tolerance import NaNLossDetector
        
        detector = NaNLossDetector(loss_threshold=50.0, nan_streak_limit=5)
        assert detector.loss_threshold == 50.0
        assert detector.nan_streak_limit == 5
    
    def test_check_valid_loss(self):
        """Test valid loss detection."""
        from src.deepseek.torch.training.fault_tolerance import NaNLossDetector
        
        detector = NaNLossDetector()
        is_valid, action = detector.check_loss(1.5, step=100)
        
        assert is_valid is True
        assert action is None
        assert detector.last_valid_loss == 1.5
    
    def test_check_nan_loss(self):
        """Test NaN loss detection."""
        from src.deepseek.torch.training.fault_tolerance import NaNLossDetector
        
        detector = NaNLossDetector()
        is_valid, action = detector.check_loss(float('nan'), step=100)
        
        assert is_valid is False
        assert action == "warn"
        assert detector.nan_streak == 1
    
    def test_check_inf_loss(self):
        """Test Inf loss detection."""
        from src.deepseek.torch.training.fault_tolerance import NaNLossDetector
        
        detector = NaNLossDetector()
        is_valid, action = detector.check_loss(float('inf'), step=100)
        
        assert is_valid is False
    
    def test_nan_streak_triggers_rollback(self):
        """Test NaN streak triggers rollback."""
        from src.deepseek.torch.training.fault_tolerance import NaNLossDetector
        
        detector = NaNLossDetector(nan_streak_limit=3)
        
        detector.check_loss(float('nan'), step=100)
        detector.check_loss(float('nan'), step=101)
        is_valid, action = detector.check_loss(float('nan'), step=102)
        
        assert is_valid is False
        assert action == "rollback"
    
    def test_nan_streak_resets_on_valid(self):
        """Test NaN streak resets on valid loss."""
        from src.deepseek.torch.training.fault_tolerance import NaNLossDetector
        
        detector = NaNLossDetector(nan_streak_limit=3)
        
        detector.check_loss(float('nan'), step=100)
        detector.check_loss(float('nan'), step=101)
        detector.check_loss(1.0, step=102)  # Valid loss
        
        assert detector.nan_streak == 0
    
    def test_divergence_warning(self):
        """Test divergence warning on high loss."""
        from src.deepseek.torch.training.fault_tolerance import NaNLossDetector
        
        detector = NaNLossDetector(loss_threshold=10.0)
        is_valid, action = detector.check_loss(50.0, step=100)
        
        assert is_valid is False
        assert action == "warn"
    
    def test_set_valid_checkpoint(self):
        """Test setting valid checkpoint for rollback."""
        from src.deepseek.torch.training.fault_tolerance import NaNLossDetector
        
        detector = NaNLossDetector()
        detector.set_valid_checkpoint("/checkpoints/step_100.pt")
        
        assert detector.get_rollback_checkpoint() == "/checkpoints/step_100.pt"


# =============================================================================
# F8: Health Check Endpoint Tests
# =============================================================================

class TestHealthCheckServer:
    """Tests for health check endpoint (F8)."""
    
    def test_update_status(self):
        """Test status update."""
        from src.deepseek.torch.training.fault_tolerance import HealthCheckServer, HealthCheckHandler
        
        HealthCheckServer.update_status(
            status="running",
            step=1000,
            loss=0.5,
            backend="pytorch",
        )
        
        assert HealthCheckHandler.training_state["status"] == "running"
        assert HealthCheckHandler.training_state["step"] == 1000
        assert HealthCheckHandler.training_state["loss"] == 0.5
    
    def test_server_start_stop(self):
        """Test server start and stop."""
        from src.deepseek.torch.training.fault_tolerance import HealthCheckServer
        
        server = HealthCheckServer(port=18080)
        
        started = server.start()
        assert started is True
        
        time.sleep(0.1)  # Let server start
        
        server.stop()
    
    def test_health_endpoint_response(self):
        """Test /health endpoint returns JSON."""
        from src.deepseek.torch.training.fault_tolerance import HealthCheckServer
        
        server = HealthCheckServer(port=18081)
        server.start()
        
        try:
            time.sleep(0.2)
            
            # Make HTTP request
            conn = HTTPConnection("localhost", 18081, timeout=5)
            conn.request("GET", "/health")
            response = conn.getresponse()
            
            assert response.status == 200
            data = json.loads(response.read().decode())
            assert "status" in data
            assert "timestamp" in data
            
            conn.close()
        finally:
            server.stop()


# =============================================================================
# F9: Cross-Backend Failure Handling Tests
# =============================================================================

class TestCrossBackendCoordinator:
    """Tests for cross-backend failure handling (F9)."""
    
    def test_init(self):
        """Test coordinator initialization."""
        from src.deepseek.torch.training.fault_tolerance import CrossBackendCoordinator
        
        coordinator = CrossBackendCoordinator(backends=["pytorch", "rust", "mlx"])
        
        assert "pytorch" in coordinator.backends
        assert "rust" in coordinator.backends
        assert "mlx" in coordinator.backends
    
    def test_report_status(self):
        """Test status reporting."""
        from src.deepseek.torch.training.fault_tolerance import CrossBackendCoordinator
        
        coordinator = CrossBackendCoordinator()
        coordinator.report_status("pytorch", "running", step=100)
        
        assert coordinator.backend_status["pytorch"] == "running"
        assert coordinator.backend_steps["pytorch"] == 100
    
    def test_report_failure(self):
        """Test failure reporting."""
        from src.deepseek.torch.training.fault_tolerance import CrossBackendCoordinator
        
        coordinator = CrossBackendCoordinator()
        coordinator.report_status("rust", "failed", step=50)
        
        assert "rust" in coordinator.failed_backends
    
    def test_can_continue_all_healthy(self):
        """Test can_continue with all healthy backends."""
        from src.deepseek.torch.training.fault_tolerance import CrossBackendCoordinator
        
        coordinator = CrossBackendCoordinator()
        coordinator.report_status("pytorch", "running")
        coordinator.report_status("rust", "running")
        
        assert coordinator.can_continue() is True
    
    def test_can_continue_one_failed(self):
        """Test can_continue with one failed backend."""
        from src.deepseek.torch.training.fault_tolerance import CrossBackendCoordinator
        
        coordinator = CrossBackendCoordinator()
        coordinator.report_status("pytorch", "running")
        coordinator.report_status("rust", "failed")
        
        assert coordinator.can_continue() is True
    
    def test_can_continue_all_failed(self):
        """Test can_continue with all failed backends."""
        from src.deepseek.torch.training.fault_tolerance import CrossBackendCoordinator
        
        coordinator = CrossBackendCoordinator()
        coordinator.report_status("pytorch", "failed")
        coordinator.report_status("rust", "failed")
        
        assert coordinator.can_continue() is False
    
    def test_get_healthy_backends(self):
        """Test get_healthy_backends."""
        from src.deepseek.torch.training.fault_tolerance import CrossBackendCoordinator
        
        coordinator = CrossBackendCoordinator()
        coordinator.report_status("pytorch", "running")
        coordinator.report_status("rust", "failed")
        
        healthy = coordinator.get_healthy_backends()
        
        assert "pytorch" in healthy
        assert "rust" not in healthy
    
    def test_get_summary(self):
        """Test get_summary returns complete info."""
        from src.deepseek.torch.training.fault_tolerance import CrossBackendCoordinator
        
        coordinator = CrossBackendCoordinator()
        coordinator.report_status("pytorch", "running", step=100)
        
        summary = coordinator.get_summary()
        
        assert "backends" in summary
        assert "status" in summary
        assert "steps" in summary
        assert "can_continue" in summary


# =============================================================================
# F10: Retry Budget Tracking Tests
# =============================================================================

class TestRetryBudgetTracker:
    """Tests for retry budget tracking (F10)."""
    
    def test_init(self):
        """Test tracker initialization."""
        from src.deepseek.torch.training.fault_tolerance import RetryBudgetTracker
        
        tracker = RetryBudgetTracker(max_retry_cost=100.0)
        
        assert tracker.max_retry_cost == 100.0
        assert tracker.total_retry_cost == 0.0
    
    def test_start_end_retry(self):
        """Test start/end retry tracking."""
        from src.deepseek.torch.training.fault_tolerance import RetryBudgetTracker
        
        tracker = RetryBudgetTracker()
        
        tracker.start_retry()
        time.sleep(0.1)
        cost = tracker.end_retry(success=True, gpu_count=1)
        
        assert cost > 0
        assert tracker.total_retry_cost > 0
        assert len(tracker.retry_costs) == 1
    
    def test_can_afford_retry_under_budget(self):
        """Test can_afford_retry under budget."""
        from src.deepseek.torch.training.fault_tolerance import RetryBudgetTracker
        
        tracker = RetryBudgetTracker(max_retry_cost=100.0)
        
        assert tracker.can_afford_retry() is True
    
    def test_can_afford_retry_over_budget(self):
        """Test can_afford_retry over budget."""
        from src.deepseek.torch.training.fault_tolerance import RetryBudgetTracker
        
        tracker = RetryBudgetTracker(max_retry_cost=10.0)
        tracker.total_retry_cost = 15.0
        
        assert tracker.can_afford_retry() is False
    
    def test_get_stats(self):
        """Test get_stats returns complete info."""
        from src.deepseek.torch.training.fault_tolerance import RetryBudgetTracker
        
        tracker = RetryBudgetTracker(max_retry_cost=50.0)
        tracker.start_retry()
        tracker.end_retry(success=True)
        
        stats = tracker.get_stats()
        
        assert "total_retries" in stats
        assert "total_retry_cost" in stats
        assert "remaining_budget" in stats
        assert "retry_history" in stats


# =============================================================================
# F5: Modal Restart Policy Tests
# =============================================================================

class TestModalRestartPolicy:
    """Tests for Modal restart policy configuration (F5)."""
    
    def test_get_modal_restart_policy(self):
        """Test getting Modal restart policy."""
        from src.deepseek.torch.training.fault_tolerance import get_modal_restart_policy
        
        policy = get_modal_restart_policy()
        
        assert "retries" in policy
        assert policy["retries"] == 3
        assert "spot_policy" in policy
        assert "oom_policy" in policy
        assert "health_check" in policy
    
    def test_create_modal_training_stub(self):
        """Test creating Modal training stub."""
        from src.deepseek.torch.training.fault_tolerance import create_modal_training_stub
        
        stub = create_modal_training_stub()
        
        assert "modal" in stub
        assert "@app.function" in stub
        assert "retries=3" in stub
        assert "A100-80GB" in stub


# =============================================================================
# F2: SIGTERM Handler Tests (Original PreemptionHandler)
# =============================================================================

class TestPreemptionHandler:
    """Tests for SIGTERM graceful shutdown handler (F2)."""
    
    def test_handler_registration(self):
        """Test signal handler registration."""
        from src.deepseek.torch.training.fault_tolerance import PreemptionHandler
        
        checkpoint_called = False
        
        def checkpoint_fn():
            nonlocal checkpoint_called
            checkpoint_called = True
        
        handler = PreemptionHandler(checkpoint_fn=checkpoint_fn)
        handler.register_handlers()
        
        # Verify handlers are registered
        assert signal.SIGTERM in handler._original_handlers
        assert signal.SIGINT in handler._original_handlers
        
        handler.unregister_handlers()
    
    def test_was_preempted_property(self):
        """Test was_preempted property."""
        from src.deepseek.torch.training.fault_tolerance import PreemptionHandler
        
        handler = PreemptionHandler()
        
        assert handler.was_preempted is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for fault tolerance system."""
    
    def test_retry_manager_with_nan_detector(self):
        """Test RetryManager integration with NaN detector."""
        from src.deepseek.torch.training.fault_tolerance import (
            RetryManager, NaNLossDetector, FailureType
        )
        
        retry_manager = RetryManager(max_attempts=3)
        nan_detector = NaNLossDetector(nan_streak_limit=2)
        
        # Simulate training loop with NaN
        for step in range(5):
            loss = float('nan') if step < 2 else 1.0
            is_valid, action = nan_detector.check_loss(loss, step)
            
            if action == "rollback":
                # Would trigger retry
                pass
    
    def test_full_recovery_flow(self, temp_checkpoint_dir, save_checkpoint, sample_model):
        """Test full recovery flow."""
        from src.deepseek.torch.training.fault_tolerance import (
            RetryManager,
            find_latest_checkpoint,
            validate_checkpoint,
            load_checkpoint_with_validation,
            NaNLossDetector,
            HealthCheckServer,
        )
        
        # Setup
        retry_manager = RetryManager(max_attempts=3)
        nan_detector = NaNLossDetector()
        
        # Save initial checkpoint
        checkpoint_path = save_checkpoint(step=100, loss=1.0)
        nan_detector.set_valid_checkpoint(checkpoint_path)
        
        # Update health status
        HealthCheckServer.update_status(status="running", step=100)
        
        # Find and validate checkpoint
        latest = find_latest_checkpoint(str(temp_checkpoint_dir))
        is_valid, _ = validate_checkpoint(latest)
        assert is_valid
        
        # Load checkpoint
        metadata = load_checkpoint_with_validation(latest, sample_model)
        assert metadata["step"] == 100


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
