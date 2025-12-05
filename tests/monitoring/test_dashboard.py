"""Tests for dashboard module."""

from datetime import datetime

import pytest

from monitoring.dashboard import TrainingDashboard, TrainingMetrics


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            step=100,
            total_steps=1000,
            loss=2.5,
            learning_rate=1e-4,
        )

        assert metrics.step == 100
        assert metrics.total_steps == 1000
        assert metrics.loss == 2.5
        assert metrics.learning_rate == 1e-4

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = TrainingMetrics()

        assert metrics.step == 0
        assert metrics.total_steps == 10000
        assert metrics.loss == 0.0
        assert metrics.stage == "pretrain"
        assert metrics.gpu_memory_total == 80.0  # H100 default

    def test_progress_percent(self) -> None:
        """Test progress percentage calculation."""
        metrics = TrainingMetrics(step=250, total_steps=1000)

        assert metrics.progress_percent == 25.0

    def test_progress_percent_zero_steps(self) -> None:
        """Test progress with zero total steps."""
        metrics = TrainingMetrics(step=100, total_steps=0)

        assert metrics.progress_percent == 0.0

    def test_gpu_memory_percent(self) -> None:
        """Test GPU memory percentage calculation."""
        metrics = TrainingMetrics(gpu_memory_used=40.0, gpu_memory_total=80.0)

        assert metrics.gpu_memory_percent == 50.0

    def test_gpu_memory_percent_zero_total(self) -> None:
        """Test GPU memory with zero total."""
        metrics = TrainingMetrics(gpu_memory_used=10.0, gpu_memory_total=0.0)

        assert metrics.gpu_memory_percent == 0.0


class TestTrainingDashboard:
    """Tests for TrainingDashboard class."""

    def test_create_dashboard(self) -> None:
        """Test creating a dashboard."""
        dashboard = TrainingDashboard()

        assert dashboard.cost_tracker is None
        assert dashboard.refresh_rate == 4
        assert dashboard.title == "DeepSeek Training Pipeline"

    def test_create_dashboard_custom_params(self) -> None:
        """Test creating dashboard with custom parameters."""
        dashboard = TrainingDashboard(
            refresh_rate=10,
            title="Custom Dashboard",
        )

        assert dashboard.refresh_rate == 10
        assert dashboard.title == "Custom Dashboard"

    def test_dashboard_has_metrics(self) -> None:
        """Test that dashboard has metrics object."""
        dashboard = TrainingDashboard()

        assert isinstance(dashboard.metrics, TrainingMetrics)
        assert dashboard.metrics.step == 0

    def test_update_metrics(self) -> None:
        """Test updating dashboard metrics without live display."""
        dashboard = TrainingDashboard()

        # Update method should work even without live display
        dashboard.metrics.step = 100
        dashboard.metrics.loss = 2.5

        assert dashboard.metrics.step == 100
        assert dashboard.metrics.loss == 2.5


class TestDashboardFormatting:
    """Tests for dashboard formatting methods."""

    def test_metrics_formatting(self) -> None:
        """Test that metrics are formatted correctly."""
        metrics = TrainingMetrics(
            step=500,
            total_steps=1000,
            loss=0.123456789,
            learning_rate=1e-5,
            tokens_per_second=12345.6,
        )

        # Test that values are stored correctly for formatting
        assert metrics.step == 500
        assert metrics.loss == pytest.approx(0.123456789)
        assert metrics.learning_rate == 1e-5
        assert metrics.tokens_per_second == pytest.approx(12345.6)

    def test_metrics_timestamp(self) -> None:
        """Test that metrics have timestamp."""
        before = datetime.now()
        metrics = TrainingMetrics()
        after = datetime.now()

        assert before <= metrics.timestamp <= after


class TestDashboardIntegration:
    """Integration tests for dashboard with cost tracker."""

    def test_dashboard_with_cost_tracker(self) -> None:
        """Test dashboard works with cost tracker."""
        from monitoring.cost_tracker import CostTracker

        tracker = CostTracker(budget_limit=100.0)
        dashboard = TrainingDashboard(cost_tracker=tracker)

        assert dashboard.cost_tracker is not None
        assert dashboard.cost_tracker.budget_limit == 100.0

    def test_dashboard_reflects_cost_changes(self) -> None:
        """Test dashboard sees cost tracker changes."""
        from monitoring.cost_tracker import CostTracker

        tracker = CostTracker(budget_limit=100.0)
        dashboard = TrainingDashboard(cost_tracker=tracker)

        # Add some cost
        tracker.add_gpu_time(hours=1.0, stage="pretrain")

        assert dashboard.cost_tracker is not None
        assert dashboard.cost_tracker.total_cost == pytest.approx(3.95)


class TestDashboardEdgeCases:
    """Edge case tests for dashboard."""

    def test_empty_loss_history(self) -> None:
        """Test dashboard handles empty loss history."""
        dashboard = TrainingDashboard()

        # Should not raise errors with no loss history
        assert len(dashboard._loss_history) == 0

    def test_metrics_with_extreme_values(self) -> None:
        """Test metrics with extreme values."""
        metrics = TrainingMetrics(
            step=999999999,
            total_steps=1000000000,
            loss=0.000001,
            learning_rate=1e-10,
            tokens_per_second=1000000.0,
        )

        assert metrics.progress_percent == pytest.approx(99.9999999)
        assert metrics.loss == 1e-6
        assert metrics.learning_rate == 1e-10

    def test_dashboard_without_starting(self) -> None:
        """Test accessing dashboard without starting."""
        dashboard = TrainingDashboard()

        # These should be None before starting
        assert dashboard._live is None
        assert dashboard._progress is None
        assert dashboard._task_id is None
