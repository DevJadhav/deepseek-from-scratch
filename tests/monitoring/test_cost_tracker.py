"""Tests for cost tracking functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from monitoring.cost_tracker import (
    AlertLevel,
    CostAlert,
    CostRecord,
    CostTracker,
    GPUType,
)


class TestGPUType:
    """Tests for GPUType enum."""

    def test_h100_rate(self) -> None:
        """Test H100 hourly rate."""
        assert GPUType.H100.hourly_rate == 3.95

    def test_a100_80gb_rate(self) -> None:
        """Test A100 80GB hourly rate."""
        assert GPUType.A100_80GB.hourly_rate == 2.78

    def test_a100_40gb_rate(self) -> None:
        """Test A100 40GB hourly rate."""
        assert GPUType.A100_40GB.hourly_rate == 2.21


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_levels(self) -> None:
        """Test alert level values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EXCEEDED.value == "exceeded"


class TestCostAlert:
    """Tests for CostAlert dataclass."""

    def test_create_alert(self) -> None:
        """Test creating a cost alert."""
        alert = CostAlert(
            level=AlertLevel.WARNING,
            threshold_percent=75.0,
            current_cost=75.0,
            budget_limit=100.0,
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.threshold_percent == 75.0
        assert "75%" in alert.message

    def test_alert_to_dict(self) -> None:
        """Test serialization to dict."""
        alert = CostAlert(
            level=AlertLevel.CRITICAL,
            threshold_percent=90.0,
            current_cost=90.0,
            budget_limit=100.0,
        )

        d = alert.to_dict()
        assert d["level"] == "critical"
        assert d["threshold_percent"] == 90.0
        assert d["current_cost"] == 90.0

    def test_alert_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "level": "warning",
            "threshold_percent": 75.0,
            "current_cost": 75.0,
            "budget_limit": 100.0,
            "timestamp": "2024-01-01T10:00:00",
            "message": "Test message",
        }

        alert = CostAlert.from_dict(data)
        assert alert.level == AlertLevel.WARNING
        assert alert.threshold_percent == 75.0


class TestCostRecord:
    """Tests for CostRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating a cost record."""
        record = CostRecord(
            stage="pretrain",
            gpu_type=GPUType.H100,
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 0, 0),
        )

        assert record.gpu_hours == pytest.approx(1.0)
        assert record.cost == pytest.approx(3.95)

    def test_record_finalize(self) -> None:
        """Test finalizing a record."""
        record = CostRecord(
            stage="pretrain",
            gpu_type=GPUType.H100,
            start_time=datetime(2024, 1, 1, 10, 0, 0),
        )
        record.finalize(datetime(2024, 1, 1, 12, 0, 0))

        assert record.gpu_hours == pytest.approx(2.0)
        assert record.cost == pytest.approx(7.90)

    def test_record_to_dict(self) -> None:
        """Test serialization to dict."""
        record = CostRecord(
            stage="pretrain",
            gpu_type=GPUType.H100,
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 11, 0, 0),
        )

        d = record.to_dict()
        assert d["stage"] == "pretrain"
        assert d["gpu_type"] == "H100"
        assert "start_time" in d
        assert "end_time" in d

    def test_record_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "stage": "finetune",
            "gpu_type": "H100",
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T11:00:00",
            "gpu_hours": 1.0,
            "cost": 3.95,
            "metadata": {},
        }

        record = CostRecord.from_dict(data)
        assert record.stage == "finetune"
        assert record.gpu_type == GPUType.H100
        assert record.gpu_hours == 1.0


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_create_tracker(self) -> None:
        """Test creating a cost tracker with defaults."""
        tracker = CostTracker()

        assert tracker.budget_limit == 1000.0
        assert tracker.total_cost == 0.0
        assert len(tracker.records) == 0

    def test_create_tracker_custom_budget(self) -> None:
        """Test creating tracker with custom budget."""
        tracker = CostTracker(budget_limit=500.0)

        assert tracker.budget_limit == 500.0

    def test_add_gpu_time(self) -> None:
        """Test adding GPU time records."""
        tracker = CostTracker()

        tracker.add_gpu_time(
            hours=1.0,
            stage="pretrain",
        )

        assert tracker.total_cost == pytest.approx(3.95)
        assert len(tracker.records) == 1

    def test_add_multiple_records(self) -> None:
        """Test accumulating multiple records."""
        tracker = CostTracker()

        tracker.add_gpu_time(
            hours=1.0,
            stage="pretrain",
        )
        tracker.add_gpu_time(
            hours=0.5,
            stage="finetune",
        )

        # 3.95 + (3.95 * 0.5) = 3.95 + 1.975 = 5.925
        assert tracker.total_cost == pytest.approx(5.925)
        assert len(tracker.records) == 2

    def test_budget_percentage(self) -> None:
        """Test budget percentage calculation."""
        tracker = CostTracker(budget_limit=100.0)

        # 1 hour = $3.95
        tracker.add_gpu_time(
            hours=1.0,
            stage="pretrain",
        )

        assert tracker.budget_percent_used == pytest.approx(3.95)

    def test_remaining_budget(self) -> None:
        """Test remaining budget calculation."""
        tracker = CostTracker(budget_limit=100.0)

        tracker.add_gpu_time(
            hours=1.0,
            stage="pretrain",
        )

        assert tracker.remaining_budget == pytest.approx(96.05)

    def test_total_gpu_hours(self) -> None:
        """Test total GPU hours calculation."""
        tracker = CostTracker()

        tracker.add_gpu_time(
            hours=2.0,
            stage="pretrain",
        )
        tracker.add_gpu_time(
            hours=1.5,
            stage="finetune",
        )

        assert tracker.total_gpu_hours == pytest.approx(3.5)

    def test_is_over_budget(self) -> None:
        """Test over budget detection."""
        tracker = CostTracker(budget_limit=5.0)

        tracker.add_gpu_time(
            hours=2.0,  # $7.90, over $5 budget
            stage="pretrain",
        )

        assert tracker.is_over_budget is True

    def test_estimated_hours_remaining(self) -> None:
        """Test estimated hours remaining."""
        tracker = CostTracker(budget_limit=100.0)

        # $100 budget / $3.95 per hour = ~25.32 hours
        assert tracker.estimated_hours_remaining == pytest.approx(25.316, rel=0.01)

    def test_session_tracking(self) -> None:
        """Test start and end session."""
        tracker = CostTracker()

        record = tracker.start_session("pretrain")
        assert record is not None
        assert record.stage == "pretrain"

        # Simulate some time passing (we can't easily test real time)
        # Just end the session
        result = tracker.end_session()
        assert result is not None
        assert len(tracker.records) == 1

    def test_session_with_custom_gpu_type(self) -> None:
        """Test session with custom GPU type."""
        tracker = CostTracker()

        tracker.start_session("pretrain", gpu_type=GPUType.A100_80GB)
        record = tracker.end_session()

        assert record is not None
        assert record.gpu_type == GPUType.A100_80GB

    def test_add_gpu_time_with_different_gpu(self) -> None:
        """Test adding GPU time with different GPU type."""
        tracker = CostTracker()

        tracker.add_gpu_time(
            hours=1.0,
            stage="pretrain",
            gpu_type=GPUType.A100_80GB,
        )

        assert tracker.total_cost == pytest.approx(2.78)

    def test_save_and_load(self) -> None:
        """Test persisting tracker to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "cost_tracking.json"

            tracker1 = CostTracker(budget_limit=500.0, persist_path=filepath)
            tracker1.add_gpu_time(
                hours=1.0,
                stage="pretrain",
            )
            tracker1.save()

            tracker2 = CostTracker(persist_path=filepath)
            tracker2.load()

            assert tracker2.budget_limit == 500.0
            assert tracker2.total_cost == pytest.approx(3.95)
            assert len(tracker2.records) == 1


class TestCostTrackerIntegration:
    """Integration tests for cost tracking."""

    def test_full_training_simulation(self) -> None:
        """Simulate a full training run with multiple stages."""
        tracker = CostTracker(budget_limit=100.0)

        # Stage 1: Data preparation (2 hours)
        tracker.add_gpu_time(
            hours=2.0,
            stage="data_prep",
        )

        # Stage 2: Pretraining (10 hours)
        tracker.add_gpu_time(
            hours=10.0,
            stage="pretrain",
        )

        # Stage 3: Fine-tuning (3 hours)
        tracker.add_gpu_time(
            hours=3.0,
            stage="finetune",
        )

        # Check totals
        # data_prep: 2 * 3.95 = 7.90
        # pretrain: 10 * 3.95 = 39.50
        # finetune: 3 * 3.95 = 11.85
        # Total: 59.25
        assert tracker.total_cost == pytest.approx(59.25)
        assert tracker.total_gpu_hours == pytest.approx(15.0)

    def test_alerts_at_thresholds(self) -> None:
        """Test that alerts are triggered at correct thresholds."""
        tracker = CostTracker(budget_limit=10.0, alert_thresholds=[50.0, 75.0, 90.0])

        # Add cost to trigger 50% threshold
        tracker.add_gpu_time(
            hours=1.4,  # ~$5.53, 55% of $10
            stage="pretrain",
        )

        # Check that we have alerts
        assert len(tracker.alerts) >= 1

    def test_persistence_with_alerts(self) -> None:
        """Test that alert state persists correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "tracking.json"

            tracker1 = CostTracker(budget_limit=10.0, persist_path=filepath)
            tracker1.add_gpu_time(
                hours=1.4,  # ~$5.53, 55% of $10
                stage="pretrain",
            )
            tracker1.save()

            tracker2 = CostTracker(persist_path=filepath)
            tracker2.load()

            # Should have same cost
            assert tracker2.total_cost == pytest.approx(5.53)


class TestCostTrackerEdgeCases:
    """Edge case tests for CostTracker."""

    def test_zero_budget(self) -> None:
        """Test with zero budget."""
        tracker = CostTracker(budget_limit=0.0)

        tracker.add_gpu_time(hours=1.0, stage="pretrain")

        assert tracker.budget_percent_used == 100.0

    def test_empty_tracker(self) -> None:
        """Test empty tracker properties."""
        tracker = CostTracker()

        assert tracker.total_cost == 0.0
        assert tracker.total_gpu_hours == 0.0
        assert tracker.remaining_budget == 1000.0

    def test_end_session_without_start(self) -> None:
        """Test ending session without starting."""
        tracker = CostTracker()

        result = tracker.end_session()
        assert result is None
