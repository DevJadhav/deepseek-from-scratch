"""Tests for downstream evaluation module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import torch

from scripts.downstream_eval import (
    EvaluationResult,
)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating evaluation result."""
        result = EvaluationResult(
            task="hellaswag",
            accuracy=0.75,
            num_samples=100,
            correct=75,
        )

        assert result.task == "hellaswag"
        assert result.accuracy == 0.75
        assert result.num_samples == 100
        assert result.correct == 75

    def test_result_with_metadata(self) -> None:
        """Test result with metadata."""
        result = EvaluationResult(
            task="lambada",
            accuracy=0.60,
            num_samples=50,
            correct=30,
            metadata={"perplexity": 12.5},
        )

        assert result.metadata["perplexity"] == 12.5

    def test_result_to_dict(self) -> None:
        """Test result serialization."""
        result = EvaluationResult(
            task="hellaswag",
            accuracy=0.80,
            num_samples=200,
            correct=160,
        )

        d = result.to_dict()
        assert d["task"] == "hellaswag"
        assert d["accuracy"] == 0.80
        assert d["num_samples"] == 200
        assert d["correct"] == 160

    def test_result_timestamp(self) -> None:
        """Test result has timestamp."""
        before = datetime.now()
        result = EvaluationResult(
            task="test",
            accuracy=0.5,
            num_samples=10,
            correct=5,
        )
        after = datetime.now()

        assert before <= result.timestamp <= after


class TestHellaSwagEvaluatorUnit:
    """Unit tests for HellaSwag evaluator without model."""

    def test_hellaswag_import(self) -> None:
        """Test HellaSwag evaluator can be imported."""
        from scripts.downstream_eval import HellaSwagEvaluator

        assert HellaSwagEvaluator is not None

    def test_hellaswag_requires_model(self) -> None:
        """Test HellaSwag requires model and tokenizer."""
        from scripts.downstream_eval import HellaSwagEvaluator

        # Create mock model and tokenizer
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=None)
        mock_tokenizer = MagicMock()

        evaluator = HellaSwagEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
        )

        assert evaluator.task_name == "hellaswag"


class TestLAMBADAEvaluatorUnit:
    """Unit tests for LAMBADA evaluator without model."""

    def test_lambada_import(self) -> None:
        """Test LAMBADA evaluator can be imported."""
        from scripts.downstream_eval import LAMBADAEvaluator

        assert LAMBADAEvaluator is not None

    def test_lambada_requires_model(self) -> None:
        """Test LAMBADA requires model and tokenizer."""
        from scripts.downstream_eval import LAMBADAEvaluator

        # Create mock model and tokenizer
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=None)
        mock_tokenizer = MagicMock()

        evaluator = LAMBADAEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=torch.device("cpu"),
        )

        assert evaluator.task_name == "lambada"


class TestEvaluatorConfig:
    """Tests for evaluator configuration."""

    def test_batch_size_config(self) -> None:
        """Test custom batch size."""
        from scripts.downstream_eval import HellaSwagEvaluator

        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=None)
        mock_tokenizer = MagicMock()

        evaluator = HellaSwagEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            batch_size=32,
            device=torch.device("cpu"),
        )

        assert evaluator.batch_size == 32

    def test_max_samples_config(self) -> None:
        """Test max samples configuration."""
        from scripts.downstream_eval import LAMBADAEvaluator

        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=None)
        mock_tokenizer = MagicMock()

        evaluator = LAMBADAEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            max_samples=100,
            device=torch.device("cpu"),
        )

        assert evaluator.max_samples == 100

    def test_dtype_config(self) -> None:
        """Test data type configuration."""
        from scripts.downstream_eval import HellaSwagEvaluator

        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=None)
        mock_tokenizer = MagicMock()

        evaluator = HellaSwagEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert evaluator.dtype == torch.float32


class TestComparisonReport:
    """Tests for model comparison functionality."""

    def test_comparison_import(self) -> None:
        """Test ComparisonReport can be imported."""
        from scripts.downstream_eval import ComparisonReport

        assert ComparisonReport is not None

    def test_comparison_add_result(self) -> None:
        """Test adding results to comparison."""
        import tempfile

        from scripts.downstream_eval import ComparisonReport

        with tempfile.TemporaryDirectory() as tmpdir:
            comparison = ComparisonReport(output_dir=tmpdir)
            result = EvaluationResult(
                task="hellaswag",
                accuracy=0.70,
                num_samples=100,
                correct=70,
            )

            comparison.add_result("model_a", result)

            assert "model_a" in comparison.results
            assert "hellaswag" in comparison.results["model_a"]

    def test_comparison_summary(self) -> None:
        """Test comparison summary generation."""
        import tempfile

        from scripts.downstream_eval import ComparisonReport

        with tempfile.TemporaryDirectory() as tmpdir:
            comparison = ComparisonReport(output_dir=tmpdir)
            result1 = EvaluationResult(
                task="hellaswag",
                accuracy=0.70,
                num_samples=100,
                correct=70,
            )
            result2 = EvaluationResult(
                task="hellaswag",
                accuracy=0.75,
                num_samples=100,
                correct=75,
            )

            comparison.add_result("model_a", result1)
            comparison.add_result("model_b", result2)

            summary = comparison.get_comparison_summary()

            assert "models" in summary
            assert "best_per_task" in summary
            assert summary["best_per_task"]["hellaswag"]["model"] == "model_b"


class TestEdgeCases:
    """Edge case tests."""

    def test_result_with_zero_samples(self) -> None:
        """Test result with zero samples."""
        result = EvaluationResult(
            task="test",
            accuracy=0.0,
            num_samples=0,
            correct=0,
        )

        assert result.num_samples == 0
        assert result.accuracy == 0.0

    def test_result_perfect_accuracy(self) -> None:
        """Test result with perfect accuracy."""
        result = EvaluationResult(
            task="test",
            accuracy=1.0,
            num_samples=100,
            correct=100,
        )

        assert result.accuracy == 1.0
        assert result.correct == result.num_samples

    def test_result_metadata_empty(self) -> None:
        """Test result with empty metadata."""
        result = EvaluationResult(
            task="test",
            accuracy=0.5,
            num_samples=10,
            correct=5,
        )

        assert result.metadata == {}


class TestResultPersistence:
    """Tests for result persistence."""

    def test_result_to_dict_round_trip(self) -> None:
        """Test result can be converted to dict and back."""

        result = EvaluationResult(
            task="hellaswag",
            accuracy=0.75,
            num_samples=100,
            correct=75,
            metadata={"batch_size": 8},
        )

        d = result.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)

        assert loaded["task"] == "hellaswag"
        assert loaded["accuracy"] == 0.75
        assert loaded["metadata"]["batch_size"] == 8

    def test_save_results_to_file(self) -> None:
        """Test saving results to file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            result = EvaluationResult(
                task="test",
                accuracy=0.80,
                num_samples=50,
                correct=40,
            )

            filepath = Path(tmpdir) / "results.json"
            with open(filepath, "w") as f:
                json.dump(result.to_dict(), f)

            with open(filepath) as f:
                loaded = json.load(f)

            assert loaded["task"] == "test"
            assert loaded["accuracy"] == 0.80
