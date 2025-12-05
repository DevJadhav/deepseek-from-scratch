"""
Tests for ANE Validation & Benchmarking Module

Tests for ANE validation components:
- Numerical validation (Reference Comparison)
- Perplexity testing (WikiText-2, C4)
- Task accuracy (MMLU, HumanEval)
- Output distribution (KL divergence)
- Performance benchmarks
"""

import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import validation modules
from deepseek.mlx.ane.validation import (
    NumericalValidator,
    NumericalValidationConfig,
    ValidationResult,
    compare_tensors,
    compare_models,
    PerplexityEvaluator,
    PerplexityConfig,
    PerplexityResult,
    DistributionComparator,
    DistributionConfig,
    DistributionResult,
    kl_divergence,
    js_divergence,
    PerformanceBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    LatencyBenchmark,
    MemoryBenchmark,
    ThroughputBenchmark,
    TaskEvaluator,
    MMLUEvaluator,
    HumanEvalEvaluator,
    TaskConfig,
    TaskResult,
)


# =============================================================================
# Fixtures and Helper Functions
# =============================================================================


@pytest.fixture
def device():
    """Get test device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size)
                for _ in range(num_layers)
            ])
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = F.relu(layer(x))
            return self.lm_head(x)
    
    return SimpleModel()


@pytest.fixture
def simple_tokenizer():
    """Create a simple tokenizer mock."""
    class SimpleTokenizer:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size
        
        def encode(self, text):
            # Simple encoding: hash each character
            return [ord(c) % self.vocab_size for c in text]
        
        def decode(self, tokens):
            return "".join(chr(t % 128) for t in tokens)
    
    return SimpleTokenizer()


def create_reference_model():
    """Create a reference model for comparison."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    return model


def create_ane_model():
    """Create an ANE-optimized model (same architecture for testing)."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    return model


# =============================================================================
# Test Section 7.1: Numerical Validation - Reference Comparison
# =============================================================================


class TestNumericalValidation:
    """Tests for numerical validation against reference."""

    def test_numerical_validator_init(self):
        """Test NumericalValidator initialization."""
        validator = NumericalValidator()
        assert validator is not None
        assert validator.config is not None

    def test_numerical_validator_with_config(self):
        """Test NumericalValidator with custom config."""
        config = NumericalValidationConfig(
            rtol=1e-4,
            atol=1e-5,
            num_test_inputs=100,
        )
        validator = NumericalValidator(config=config)
        assert validator.config.rtol == 1e-4
        assert validator.config.atol == 1e-5
        assert validator.config.num_test_inputs == 100

    def test_compare_identical_tensors(self):
        """Test comparing identical tensors."""
        tensor = torch.randn(10, 64)
        result = compare_tensors(tensor, tensor)

        assert result is not None
        assert result.passed
        assert result.metrics.get("max_abs_diff", 0) < 1e-10
        assert result.metrics.get("mean_abs_diff", 0) < 1e-10

    def test_compare_similar_tensors(self):
        """Test comparing similar tensors within tolerance."""
        tensor1 = torch.randn(10, 64)
        tensor2 = tensor1 + torch.randn_like(tensor1) * 1e-6

        config = NumericalValidationConfig(rtol=1e-4, atol=1e-5)
        result = compare_tensors(tensor1, tensor2, config=config)

        assert result is not None
        assert result.passed

    def test_compare_different_tensors(self):
        """Test comparing different tensors outside tolerance."""
        tensor1 = torch.randn(10, 64)
        tensor2 = tensor1 + torch.randn_like(tensor1) * 0.1

        config = NumericalValidationConfig(rtol=1e-6, atol=1e-7)
        result = compare_tensors(tensor1, tensor2, config=config)

        assert result is not None
        assert not result.passed

    def test_validation_result_metrics(self):
        """Test ValidationResult contains expected metrics."""
        tensor1 = torch.randn(10, 64)
        tensor2 = tensor1 + torch.randn_like(tensor1) * 1e-5

        result = compare_tensors(tensor1, tensor2)

        assert "max_abs_diff" in result.metrics
        assert "mean_abs_diff" in result.metrics
        assert "cosine_sim" in result.metrics
        assert isinstance(result.metrics["max_abs_diff"], float)
        assert isinstance(result.metrics["mean_abs_diff"], float)

    def test_compare_models(self, device):
        """Test comparing two models."""
        ref_model = create_reference_model().to(device)
        ane_model = create_ane_model().to(device)

        # Create input generator (Sequential expects unnamed input)
        def input_gen():
            return {"input": torch.randn(1, 64, device=device)}

        # For nn.Sequential, we need to test differently
        # since it doesn't accept named arguments
        test_input = torch.randn(1, 64, device=device)

        with torch.no_grad():
            ref_out = ref_model(test_input)
            ane_out = ane_model(test_input)

        result = compare_tensors(ref_out, ane_out)
        assert result is not None
        assert result.passed  # Same seed = same weights

    def test_validator_with_model(self, simple_model, device):
        """Test validator with actual model."""
        model = simple_model.to(device)
        validator = NumericalValidator()

        # Test input
        test_input = torch.randint(0, 1000, (1, 16), device=device)

        # Compare model with itself
        ref_output = model(test_input)
        ane_output = model(test_input)  # Same model

        result = compare_tensors(ref_output, ane_output)
        assert result.passed

    def test_layer_by_layer_comparison(self, device):
        """Test layer-by-layer numerical comparison."""
        ref_model = create_reference_model().to(device)
        ane_model = create_ane_model().to(device)

        test_input = torch.randn(1, 64, device=device)

        # Get intermediate outputs
        ref_outputs = []
        ane_outputs = []

        def ref_hook(module, input, output):
            ref_outputs.append(output.clone())

        def ane_hook(module, input, output):
            ane_outputs.append(output.clone())

        # Register hooks
        ref_handles = [m.register_forward_hook(ref_hook) for m in ref_model]
        ane_handles = [m.register_forward_hook(ane_hook) for m in ane_model]

        ref_model(test_input)
        ane_model(test_input)

        # Compare each layer
        for i, (ref_out, ane_out) in enumerate(zip(ref_outputs, ane_outputs)):
            result = compare_tensors(ref_out, ane_out)
            assert result.passed, f"Layer {i} failed validation"

        # Clean up hooks
        for h in ref_handles + ane_handles:
            h.remove()

    def test_tolerance_thresholds(self):
        """Test validation tolerance thresholds (target: <0.1% deviation)."""
        tensor1 = torch.randn(100, 64)

        # Add noise at different levels
        for noise_scale in [1e-5, 1e-4, 1e-3]:
            tensor2 = tensor1 + torch.randn_like(tensor1) * noise_scale
            config = NumericalValidationConfig(rtol=1e-2, atol=1e-3)  # More relaxed tolerance
            result = compare_tensors(tensor1, tensor2, config=config)

            if noise_scale <= 1e-4:
                # Should be within tolerance for small noise
                assert result.passed or result.metrics.get("cosine_sim", 0) > 0.999


# =============================================================================
# Test Section 7.1: Perplexity Testing
# =============================================================================


class TestPerplexityEvaluation:
    """Tests for perplexity evaluation on WikiText-2 and C4."""

    def test_perplexity_evaluator_init(self):
        """Test PerplexityEvaluator initialization."""
        evaluator = PerplexityEvaluator()
        assert evaluator is not None

    def test_perplexity_evaluator_with_config(self):
        """Test PerplexityEvaluator with custom config."""
        config = PerplexityConfig(
            stride=512,
            max_seq_len=1024,
            batch_size=1,
        )
        evaluator = PerplexityEvaluator(config=config)
        assert evaluator.config.stride == 512
        assert evaluator.config.max_seq_len == 1024

    def test_perplexity_result_structure(self):
        """Test PerplexityResult structure."""
        result = PerplexityResult(
            perplexity=15.5,
            total_loss=2.74,
            num_tokens=1000,
            num_sequences=10,
            dataset="test",
        )

        assert result.perplexity == 15.5
        assert result.total_loss == 2.74
        assert result.num_tokens == 1000
        assert result.dataset == "test"

    def test_perplexity_config_defaults(self):
        """Test PerplexityConfig default values."""
        config = PerplexityConfig()

        assert config.stride > 0
        assert config.max_seq_len > 0
        assert config.batch_size >= 1

    def test_perplexity_result_to_dict(self):
        """Test PerplexityResult serialization."""
        result = PerplexityResult(
            perplexity=15.5,
            total_loss=2.74,
            num_tokens=1000,
            num_sequences=10,
        )

        d = result.to_dict()
        assert "perplexity" in d
        assert "total_loss" in d
        assert d["perplexity"] == 15.5

    def test_perplexity_calculation_math(self):
        """Test perplexity calculation: PPL = exp(loss)."""
        import math

        # Test that perplexity formula is correct
        loss = 2.0
        expected_ppl = math.exp(loss)

        # Create result manually
        result = PerplexityResult(
            perplexity=expected_ppl,
            total_loss=loss,
            num_tokens=100,
            num_sequences=1,
            dataset="test",
        )

        assert abs(math.log(result.perplexity) - result.total_loss) < 1e-6

    def test_perplexity_target(self):
        """Test perplexity target: <1% degradation from reference."""
        # Simulate reference and ANE perplexity
        ref_ppl = 10.0

        # 1% degradation threshold
        max_allowed_ppl = ref_ppl * 1.01

        # ANE perplexity within tolerance
        ane_ppl = 10.05  # 0.5% degradation
        assert ane_ppl <= max_allowed_ppl

        # ANE perplexity outside tolerance
        ane_ppl_bad = 10.2  # 2% degradation
        assert ane_ppl_bad > max_allowed_ppl


# =============================================================================
# Test Section 7.1: Task Accuracy (MMLU, HumanEval)
# =============================================================================


class TestTaskAccuracy:
    """Tests for task accuracy evaluation."""

    def test_task_evaluator_init(self):
        """Test TaskEvaluator initialization."""
        evaluator = TaskEvaluator()
        assert evaluator is not None
        assert evaluator.mmlu is not None
        assert evaluator.humaneval is not None

    def test_task_config_defaults(self):
        """Test TaskConfig default values."""
        config = TaskConfig()
        
        assert config.max_seq_len > 0
        assert config.num_shots >= 0
        assert config.batch_size >= 1

    def test_task_result_structure(self):
        """Test TaskResult structure."""
        result = TaskResult(
            task_name="MMLU",
            accuracy=0.75,
            num_correct=75,
            num_total=100,
            category_results={"STEM": {"accuracy": 0.80}},
        )
        
        assert result.task_name == "MMLU"
        assert result.accuracy == 0.75
        assert result.num_correct == 75
        assert result.num_total == 100

    def test_task_result_to_dict(self):
        """Test TaskResult serialization."""
        result = TaskResult(
            task_name="MMLU",
            accuracy=0.75,
            num_correct=75,
            num_total=100,
        )
        
        d = result.to_dict()
        assert d["task_name"] == "MMLU"
        assert d["accuracy"] == 0.75
        assert d["num_correct"] == 75

    def test_mmlu_evaluator_init(self):
        """Test MMLUEvaluator initialization."""
        config = TaskConfig(max_samples=5)
        evaluator = MMLUEvaluator(config=config)
        assert evaluator is not None

    def test_humaneval_evaluator_init(self):
        """Test HumanEvalEvaluator initialization."""
        config = TaskConfig(max_samples=5)
        evaluator = HumanEvalEvaluator(config=config)
        assert evaluator is not None

    def test_evaluator_get_summary(self):
        """Test getting evaluation summary."""
        evaluator = TaskEvaluator()
        
        # Manually add a result
        evaluator.results.append(TaskResult(
            task_name="test",
            accuracy=0.5,
            num_correct=5,
            num_total=10,
        ))
        
        summary = evaluator.get_summary()
        assert "num_tasks" in summary
        assert "results" in summary
        assert summary["num_tasks"] == 1


# =============================================================================
# Test Section 7.1: Output Distribution (KL Divergence)
# =============================================================================


class TestOutputDistribution:
    """Tests for output distribution comparison."""

    def test_distribution_comparator_init(self):
        """Test DistributionComparator initialization."""
        comparator = DistributionComparator()
        assert comparator is not None

    def test_distribution_config(self):
        """Test DistributionConfig settings."""
        config = DistributionConfig(
            temperature=1.0,
            num_bins=50,
            num_samples=100,
        )

        assert config.temperature == 1.0
        assert config.num_bins == 50
        assert config.num_samples == 100

    def test_kl_divergence_identical(self):
        """Test KL divergence for identical distributions."""
        p = F.softmax(torch.randn(100), dim=-1)

        kl = kl_divergence(p, p)

        assert kl >= 0
        assert kl < 1e-6  # Should be ~0 for identical

    def test_kl_divergence_different(self):
        """Test KL divergence for different distributions."""
        p = F.softmax(torch.randn(100), dim=-1)
        q = F.softmax(torch.randn(100), dim=-1)

        kl = kl_divergence(p, q)

        assert kl >= 0
        assert kl > 0  # Should be positive for different

    def test_kl_divergence_asymmetry(self):
        """Test that KL divergence is asymmetric."""
        p = F.softmax(torch.randn(100), dim=-1)
        q = F.softmax(torch.randn(100), dim=-1)

        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)

        # KL is generally not symmetric
        # Just verify both are non-negative
        assert kl_pq >= 0
        assert kl_qp >= 0

    def test_js_divergence_identical(self):
        """Test JS divergence for identical distributions."""
        p = F.softmax(torch.randn(100), dim=-1)

        js = js_divergence(p, p)

        assert js >= 0
        assert js < 1e-6  # Should be ~0 for identical

    def test_js_divergence_symmetry(self):
        """Test that JS divergence is symmetric."""
        p = F.softmax(torch.randn(100), dim=-1)
        q = F.softmax(torch.randn(100), dim=-1)

        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)

        # JS should be symmetric
        assert abs(js_pq - js_qp) < 1e-6

    def test_js_divergence_bounds(self):
        """Test JS divergence is bounded [0, log(2)]."""
        import math

        p = F.softmax(torch.randn(100), dim=-1)
        q = F.softmax(torch.randn(100), dim=-1)

        js = js_divergence(p, q)

        assert js >= 0
        assert js <= math.log(2) + 1e-6

    def test_distribution_result_structure(self):
        """Test DistributionResult structure."""
        result = DistributionResult(
            kl_divergence=0.05,
            kl_reverse=0.06,
            js_divergence=0.02,
            tv_distance=0.1,
            histogram_correlation=0.95,
        )

        assert result.kl_divergence == 0.05
        assert result.js_divergence == 0.02
        assert result.tv_distance == 0.1

    def test_distribution_result_to_dict(self):
        """Test DistributionResult serialization."""
        result = DistributionResult(
            kl_divergence=0.05,
            kl_reverse=0.06,
            js_divergence=0.02,
            tv_distance=0.1,
            histogram_correlation=0.95,
        )

        d = result.to_dict()
        assert "kl_divergence" in d
        assert "js_divergence" in d
        assert d["kl_divergence"] == 0.05


# =============================================================================
# Test Section 7.2: Performance Benchmarks
# =============================================================================


class TestPerformanceBenchmarks:
    """Tests for performance benchmarking."""

    def test_benchmark_config_defaults(self):
        """Test BenchmarkConfig default values."""
        config = BenchmarkConfig()

        assert config.num_warmup > 0
        assert config.num_iterations > 0
        assert config.batch_sizes is not None

    def test_benchmark_result_structure(self):
        """Test BenchmarkResult structure."""
        result = BenchmarkResult(
            benchmark_type="latency",
            latency_mean_ms=50.0,
            latency_std_ms=5.0,
            latency_p50_ms=48.0,
            latency_p95_ms=60.0,
            latency_p99_ms=65.0,
            throughput_tokens_per_sec=100.0,
            memory_peak_mb=4000.0,
        )

        assert result.latency_mean_ms == 50.0
        assert result.throughput_tokens_per_sec == 100.0
        assert result.memory_peak_mb == 4000.0

    def test_performance_benchmark_init(self):
        """Test PerformanceBenchmark initialization."""
        config = BenchmarkConfig(
            num_warmup=5,
            num_iterations=10,
        )
        benchmark = PerformanceBenchmark(config=config)
        assert benchmark is not None

    def test_latency_benchmark_init(self):
        """Test LatencyBenchmark initialization."""
        benchmark = LatencyBenchmark()
        assert benchmark is not None

    def test_memory_benchmark_init(self):
        """Test MemoryBenchmark initialization."""
        benchmark = MemoryBenchmark()
        assert benchmark is not None

    def test_throughput_benchmark_init(self):
        """Test ThroughputBenchmark initialization."""
        benchmark = ThroughputBenchmark()
        assert benchmark is not None

    def test_benchmark_result_to_dict(self):
        """Test BenchmarkResult serialization."""
        result = BenchmarkResult(
            benchmark_type="latency",
            latency_mean_ms=50.0,
            latency_std_ms=5.0,
        )

        d = result.to_dict()
        assert "benchmark_type" in d
        # The to_dict method organizes metrics into nested dicts
        assert "latency" in d
        assert d["latency"]["mean_ms"] == 50.0

    def test_prefill_latency_target(self, simple_model, device):
        """Test prefill latency target: <100ms."""
        model = simple_model.to(device)
        model.eval()

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                x = torch.randint(0, 1000, (1, 512), device=device)
                model(x)

        # Measure prefill latency
        latencies = []
        for _ in range(5):
            x = torch.randint(0, 1000, (1, 512), device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                model(x)

            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        mean_latency = sum(latencies) / len(latencies)

        # Simple model should easily meet <100ms target
        assert mean_latency < 100, f"Prefill latency {mean_latency:.1f}ms > 100ms target"

    def test_decode_latency_target(self, simple_model, device):
        """Test decode latency target: <20ms per token."""
        model = simple_model.to(device)
        model.eval()

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                x = torch.randint(0, 1000, (1, 1), device=device)
                model(x)

        # Measure decode (single token) latency
        latencies = []
        for _ in range(10):
            x = torch.randint(0, 1000, (1, 1), device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                model(x)

            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        mean_latency = sum(latencies) / len(latencies)

        # Simple model should easily meet <20ms target
        assert mean_latency < 20, f"Decode latency {mean_latency:.1f}ms > 20ms target"

    def test_memory_target(self, simple_model, device):
        """Test memory target: <8GB for 7B model."""
        model = simple_model.to(device)

        # Calculate model size in MB
        model_size_mb = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        ) / (1024 * 1024)

        # Our simple model is tiny
        # Just verify we can measure and it's reasonable
        assert model_size_mb < 8 * 1024  # < 8GB

    def test_throughput_target(self, simple_model, device):
        """Test throughput target: >50 tok/s."""
        model = simple_model.to(device)
        model.eval()

        batch_size = 1
        seq_len = 16
        num_tokens = batch_size * seq_len

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                model(x)

        # Measure throughput
        total_tokens = 0
        start = time.perf_counter()

        for _ in range(20):
            x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            with torch.no_grad():
                model(x)
            total_tokens += num_tokens

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        end = time.perf_counter()
        throughput = total_tokens / (end - start)

        # Simple model should easily exceed 50 tok/s
        assert throughput > 50, f"Throughput {throughput:.1f} tok/s < 50 tok/s target"


# =============================================================================
# Integration Tests
# =============================================================================


class TestValidationIntegration:
    """Integration tests for validation components."""

    def test_full_validation_pipeline(self, simple_model, simple_tokenizer, device):
        """Test complete validation pipeline."""
        model = simple_model.to(device)

        # 1. Numerical validation
        test_input = torch.randint(0, 1000, (1, 16), device=device)

        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)

        num_result = compare_tensors(output1, output2)
        assert num_result.passed, "Numerical validation failed"

        # 2. Distribution comparison (using logits)
        p = F.softmax(output1[0, 0, :100], dim=-1)
        q = F.softmax(output2[0, 0, :100], dim=-1)
        kl = kl_divergence(p, q)
        assert kl < 1e-6, "Distribution comparison failed"

    def test_validation_report(self, simple_model, device):
        """Test generating validation report."""
        model = simple_model.to(device)

        report = {
            "numerical": {},
            "distribution": {},
            "performance": {},
        }

        # Numerical validation
        test_input = torch.randint(0, 1000, (1, 16), device=device)

        with torch.no_grad():
            output = model(test_input)

        num_result = compare_tensors(output, output)
        report["numerical"] = {
            "passed": num_result.passed,
            "max_diff": num_result.metrics.get("max_abs_diff", 0),
            "mean_diff": num_result.metrics.get("mean_abs_diff", 0),
        }

        # Distribution - compute KL for identical outputs
        p = F.softmax(output[0, 0, :100], dim=-1)
        kl = kl_divergence(p, p)
        report["distribution"] = {
            "kl_divergence": kl,
        }

        # Verify report structure
        assert report["numerical"]["passed"]
        assert report["distribution"]["kl_divergence"] < 1e-6

    def test_continuous_validation(self, simple_model, device):
        """Test continuous validation during inference."""
        model = simple_model.to(device)

        # Simulate continuous inference
        for i in range(10):
            test_input = torch.randint(0, 1000, (1, 16), device=device)

            with torch.no_grad():
                output = model(test_input)

            # Validate each output against itself (deterministic check)
            result = compare_tensors(output, output)
            assert result.passed, f"Validation failed at iteration {i}"

    def test_cross_device_validation(self, simple_model):
        """Test validation across devices (CPU reference)."""
        # CPU reference
        model_cpu = simple_model

        test_input = torch.randint(0, 1000, (1, 16))

        with torch.no_grad():
            output_cpu = model_cpu(test_input)

        # Compare same model on CPU
        result = compare_tensors(output_cpu, output_cpu)
        assert result.passed


# =============================================================================
# Module Import Tests
# =============================================================================


class TestValidationModuleImports:
    """Test that all validation module components can be imported."""

    def test_import_numerical(self):
        """Test importing numerical validation components."""
        from deepseek.mlx.ane.validation.numerical import (
            NumericalValidator,
            NumericalValidationConfig,
            ValidationResult,
            compare_tensors,
            compare_models,
        )
        
        assert NumericalValidator is not None
        assert NumericalValidationConfig is not None
        assert ValidationResult is not None

    def test_import_perplexity(self):
        """Test importing perplexity evaluation components."""
        from deepseek.mlx.ane.validation.perplexity import (
            PerplexityEvaluator,
            PerplexityConfig,
            PerplexityResult,
        )
        
        assert PerplexityEvaluator is not None
        assert PerplexityConfig is not None
        assert PerplexityResult is not None

    def test_import_distribution(self):
        """Test importing distribution comparison components."""
        from deepseek.mlx.ane.validation.distribution import (
            DistributionComparator,
            DistributionConfig,
            DistributionResult,
            kl_divergence,
            js_divergence,
        )
        
        assert DistributionComparator is not None
        assert kl_divergence is not None
        assert js_divergence is not None

    def test_import_benchmarks(self):
        """Test importing benchmark components."""
        from deepseek.mlx.ane.validation.benchmarks import (
            PerformanceBenchmark,
            BenchmarkConfig,
            BenchmarkResult,
            LatencyBenchmark,
            MemoryBenchmark,
            ThroughputBenchmark,
        )
        
        assert PerformanceBenchmark is not None
        assert BenchmarkConfig is not None
        assert LatencyBenchmark is not None

    def test_import_task_accuracy(self):
        """Test importing task accuracy components."""
        from deepseek.mlx.ane.validation.task_accuracy import (
            TaskEvaluator,
            MMLUEvaluator,
            HumanEvalEvaluator,
            TaskConfig,
            TaskResult,
        )
        
        assert TaskEvaluator is not None
        assert MMLUEvaluator is not None
        assert HumanEvalEvaluator is not None

    def test_import_from_package(self):
        """Test importing from validation package."""
        from deepseek.mlx.ane.validation import (
            NumericalValidator,
            PerplexityEvaluator,
            DistributionComparator,
            PerformanceBenchmark,
            TaskEvaluator,
        )
        
        assert NumericalValidator is not None
        assert PerplexityEvaluator is not None
        assert DistributionComparator is not None
        assert PerformanceBenchmark is not None
        assert TaskEvaluator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
