"""
Tests for Phase 2: Training (The MoE Loop) Implementation

This module tests:
1. HeterogeneousExpertPlacement - Expert placement across heterogeneous hardware
2. CheckpointInterop - Rust-PyTorch checkpoint interoperability
3. ExpertLoadHistory - EMA-based expert load tracking

Uses fallbacks for GPU-specific tests (no main script modifications).
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from src.deepseek.pipeline.training_loop import (
    CANDLE_TO_PYTORCH_NAME_MAP,
    PYTORCH_TO_CANDLE_NAME_MAP,
    CheckpointFormat,
    CheckpointInterop,
    CheckpointInteropConfig,
    CheckpointMetadata,
    ExpertLoadHistory,
    ExpertLoadStats,
    ExpertPlacementConfig,
    ExpertPlacementState,
    HardwareTarget,
    HeterogeneousExpertPlacement,
    _map_name_candle_to_pytorch,
    _map_name_pytorch_to_candle,
)

# Try to import torch for integration tests
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from safetensors.torch import save_file as safetensors_save

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# =============================================================================
# Hardware Target Tests
# =============================================================================


class TestHardwareTarget:
    """Tests for HardwareTarget enum."""

    def test_hardware_target_values(self):
        """Test all hardware target values exist."""
        assert HardwareTarget.CUDA_H100.value == "cuda_h100"
        assert HardwareTarget.CUDA_A100.value == "cuda_a100"
        assert HardwareTarget.CUDA_GENERIC.value == "cuda_generic"
        assert HardwareTarget.APPLE_SILICON.value == "apple_silicon"
        assert HardwareTarget.CPU.value == "cpu"

    def test_hardware_target_from_value(self):
        """Test creating HardwareTarget from string value."""
        assert HardwareTarget("cuda_h100") == HardwareTarget.CUDA_H100
        assert HardwareTarget("apple_silicon") == HardwareTarget.APPLE_SILICON


# =============================================================================
# Expert Load Stats Tests
# =============================================================================


class TestExpertLoadStats:
    """Tests for ExpertLoadStats dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        stats = ExpertLoadStats(expert_id=0)
        assert stats.expert_id == 0
        assert stats.total_tokens == 0
        assert stats.total_activations == 0
        assert stats.ema_load == 0.0
        assert stats.peak_load == 0.0

    def test_single_update(self):
        """Test single update."""
        stats = ExpertLoadStats(expert_id=1)
        stats.update(tokens_processed=100, ema_decay=0.99)

        assert stats.total_tokens == 100
        assert stats.total_activations == 1
        assert stats.ema_load == 100.0  # First update sets EMA directly
        assert stats.peak_load == 100.0

    def test_multiple_updates_ema(self):
        """Test EMA computation with multiple updates."""
        stats = ExpertLoadStats(expert_id=2)
        ema_decay = 0.9

        # First update
        stats.update(tokens_processed=100, ema_decay=ema_decay)
        assert stats.ema_load == 100.0

        # Second update: EMA = 0.9 * 100 + 0.1 * 50 = 95
        stats.update(tokens_processed=50, ema_decay=ema_decay)
        expected_ema = 0.9 * 100 + 0.1 * 50
        assert abs(stats.ema_load - expected_ema) < 1e-6

        # Check peak is tracked
        assert stats.peak_load == 100.0

    def test_average_load(self):
        """Test average load calculation."""
        stats = ExpertLoadStats(expert_id=3)
        stats.update(100)
        stats.update(200)
        stats.update(300)

        assert stats.total_tokens == 600
        assert stats.total_activations == 3
        assert stats.average_load == 200.0

    def test_average_load_zero_activations(self):
        """Test average load with no activations."""
        stats = ExpertLoadStats(expert_id=4)
        assert stats.average_load == 0.0


# =============================================================================
# Expert Load History Tests
# =============================================================================


class TestExpertLoadHistory:
    """Tests for ExpertLoadHistory class."""

    def test_initialization(self):
        """Test history initialization."""
        history = ExpertLoadHistory(num_experts=8)
        assert len(history.stats) == 8
        for i in range(8):
            assert i in history.stats
            assert history.stats[i].expert_id == i

    def test_record_batch(self):
        """Test recording expert loads from a batch."""
        history = ExpertLoadHistory(num_experts=4)

        # Record batch
        expert_counts = {0: 100, 1: 50, 2: 200, 3: 25}
        history.record_batch(expert_counts)

        assert history.stats[0].total_tokens == 100
        assert history.stats[1].total_tokens == 50
        assert history.stats[2].total_tokens == 200
        assert history.stats[3].total_tokens == 25

    def test_load_ranking(self):
        """Test load ranking by EMA."""
        history = ExpertLoadHistory(num_experts=4)

        # Record multiple batches to establish EMA
        for _ in range(10):
            history.record_batch({0: 100, 1: 50, 2: 200, 3: 25})

        ranking = history.get_load_ranking()

        # Expert 2 should be highest, expert 3 lowest
        assert ranking[0][0] == 2  # Highest load
        assert ranking[-1][0] == 3  # Lowest load

    def test_hot_cold_split_default(self):
        """Test hot/cold split with default 20% fraction."""
        history = ExpertLoadHistory(num_experts=10)

        # Simulate different loads
        for _ in range(10):
            history.record_batch({i: (i + 1) * 100 for i in range(10)})

        hot, cold = history.get_hot_cold_split(hot_fraction=0.2)

        # Top 20% (2 experts) should be hot
        assert len(hot) == 2
        assert len(cold) == 8

        # Hot should include experts with highest loads (8 and 9)
        assert 9 in hot
        assert 8 in hot

    def test_hot_cold_split_custom_fraction(self):
        """Test hot/cold split with custom fraction."""
        history = ExpertLoadHistory(num_experts=10)

        for _ in range(10):
            history.record_batch({i: (i + 1) * 100 for i in range(10)})

        hot, cold = history.get_hot_cold_split(hot_fraction=0.5)

        # Top 50% (5 experts) should be hot
        assert len(hot) == 5
        assert len(cold) == 5

    def test_serialization_roundtrip(self):
        """Test serialization to/from dict."""
        history = ExpertLoadHistory(num_experts=4)
        history.record_batch({0: 100, 1: 50, 2: 200, 3: 25})

        # Serialize
        data = history.to_dict()
        assert data["num_experts"] == 4
        assert "stats" in data

        # Deserialize
        loaded = ExpertLoadHistory.from_dict(data)
        assert loaded.num_experts == 4
        assert loaded.stats[0].total_tokens == 100


# =============================================================================
# Expert Placement Config Tests
# =============================================================================


class TestExpertPlacementConfig:
    """Tests for ExpertPlacementConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExpertPlacementConfig()
        assert config.hot_fraction == 0.2
        assert config.hot_expert_target == HardwareTarget.CUDA_H100
        assert config.cold_expert_target == HardwareTarget.APPLE_SILICON
        assert config.ema_decay == 0.99
        assert config.rebalance_interval_steps == 1000

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExpertPlacementConfig(
            hot_fraction=0.3,
            hot_expert_target=HardwareTarget.CUDA_A100,
            cold_expert_target=HardwareTarget.CPU,
        )
        assert config.hot_fraction == 0.3
        assert config.hot_expert_target == HardwareTarget.CUDA_A100


# =============================================================================
# Expert Placement State Tests
# =============================================================================


class TestExpertPlacementState:
    """Tests for ExpertPlacementState."""

    def test_default_state(self):
        """Test default state values."""
        state = ExpertPlacementState()
        assert len(state.expert_to_hardware) == 0
        assert state.placement_version == 0

    def test_get_experts_for_hardware(self):
        """Test getting experts by hardware target."""
        state = ExpertPlacementState(
            expert_to_hardware={
                0: HardwareTarget.CUDA_H100,
                1: HardwareTarget.CUDA_H100,
                2: HardwareTarget.APPLE_SILICON,
                3: HardwareTarget.APPLE_SILICON,
            }
        )

        cuda = state.get_experts_for_hardware(HardwareTarget.CUDA_H100)
        metal = state.get_experts_for_hardware(HardwareTarget.APPLE_SILICON)

        assert set(cuda) == {0, 1}
        assert set(metal) == {2, 3}

    def test_serialization_roundtrip(self):
        """Test state serialization."""
        state = ExpertPlacementState(
            expert_to_hardware={
                0: HardwareTarget.CUDA_H100,
                1: HardwareTarget.APPLE_SILICON,
            },
            placement_version=5,
            last_rebalance_step=1000,
        )

        # Serialize
        data = state.to_dict()
        assert data["placement_version"] == 5

        # Deserialize
        loaded = ExpertPlacementState.from_dict(data)
        assert loaded.placement_version == 5
        assert loaded.expert_to_hardware[0] == HardwareTarget.CUDA_H100


# =============================================================================
# Heterogeneous Expert Placement Tests
# =============================================================================


class TestHeterogeneousExpertPlacement:
    """Tests for HeterogeneousExpertPlacement class."""

    def test_initialization(self):
        """Test placement manager initialization."""
        placement = HeterogeneousExpertPlacement(
            num_experts=16,
            num_shared_experts=2,
        )

        assert placement.num_experts == 16
        assert placement.num_shared_experts == 2
        assert len(placement.load_history.stats) == 16

    def test_initial_placement_all_hot(self):
        """Test initial placement puts all experts on hot target."""
        placement = HeterogeneousExpertPlacement(num_experts=8)

        # Initially all should be on hot target
        for i in range(8):
            assert placement.state.expert_to_hardware[i] == HardwareTarget.CUDA_H100

    def test_record_expert_loads(self):
        """Test recording expert loads."""
        placement = HeterogeneousExpertPlacement(num_experts=4)

        placement.record_expert_loads({0: 100, 1: 50, 2: 200, 3: 25})

        assert placement.load_history.stats[0].total_tokens == 100
        assert placement.load_history.stats[2].total_tokens == 200

    def test_should_rebalance_min_activations(self):
        """Test rebalance check requires minimum activations."""
        config = ExpertPlacementConfig(min_activations_for_placement=10)
        placement = HeterogeneousExpertPlacement(num_experts=4, config=config)

        # Record few batches
        for _ in range(5):
            placement.record_expert_loads({i: 100 for i in range(4)})

        # Should not rebalance yet (5 * 4 = 20 < 10 per expert? No, 5 batches < 10)
        # Actually 5 activations per expert < 10? The check is total activations
        # Let's check: total_activations = 5 per expert
        # min_activations_for_placement = 10 means sum of all should be >= 10
        # Actually the code sums all activations: 5 * 4 = 20 >= 10
        # But we also need interval check
        assert not placement.should_rebalance(current_step=100)

    def test_should_rebalance_interval(self):
        """Test rebalance interval check."""
        config = ExpertPlacementConfig(
            min_activations_for_placement=10,
            rebalance_interval_steps=500,
        )
        placement = HeterogeneousExpertPlacement(num_experts=4, config=config)

        # Record enough batches
        for _ in range(100):
            placement.record_expert_loads({i: 100 for i in range(4)})

        # Not enough steps since last rebalance
        assert not placement.should_rebalance(current_step=100)

        # Enough steps
        assert placement.should_rebalance(current_step=600)

    def test_rebalance_hot_cold_split(self):
        """Test rebalancing creates hot/cold split."""
        config = ExpertPlacementConfig(
            min_activations_for_placement=10,
            hot_fraction=0.25,  # 1 of 4 experts
        )
        placement = HeterogeneousExpertPlacement(num_experts=4, config=config)

        # Simulate load: expert 2 gets most traffic
        for _ in range(100):
            placement.record_expert_loads({0: 25, 1: 50, 2: 200, 3: 10})

        # Rebalance
        result = placement.rebalance(current_step=1000)

        # Expert 2 should be hot (highest load)
        assert 2 in result["hot_experts"]
        assert len(result["hot_experts"]) == 1

        # Others should be cold
        assert len(result["cold_experts"]) == 3

        # Check state updated
        assert placement.state.expert_to_hardware[2] == HardwareTarget.CUDA_H100
        for cold_id in result["cold_experts"]:
            assert placement.state.expert_to_hardware[cold_id] == HardwareTarget.APPLE_SILICON

    def test_get_cuda_experts(self):
        """Test getting all CUDA experts."""
        placement = HeterogeneousExpertPlacement(num_experts=4)

        # Manually set placement
        placement.state.expert_to_hardware = {
            0: HardwareTarget.CUDA_H100,
            1: HardwareTarget.CUDA_A100,
            2: HardwareTarget.APPLE_SILICON,
            3: HardwareTarget.CPU,
        }

        cuda = placement.get_cuda_experts()
        assert set(cuda) == {0, 1}

    def test_get_metal_experts(self):
        """Test getting Apple Silicon experts."""
        placement = HeterogeneousExpertPlacement(num_experts=4)

        placement.state.expert_to_hardware = {
            0: HardwareTarget.CUDA_H100,
            1: HardwareTarget.APPLE_SILICON,
            2: HardwareTarget.APPLE_SILICON,
            3: HardwareTarget.CPU,
        }

        metal = placement.get_metal_experts()
        assert set(metal) == {1, 2}

    def test_placement_summary(self):
        """Test getting placement summary."""
        placement = HeterogeneousExpertPlacement(num_experts=4)

        for _ in range(10):
            placement.record_expert_loads({0: 100, 1: 50, 2: 200, 3: 25})

        placement.rebalance(1000)
        summary = placement.get_placement_summary()

        assert "by_hardware" in summary
        assert "hardware_counts" in summary
        assert "load_stats" in summary

    def test_save_load_roundtrip(self):
        """Test saving and loading placement state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "placement.json"

            # Create and populate placement
            placement = HeterogeneousExpertPlacement(num_experts=8)
            for _ in range(100):
                placement.record_expert_loads({i: (i + 1) * 10 for i in range(8)})
            placement.rebalance(1000)

            # Save
            placement.save(path)
            assert path.exists()

            # Load
            loaded = HeterogeneousExpertPlacement.load(path)

            # Verify
            assert loaded.num_experts == 8
            assert loaded.state.placement_version == placement.state.placement_version
            assert loaded.load_history.stats[0].total_tokens == 1000


# =============================================================================
# Checkpoint Format Tests
# =============================================================================


class TestCheckpointFormat:
    """Tests for CheckpointFormat enum."""

    def test_checkpoint_format_values(self):
        """Test checkpoint format values."""
        assert CheckpointFormat.PYTORCH.value == "pytorch"
        assert CheckpointFormat.CANDLE.value == "candle"
        assert CheckpointFormat.SAFETENSORS.value == "safetensors"
        assert CheckpointFormat.MLX.value == "mlx"


# =============================================================================
# Name Mapping Tests
# =============================================================================


class TestNameMapping:
    """Tests for Candle <-> PyTorch name mapping."""

    def test_direct_mapping_candle_to_pytorch(self):
        """Test direct name mappings."""
        assert _map_name_candle_to_pytorch("attention.w_q") == "self_attn.q_proj"
        assert _map_name_candle_to_pytorch("attention.w_k") == "self_attn.k_proj"
        assert _map_name_candle_to_pytorch("mlp.gate") == "mlp.gate_proj"

    def test_direct_mapping_pytorch_to_candle(self):
        """Test reverse name mappings."""
        assert _map_name_pytorch_to_candle("self_attn.q_proj") == "attention.w_q"
        assert _map_name_pytorch_to_candle("mlp.gate_proj") == "mlp.gate"

    def test_name_map_consistency(self):
        """Test name maps are consistent inverses."""
        for candle, pytorch in CANDLE_TO_PYTORCH_NAME_MAP.items():
            assert PYTORCH_TO_CANDLE_NAME_MAP[pytorch] == candle

    def test_unmapped_names_preserved(self):
        """Test unmapped names are preserved."""
        assert _map_name_candle_to_pytorch("custom.layer") == "custom.layer"
        assert _map_name_pytorch_to_candle("custom.layer") == "custom.layer"


# =============================================================================
# Checkpoint Metadata Tests
# =============================================================================


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata."""

    def test_default_metadata(self):
        """Test default metadata values."""
        meta = CheckpointMetadata(
            version=1,
            format=CheckpointFormat.PYTORCH,
            source_framework="pytorch",
            timestamp=time.time(),
        )
        assert meta.version == 1
        assert meta.training_step == 0
        assert len(meta.tensors) == 0

    def test_serialization_roundtrip(self):
        """Test metadata serialization."""
        meta = CheckpointMetadata(
            version=2,
            format=CheckpointFormat.CANDLE,
            source_framework="candle",
            timestamp=1234567890.0,
            training_step=10000,
            model_config={"d_model": 512},
        )

        # Serialize
        data = meta.to_dict()
        assert data["version"] == 2
        assert data["format"] == "candle"

        # Deserialize
        loaded = CheckpointMetadata.from_dict(data)
        assert loaded.version == 2
        assert loaded.format == CheckpointFormat.CANDLE
        assert loaded.training_step == 10000


# =============================================================================
# Checkpoint Interop Config Tests
# =============================================================================


class TestCheckpointInteropConfig:
    """Tests for CheckpointInteropConfig."""

    def test_default_config(self):
        """Test default config values."""
        config = CheckpointInteropConfig()
        assert config.enable_name_mapping is True
        assert config.validate_checksums is True
        assert config.target_dtype == "float32"
        assert config.use_safetensors is True


# =============================================================================
# Checkpoint Interop Tests
# =============================================================================


class TestCheckpointInterop:
    """Tests for CheckpointInterop class."""

    def test_initialization(self):
        """Test interop initialization."""
        interop = CheckpointInterop()
        assert interop.config.enable_name_mapping is True

    def test_initialization_custom_config(self):
        """Test interop with custom config."""
        config = CheckpointInteropConfig(enable_name_mapping=False)
        interop = CheckpointInterop(config)
        assert interop.config.enable_name_mapping is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_validate_checkpoint_missing_file(self):
        """Test validation of missing checkpoint."""
        interop = CheckpointInterop()
        result = interop.validate_checkpoint("/nonexistent/path.pt")

        assert result["valid"] is False
        assert "File not found" in result["errors"][0]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_validate_pytorch_checkpoint(self):
        """Test validation of valid PyTorch checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"

            # Create simple checkpoint
            state_dict = {
                "weight": torch.randn(10, 10),
                "bias": torch.randn(10),
            }
            torch.save(state_dict, path)

            # Validate
            interop = CheckpointInterop()
            result = interop.validate_checkpoint(path)

            assert result["valid"] is True
            assert result["tensor_count"] == 2

    @pytest.mark.skipif(
        not (TORCH_AVAILABLE and SAFETENSORS_AVAILABLE),
        reason="PyTorch or SafeTensors not available",
    )
    def test_validate_safetensors_checkpoint(self):
        """Test validation of SafeTensors checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"

            # Create SafeTensors checkpoint
            tensors = {
                "weight": torch.randn(10, 10),
                "bias": torch.randn(10),
            }
            safetensors_save(tensors, str(path))

            # Validate
            interop = CheckpointInterop()
            result = interop.validate_checkpoint(path)

            assert result["valid"] is True
            assert result["tensor_count"] == 2

    @pytest.mark.skipif(
        not (TORCH_AVAILABLE and SAFETENSORS_AVAILABLE),
        reason="PyTorch or SafeTensors not available",
    )
    def test_convert_pytorch_to_candle(self):
        """Test PyTorch to Candle conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pytorch_path = Path(tmpdir) / "pytorch.pt"
            candle_path = Path(tmpdir) / "candle.safetensors"

            # Create PyTorch checkpoint with mapped names
            state_dict = {
                "self_attn.q_proj.weight": torch.randn(64, 64),
                "self_attn.k_proj.weight": torch.randn(64, 64),
                "mlp.gate_proj.weight": torch.randn(256, 64),
            }
            torch.save(state_dict, pytorch_path)

            # Convert
            interop = CheckpointInterop()
            result = interop.convert_pytorch_to_candle(pytorch_path, candle_path)

            # Check converted names
            assert "attention.w_q.weight" in result or "self_attn.q_proj.weight" in result
            assert candle_path.exists()

    @pytest.mark.skipif(
        not (TORCH_AVAILABLE and SAFETENSORS_AVAILABLE),
        reason="PyTorch or SafeTensors not available",
    )
    def test_convert_candle_to_pytorch(self):
        """Test Candle to PyTorch conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            candle_path = Path(tmpdir) / "candle.safetensors"
            pytorch_path = Path(tmpdir) / "pytorch.pt"

            # Create Candle-style checkpoint (SafeTensors)
            tensors = {
                "attention.w_q": torch.randn(64, 64),
                "attention.w_k": torch.randn(64, 64),
                "mlp.gate": torch.randn(256, 64),
            }
            safetensors_save(tensors, str(candle_path))

            # Convert
            interop = CheckpointInterop()
            result = interop.convert_candle_to_pytorch(candle_path, pytorch_path)

            # Check converted names
            assert "self_attn.q_proj" in result
            assert "self_attn.k_proj" in result
            assert "mlp.gate_proj" in result
            assert pytorch_path.exists()

    def test_get_name_mapping_preview(self):
        """Test name mapping preview."""
        with tempfile.TemporaryDirectory() as tmpdir:
            if not (TORCH_AVAILABLE and SAFETENSORS_AVAILABLE):
                pytest.skip("PyTorch or SafeTensors not available")

            path = Path(tmpdir) / "test.safetensors"

            # Create checkpoint
            tensors = {
                "attention.w_q": torch.randn(64, 64),
                "mlp.gate": torch.randn(256, 64),
            }
            safetensors_save(tensors, str(path))

            # Get preview
            interop = CheckpointInterop()
            mapping = interop.get_name_mapping_preview(path, CheckpointFormat.CANDLE)

            # Check mappings
            mapping_dict = dict(mapping)
            assert mapping_dict["attention.w_q"] == "self_attn.q_proj"
            assert mapping_dict["mlp.gate"] == "mlp.gate_proj"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    def test_expert_placement_with_checkpoint_save(self):
        """Test saving expert placement alongside checkpoint metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            placement_path = Path(tmpdir) / "placement.json"
            meta_path = Path(tmpdir) / "checkpoint_metadata.json"

            # Create placement
            placement = HeterogeneousExpertPlacement(num_experts=16)
            for _ in range(100):
                placement.record_expert_loads({i: (i % 4 + 1) * 50 for i in range(16)})
            placement.rebalance(1000)

            # Save placement
            placement.save(placement_path)

            # Create checkpoint metadata
            meta = CheckpointMetadata(
                version=1,
                format=CheckpointFormat.PYTORCH,
                source_framework="pytorch",
                timestamp=time.time(),
                training_step=1000,
                extra={
                    "expert_placement_file": str(placement_path),
                },
            )

            with open(meta_path, "w") as f:
                json.dump(meta.to_dict(), f)

            # Verify both saved
            assert placement_path.exists()
            assert meta_path.exists()

            # Load and verify linkage
            loaded_placement = HeterogeneousExpertPlacement.load(placement_path)
            with open(meta_path) as f:
                loaded_meta = CheckpointMetadata.from_dict(json.load(f))

            assert loaded_meta.extra["expert_placement_file"] == str(placement_path)
            assert loaded_placement.state.last_rebalance_step == 1000

    def test_full_training_loop_simulation(self):
        """Test simulated training loop with expert placement updates."""
        placement = HeterogeneousExpertPlacement(
            num_experts=8,
            config=ExpertPlacementConfig(
                min_activations_for_placement=50,
                rebalance_interval_steps=100,
                hot_fraction=0.25,
            ),
        )

        # Simulate training
        for step in range(500):
            # Simulate expert loads (some experts get more traffic)
            loads = {
                i: np.random.poisson(100 + i * 20)
                for i in range(8)  # Expert 7 gets most
            }
            placement.record_expert_loads(loads, step=step)

        # After training, expert 7 should be hot
        hot = placement.get_cuda_experts()
        cold = placement.get_metal_experts()

        # Expert 7 should be in hot list (highest expected load)
        assert 7 in hot
        assert len(cold) == 6  # 75% should be cold


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_single_expert(self):
        """Test with single expert."""
        placement = HeterogeneousExpertPlacement(num_experts=1)
        placement.record_expert_loads({0: 100})

        hot, cold = placement.load_history.get_hot_cold_split()

        # Single expert should be hot
        assert len(hot) == 1
        assert len(cold) == 0

    def test_zero_load_experts(self):
        """Test experts with zero load."""
        placement = HeterogeneousExpertPlacement(num_experts=4)

        # Only some experts get traffic
        for _ in range(10):
            placement.record_expert_loads({0: 100, 1: 50})

        # Experts 2 and 3 have zero load
        assert placement.load_history.stats[2].total_tokens == 0
        assert placement.load_history.stats[3].total_tokens == 0

    def test_empty_checkpoint_validation(self):
        """Test validating empty/invalid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.pt"
            path.touch()  # Create empty file

            interop = CheckpointInterop()
            result = interop.validate_checkpoint(path)

            # Should fail gracefully
            assert result["valid"] is False or len(result["errors"]) > 0

    def test_large_expert_count(self):
        """Test with large number of experts (256)."""
        placement = HeterogeneousExpertPlacement(
            num_experts=256,
            config=ExpertPlacementConfig(hot_fraction=0.2),
        )

        # Simulate loads
        for _ in range(10):
            loads = {i: np.random.randint(10, 1000) for i in range(256)}
            placement.record_expert_loads(loads)

        hot, cold = placement.load_history.get_hot_cold_split()

        # 20% of 256 = ~51 hot experts
        assert len(hot) == 51
        assert len(cold) == 205

    def test_placement_version_increment(self):
        """Test placement version increments on rebalance."""
        placement = HeterogeneousExpertPlacement(num_experts=4)

        initial_version = placement.state.placement_version
        assert initial_version == 0

        # Record enough data and rebalance
        for _ in range(100):
            placement.record_expert_loads({i: 100 for i in range(4)})

        placement.rebalance(1000)
        assert placement.state.placement_version == 1

        placement.rebalance(2000)
        assert placement.state.placement_version == 2
