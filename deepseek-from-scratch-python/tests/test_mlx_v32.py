"""
Tests for DeepSeek-V3.2 MLX implementations.

Tests the Phase 1 components:
- Extended RoPE with NTK/YaRN
- DeepSeek Sparse Attention (DSA)
- DeepSeek MoE V3 (256 experts, 8 active)
- DualPipe scheduler
- DSA alignment loss
"""

import pytest
import sys
import os

# Add mlx_impl to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip all tests if MLX is not available
mlx_available = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    mlx_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not mlx_available, reason="MLX not available")


# ============================================================================
# Extended RoPE Tests
# ============================================================================

class TestExtendedRoPE:
    """Tests for Extended RoPE with NTK/YaRN scaling."""
    
    def test_config_creation(self):
        """Test config creation for different context lengths."""
        from mlx_impl.attention import ExtendedRoPEConfig, RoPEScalingType
        
        config = ExtendedRoPEConfig.for_128k()
        assert config.max_seq_len == 131072
        assert config.scaling_type == RoPEScalingType.NTK_AWARE
        
        config_yarn = ExtendedRoPEConfig.for_128k_yarn()
        assert config_yarn.scaling_type == RoPEScalingType.YARN
    
    def test_ntk_forward(self):
        """Test NTK-aware RoPE forward pass."""
        from mlx_impl.attention import ExtendedRotaryPositionalEncoding, ExtendedRoPEConfig
        
        config = ExtendedRoPEConfig(
            d_head=64,
            max_seq_len=8192,
        )
        rope = ExtendedRotaryPositionalEncoding(config)
        
        batch, heads, seq_len, d_head = 2, 4, 128, 64
        x = mx.random.normal((batch, heads, seq_len, d_head))
        
        out = rope(x)
        
        assert out.shape == x.shape
        # Verify no NaN
        assert not mx.any(mx.isnan(out)).item()
    
    def test_yarn_forward(self):
        """Test YaRN RoPE forward pass."""
        from mlx_impl.attention import (
            ExtendedRotaryPositionalEncoding, 
            ExtendedRoPEConfig,
            RoPEScalingType
        )
        
        config = ExtendedRoPEConfig(
            d_head=64,
            max_seq_len=8192,
            scaling_type=RoPEScalingType.YARN,
        )
        rope = ExtendedRotaryPositionalEncoding(config)
        
        batch, heads, seq_len, d_head = 2, 4, 128, 64
        x = mx.random.normal((batch, heads, seq_len, d_head))
        
        out = rope(x)
        
        assert out.shape == x.shape
        assert not mx.any(mx.isnan(out)).item()
    
    def test_rope_with_offset(self):
        """Test RoPE with position offset."""
        from mlx_impl.attention import ExtendedRotaryPositionalEncoding, ExtendedRoPEConfig
        
        config = ExtendedRoPEConfig(d_head=32)
        rope = ExtendedRotaryPositionalEncoding(config)
        
        x = mx.random.normal((1, 2, 16, 32))
        
        out_no_offset = rope(x, offset=0)
        out_with_offset = rope(x, offset=100)
        
        # Outputs should be different due to offset
        assert not mx.allclose(out_no_offset, out_with_offset).item()


# ============================================================================
# Sparse Attention Tests
# ============================================================================

class TestSparseAttention:
    """Tests for DeepSeek Sparse Attention."""
    
    def test_dsa_config(self):
        """Test DSA configuration."""
        from mlx_impl.sparse_attention import DSAConfig
        
        config = DSAConfig.for_128k_context()
        assert config.max_seq_len == 131072
        assert config.window_size == 4096
        assert config.effective_kv_budget < config.max_seq_len
    
    def test_sparse_pattern_generation(self):
        """Test sparse attention pattern generation."""
        from mlx_impl.sparse_attention import DSAConfig, SparseAttentionPattern
        
        config = DSAConfig(window_size=8, dilation_stride=4, causal=True)
        pattern = SparseAttentionPattern(config)
        
        mask = pattern.get_mask(16)
        
        assert mask.shape == (16, 16)
        # Causal: upper triangle should be False (except for sparse pattern)
        # Verify mask is boolean
        assert mask.dtype == mx.bool_
    
    def test_dsa_forward(self):
        """Test DSA forward pass."""
        from mlx_impl.sparse_attention import DSAConfig, DeepSeekSparseAttention
        
        config = DSAConfig(
            d_model=64,
            num_heads=4,
            d_latent=32,
            d_rope=16,
            window_size=8,
            max_seq_len=64,
        )
        dsa = DeepSeekSparseAttention(config)
        
        batch, seq_len = 2, 16
        x = mx.random.normal((batch, seq_len, 64))
        
        out = dsa(x)
        
        assert out.shape == x.shape
        assert not mx.any(mx.isnan(out)).item()
    
    def test_block_sparse_attention(self):
        """Test block-sparse attention."""
        from mlx_impl.sparse_attention import DSAConfig, BlockSparseAttention
        
        config = DSAConfig(
            d_model=64,
            num_heads=4,
            block_size=4,
            window_size=8,
            max_seq_len=32,
        )
        attn = BlockSparseAttention(config)
        
        batch, seq_len = 2, 16
        x = mx.random.normal((batch, seq_len, 64))
        
        out = attn(x)
        
        assert out.shape == x.shape


# ============================================================================
# MoE V3 Tests
# ============================================================================

class TestMoEV3:
    """Tests for DeepSeek MoE V3."""
    
    def test_moe_v3_config(self):
        """Test MoE V3 configuration."""
        from mlx_impl.moe import DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.small_16_2()
        assert config.n_routed_experts == 16
        assert config.top_k == 2
        assert config.n_expert_groups == 4
        
        config_full = DeepSeekMoEV3Config.v3_256_8()
        assert config_full.n_routed_experts == 256
        assert config_full.top_k == 8
    
    def test_load_balancing_state(self):
        """Test load balancing state updates."""
        from mlx_impl.moe import DeepSeekMoEV3Config, LoadBalancingState
        
        config = DeepSeekMoEV3Config.small_16_2()
        state = LoadBalancingState(config)
        
        # Simulate uneven expert usage using array creation
        counts = mx.array([10.0, 0.5] + [1.0] * 14)  # Expert 0 overused, expert 1 underused
        
        state.update(counts)
        
        # Bias should adjust: overused expert gets negative bias, underused gets positive
        assert state.bias[0] < state.bias[1]
    
    def test_moe_v3_forward(self):
        """Test MoE V3 forward pass."""
        from mlx_impl.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        config = DeepSeekMoEV3Config.small_16_2()
        config.d_model = 64
        config.routed_hidden_mult = 2.0
        config.shared_hidden_mult = 2.0
        
        moe = DeepSeekMoEV3(config)
        
        batch, seq_len = 2, 8
        x = mx.random.normal((batch, seq_len, 64))
        
        out = moe(x)
        
        assert out.shape == x.shape
        assert not mx.any(mx.isnan(out)).item()
    
    def test_moe_v3_load_balance_stats(self):
        """Test load balance statistics."""
        from mlx_impl.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        config = DeepSeekMoEV3Config.small_16_2()
        config.d_model = 32
        
        moe = DeepSeekMoEV3(config)
        
        # Run a few iterations
        for _ in range(5):
            x = mx.random.normal((2, 8, 32))
            moe(x)
        
        mean, imbalance, steps = moe.get_load_balance_stats()
        
        assert mean >= 0
        assert imbalance >= 1.0
        assert steps >= 0


# ============================================================================
# DualPipe Tests
# ============================================================================

class TestDualPipe:
    """Tests for DualPipe scheduler."""
    
    def test_dualpipe_creation(self):
        """Test DualPipe scheduler creation."""
        from mlx_impl.pipeline import DualPipeScheduler, DualPipePhase
        
        scheduler = DualPipeScheduler(num_micro_batches=8)
        
        assert scheduler.phase == DualPipePhase.WARMUP
        assert not scheduler.is_done()
    
    def test_dualpipe_schedule(self):
        """Test DualPipe produces correct schedule."""
        from mlx_impl.pipeline import DualPipeScheduler
        
        scheduler = DualPipeScheduler(num_micro_batches=8)
        
        regular_fwd = 0
        regular_bwd = 0
        reverse_fwd = 0
        reverse_bwd = 0
        steps = 0
        
        while not scheduler.is_done() and steps < 100:
            action = scheduler.next_action()
            if action.regular_fwd is not None:
                regular_fwd += 1
            if action.regular_bwd is not None:
                regular_bwd += 1
            if action.reverse_fwd is not None:
                reverse_fwd += 1
            if action.reverse_bwd is not None:
                reverse_bwd += 1
            steps += 1
        
        # Each stream should process 4 micro-batches
        assert regular_fwd == 4
        assert regular_bwd == 4
        assert reverse_fwd == 4
        assert reverse_bwd == 4
    
    def test_dualpipe_reset(self):
        """Test DualPipe reset."""
        from mlx_impl.pipeline import DualPipeScheduler, DualPipePhase
        
        scheduler = DualPipeScheduler(num_micro_batches=4)
        
        # Run some steps
        for _ in range(5):
            scheduler.next_action()
        
        scheduler.reset()
        
        assert scheduler.phase == DualPipePhase.WARMUP
        assert not scheduler.is_done()
    
    def test_dualpipe_bubble_ratio(self):
        """Test bubble ratio calculation."""
        from mlx_impl.pipeline import DualPipeScheduler
        
        scheduler = DualPipeScheduler(num_micro_batches=8, num_stages=4)
        
        bubble = scheduler.bubble_ratio()
        
        assert 0.0 <= bubble <= 1.0


# ============================================================================
# DSA Alignment Loss Tests
# ============================================================================

class TestDSAAlignmentLoss:
    """Tests for DSA alignment loss."""
    
    def test_mse_loss(self):
        """Test MSE alignment loss."""
        from mlx_impl.pipeline import DSAAlignmentLoss, DSAAlignmentConfig, DSAAlignmentLossType
        
        config = DSAAlignmentConfig(loss_type=DSAAlignmentLossType.MSE)
        loss_fn = DSAAlignmentLoss(config)
        
        batch, seq_len, d_model = 2, 16, 64
        sparse = mx.random.normal((batch, seq_len, d_model))
        dense = mx.random.normal((batch, seq_len, d_model))
        
        loss = loss_fn.compute(sparse, dense)
        
        assert loss.shape == ()  # Scalar
        assert loss.item() > 0
    
    def test_cosine_loss(self):
        """Test cosine similarity alignment loss."""
        from mlx_impl.pipeline import DSAAlignmentLoss, DSAAlignmentConfig, DSAAlignmentLossType
        
        config = DSAAlignmentConfig(loss_type=DSAAlignmentLossType.COSINE)
        loss_fn = DSAAlignmentLoss(config)
        
        batch, seq_len, d_model = 2, 16, 64
        x = mx.random.normal((batch, seq_len, d_model))
        
        # Same input should have ~0 loss
        loss = loss_fn.compute(x, x)
        
        assert loss.item() < 0.01
    
    def test_combined_loss(self):
        """Test combined alignment loss."""
        from mlx_impl.pipeline import DSAAlignmentLoss, DSAAlignmentConfig, DSAAlignmentLossType
        
        config = DSAAlignmentConfig(loss_type=DSAAlignmentLossType.COMBINED)
        loss_fn = DSAAlignmentLoss(config)
        
        batch, seq_len, d_model = 2, 16, 64
        sparse = mx.random.normal((batch, seq_len, d_model))
        dense = mx.random.normal((batch, seq_len, d_model))
        
        loss = loss_fn.compute(sparse, dense)
        
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_sampled_loss(self):
        """Test loss with position sampling."""
        from mlx_impl.pipeline import DSAAlignmentLoss, DSAAlignmentConfig
        
        config = DSAAlignmentConfig(sample_fraction=0.5)
        loss_fn = DSAAlignmentLoss(config)
        
        batch, seq_len, d_model = 2, 32, 64
        sparse = mx.random.normal((batch, seq_len, d_model))
        dense = mx.random.normal((batch, seq_len, d_model))
        
        loss = loss_fn.compute(sparse, dense)
        
        assert loss.shape == ()


# ============================================================================
# Integration Test
# ============================================================================

class TestV32Integration:
    """Integration test for all V3.2 components."""
    
    def test_full_pipeline(self):
        """Test all components working together."""
        from mlx_impl.attention import ExtendedRotaryPositionalEncoding, ExtendedRoPEConfig
        from mlx_impl.sparse_attention import DSAConfig, DeepSeekSparseAttention
        from mlx_impl.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        from mlx_impl.pipeline import DualPipeScheduler
        
        # Config
        batch_size = 2
        seq_len = 16
        d_model = 64
        num_heads = 4
        
        # Create components
        rope_config = ExtendedRoPEConfig(d_head=d_model // num_heads)
        rope = ExtendedRotaryPositionalEncoding(rope_config)
        
        dsa_config = DSAConfig(
            d_model=d_model,
            num_heads=num_heads,
            d_latent=32,
            d_rope=16,
            window_size=8,
            max_seq_len=64,
        )
        dsa = DeepSeekSparseAttention(dsa_config)
        
        moe_config = DeepSeekMoEV3Config.small_16_2()
        moe_config.d_model = d_model
        moe = DeepSeekMoEV3(moe_config)
        
        scheduler = DualPipeScheduler(num_micro_batches=4)
        
        # Forward pass
        x = mx.random.normal((batch_size, seq_len, d_model))
        
        # DSA
        attn_out = dsa(x)
        assert attn_out.shape == x.shape
        
        # MoE
        moe_out = moe(attn_out)
        assert moe_out.shape == x.shape
        
        # RoPE on attention heads
        x_heads = x.reshape(batch_size, seq_len, num_heads, d_model // num_heads)
        x_heads = x_heads.transpose(0, 2, 1, 3)
        rope_out = rope(x_heads)
        assert rope_out.shape == x_heads.shape
        
        # Scheduler
        action = scheduler.next_action()
        assert action.regular_fwd is not None or action.reverse_fwd is not None
        
        print("\n✅ DeepSeek-V3.2 MLX Integration Test Passed!")
        print(f"   DSA output: {attn_out.shape}")
        print(f"   MoE output: {moe_out.shape}")
        print(f"   RoPE output: {rope_out.shape}")


# ============================================================================
# Phase 2: Capacity Metrics Tests
# ============================================================================

class TestCapacityMetrics:
    """Tests for Phase 2 capacity metrics in MLX."""
    
    def test_capacity_metrics_creation(self):
        """Test CapacityMetrics class creation."""
        from mlx_impl.moe import CapacityMetrics
        
        metrics = CapacityMetrics()
        metrics.reset(n_experts=16)
        
        assert len(metrics.expert_overflow) == 16
        assert len(metrics.expert_utilization) == 16
        assert metrics.total_tokens == 0
        assert metrics.dropped_tokens == 0
    
    def test_capacity_metrics_drop_rate(self):
        """Test drop rate calculation."""
        from mlx_impl.moe import CapacityMetrics
        
        metrics = CapacityMetrics()
        metrics.reset(n_experts=4)
        
        # Simulate some dispatches with overflow
        metrics.record_dispatch(expert_id=0, tokens_routed=100, capacity=80)  # 20 dropped
        metrics.record_dispatch(expert_id=1, tokens_routed=50, capacity=60)   # no drop
        metrics.record_dispatch(expert_id=2, tokens_routed=30, capacity=20)   # 10 dropped
        
        drop_rate = metrics.drop_rate()
        
        assert drop_rate > 0  # Some tokens dropped
        assert drop_rate < 1.0  # Not all dropped
    
    def test_capacity_metrics_utilization(self):
        """Test utilization tracking."""
        from mlx_impl.moe import CapacityMetrics
        
        metrics = CapacityMetrics()
        metrics.reset(n_experts=4)
        
        # Expert 0 gets 80 tokens, capacity 100 -> 0.8 util
        metrics.record_dispatch(expert_id=0, tokens_routed=80, capacity=100)
        # Expert 1 gets 120 tokens, capacity 100 -> 1.2 util (overflow)
        metrics.record_dispatch(expert_id=1, tokens_routed=120, capacity=100)
        
        assert abs(metrics.expert_utilization[0] - 0.8) < 0.01
        assert abs(metrics.expert_utilization[1] - 1.2) < 0.01
    
    def test_capacity_metrics_record_dispatch(self):
        """Test recording dispatch statistics."""
        from mlx_impl.moe import CapacityMetrics
        
        metrics = CapacityMetrics()
        metrics.reset(n_experts=4)
        
        # Expert 0: 100 routed, 80 capacity -> 20 dropped
        metrics.record_dispatch(expert_id=0, tokens_routed=100, capacity=80)
        
        assert metrics.total_tokens == 80  # Only capacity processed
        assert metrics.dropped_tokens == 20
        assert metrics.expert_overflow[0] == 20


class TestEfficientBatchedDispatch:
    """Tests for efficient batched expert dispatch in MLX."""
    
    def test_moe_with_capacity_factor(self):
        """Test MoE respects capacity factor."""
        from mlx_impl.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        config = DeepSeekMoEV3Config.small_16_2()
        config.capacity_factor = 1.5  # Allow 50% overflow
        
        moe = DeepSeekMoEV3(config)
        
        batch, seq_len = 4, 16  # 64 tokens
        x = mx.random.normal((batch, seq_len, config.d_model))
        
        out = moe(x)
        
        assert out.shape == x.shape
        assert not mx.isnan(out).any()
    
    def test_capacity_metrics_tracking(self):
        """Test that capacity metrics are tracked during forward."""
        from mlx_impl.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        config = DeepSeekMoEV3Config.small_16_2()
        
        moe = DeepSeekMoEV3(config)
        
        # Run forward
        x = mx.random.normal((2, 8, config.d_model))
        moe(x)
        
        # Check capacity metrics exist and are initialized
        assert hasattr(moe, 'capacity_metrics')
        assert len(moe.capacity_metrics.expert_overflow) == 16
    
    def test_capacity_stats_method(self):
        """Test get_capacity_stats method."""
        from mlx_impl.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        config = DeepSeekMoEV3Config.small_16_2()
        moe = DeepSeekMoEV3(config)
        
        # Run forward
        x = mx.random.normal((2, 8, config.d_model))
        moe(x)
        
        # Get stats
        drop_rate, avg_util = moe.get_capacity_stats()
        
        assert isinstance(drop_rate, float)
        assert isinstance(avg_util, float)
        assert drop_rate >= 0
        assert avg_util >= 0
    
    def test_large_expert_count(self):
        """Test MoE with larger expert count."""
        from mlx_impl.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        # Use medium config with 64 experts (reduce d_model for speed)
        config = DeepSeekMoEV3Config.medium_64_4()
        config.d_model = 128
        config.routed_hidden_mult = 1.0
        
        moe = DeepSeekMoEV3(config)
        
        batch, seq_len = 2, 8
        x = mx.random.normal((batch, seq_len, config.d_model))
        
        out = moe(x)
        
        assert out.shape == x.shape
        assert not mx.isnan(out).any()


# ============================================================================
# Phase 3: Distributed Training Infrastructure Tests
# ============================================================================

class TestActivationCheckpointing:
    """Tests for Phase 3 activation checkpointing in MLX."""
    
    def test_checkpointing_none(self):
        """Test no checkpointing strategy."""
        from mlx_impl.pipeline import ActivationCheckpointing, CheckpointingStrategy
        
        ac = ActivationCheckpointing(
            strategy=CheckpointingStrategy.NONE,
            num_layers=32
        )
        
        assert ac.num_checkpointed_layers() == 0
        assert not ac.should_checkpoint(0)
    
    def test_checkpointing_every_n(self):
        """Test every-N-layers checkpointing."""
        from mlx_impl.pipeline import ActivationCheckpointing, CheckpointingStrategy
        
        ac = ActivationCheckpointing(
            strategy=CheckpointingStrategy.EVERY_N_LAYERS,
            num_layers=32,
            checkpoint_interval=4
        )
        
        assert ac.num_checkpointed_layers() == 8
        assert ac.should_checkpoint(0)
        assert not ac.should_checkpoint(1)
        assert ac.should_checkpoint(4)
    
    def test_checkpointing_optimal(self):
        """Test optimal checkpointing strategy."""
        from mlx_impl.pipeline import ActivationCheckpointing
        
        ac = ActivationCheckpointing.optimal(64)
        
        assert ac.num_checkpointed_layers() > 0
        assert ac.num_checkpointed_layers() <= 64


class TestZeROOptimizer:
    """Tests for Phase 3 ZeRO optimizer sharding in MLX."""
    
    def test_zero_stage0(self):
        """Test ZeRO Stage 0."""
        from mlx_impl.pipeline import ZeROOptimizer, ZeROStage
        
        zero = ZeROOptimizer(
            stage=ZeROStage.STAGE_0,
            dp_size=8,
            dp_rank=0,
            total_params=1000000
        )
        
        assert not zero.grads_sharded()
        assert not zero.params_sharded()
        assert zero.memory_per_param() == 16.0
    
    def test_zero_stage3(self):
        """Test ZeRO Stage 3."""
        from mlx_impl.pipeline import ZeROOptimizer, ZeROStage
        
        zero = ZeROOptimizer(
            stage=ZeROStage.STAGE_3,
            dp_size=8,
            dp_rank=0,
            total_params=1000000
        )
        
        assert zero.grads_sharded()
        assert zero.params_sharded()
        assert zero.memory_per_param() == 2.0  # 16/8
    
    def test_param_partition(self):
        """Test parameter partitioning."""
        from mlx_impl.pipeline import ZeROOptimizer, ZeROStage
        
        zero = ZeROOptimizer(
            stage=ZeROStage.STAGE_1,
            dp_size=4,
            dp_rank=0,
            total_params=1000
        )
        
        start, end = zero.get_param_partition()
        assert start == 0
        assert end == 250
        assert zero.owns_param(0)


class TestHierarchicalAllToAll:
    """Tests for Phase 3 hierarchical all-to-all in MLX."""
    
    def test_node_topology(self):
        """Test node topology calculation."""
        from mlx_impl.pipeline import HierarchicalAllToAll
        
        hier = HierarchicalAllToAll(
            world_size=16,
            ranks_per_node=8,
            global_rank=0
        )
        
        assert hier.num_nodes == 2
        assert hier.node_id == 0
        assert hier.is_node_leader()
    
    def test_intra_node_ranks(self):
        """Test intra-node rank computation."""
        from mlx_impl.pipeline import HierarchicalAllToAll
        
        hier = HierarchicalAllToAll(
            world_size=16,
            ranks_per_node=8,
            global_rank=10
        )
        
        intra = hier.intra_node_ranks()
        assert len(intra) == 8
        assert intra[0] == 8
    
    def test_send_order(self):
        """Test send order computation."""
        from mlx_impl.pipeline import HierarchicalAllToAll
        
        hier = HierarchicalAllToAll(
            world_size=16,
            ranks_per_node=8,
            global_rank=0
        )
        
        # Same node
        intra, inter = hier.compute_send_order(2)
        assert intra
        assert not inter


# ============================================================================
# Phase 4: FP8 Mixed-Precision Training Tests
# ============================================================================

class TestFP8Format:
    """Tests for FP8 format constants and configuration."""
    
    def test_fp8_format_constants(self):
        """Test FP8 format constants."""
        from mlx_impl.quantization import (
            E4M3_MAX, E4M3_MIN, E5M2_MAX, E5M2_MIN, FP8Format
        )
        
        assert E4M3_MAX == 448.0
        assert E4M3_MIN == -448.0
        assert E5M2_MAX == 57344.0
        assert E5M2_MIN == -57344.0
        
        # Test enum properties
        assert FP8Format.E4M3.max_value == 448.0
        assert FP8Format.E5M2.max_value == 57344.0
    
    def test_tile_scaling_config(self):
        """Test TileScalingConfig defaults."""
        from mlx_impl.quantization import TileScalingConfig, FP8Format
        
        config = TileScalingConfig()
        assert config.tile_rows == 128
        assert config.tile_cols == 128
        assert config.forward_format == FP8Format.E4M3
        assert config.backward_format == FP8Format.E5M2


class TestTileScaling:
    """Tests for per-tile scaling infrastructure."""
    
    def test_compute_tile_scales(self):
        """Test tile scale computation."""
        from mlx_impl.quantization import TileScalingConfig, TileScalingState, FP8Format
        
        config = TileScalingConfig(tile_rows=4, tile_cols=4)
        state = TileScalingState(config)
        
        # Create 8x8 tensor (2x2 tiles)
        tensor = mx.random.normal((8, 8))
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        assert scales.shape == (2, 2)
        assert mx.all(scales > 0)
    
    def test_amax_history(self):
        """Test amax history tracking."""
        from mlx_impl.quantization import TileScalingConfig, TileScalingState, FP8Format
        
        config = TileScalingConfig(tile_rows=4, tile_cols=4, amax_history_len=4)
        state = TileScalingState(config)
        
        for _ in range(6):
            tensor = mx.random.normal((8, 8))
            state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        assert len(state.amax_history) == 4
        
        smoothed = state.get_smoothed_amax()
        assert smoothed is not None
        assert smoothed.shape == (2, 2)


class TestFP8QuantizeDequantize:
    """Tests for FP8 quantization and dequantization."""
    
    def test_quantize_fp8_tiled(self):
        """Test FP8 tiled quantization."""
        from mlx_impl.quantization import (
            quantize_fp8_tiled, TileScalingConfig, TileScalingState, FP8Format
        )
        
        tensor = mx.random.normal((16, 16)) * 100
        
        config = TileScalingConfig(tile_rows=4, tile_cols=4)
        state = TileScalingState(config)
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        quantized = quantize_fp8_tiled(tensor, scales, FP8Format.E4M3, 4, 4)
        
        assert quantized.shape == tensor.shape
        assert mx.max(mx.abs(quantized)) <= FP8Format.E4M3.max_value
    
    def test_dequantize_fp8_tiled(self):
        """Test FP8 tiled dequantization."""
        from mlx_impl.quantization import (
            quantize_fp8_tiled, dequantize_fp8_tiled,
            TileScalingConfig, TileScalingState, FP8Format
        )
        
        tensor = mx.random.normal((8, 8)) * 10
        
        config = TileScalingConfig(tile_rows=4, tile_cols=4)
        state = TileScalingState(config)
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        quantized = quantize_fp8_tiled(tensor, scales, FP8Format.E4M3, 4, 4)
        dequantized = dequantize_fp8_tiled(quantized, scales, 4, 4)
        
        assert dequantized.shape == tensor.shape
        
        diff = float(mx.max(mx.abs(tensor - dequantized)))
        assert diff < 1.0


class TestFP8OptimizerState:
    """Tests for FP8 optimizer state quantization."""
    
    def test_optimizer_state_creation(self):
        """Test FP8OptimizerState creation."""
        from mlx_impl.quantization import FP8OptimizerState, FP8OptimizerConfig
        
        config = FP8OptimizerConfig()
        
        m = mx.random.normal((32, 32)) * 0.1
        v = mx.abs(mx.random.normal((32, 32))) * 0.01
        
        state = FP8OptimizerState(m, v, config)
        
        assert state.m_quantized.shape == m.shape
        assert state.v_quantized.shape == v.shape
        assert state.step == 0
    
    def test_optimizer_state_to_fp32(self):
        """Test dequantization of optimizer state."""
        from mlx_impl.quantization import FP8OptimizerState, FP8OptimizerConfig
        
        config = FP8OptimizerConfig()
        
        m = mx.random.normal((16, 16)) * 0.1
        v = mx.abs(mx.random.normal((16, 16))) * 0.01
        
        state = FP8OptimizerState(m, v, config)
        m_dq, v_dq = state.to_fp32()
        
        m_diff = float(mx.max(mx.abs(m - m_dq)))
        assert m_diff < 0.01
    
    def test_optimizer_state_update(self):
        """Test Adam-style update with FP8 state."""
        from mlx_impl.quantization import FP8OptimizerState, FP8OptimizerConfig
        
        config = FP8OptimizerConfig()
        
        m = mx.zeros((16, 16))
        v = mx.zeros((16, 16))
        
        state = FP8OptimizerState(m, v, config)
        
        grad = mx.random.normal((16, 16)) * 0.5
        m_new, v_new = state.update(grad, beta1=0.9, beta2=0.999)
        
        assert state.step == 1
        assert m_new.shape == grad.shape
        assert v_new.shape == grad.shape


class TestFP8MixedPrecisionManager:
    """Tests for FP8 mixed-precision training manager."""
    
    def test_manager_creation(self):
        """Test FP8MixedPrecisionManager creation."""
        from mlx_impl.quantization import FP8MixedPrecisionManager, FP8TrainingConfig
        
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        assert manager.loss_scale == 65536.0
        assert manager.growth_tracker == 0
    
    def test_loss_scaling(self):
        """Test loss scaling."""
        from mlx_impl.quantization import FP8MixedPrecisionManager, FP8TrainingConfig
        
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        loss = mx.array(1.0)
        scaled_loss = manager.scale_loss(loss)
        
        assert float(scaled_loss) == 65536.0
    
    def test_loss_scale_update(self):
        """Test loss scale update on overflow."""
        from mlx_impl.quantization import FP8MixedPrecisionManager, FP8TrainingConfig
        
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        initial_scale = manager.loss_scale
        manager.update_loss_scale(overflow=True)
        
        assert manager.loss_scale == initial_scale * 0.5
    
    def test_quantize_activation(self):
        """Test activation quantization."""
        from mlx_impl.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig, TileScalingConfig
        )
        
        config = FP8TrainingConfig(
            tile_config=TileScalingConfig(tile_rows=4, tile_cols=4)
        )
        manager = FP8MixedPrecisionManager(config)
        
        tensor = mx.random.normal((8, 8)) * 100
        quantized, scales = manager.quantize_activation("layer1", tensor)
        
        assert quantized.shape == tensor.shape
        assert scales.shape == (2, 2)
    
    def test_init_optimizer_state(self):
        """Test optimizer state initialization."""
        from mlx_impl.quantization import FP8MixedPrecisionManager, FP8TrainingConfig
        
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        param = mx.random.normal((64, 64))
        manager.init_optimizer_state("weight", param)
        
        assert "weight" in manager.optimizer_states
        state = manager.get_optimizer_state("weight")
        assert state is not None


class TestFP8Linear:
    """Tests for FP8Linear module."""
    
    def test_fp8_linear_creation(self):
        """Test FP8Linear creation."""
        from mlx_impl.quantization import FP8Linear
        
        layer = FP8Linear(64, 32, tile_size=16)
        
        assert layer.input_dim == 64
        assert layer.output_dim == 32
        assert layer.weight.shape == (32, 64)
    
    def test_fp8_linear_forward(self):
        """Test FP8Linear forward pass."""
        from mlx_impl.quantization import FP8Linear
        
        layer = FP8Linear(64, 32)
        
        x = mx.random.normal((4, 64))
        out = layer(x)
        
        assert out.shape == (4, 32)


# ============================================================================
# Phase 5: Agent/Tool-Use Training Tests (MLX)
# ============================================================================

class TestMLXToolCallFormat:
    """Tests for tool call JSON format specification (MLX)."""
    
    def test_tool_call_creation(self):
        """Test ToolCall creation and serialization."""
        from mlx_impl.agent import ToolCall, ToolType
        
        call = ToolCall(
            id="call_001",
            tool_type=ToolType.CODE_EXECUTION,
            function_name="run_python",
            arguments={"code": "print('hello')"}
        )
        
        assert call.id == "call_001"
        assert call.tool_type == ToolType.CODE_EXECUTION
        assert call.function_name == "run_python"
        
        json_str = call.to_json()
        assert "call_001" in json_str
        assert "CODE_EXECUTION" in json_str
    
    def test_tool_call_from_json(self):
        """Test ToolCall parsing from JSON."""
        from mlx_impl.agent import ToolCall, ToolType
        
        json_str = '''
        {
            "id": "call_002",
            "tool_type": "WEB_SEARCH",
            "function_name": "search",
            "arguments": {"query": "deepseek v3"}
        }
        '''
        
        call = ToolCall.from_json(json_str)
        assert call is not None
        assert call.id == "call_002"
        assert call.tool_type == ToolType.WEB_SEARCH
    
    def test_tool_response_creation(self):
        """Test ToolResponse creation."""
        from mlx_impl.agent import ToolResponse, ToolStatus
        
        response = ToolResponse(
            call_id="call_001",
            status=ToolStatus.SUCCESS,
            content="Execution complete",
            execution_time_ms=100
        )
        
        assert response.call_id == "call_001"
        assert response.status == ToolStatus.SUCCESS


class TestMLXAgentTrajectory:
    """Tests for agent trajectory management (MLX)."""
    
    def test_trajectory_creation(self):
        """Test AgentTrajectory creation."""
        from mlx_impl.agent import AgentTrajectory, TaskTier
        
        trajectory = AgentTrajectory(
            prompt="Calculate 2+2",
            task_tier=TaskTier.SINGLE_TOOL
        )
        
        assert trajectory.prompt == "Calculate 2+2"
        assert trajectory.task_tier == TaskTier.SINGLE_TOOL
        assert len(trajectory.steps) == 0
    
    def test_trajectory_with_steps(self):
        """Test trajectory with multiple steps."""
        from mlx_impl.agent import (
            AgentTrajectory, AgentStep, AgentAction, AgentActionType,
            ToolCall, ToolResponse, ToolType, ToolStatus, TaskTier
        )
        
        trajectory = AgentTrajectory(
            prompt="Calculate 2+2",
            task_tier=TaskTier.SINGLE_TOOL
        )
        
        # Add tool call step
        tool_call = ToolCall(
            id="call_1",
            tool_type=ToolType.REASONING,
            function_name="calculate",
            arguments={"expression": "2+2"}
        )
        tool_response = ToolResponse(
            call_id="call_1",
            status=ToolStatus.SUCCESS,
            content="Result: 4",
            execution_time_ms=10
        )
        tool_step = AgentStep(
            action=AgentAction(
                action_type=AgentActionType.TOOL_CALL,
                content="Calling calculator",
                tool_call=tool_call,
                tool_response=tool_response
            )
        )
        trajectory.steps.append(tool_step)
        trajectory.task_completed = True
        
        assert trajectory.num_tool_calls == 1
        assert trajectory.successful_tool_calls == 1


class TestMLXTaskTiers:
    """Tests for task tier specification (MLX)."""
    
    def test_tier_expected_calls(self):
        """Test tier expected tool call ranges."""
        from mlx_impl.agent import TaskTier
        
        min_calls, max_calls = TaskTier.SINGLE_TOOL.expected_tool_calls()
        assert min_calls == 1
        assert max_calls == 1
        
        min_calls, max_calls = TaskTier.COMPLEX_WORKFLOW.expected_tool_calls()
        assert min_calls == 10
        assert max_calls == 50
    
    def test_tier_environment_counts(self):
        """Test tier environment counts (~1,800 total)."""
        from mlx_impl.agent import TaskTier
        
        total_envs = sum(tier.environment_count() for tier in TaskTier)
        assert total_envs == 1800


class TestMLXRewardComputation:
    """Tests for multi-objective reward computation (MLX)."""
    
    def test_reward_weights(self):
        """Test reward weight validation."""
        from mlx_impl.agent import RewardWeights
        
        weights = RewardWeights()
        total = weights.correctness + weights.format + weights.efficiency + weights.safety
        assert abs(total - 1.0) < 1e-6
        
        # R_total = 0.5*R_correct + 0.2*R_format + 0.15*R_efficiency + 0.15*R_safety
        assert weights.correctness == 0.5
        assert weights.format == 0.2
        assert weights.efficiency == 0.15
        assert weights.safety == 0.15
    
    def test_reward_breakdown(self):
        """Test reward breakdown computation."""
        from mlx_impl.agent import AgentRewardComputer, AgentTrajectory, TaskTier
        
        computer = AgentRewardComputer()
        
        trajectory = AgentTrajectory(
            prompt="Calculate 2+2",
            task_tier=TaskTier.SINGLE_TOOL
        )
        trajectory.final_output = "4"
        trajectory.task_completed = True
        
        breakdown = computer.compute_reward(trajectory, ground_truth="4")
        
        assert breakdown.correctness == 1.0
        assert breakdown.total > 0.8
    
    def test_batch_rewards_mlx(self):
        """Test batch reward computation returns MLX array."""
        from mlx_impl.agent import (
            AgentRewardComputer, AgentTrajectory, TaskTier
        )
        
        computer = AgentRewardComputer()
        
        trajectories = []
        for i in range(4):
            traj = AgentTrajectory(
                prompt=f"Task {i}",
                task_tier=TaskTier.SINGLE_TOOL
            )
            traj.final_output = str(i)
            traj.task_completed = True
            trajectories.append(traj)
        
        rewards = computer.compute_batch_rewards(trajectories)
        
        assert rewards.shape == (4,)
        assert isinstance(rewards, mx.array)


class TestMLXTaskGenerator:
    """Tests for synthetic task generation (MLX)."""
    
    def test_generator_creation(self):
        """Test TaskGenerator creation."""
        from mlx_impl.agent import TaskGenerator
        
        generator = TaskGenerator(seed=42)
        assert generator is not None
    
    def test_generate_single_task(self):
        """Test single task generation."""
        from mlx_impl.agent import TaskGenerator, TaskTier
        
        generator = TaskGenerator(seed=42)
        
        prompt, tier = generator.generate_task(TaskTier.SINGLE_TOOL)
        
        assert len(prompt) > 0
        assert tier == TaskTier.SINGLE_TOOL
    
    def test_generate_batch(self):
        """Test batch task generation."""
        from mlx_impl.agent import TaskGenerator, TaskTier
        
        generator = TaskGenerator(seed=42)
        
        batch = generator.generate_batch(10, TaskTier.MULTI_TOOL_SEQ)
        
        assert len(batch) == 10
        for prompt, tier in batch:
            assert tier == TaskTier.MULTI_TOOL_SEQ
    
    def test_tier_distribution(self):
        """Test tier distribution matches spec."""
        from mlx_impl.agent import TaskGenerator, TaskTier
        
        generator = TaskGenerator()
        dist = generator.get_tier_distribution()
        
        assert sum(dist.values()) == 1800


class TestMLXToolCallParser:
    """Tests for tool call parsing (MLX)."""
    
    def test_parse_json_code_block(self):
        """Test parsing tool call from JSON code block."""
        from mlx_impl.agent import ToolCallParser, ToolType
        
        text = '''
        I'll use a calculator for this.
        
        ```json
        {
            "id": "call_123",
            "tool_type": "REASONING",
            "function_name": "calculate",
            "arguments": {"expression": "2+2"}
        }
        ```
        '''
        
        calls = ToolCallParser.parse(text)
        
        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.REASONING
    
    def test_parse_multiple_calls(self):
        """Test parsing multiple tool calls."""
        from mlx_impl.agent import ToolCallParser
        
        text = '''
        First: ```json {"function_name": "a", "tool_type": "WEB_SEARCH", "arguments": {}} ```
        Second: ```json {"function_name": "b", "tool_type": "WEB_SEARCH", "arguments": {}} ```
        '''
        
        calls = ToolCallParser.parse(text)
        assert len(calls) == 2


class TestMLXAgentGRPOTrainer:
    """Tests for Agent GRPO trainer (MLX)."""
    
    def test_trainer_creation(self):
        """Test AgentGRPOTrainer creation."""
        from mlx_impl.agent import AgentGRPOTrainer, AgentGRPOConfig
        
        config = AgentGRPOConfig(
            beta=0.04,
            gamma=0.99,
            group_size=8,
        )
        trainer = AgentGRPOTrainer(config)
        
        assert trainer.beta == 0.04
        assert trainer.gamma == 0.99
    
    def test_compute_loss_mlx(self):
        """Test GRPO loss computation with MLX arrays."""
        from mlx_impl.agent import AgentGRPOTrainer, AgentGRPOConfig
        
        config = AgentGRPOConfig(beta=0.04, gamma=0.99)
        trainer = AgentGRPOTrainer(config)
        
        G, Seq, Vocab = 4, 32, 100
        
        logits = mx.random.normal((G, Seq, Vocab))
        input_ids = mx.random.randint(0, Vocab, (G, Seq))
        rewards = mx.random.normal((G,))
        ref_logits = mx.random.normal((G, Seq, Vocab))
        
        loss = trainer.compute_loss(logits, input_ids, rewards, ref_logits)
        
        assert loss.shape == ()  # Scalar
        assert not mx.isnan(loss)
    
    def test_curriculum_update(self):
        """Test curriculum tier progression."""
        from mlx_impl.agent import AgentGRPOTrainer, AgentGRPOConfig, TaskTier
        
        config = AgentGRPOConfig(use_curriculum=True)
        trainer = AgentGRPOTrainer(config)
        trainer.tier_thresholds = {
            TaskTier.SINGLE_TOOL: 10,
            TaskTier.MULTI_TOOL_SEQ: 20,
            TaskTier.MULTI_TOOL_PARALLEL: 30,
            TaskTier.COMPLEX_WORKFLOW: float('inf'),
        }
        
        assert trainer.get_current_tier() == TaskTier.SINGLE_TOOL
        
        for _ in range(15):
            trainer.update_curriculum()
        
        assert trainer.get_current_tier() == TaskTier.MULTI_TOOL_SEQ
    
    def test_training_progress(self):
        """Test training progress reporting."""
        from mlx_impl.agent import AgentGRPOTrainer, AgentGRPOConfig
        
        config = AgentGRPOConfig(use_curriculum=True)
        trainer = AgentGRPOTrainer(config)
        
        progress = trainer.get_training_progress()
        
        assert "current_tier" in progress
        assert "steps_completed" in progress


class TestMLXAgentEnvironment:
    """Tests for agent environment simulation (MLX)."""
    
    def test_environment_creation(self):
        """Test AgentEnvironment creation."""
        from mlx_impl.agent import AgentEnvironment
        
        env = AgentEnvironment(max_steps=50)
        assert env.max_steps == 50
    
    def test_execute_tool_call(self):
        """Test tool call execution."""
        from mlx_impl.agent import (
            AgentEnvironment, ToolCall, ToolType, ToolStatus
        )
        
        env = AgentEnvironment()
        
        call = ToolCall(
            id="test_call",
            tool_type=ToolType.REASONING,
            function_name="calculate",
            arguments={"expression": "2+2"}
        )
        
        response = env.execute_tool_call(call)
        
        assert response.call_id == "test_call"
        assert response.status == ToolStatus.SUCCESS


class TestMLXAgentGroupSampler:
    """Tests for agent group sampler (MLX)."""
    
    def test_sampler_creation(self):
        """Test AgentGroupSampler creation."""
        from mlx_impl.agent import AgentGroupSampler
        
        sampler = AgentGroupSampler(group_size=8)
        assert sampler.group_size == 8
    
    def test_sample_group(self):
        """Test sampling a group of trajectories."""
        from mlx_impl.agent import AgentGroupSampler, TaskTier
        
        sampler = AgentGroupSampler(group_size=4)
        
        def mock_model(prompt):
            return "The answer is 4"
        
        trajectories = sampler.sample_group(
            prompt="Calculate 2+2",
            model_fn=mock_model,
            tier=TaskTier.SINGLE_TOOL
        )
        
        assert len(trajectories) == 4
        for traj in trajectories:
            assert traj.task_completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
