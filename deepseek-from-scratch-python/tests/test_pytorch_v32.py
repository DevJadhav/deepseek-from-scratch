"""
Tests for DeepSeek-V3.2 PyTorch implementations.

Tests sparse attention, extended RoPE, MoE V3, DualPipe scheduler,
and DSA alignment loss for MPS/CUDA compatibility.
"""

import pytest
import torch
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# Extended RoPE Tests
# ============================================================================

class TestExtendedRoPE:
    """Tests for Extended RoPE with NTK/YaRN scaling."""
    
    def test_config_creation(self):
        """Test ExtendedRoPEConfig creation."""
        from deepseek.model.mla import ExtendedRoPEConfig
        
        config = ExtendedRoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type="yarn",
        )
        assert config.d_head == 64
        assert config.max_seq_len == 131072
        assert config.scaling_type == "yarn"
    
    def test_ntk_forward(self):
        """Test NTK-aware RoPE forward pass."""
        from deepseek.model.mla import ExtendedRoPEConfig, ExtendedRotaryPositionalEncoding
        
        device = get_device()
        config = ExtendedRoPEConfig(
            d_head=64,
            max_seq_len=8192,
            scaling_type="ntk",
            ntk_alpha=8.0,
        )
        rope = ExtendedRotaryPositionalEncoding(config).to(device)
        
        batch, heads, seq_len = 2, 8, 128
        x = torch.randn(batch, heads, seq_len, 64, device=device)
        
        out = rope(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_yarn_forward(self):
        """Test YaRN interpolation forward pass."""
        from deepseek.model.mla import ExtendedRoPEConfig, ExtendedRotaryPositionalEncoding
        
        device = get_device()
        config = ExtendedRoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type="yarn",
            yarn_original_max_position=4096,
        )
        rope = ExtendedRotaryPositionalEncoding(config).to(device)
        
        batch, heads, seq_len = 2, 8, 256
        x = torch.randn(batch, heads, seq_len, 64, device=device)
        
        out = rope(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        # YaRN should apply attention scaling factor
        assert rope.yarn_attn_factor > 1.0
    
    def test_rope_with_offset(self):
        """Test RoPE with position offset for KV cache."""
        from deepseek.model.mla import ExtendedRoPEConfig, ExtendedRotaryPositionalEncoding
        
        device = get_device()
        config = ExtendedRoPEConfig(d_head=64, scaling_type="none")
        rope = ExtendedRotaryPositionalEncoding(config).to(device)
        
        batch, heads, seq_len = 2, 8, 32
        x = torch.randn(batch, heads, seq_len, 64, device=device)
        
        out_no_offset = rope(x, offset=0)
        out_with_offset = rope(x, offset=100)
        
        # Outputs should be different due to offset
        assert not torch.allclose(out_no_offset, out_with_offset)


# ============================================================================
# Sparse Attention Tests
# ============================================================================

class TestSparseAttention:
    """Tests for DeepSeek Sparse Attention."""
    
    def test_dsa_config(self):
        """Test DSA configuration."""
        from deepseek.model.sparse_attention import DSAConfig
        
        config = DSAConfig.for_128k_context()
        assert config.max_seq_len == 131072
        assert config.window_size == 4096
        assert config.effective_kv_budget < config.max_seq_len
    
    def test_sparse_pattern_generation(self):
        """Test sparse attention pattern generation."""
        from deepseek.model.sparse_attention import DSAConfig, SparseAttentionPattern
        
        device = get_device()
        config = DSAConfig(window_size=8, dilation_stride=4, causal=True)
        pattern = SparseAttentionPattern(config).to(device)
        
        mask = pattern(16, device)
        
        assert mask.shape == (16, 16)
        assert mask.dtype == torch.bool
    
    def test_dsa_forward(self):
        """Test DSA forward pass."""
        from deepseek.model.sparse_attention import DSAConfig, DeepSeekSparseAttention
        
        device = get_device()
        config = DSAConfig(
            d_model=64,
            num_heads=4,
            d_latent=32,
            d_rope=16,
            window_size=8,
            max_seq_len=64,
        )
        dsa = DeepSeekSparseAttention(config).to(device)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, 64, device=device)
        
        out = dsa(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_block_sparse_attention(self):
        """Test block-sparse attention."""
        from deepseek.model.sparse_attention import DSAConfig, BlockSparseAttention
        
        device = get_device()
        config = DSAConfig(
            d_model=64,
            num_heads=4,
            window_size=16,
            max_seq_len=128,
        )
        bsa = BlockSparseAttention(config, block_size=8).to(device)
        
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, 64, device=device)
        
        out = bsa(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


# ============================================================================
# MoE V3 Tests
# ============================================================================

class TestMoEV3:
    """Tests for DeepSeek-V3 MoE."""
    
    def test_moe_v3_config(self):
        """Test MoE V3 configuration."""
        from deepseek.model.moe import DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config.v3_256_8()
        assert config.n_routed_experts == 256
        assert config.top_k == 8
        assert config.n_expert_groups == 8
        assert config.experts_per_group == 32
    
    def test_load_balancing_state(self):
        """Test load balancing state updates."""
        from deepseek.model.moe import DeepSeekMoEV3Config, LoadBalancingState
        
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        state = LoadBalancingState(config, device=device)
        
        # Simulate uneven expert usage
        counts = torch.tensor([10.0, 0.5] + [1.0] * 14, device=device)
        
        state.update(counts)
        
        # Bias should adjust: overused gets negative, underused gets positive
        assert state.bias[0] < state.bias[1]
    
    def test_moe_v3_forward(self):
        """Test MoE V3 forward pass."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        config.d_model = 64
        config.routed_hidden_mult = 2.0
        config.shared_hidden_mult = 2.0
        
        moe = DeepSeekMoEV3(config).to(device)
        
        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, 64, device=device)
        
        out = moe(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_moe_v3_load_balance_stats(self):
        """Test load balance statistics."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        config.d_model = 32
        
        moe = DeepSeekMoEV3(config).to(device)
        
        # Run a few iterations
        for _ in range(5):
            x = torch.randn(2, 8, 32, device=device)
            moe(x)
        
        mean, imbalance, steps = moe.get_load_balance_stats()
        
        assert mean > 0
        assert steps == 5


# ============================================================================
# DualPipe Tests
# ============================================================================

class TestDualPipe:
    """Tests for DualPipe scheduler."""
    
    def test_dualpipe_creation(self):
        """Test DualPipe scheduler creation."""
        from deepseek.training.pipeline import DualPipeScheduler
        
        scheduler = DualPipeScheduler(num_stages=4, num_micro_batches=8)
        
        assert scheduler.num_stages == 4
        assert scheduler.num_micro_batches == 8
    
    def test_dualpipe_schedule(self):
        """Test DualPipe schedule generation."""
        from deepseek.training.pipeline import DualPipeScheduler, DualPipePhase
        
        scheduler = DualPipeScheduler(num_stages=4, num_micro_batches=8)
        schedule = scheduler.build_schedule()
        
        assert len(schedule) > 0
        
        # Check that schedule covers all micro-batches
        forward_batches = set()
        backward_batches = set()
        
        for step in schedule:
            for slot in step:
                if slot.direction == "forward":
                    forward_batches.add(slot.micro_batch_id)
                else:
                    backward_batches.add(slot.micro_batch_id)
        
        # All micro-batches should appear in forward
        assert len(forward_batches) == 8
    
    def test_dualpipe_reset(self):
        """Test DualPipe reset."""
        from deepseek.training.pipeline import DualPipeScheduler
        
        scheduler = DualPipeScheduler(num_stages=4, num_micro_batches=8)
        scheduler.build_schedule()
        scheduler.step()
        scheduler.step()
        
        assert scheduler.current_step == 2
        
        scheduler.reset()
        
        assert scheduler.current_step == 0
    
    def test_dualpipe_bubble_ratio(self):
        """Test DualPipe bubble ratio calculation."""
        from deepseek.training.pipeline import DualPipeScheduler
        
        scheduler = DualPipeScheduler(num_stages=4, num_micro_batches=16)
        
        bubble_ratio = scheduler.get_bubble_ratio()
        
        # DualPipe: (P-1)/(2M) = 3/32 = 0.09375
        expected = (4 - 1) / (2 * 16)
        assert abs(bubble_ratio - expected) < 1e-6


# ============================================================================
# DSA Alignment Loss Tests
# ============================================================================

class TestDSAAlignmentLoss:
    """Tests for DSA alignment loss."""
    
    def test_mse_loss(self):
        """Test MSE component of alignment loss."""
        from deepseek.training.training import DSAAlignmentLoss
        
        device = get_device()
        loss_fn = DSAAlignmentLoss(mse_weight=1.0, cosine_weight=0.0).to(device)
        
        sparse = torch.randn(2, 16, 64, device=device)
        full = torch.randn(2, 16, 64, device=device)
        
        loss = loss_fn(sparse, full)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_cosine_loss(self):
        """Test cosine similarity component."""
        from deepseek.training.training import DSAAlignmentLoss
        
        device = get_device()
        loss_fn = DSAAlignmentLoss(mse_weight=0.0, cosine_weight=1.0).to(device)
        
        sparse = torch.randn(2, 16, 64, device=device)
        full = sparse.clone()  # Same vectors
        
        loss = loss_fn(sparse, full)
        
        # Identical vectors should have ~0 cosine loss
        assert loss.item() < 0.01
    
    def test_combined_loss(self):
        """Test combined MSE + cosine loss."""
        from deepseek.training.training import DSAAlignmentLoss
        
        device = get_device()
        loss_fn = DSAAlignmentLoss(mse_weight=1.0, cosine_weight=0.5).to(device)
        
        sparse = torch.randn(2, 16, 64, device=device)
        full = torch.randn(2, 16, 64, device=device)
        
        loss = loss_fn(sparse, full)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_sampled_loss(self):
        """Test loss with position sampling."""
        from deepseek.training.training import DSAAlignmentLoss
        
        device = get_device()
        loss_fn = DSAAlignmentLoss(sample_ratio=0.1).to(device)
        
        sparse = torch.randn(2, 100, 64, device=device)
        full = torch.randn(2, 100, 64, device=device)
        
        loss = loss_fn(sparse, full)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


# ============================================================================
# Integration Tests
# ============================================================================

class TestV32Integration:
    """Integration tests for V3.2 components."""
    
    def test_full_pipeline(self):
        """Test all V3.2 components together."""
        from deepseek.model.sparse_attention import DSAConfig, DeepSeekSparseAttention
        from deepseek.model.mla import ExtendedRoPEConfig, ExtendedRotaryPositionalEncoding
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        from deepseek.training.pipeline import DualPipeScheduler
        
        device = get_device()
        d_model = 64
        num_heads = 4
        batch_size = 2
        seq_len = 16
        
        # Extended RoPE
        rope_config = ExtendedRoPEConfig(d_head=d_model // num_heads)
        rope = ExtendedRotaryPositionalEncoding(rope_config).to(device)
        
        # DSA
        dsa_config = DSAConfig(
            d_model=d_model,
            num_heads=num_heads,
            d_latent=32,
            d_rope=16,
            window_size=8,
            max_seq_len=64,
        )
        dsa = DeepSeekSparseAttention(dsa_config).to(device)
        
        # MoE V3
        moe_config = DeepSeekMoEV3Config.small_16_2()
        moe_config.d_model = d_model
        moe = DeepSeekMoEV3(moe_config).to(device)
        
        # DualPipe
        scheduler = DualPipeScheduler(num_micro_batches=4)
        
        # Forward pass
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # DSA
        attn_out = dsa(x)
        assert attn_out.shape == x.shape
        
        # MoE
        moe_out = moe(attn_out)
        assert moe_out.shape == x.shape
        
        # No NaNs
        assert not torch.isnan(attn_out).any()
        assert not torch.isnan(moe_out).any()
        
        print(f"V3.2 integration test passed on {device}")


# ============================================================================
# Phase 2: Capacity Metrics Tests
# ============================================================================

class TestCapacityMetrics:
    """Tests for Phase 2 capacity metrics."""
    
    def test_capacity_metrics_creation(self):
        """Test CapacityMetrics class."""
        from deepseek.model.moe import CapacityMetrics
        
        metrics = CapacityMetrics()
        metrics.reset(n_experts=16)
        
        assert len(metrics.expert_overflow) == 16
        assert len(metrics.expert_utilization) == 16
        assert metrics.total_tokens == 0
        assert metrics.dropped_tokens == 0
    
    def test_capacity_metrics_drop_rate(self):
        """Test drop rate calculation."""
        from deepseek.model.moe import CapacityMetrics
        
        metrics = CapacityMetrics()
        metrics.reset(n_experts=4)
        
        # Simulate some dispatches with overflow
        metrics.record_dispatch(expert_id=0, tokens_routed=100, capacity=80)  # 20 dropped
        metrics.record_dispatch(expert_id=1, tokens_routed=50, capacity=60)   # no drop
        metrics.record_dispatch(expert_id=2, tokens_routed=30, capacity=20)   # 10 dropped
        
        drop_rate = metrics.drop_rate()
        
        # 30 dropped out of 180 total routed
        assert drop_rate > 0  # Some tokens dropped
        assert drop_rate < 1.0  # Not all dropped
    
    def test_capacity_metrics_utilization(self):
        """Test utilization tracking."""
        from deepseek.model.moe import CapacityMetrics
        
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
        from deepseek.model.moe import CapacityMetrics
        
        metrics = CapacityMetrics()
        metrics.reset(n_experts=4)
        
        # Expert 0: 100 routed, 80 capacity -> 20 dropped
        metrics.record_dispatch(expert_id=0, tokens_routed=100, capacity=80)
        
        assert metrics.total_tokens == 80  # Only capacity processed
        assert metrics.dropped_tokens == 20
        assert metrics.expert_overflow[0] == 20


class TestEfficientBatchedDispatch:
    """Tests for efficient batched expert dispatch."""
    
    def test_moe_with_capacity_factor(self):
        """Test MoE respects capacity factor."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        config.d_model = 64
        config.capacity_factor = 1.5  # Allow 50% overflow
        
        moe = DeepSeekMoEV3(config).to(device)
        
        batch, seq_len = 4, 16  # 64 tokens
        x = torch.randn(batch, seq_len, 64, device=device)
        
        out = moe(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_capacity_metrics_tracking(self):
        """Test that capacity metrics are tracked during forward."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        config = DeepSeekMoEV3Config.small_16_2()
        config.d_model = 64
        
        moe = DeepSeekMoEV3(config).to(device)
        
        # Run forward
        x = torch.randn(2, 8, 64, device=device)
        moe(x)
        
        # Check capacity metrics exist and are initialized
        assert hasattr(moe, 'capacity_metrics')
        assert len(moe.capacity_metrics.expert_overflow) == 16
    
    def test_large_expert_count(self):
        """Test MoE with larger expert count (simulating 256 experts)."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        # Use medium config with 64 experts
        config = DeepSeekMoEV3Config.medium_64_4()
        config.d_model = 32
        config.routed_hidden_mult = 1.0
        
        moe = DeepSeekMoEV3(config).to(device)
        
        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, 32, device=device)
        
        out = moe(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


# ============================================================================
# Phase 3: Distributed Training Infrastructure Tests
# ============================================================================

class TestActivationCheckpointing:
    """Tests for Phase 3 activation checkpointing."""
    
    def test_checkpointing_none(self):
        """Test no checkpointing strategy."""
        from deepseek.training.pipeline import (
            ActivationCheckpointing, 
            CheckpointingStrategy
        )
        
        ac = ActivationCheckpointing(
            strategy=CheckpointingStrategy.NONE,
            num_layers=32
        )
        
        assert ac.num_checkpointed_layers() == 0
        assert not ac.should_checkpoint(0)
        assert not ac.should_checkpoint(15)
    
    def test_checkpointing_every_n(self):
        """Test every-N-layers checkpointing strategy."""
        from deepseek.training.pipeline import (
            ActivationCheckpointing, 
            CheckpointingStrategy
        )
        
        ac = ActivationCheckpointing(
            strategy=CheckpointingStrategy.EVERY_N_LAYERS,
            num_layers=32,
            checkpoint_interval=4
        )
        
        assert ac.num_checkpointed_layers() == 8
        assert ac.should_checkpoint(0)
        assert not ac.should_checkpoint(1)
        assert ac.should_checkpoint(4)
        assert ac.should_checkpoint(8)
    
    def test_checkpointing_selective(self):
        """Test selective layer checkpointing."""
        from deepseek.training.pipeline import (
            ActivationCheckpointing, 
            CheckpointingStrategy
        )
        
        ac = ActivationCheckpointing(
            strategy=CheckpointingStrategy.SELECTIVE,
            num_layers=32,
            checkpoint_layers=[0, 15, 31]
        )
        
        assert ac.num_checkpointed_layers() == 3
        assert ac.should_checkpoint(0)
        assert not ac.should_checkpoint(1)
        assert ac.should_checkpoint(15)
        assert ac.should_checkpoint(31)
    
    def test_checkpointing_optimal(self):
        """Test optimal checkpointing strategy."""
        from deepseek.training.pipeline import ActivationCheckpointing
        
        ac = ActivationCheckpointing.optimal(64)
        
        assert ac.num_checkpointed_layers() > 0
        assert ac.num_checkpointed_layers() <= 64
    
    def test_memory_savings_estimate(self):
        """Test memory savings estimation."""
        from deepseek.training.pipeline import (
            ActivationCheckpointing,
            CheckpointingStrategy
        )
        
        ac = ActivationCheckpointing(
            strategy=CheckpointingStrategy.FULL,
            num_layers=32
        )
        
        savings = ac.estimate_memory_savings(
            hidden_dim=4096,
            seq_len=2048,
            batch_size=4,
            dtype_bytes=4  # float32
        )
        
        assert savings > 0
        # Each layer saves ~2 * 4096 * 2048 * 4 * 4 = 268MB
        assert savings == 32 * 2 * 4096 * 2048 * 4 * 4


class TestZeROOptimizer:
    """Tests for Phase 3 ZeRO optimizer sharding."""
    
    def test_zero_stage0(self):
        """Test ZeRO Stage 0 (no sharding)."""
        from deepseek.training.pipeline import ZeROOptimizer, ZeROStage
        
        zero = ZeROOptimizer(
            stage=ZeROStage.STAGE_0,
            dp_size=8,
            dp_rank=0,
            total_params=1000000
        )
        
        assert not zero.grads_sharded()
        assert not zero.params_sharded()
        assert zero.needs_grad_reduce()
        assert zero.memory_per_param() == 16.0
    
    def test_zero_stage3(self):
        """Test ZeRO Stage 3 (full sharding)."""
        from deepseek.training.pipeline import ZeROOptimizer, ZeROStage
        
        zero = ZeROOptimizer(
            stage=ZeROStage.STAGE_3,
            dp_size=8,
            dp_rank=0,
            total_params=1000000
        )
        
        assert zero.grads_sharded()
        assert zero.params_sharded()
        assert not zero.needs_grad_reduce()
        assert zero.memory_per_param() == 16.0 / 8.0
    
    def test_param_partition(self):
        """Test parameter partitioning."""
        from deepseek.training.pipeline import ZeROOptimizer, ZeROStage
        
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
        assert zero.owns_param(249)
        assert not zero.owns_param(250)
    
    def test_memory_savings(self):
        """Test memory savings across ZeRO stages."""
        from deepseek.training.pipeline import ZeROOptimizer, ZeROStage
        
        mem0 = ZeROOptimizer(
            stage=ZeROStage.STAGE_0, dp_size=8, dp_rank=0, total_params=1000000
        ).memory_per_param()
        
        mem3 = ZeROOptimizer(
            stage=ZeROStage.STAGE_3, dp_size=8, dp_rank=0, total_params=1000000
        ).memory_per_param()
        
        assert mem3 < mem0  # Stage 3 uses less memory
        assert mem3 == mem0 / 8  # 8x reduction


class TestHierarchicalAllToAll:
    """Tests for Phase 3 hierarchical all-to-all."""
    
    def test_node_topology(self):
        """Test node topology calculation."""
        from deepseek.training.pipeline import HierarchicalAllToAll
        
        hier = HierarchicalAllToAll(
            world_size=16,
            ranks_per_node=8,
            global_rank=0
        )
        
        assert hier.num_nodes == 2
        assert hier.node_id == 0
        assert hier.local_rank == 0
        assert hier.is_node_leader()
    
    def test_intra_node_ranks(self):
        """Test intra-node rank computation."""
        from deepseek.training.pipeline import HierarchicalAllToAll
        
        hier = HierarchicalAllToAll(
            world_size=16,
            ranks_per_node=8,
            global_rank=10  # Node 1, local rank 2
        )
        
        intra = hier.intra_node_ranks()
        assert len(intra) == 8
        assert intra[0] == 8
        assert intra[7] == 15
        assert hier.node_id == 1
        assert hier.local_rank == 2
    
    def test_inter_node_leaders(self):
        """Test inter-node leader ranks."""
        from deepseek.training.pipeline import HierarchicalAllToAll
        
        hier = HierarchicalAllToAll(
            world_size=16,
            ranks_per_node=8,
            global_rank=0
        )
        
        leaders = hier.inter_node_leader_ranks()
        assert len(leaders) == 2
        assert leaders[0] == 0
        assert leaders[1] == 8
    
    def test_send_order(self):
        """Test send order computation."""
        from deepseek.training.pipeline import HierarchicalAllToAll
        
        hier = HierarchicalAllToAll(
            world_size=16,
            ranks_per_node=8,
            global_rank=0
        )
        
        # Same node
        intra, inter = hier.compute_send_order(2)
        assert intra
        assert not inter
        
        # Different node
        intra, inter = hier.compute_send_order(10)
        assert not intra
        assert inter


# ============================================================================
# Phase 4: FP8 Mixed-Precision Training Tests
# ============================================================================

class TestFP8Format:
    """Tests for FP8 format constants and configuration."""
    
    def test_fp8_format_constants(self):
        """Test FP8 format constants."""
        from deepseek.model.quantization import (
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
        from deepseek.model.quantization import TileScalingConfig, FP8Format
        
        config = TileScalingConfig()
        assert config.tile_rows == 128
        assert config.tile_cols == 128
        assert config.forward_format == FP8Format.E4M3
        assert config.backward_format == FP8Format.E5M2
        assert config.dynamic_scaling is True


class TestTileScaling:
    """Tests for per-tile scaling infrastructure."""
    
    def test_compute_tile_scales(self):
        """Test tile scale computation."""
        from deepseek.model.quantization import (
            TileScalingConfig, TileScalingState, FP8Format
        )
        
        device = get_device()
        config = TileScalingConfig(tile_rows=4, tile_cols=4)
        state = TileScalingState(config)
        
        # Create 8x8 tensor (2x2 tiles)
        tensor = torch.randn(8, 8, device=device)
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        assert scales.shape == (2, 2)
        assert (scales > 0).all()
    
    def test_amax_history(self):
        """Test amax history tracking."""
        from deepseek.model.quantization import (
            TileScalingConfig, TileScalingState, FP8Format
        )
        
        device = get_device()
        config = TileScalingConfig(tile_rows=4, tile_cols=4, amax_history_len=4)
        state = TileScalingState(config)
        
        # Multiple updates
        for _ in range(6):
            tensor = torch.randn(8, 8, device=device)
            state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        # History should be capped at 4
        assert len(state.amax_history) == 4
        
        # Smoothed amax should exist
        smoothed = state.get_smoothed_amax()
        assert smoothed is not None
        assert smoothed.shape == (2, 2)


class TestFP8QuantizeDequantize:
    """Tests for FP8 quantization and dequantization."""
    
    def test_quantize_fp8_tiled(self):
        """Test FP8 tiled quantization."""
        from deepseek.model.quantization import (
            quantize_fp8_tiled, TileScalingConfig, TileScalingState, FP8Format
        )
        
        device = get_device()
        tensor = torch.randn(16, 16, device=device) * 100
        
        config = TileScalingConfig(tile_rows=4, tile_cols=4)
        state = TileScalingState(config)
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        quantized = quantize_fp8_tiled(tensor, scales, FP8Format.E4M3, 4, 4)
        
        assert quantized.shape == tensor.shape
        assert quantized.abs().max() <= FP8Format.E4M3.max_value
    
    def test_dequantize_fp8_tiled(self):
        """Test FP8 tiled dequantization."""
        from deepseek.model.quantization import (
            quantize_fp8_tiled, dequantize_fp8_tiled,
            TileScalingConfig, TileScalingState, FP8Format
        )
        
        device = get_device()
        tensor = torch.randn(8, 8, device=device) * 10
        
        config = TileScalingConfig(tile_rows=4, tile_cols=4)
        state = TileScalingState(config)
        scales = state.compute_tile_scales(tensor, FP8Format.E4M3)
        
        quantized = quantize_fp8_tiled(tensor, scales, FP8Format.E4M3, 4, 4)
        dequantized = dequantize_fp8_tiled(quantized, scales, 4, 4)
        
        assert dequantized.shape == tensor.shape
        
        # Should be close (within quantization error)
        diff = (tensor - dequantized).abs().max()
        assert diff < 1.0, f"Quantization error too large: {diff}"
    
    def test_e5m2_for_gradients(self):
        """Test E5M2 format for gradients (larger range)."""
        from deepseek.model.quantization import (
            quantize_fp8_tiled, TileScalingConfig, TileScalingState, FP8Format
        )
        
        device = get_device()
        # Large gradient values
        tensor = torch.randn(8, 8, device=device) * 1000
        
        config = TileScalingConfig(tile_rows=4, tile_cols=4)
        state = TileScalingState(config)
        scales = state.compute_tile_scales(tensor, FP8Format.E5M2)
        
        quantized = quantize_fp8_tiled(tensor, scales, FP8Format.E5M2, 4, 4)
        
        assert quantized.abs().max() <= FP8Format.E5M2.max_value


class TestFP8OptimizerState:
    """Tests for FP8 optimizer state quantization."""
    
    def test_optimizer_state_creation(self):
        """Test FP8OptimizerState creation."""
        from deepseek.model.quantization import (
            FP8OptimizerState, FP8OptimizerConfig
        )
        
        device = get_device()
        config = FP8OptimizerConfig()
        
        m = torch.randn(32, 32, device=device) * 0.1
        v = torch.randn(32, 32, device=device).abs() * 0.01
        
        state = FP8OptimizerState(m, v, config)
        
        assert state.m_quantized.shape == m.shape
        assert state.v_quantized.shape == v.shape
        assert state.step == 0
    
    def test_optimizer_state_to_fp32(self):
        """Test dequantization of optimizer state."""
        from deepseek.model.quantization import (
            FP8OptimizerState, FP8OptimizerConfig
        )
        
        device = get_device()
        config = FP8OptimizerConfig()
        
        m = torch.randn(16, 16, device=device) * 0.1
        v = torch.randn(16, 16, device=device).abs() * 0.01
        
        state = FP8OptimizerState(m, v, config)
        m_dq, v_dq = state.to_fp32()
        
        # Should be close to original
        m_diff = (m - m_dq).abs().max()
        v_diff = (v - v_dq).abs().max()
        
        assert m_diff < 0.01, f"m quantization error: {m_diff}"
        assert v_diff < 0.001, f"v quantization error: {v_diff}"
    
    def test_optimizer_state_update(self):
        """Test Adam-style update with FP8 state."""
        from deepseek.model.quantization import (
            FP8OptimizerState, FP8OptimizerConfig
        )
        
        device = get_device()
        config = FP8OptimizerConfig()
        
        m = torch.zeros(16, 16, device=device)
        v = torch.zeros(16, 16, device=device)
        
        state = FP8OptimizerState(m, v, config)
        
        # Simulate gradient
        grad = torch.randn(16, 16, device=device) * 0.5
        
        m_new, v_new = state.update(grad, beta1=0.9, beta2=0.999)
        
        assert state.step == 1
        assert m_new.shape == grad.shape
        assert v_new.shape == grad.shape
        assert (v_new >= 0).all()  # v should be non-negative
    
    def test_bias_correction(self):
        """Test bias correction for early steps."""
        from deepseek.model.quantization import (
            FP8OptimizerState, FP8OptimizerConfig
        )
        
        device = get_device()
        config = FP8OptimizerConfig()
        
        m = torch.ones(4, 4, device=device)
        v = torch.ones(4, 4, device=device) * 0.1
        
        state = FP8OptimizerState(m, v, config)
        state.step = 10
        
        m_hat, v_hat = state.get_bias_corrected(0.9, 0.999)
        
        # Bias correction should increase magnitude for early steps
        m_dq, _ = state.to_fp32()
        assert m_hat.sum() > m_dq.sum()


class TestFP8MixedPrecisionManager:
    """Tests for FP8 mixed-precision training manager."""
    
    def test_manager_creation(self):
        """Test FP8MixedPrecisionManager creation."""
        from deepseek.model.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig
        )
        
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        assert manager.loss_scale == 65536.0
        assert manager.growth_tracker == 0
        assert len(manager.optimizer_states) == 0
    
    def test_loss_scaling(self):
        """Test loss scaling."""
        from deepseek.model.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig
        )
        
        device = get_device()
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        loss = torch.tensor(1.0, device=device)
        scaled_loss = manager.scale_loss(loss)
        
        assert scaled_loss.item() == 65536.0
    
    def test_loss_scale_update_overflow(self):
        """Test loss scale reduction on overflow."""
        from deepseek.model.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig
        )
        
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        initial_scale = manager.loss_scale
        manager.update_loss_scale(overflow=True)
        
        assert manager.loss_scale == initial_scale * 0.5
        assert manager.growth_tracker == 0
    
    def test_loss_scale_growth(self):
        """Test loss scale growth after successful steps."""
        from deepseek.model.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig
        )
        
        config = FP8TrainingConfig(growth_interval=100)
        manager = FP8MixedPrecisionManager(config)
        
        initial_scale = manager.loss_scale
        
        # Simulate many successful steps
        for _ in range(100):
            manager.update_loss_scale(overflow=False)
        
        assert manager.loss_scale == initial_scale * 2.0
    
    def test_quantize_activation(self):
        """Test activation quantization."""
        from deepseek.model.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig, TileScalingConfig
        )
        
        device = get_device()
        config = FP8TrainingConfig(
            tile_config=TileScalingConfig(tile_rows=4, tile_cols=4)
        )
        manager = FP8MixedPrecisionManager(config)
        
        tensor = torch.randn(8, 8, device=device) * 100
        quantized, scales = manager.quantize_activation("layer1", tensor)
        
        assert quantized.shape == tensor.shape
        assert scales.shape == (2, 2)
    
    def test_quantize_gradient(self):
        """Test gradient quantization (uses E5M2)."""
        from deepseek.model.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig, TileScalingConfig
        )
        
        device = get_device()
        config = FP8TrainingConfig(
            tile_config=TileScalingConfig(tile_rows=4, tile_cols=4)
        )
        manager = FP8MixedPrecisionManager(config)
        
        tensor = torch.randn(8, 8, device=device) * 1000
        quantized, scales = manager.quantize_gradient("layer1", tensor)
        
        assert quantized.shape == tensor.shape
    
    def test_init_optimizer_state(self):
        """Test optimizer state initialization."""
        from deepseek.model.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig
        )
        
        device = get_device()
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        param = torch.randn(64, 64, device=device)
        manager.init_optimizer_state("weight", param)
        
        assert "weight" in manager.optimizer_states
        state = manager.get_optimizer_state("weight")
        assert state is not None
    
    def test_memory_savings_report(self):
        """Test memory savings report."""
        from deepseek.model.quantization import (
            FP8MixedPrecisionManager, FP8TrainingConfig
        )
        
        device = get_device()
        config = FP8TrainingConfig()
        manager = FP8MixedPrecisionManager(config)
        
        # Initialize several parameters
        for i in range(3):
            param = torch.randn(128, 128, device=device)
            manager.init_optimizer_state(f"param_{i}", param)
        
        report = manager.memory_savings_report()
        
        assert report["optimizer_states_count"] == 3
        assert report["savings_ratio"] > 0


# ============================================================================
# Phase 5: Agent/Tool-Use Training Tests
# ============================================================================

class TestToolCallFormat:
    """Tests for tool call JSON format specification."""
    
    def test_tool_call_creation(self):
        """Test ToolCall creation and serialization."""
        from deepseek.training.agent import ToolCall, ToolType
        
        call = ToolCall(
            id="call_001",
            tool_type=ToolType.CODE_EXECUTION,
            function_name="run_python",
            arguments={"code": "print('hello')"}
        )
        
        assert call.id == "call_001"
        assert call.tool_type == ToolType.CODE_EXECUTION
        assert call.function_name == "run_python"
        
        # Test JSON serialization
        json_str = call.to_json()
        assert "call_001" in json_str
        assert "CODE_EXECUTION" in json_str
        assert "run_python" in json_str
    
    def test_tool_call_from_json(self):
        """Test ToolCall parsing from JSON."""
        from deepseek.training.agent import ToolCall, ToolType
        
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
        assert call.function_name == "search"
    
    def test_tool_response_creation(self):
        """Test ToolResponse creation."""
        from deepseek.training.agent import ToolResponse, ToolStatus
        
        response = ToolResponse(
            call_id="call_001",
            status=ToolStatus.SUCCESS,
            content="Execution complete",
            execution_time_ms=100
        )
        
        assert response.call_id == "call_001"
        assert response.status == ToolStatus.SUCCESS
        assert response.execution_time_ms == 100


class TestAgentTrajectory:
    """Tests for agent trajectory management."""
    
    def test_trajectory_creation(self):
        """Test AgentTrajectory creation."""
        from deepseek.training.agent import (
            AgentTrajectory, AgentStep, AgentAction, AgentActionType, TaskTier
        )
        
        trajectory = AgentTrajectory(
            prompt="Calculate 2+2",
            task_tier=TaskTier.SINGLE_TOOL
        )
        
        assert trajectory.prompt == "Calculate 2+2"
        assert trajectory.task_tier == TaskTier.SINGLE_TOOL
        assert len(trajectory.steps) == 0
    
    def test_trajectory_with_steps(self):
        """Test trajectory with multiple steps."""
        from deepseek.training.agent import (
            AgentTrajectory, AgentStep, AgentAction, AgentActionType,
            ToolCall, ToolResponse, ToolType, ToolStatus, TaskTier
        )
        
        trajectory = AgentTrajectory(
            prompt="Calculate 2+2",
            task_tier=TaskTier.SINGLE_TOOL
        )
        
        # Add thinking step
        think_step = AgentStep(
            action=AgentAction(
                action_type=AgentActionType.THINK,
                content="I need to calculate 2+2"
            )
        )
        trajectory.steps.append(think_step)
        
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
        
        # Add response step
        response_step = AgentStep(
            action=AgentAction(
                action_type=AgentActionType.RESPOND,
                content="The answer is 4"
            )
        )
        trajectory.steps.append(response_step)
        trajectory.final_output = "The answer is 4"
        trajectory.task_completed = True
        
        assert trajectory.num_tool_calls == 1
        assert trajectory.successful_tool_calls == 1
        assert trajectory.task_completed


class TestTaskTiers:
    """Tests for task tier specification."""
    
    def test_tier_expected_calls(self):
        """Test tier expected tool call ranges."""
        from deepseek.training.agent import TaskTier
        
        # Tier 1: Single tool
        min_calls, max_calls = TaskTier.SINGLE_TOOL.expected_tool_calls()
        assert min_calls == 1
        assert max_calls == 1
        
        # Tier 2: Multi-tool sequential
        min_calls, max_calls = TaskTier.MULTI_TOOL_SEQ.expected_tool_calls()
        assert min_calls == 2
        assert max_calls == 5
        
        # Tier 3: Multi-tool parallel
        min_calls, max_calls = TaskTier.MULTI_TOOL_PARALLEL.expected_tool_calls()
        assert min_calls == 3
        assert max_calls == 8
        
        # Tier 4: Complex workflow
        min_calls, max_calls = TaskTier.COMPLEX_WORKFLOW.expected_tool_calls()
        assert min_calls == 10
        assert max_calls == 50
    
    def test_tier_environment_counts(self):
        """Test tier environment counts (~1,800 total)."""
        from deepseek.training.agent import TaskTier
        
        total_envs = sum(tier.environment_count() for tier in TaskTier)
        assert total_envs == 1800
        
        assert TaskTier.SINGLE_TOOL.environment_count() == 600
        assert TaskTier.MULTI_TOOL_SEQ.environment_count() == 500
        assert TaskTier.MULTI_TOOL_PARALLEL.environment_count() == 400
        assert TaskTier.COMPLEX_WORKFLOW.environment_count() == 300


class TestRewardComputation:
    """Tests for multi-objective reward computation."""
    
    def test_reward_weights(self):
        """Test reward weight validation."""
        from deepseek.training.agent import RewardWeights
        
        # Default weights should sum to 1.0
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
        from deepseek.training.agent import (
            AgentRewardComputer, AgentTrajectory, AgentStep, AgentAction,
            AgentActionType, TaskTier
        )
        
        computer = AgentRewardComputer()
        
        # Create a completed trajectory
        trajectory = AgentTrajectory(
            prompt="Calculate 2+2",
            task_tier=TaskTier.SINGLE_TOOL
        )
        trajectory.final_output = "4"
        trajectory.task_completed = True
        
        breakdown = computer.compute_reward(trajectory, ground_truth="4")
        
        # Perfect match should give high correctness
        assert breakdown.correctness == 1.0
        # No tool calls but task completed - efficiency should be ok
        assert breakdown.format == 1.0  # No tool calls to validate
        assert breakdown.safety == 1.0  # No dangerous ops
        assert breakdown.total > 0.8
    
    def test_efficiency_penalty(self):
        """Test efficiency penalty for excess tool calls."""
        from deepseek.training.agent import (
            AgentRewardComputer, AgentTrajectory, AgentStep, AgentAction,
            AgentActionType, ToolCall, ToolResponse, ToolType, ToolStatus, TaskTier
        )
        
        computer = AgentRewardComputer()
        
        # Create trajectory with too many tool calls for Tier 1
        trajectory = AgentTrajectory(
            prompt="Simple task",
            task_tier=TaskTier.SINGLE_TOOL
        )
        
        # Add 5 tool calls (way too many for Tier 1)
        for i in range(5):
            tool_call = ToolCall(
                id=f"call_{i}",
                tool_type=ToolType.REASONING,
                function_name="calculate",
                arguments={}
            )
            tool_response = ToolResponse(
                call_id=f"call_{i}",
                status=ToolStatus.SUCCESS,
                content="Result",
                execution_time_ms=10
            )
            step = AgentStep(
                action=AgentAction(
                    action_type=AgentActionType.TOOL_CALL,
                    content="Call",
                    tool_call=tool_call,
                    tool_response=tool_response
                )
            )
            trajectory.steps.append(step)
        
        trajectory.task_completed = True
        
        breakdown = computer.compute_reward(trajectory)
        
        # Efficiency should be penalized
        assert breakdown.efficiency < 1.0
    
    def test_safety_detection(self):
        """Test safety check for dangerous operations."""
        from deepseek.training.agent import (
            AgentRewardComputer, AgentTrajectory, AgentStep, AgentAction,
            AgentActionType, ToolCall, ToolResponse, ToolType, ToolStatus, TaskTier
        )
        
        computer = AgentRewardComputer()
        
        trajectory = AgentTrajectory(
            prompt="Delete files",
            task_tier=TaskTier.SINGLE_TOOL
        )
        
        # Add dangerous tool call
        tool_call = ToolCall(
            id="call_danger",
            tool_type=ToolType.CODE_EXECUTION,
            function_name="execute",
            arguments={"command": "rm -rf /"}
        )
        tool_response = ToolResponse(
            call_id="call_danger",
            status=ToolStatus.SUCCESS,
            content="Deleted",
            execution_time_ms=100
        )
        step = AgentStep(
            action=AgentAction(
                action_type=AgentActionType.TOOL_CALL,
                content="Executing dangerous command",
                tool_call=tool_call,
                tool_response=tool_response
            )
        )
        trajectory.steps.append(step)
        trajectory.task_completed = True
        
        breakdown = computer.compute_reward(trajectory)
        
        # Safety should be zero for dangerous ops
        assert breakdown.safety == 0.0


class TestTaskGenerator:
    """Tests for synthetic task generation."""
    
    def test_generator_creation(self):
        """Test TaskGenerator creation."""
        from deepseek.training.agent import TaskGenerator
        
        generator = TaskGenerator(seed=42)
        assert generator is not None
    
    def test_generate_single_task(self):
        """Test single task generation."""
        from deepseek.training.agent import TaskGenerator, TaskTier
        
        generator = TaskGenerator(seed=42)
        
        prompt, tier = generator.generate_task(TaskTier.SINGLE_TOOL)
        
        assert len(prompt) > 0
        assert tier == TaskTier.SINGLE_TOOL
    
    def test_generate_batch(self):
        """Test batch task generation."""
        from deepseek.training.agent import TaskGenerator, TaskTier
        
        generator = TaskGenerator(seed=42)
        
        batch = generator.generate_batch(10, TaskTier.MULTI_TOOL_SEQ)
        
        assert len(batch) == 10
        for prompt, tier in batch:
            assert tier == TaskTier.MULTI_TOOL_SEQ
            assert len(prompt) > 0
    
    def test_tier_distribution(self):
        """Test tier distribution matches spec."""
        from deepseek.training.agent import TaskGenerator, TaskTier
        
        generator = TaskGenerator()
        dist = generator.get_tier_distribution()
        
        assert dist[TaskTier.SINGLE_TOOL] == 600
        assert dist[TaskTier.MULTI_TOOL_SEQ] == 500
        assert dist[TaskTier.MULTI_TOOL_PARALLEL] == 400
        assert dist[TaskTier.COMPLEX_WORKFLOW] == 300
        assert sum(dist.values()) == 1800


class TestToolCallParser:
    """Tests for tool call parsing from model output."""
    
    def test_parse_json_code_block(self):
        """Test parsing tool call from JSON code block."""
        from deepseek.training.agent import ToolCallParser, ToolType
        
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
        assert calls[0].id == "call_123"
        assert calls[0].tool_type == ToolType.REASONING
        assert calls[0].function_name == "calculate"
    
    def test_parse_xml_format(self):
        """Test parsing tool call from XML-style tags."""
        from deepseek.training.agent import ToolCallParser, ToolType
        
        text = '''
        Let me search for that.
        
        <tool_call>
        {"function_name": "search", "tool_type": "WEB_SEARCH", "arguments": {"query": "test"}}
        </tool_call>
        '''
        
        calls = ToolCallParser.parse(text)
        
        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.WEB_SEARCH
    
    def test_parse_multiple_calls(self):
        """Test parsing multiple tool calls."""
        from deepseek.training.agent import ToolCallParser
        
        text = '''
        First search:
        ```json
        {"function_name": "search", "tool_type": "WEB_SEARCH", "arguments": {"query": "a"}}
        ```
        
        Second search:
        ```json
        {"function_name": "search", "tool_type": "WEB_SEARCH", "arguments": {"query": "b"}}
        ```
        '''
        
        calls = ToolCallParser.parse(text)
        
        assert len(calls) == 2


class TestAgentGRPOTrainer:
    """Tests for Agent GRPO trainer with multi-turn credit assignment."""
    
    def test_trainer_creation(self):
        """Test AgentGRPOTrainer creation."""
        from deepseek.training.agent import AgentGRPOTrainer, AgentGRPOConfig
        
        config = AgentGRPOConfig(
            beta=0.04,
            gamma=0.99,
            group_size=8,
        )
        trainer = AgentGRPOTrainer(config)
        
        assert trainer.beta == 0.04
        assert trainer.gamma == 0.99
    
    def test_compute_loss(self):
        """Test GRPO loss computation."""
        from deepseek.training.agent import AgentGRPOTrainer, AgentGRPOConfig
        
        device = get_device()
        
        config = AgentGRPOConfig(beta=0.04, gamma=0.99)
        trainer = AgentGRPOTrainer(config)
        
        G, Seq, Vocab = 4, 32, 100
        
        logits = torch.randn(G, Seq, Vocab, device=device)
        input_ids = torch.randint(0, Vocab, (G, Seq), device=device)
        rewards = torch.randn(G, device=device)
        ref_logits = torch.randn(G, Seq, Vocab, device=device)
        
        loss = trainer.compute_loss(logits, input_ids, rewards, ref_logits)
        
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
    
    def test_turn_credit_assignment(self):
        """Test multi-turn credit assignment."""
        from deepseek.training.agent import AgentGRPOTrainer, AgentGRPOConfig
        
        device = get_device()
        
        config = AgentGRPOConfig(
            beta=0.04,
            gamma=0.99,
            turn_credit_method="exponential"
        )
        trainer = AgentGRPOTrainer(config)
        
        G, Seq, Vocab = 2, 64, 100
        
        logits = torch.randn(G, Seq, Vocab, device=device)
        input_ids = torch.randint(0, Vocab, (G, Seq), device=device)
        rewards = torch.randn(G, device=device)
        ref_logits = torch.randn(G, Seq, Vocab, device=device)
        
        # Define turn boundaries
        turn_boundaries = [
            [(0, 20), (20, 40), (40, 64)],  # 3 turns for sample 1
            [(0, 32), (32, 64)],             # 2 turns for sample 2
        ]
        
        loss = trainer.compute_loss(
            logits, input_ids, rewards, ref_logits,
            turn_boundaries=turn_boundaries
        )
        
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_curriculum_update(self):
        """Test curriculum tier progression."""
        from deepseek.training.agent import AgentGRPOTrainer, AgentGRPOConfig, TaskTier
        
        config = AgentGRPOConfig(
            use_curriculum=True
        )
        # Override tier thresholds for testing
        trainer = AgentGRPOTrainer(config)
        trainer.tier_thresholds = {
            TaskTier.SINGLE_TOOL: 10,
            TaskTier.MULTI_TOOL_SEQ: 20,
            TaskTier.MULTI_TOOL_PARALLEL: 30,
            TaskTier.COMPLEX_WORKFLOW: float('inf'),
        }
        
        assert trainer.get_current_tier() == TaskTier.SINGLE_TOOL
        
        # Simulate training steps
        for _ in range(15):
            trainer.update_curriculum()
        
        # Should have advanced to Tier 2
        assert trainer.get_current_tier() == TaskTier.MULTI_TOOL_SEQ
    
    def test_training_progress(self):
        """Test training progress reporting."""
        from deepseek.training.agent import AgentGRPOTrainer, AgentGRPOConfig
        
        config = AgentGRPOConfig(use_curriculum=True)
        trainer = AgentGRPOTrainer(config)
        
        progress = trainer.get_training_progress()
        
        assert "current_tier" in progress
        assert "steps_completed" in progress
        assert "tier_progress" in progress


class TestAgentEnvironment:
    """Tests for agent environment simulation."""
    
    def test_environment_creation(self):
        """Test AgentEnvironment creation."""
        from deepseek.training.agent import AgentEnvironment
        
        env = AgentEnvironment(max_steps=50)
        assert env.max_steps == 50
    
    def test_execute_tool_call(self):
        """Test tool call execution."""
        from deepseek.training.agent import (
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
        assert "4" in response.content


class TestAgentGroupSampler:
    """Tests for agent group sampler."""
    
    def test_sampler_creation(self):
        """Test AgentGroupSampler creation."""
        from deepseek.training.agent import AgentGroupSampler
        
        sampler = AgentGroupSampler(group_size=8)
        assert sampler.group_size == 8
    
    def test_sample_group(self):
        """Test sampling a group of trajectories."""
        from deepseek.training.agent import AgentGroupSampler, TaskTier
        
        sampler = AgentGroupSampler(group_size=4)
        
        # Mock model function that returns a final response
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

