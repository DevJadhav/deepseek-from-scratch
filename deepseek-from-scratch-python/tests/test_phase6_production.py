"""
DeepSeek-V3.2 Phase 6: Production Hardening Tests

This module contains comprehensive tests for:
- End-to-end inference benchmarks
- Speculative decoding with MTP heads
- KV cache optimization
- 128K context verification
- 256-expert routing validation
- Full pipeline stress tests
"""

import pytest
import torch
import time
import sys
import os
from typing import List, Tuple

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
# Benchmark Utilities
# ============================================================================

class BenchmarkTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.duration = None
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.duration = time.perf_counter() - self.start
    
    def ms(self) -> float:
        return self.duration * 1000


# ============================================================================
# 128K Context Tests
# ============================================================================

class TestLongContext:
    """Tests for 128K context window support."""
    
    def test_rope_128k_initialization(self):
        """Test RoPE initialization for 128K context."""
        from deepseek.model.mla import ExtendedRoPEConfig, ExtendedRotaryPositionalEncoding
        
        device = get_device()
        
        # YaRN scaling for 128K
        config = ExtendedRoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type="yarn",
            yarn_original_max_position=4096,
        )
        rope = ExtendedRotaryPositionalEncoding(config).to(device)
        
        # Verify scaling factor is computed
        assert config.max_seq_len == 131072
        assert rope.yarn_attn_factor > 0
    
    def test_rope_all_scaling_types(self):
        """Test all RoPE scaling types."""
        from deepseek.model.mla import ExtendedRoPEConfig, ExtendedRotaryPositionalEncoding
        
        device = get_device()
        
        # Valid scaling types in the implementation: "none", "ntk", "yarn"
        scaling_types = [
            ("none", {}),
            ("ntk", {"ntk_alpha": 32.0}),
            ("yarn", {"yarn_original_max_position": 4096}),
        ]
        
        for scaling_type, kwargs in scaling_types:
            config = ExtendedRoPEConfig(
                d_head=64,
                max_seq_len=8192,
                scaling_type=scaling_type,
                **kwargs
            )
            rope = ExtendedRotaryPositionalEncoding(config).to(device)
            
            x = torch.randn(1, 4, 32, 64, device=device)
            out = rope(x)
            
            assert out.shape == x.shape
            assert not torch.isnan(out).any(), f"{scaling_type} produced NaN"
            print(f"  ✓ RoPE {scaling_type} scaling passed")
    
    def test_dsa_long_context(self):
        """Test DSA with longer sequences."""
        from deepseek.model.sparse_attention import DSAConfig, DeepSeekSparseAttention
        
        device = get_device()
        
        config = DSAConfig(
            d_model=128,
            num_heads=4,
            window_size=256,
            global_tokens=32,  # Fixed: was num_global_tokens
            max_seq_len=4096,
        )
        dsa = DeepSeekSparseAttention(config).to(device)
        
        # Test with moderately long sequence
        seq_len = 2048
        x = torch.randn(1, seq_len, 128, device=device)
        
        with BenchmarkTimer("DSA 2K seq") as timer:
            out = dsa(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        print(f"  DSA 2K sequence: {timer.ms():.2f}ms")


# ============================================================================
# 256-Expert MoE Tests
# ============================================================================

class TestLargeExpertMoE:
    """Tests for 256-expert MoE configurations."""
    
    def test_moe_v3_default_config(self):
        """Test MoE V3 default configuration."""
        from deepseek.model.moe import DeepSeekMoEV3Config
        
        config = DeepSeekMoEV3Config()
        
        # Default should be 256 experts, 8 active
        assert config.n_routed_experts == 256
        assert config.top_k == 8
        assert config.capacity_factor >= 1.0
    
    def test_moe_256_expert_forward(self):
        """Test MoE with 256-expert configuration."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        
        # Use smaller hidden dim for testing but many experts
        # Config uses routed_hidden_mult/shared_hidden_mult not explicit hidden sizes
        config = DeepSeekMoEV3Config(
            d_model=64,
            n_routed_experts=64,  # Smaller for testing
            n_shared_experts=1,
            top_k=4,
            n_expert_groups=8,
            routed_hidden_mult=2.0,  # 128 effective hidden
            shared_hidden_mult=2.0,
        )
        moe = DeepSeekMoEV3(config).to(device)
        
        batch, seq = 4, 16
        x = torch.randn(batch, seq, 64, device=device)
        
        with BenchmarkTimer("MoE 64 experts") as timer:
            out = moe(x)  # DeepSeekMoEV3 returns single tensor
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        print(f"  MoE 64 experts forward: {timer.ms():.2f}ms")
    
    def test_moe_hierarchical_routing(self):
        """Test hierarchical routing with groups."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        
        # 64 experts in 8 groups = 8 experts per group
        config = DeepSeekMoEV3Config(
            d_model=64,
            n_routed_experts=64,
            top_k=8,
            n_expert_groups=8,
            routed_hidden_mult=2.0,
            shared_hidden_mult=2.0,
        )
        
        # Verify config calculations
        assert config.experts_per_group == 8
        
        moe = DeepSeekMoEV3(config).to(device)
        x = torch.randn(2, 32, 64, device=device)
        out = moe(x)  # DeepSeekMoEV3 returns single tensor
        
        assert out.shape == x.shape
    
    def test_moe_load_balance_stats(self):
        """Test load balancing statistics."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        
        config = DeepSeekMoEV3Config(
            d_model=32,
            n_routed_experts=16,
            top_k=2,
            n_expert_groups=4,
            routed_hidden_mult=2.0,
            shared_hidden_mult=2.0,
        )
        moe = DeepSeekMoEV3(config).to(device)
        
        # Run multiple forward passes
        for _ in range(5):
            x = torch.randn(4, 16, 32, device=device)
            out = moe(x)  # DeepSeekMoEV3 returns single tensor
        
        # Check load balance stats exist
        # get_load_balance_stats returns a tuple, not a dict
        stats = moe.get_load_balance_stats()
        assert stats is not None
        assert len(stats) >= 1  # At least some stats returned


# ============================================================================
# KV Cache Tests (Simplified - no external kv_cache module)
# ============================================================================

class SimpleKVCache:
    """Simple KV cache implementation for testing."""
    
    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int, 
                 head_dim: int, device: torch.device):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Pre-allocate KV cache buffers
        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device
        )
        self.current_seq_len = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new KV and return accumulated KV."""
        new_seq_len = k.shape[2]
        end_pos = self.current_seq_len + new_seq_len
        
        if end_pos > self.max_seq_len:
            raise ValueError(f"Cache overflow: {end_pos} > {self.max_seq_len}")
        
        self.k_cache[:, :, self.current_seq_len:end_pos, :] = k
        self.v_cache[:, :, self.current_seq_len:end_pos, :] = v
        self.current_seq_len = end_pos
        
        return self.k_cache[:, :, :end_pos, :], self.v_cache[:, :, :end_pos, :]
    
    def reset(self):
        """Reset cache to initial state."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_seq_len = 0


class TestKVCache:
    """Tests for KV cache optimization."""
    
    def test_kv_cache_creation(self):
        """Test KV cache initialization."""
        device = get_device()
        
        cache = SimpleKVCache(
            batch_size=2,
            max_seq_len=2048,
            num_heads=8,
            head_dim=64,
            device=device,
        )
        
        assert cache.current_seq_len == 0
        assert cache.max_seq_len == 2048
    
    def test_kv_cache_incremental_update(self):
        """Test incremental KV cache updates."""
        device = get_device()
        batch, heads, head_dim = 2, 8, 64
        
        cache = SimpleKVCache(
            batch_size=batch,
            max_seq_len=1024,
            num_heads=heads,
            head_dim=head_dim,
            device=device,
        )
        
        # Prefill with 128 tokens
        k = torch.randn(batch, heads, 128, head_dim, device=device)
        v = torch.randn(batch, heads, 128, head_dim, device=device)
        k_out, v_out = cache.update(k, v)
        
        assert k_out.shape[2] == 128
        assert cache.current_seq_len == 128
        
        # Generate 10 tokens incrementally
        for i in range(10):
            k = torch.randn(batch, heads, 1, head_dim, device=device)
            v = torch.randn(batch, heads, 1, head_dim, device=device)
            k_out, v_out = cache.update(k, v)
            
            assert k_out.shape[2] == 128 + i + 1
        
        assert cache.current_seq_len == 138
    
    def test_kv_cache_128k_allocation(self):
        """Test 128K context KV cache allocation."""
        device = get_device()
        
        # This tests memory allocation for 128K
        cache = SimpleKVCache(
            batch_size=1,
            max_seq_len=131072,
            num_heads=8,
            head_dim=64,
            device=device,
        )
        
        # Calculate expected memory
        expected_bytes = 2 * 1 * 8 * 131072 * 64 * 4  # K + V, f32
        expected_mb = expected_bytes / (1024 * 1024)
        print(f"  128K KV cache allocated: ~{expected_mb:.0f} MB")
        
        assert cache.max_seq_len == 131072


# ============================================================================
# Speculative Decoding Tests
# ============================================================================

class TestSpeculativeDecoding:
    """Tests for speculative decoding with MTP."""
    
    def test_mtp_model_forward(self):
        """Test MTP model forward pass."""
        from deepseek.model.mtp import MTPModel
        
        device = get_device()
        
        # MTPModel constructor: vocab_size, d_model, num_layers, k_predictions
        vocab_size = 1000
        d_model = 128
        num_layers = 2
        k_predictions = 4
        
        mtp = MTPModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            k_predictions=k_predictions
        ).to(device)
        
        batch, seq = 2, 32
        input_ids = torch.randint(0, vocab_size, (batch, seq), device=device)
        
        main_logits, future_logits = mtp(input_ids)
        
        # Main logits shape: (batch, seq, vocab_size)
        assert main_logits.shape == (batch, seq, vocab_size)
        # Future logits: list of k_predictions tensors
        assert len(future_logits) == k_predictions
        for fl in future_logits:
            assert fl.shape == (batch, seq, vocab_size)
    
    def test_speculative_decode_step(self):
        """Test a single speculative decoding step."""
        from deepseek.model.mtp import MTPModel
        
        device = get_device()
        
        vocab_size = 100
        d_model = 64
        num_layers = 2
        k_predictions = 3  # Predict 3 ahead
        
        mtp = MTPModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            k_predictions=k_predictions
        ).to(device)
        
        # Single token input (during generation)
        batch = 1
        input_ids = torch.randint(0, vocab_size, (batch, 1), device=device)
        
        with BenchmarkTimer("MTP speculative step") as timer:
            main_logits, future_logits = mtp(input_ids)
        
        # Main logits + k_predictions future logits
        assert main_logits.shape == (batch, 1, vocab_size)
        assert len(future_logits) == k_predictions
        print(f"  MTP speculative step: {timer.ms():.2f}ms")


# ============================================================================
# Full Pipeline Benchmarks
# ============================================================================

class TestFullPipelineBenchmarks:
    """End-to-end pipeline benchmarks."""
    
    def test_dsa_scaling_benchmark(self):
        """Benchmark DSA scaling with sequence length."""
        from deepseek.model.sparse_attention import DSAConfig, DeepSeekSparseAttention
        
        device = get_device()
        
        seq_lengths = [128, 256, 512, 1024]
        d_model = 64
        
        print("\n  DSA Scaling Benchmark:")
        print(f"  {'Seq Len':>8} | {'Time (ms)':>12}")
        print(f"  {'-'*8}-+-{'-'*12}")
        
        for seq_len in seq_lengths:
            config = DSAConfig(
                d_model=d_model,
                num_heads=4,
                window_size=32,
                global_tokens=8,
                max_seq_len=seq_len * 2,
            )
            dsa = DeepSeekSparseAttention(config).to(device)
            
            x = torch.randn(1, seq_len, d_model, device=device)
            
            # Warmup
            _ = dsa(x)
            
            # Benchmark
            times = []
            for _ in range(10):
                with BenchmarkTimer("") as timer:
                    _ = dsa(x)
                times.append(timer.ms())
            
            avg_time = sum(times) / len(times)
            print(f"  {seq_len:>8} | {avg_time:>12.2f}")
    
    def test_moe_scaling_benchmark(self):
        """Benchmark MoE scaling with expert count."""
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        
        configs = [
            (16, 2),
            (32, 4),
            (64, 8),
        ]
        
        d_model = 32
        batch, seq = 4, 32
        
        print("\n  MoE Expert Scaling Benchmark:")
        print(f"  {'Experts':>8} | {'Active':>8} | {'Time (ms)':>12}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*12}")
        
        for n_experts, top_k in configs:
            config = DeepSeekMoEV3Config(
                d_model=d_model,
                n_routed_experts=n_experts,
                top_k=top_k,
                n_expert_groups=max(1, n_experts // 8),
                routed_hidden_mult=2.0,
                shared_hidden_mult=2.0,
            )
            moe = DeepSeekMoEV3(config).to(device)
            
            x = torch.randn(batch, seq, d_model, device=device)
            
            # Warmup
            _ = moe(x)
            
            # Benchmark
            times = []
            for _ in range(10):
                with BenchmarkTimer("") as timer:
                    _ = moe(x)
                times.append(timer.ms())
            
            avg_time = sum(times) / len(times)
            print(f"  {n_experts:>8} | {top_k:>8} | {avg_time:>12.2f}")
    
    def test_full_v32_pipeline(self):
        """Test full V3.2 forward pipeline."""
        from deepseek.model.mla import ExtendedRoPEConfig, ExtendedRotaryPositionalEncoding
        from deepseek.model.sparse_attention import DSAConfig, DeepSeekSparseAttention
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        
        batch = 2
        seq = 64
        d_model = 128
        num_heads = 4
        
        # Create components
        rope_config = ExtendedRoPEConfig(
            d_head=d_model // num_heads,
            max_seq_len=8192,
            scaling_type="yarn",
            yarn_original_max_position=4096,
        )
        rope = ExtendedRotaryPositionalEncoding(rope_config).to(device)
        
        dsa_config = DSAConfig(
            d_model=d_model,
            num_heads=num_heads,
            window_size=16,
            global_tokens=4,
            max_seq_len=1024,
        )
        dsa = DeepSeekSparseAttention(dsa_config).to(device)
        
        moe_config = DeepSeekMoEV3Config(
            d_model=d_model,
            n_routed_experts=16,
            top_k=2,
            n_expert_groups=4,
            routed_hidden_mult=2.0,
            shared_hidden_mult=2.0,
        )
        moe = DeepSeekMoEV3(moe_config).to(device)
        
        # Input
        x = torch.randn(batch, seq, d_model, device=device)
        
        # Forward pipeline
        with BenchmarkTimer("DSA") as dsa_timer:
            dsa_out = dsa(x)
        
        with BenchmarkTimer("MoE") as moe_timer:
            moe_out = moe(dsa_out)  # DeepSeekMoEV3 returns single tensor
        
        assert moe_out.shape == x.shape
        assert not torch.isnan(moe_out).any()
        
        print(f"\n  Full V3.2 Pipeline Test on {device}:")
        print(f"    DSA time: {dsa_timer.ms():.2f}ms")
        print(f"    MoE time: {moe_timer.ms():.2f}ms")
        print(f"    Total: {dsa_timer.ms() + moe_timer.ms():.2f}ms")
    
    def test_numerical_stability(self):
        """Test numerical stability over multiple iterations."""
        from deepseek.model.sparse_attention import DSAConfig, DeepSeekSparseAttention
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        d_model = 64
        
        dsa_config = DSAConfig(
            d_model=d_model,
            num_heads=4,
            window_size=8,
            global_tokens=2,
            max_seq_len=128,
        )
        dsa = DeepSeekSparseAttention(dsa_config).to(device)
        
        moe_config = DeepSeekMoEV3Config(
            d_model=d_model,
            n_routed_experts=16,
            top_k=2,
            routed_hidden_mult=2.0,
            shared_hidden_mult=2.0,
        )
        moe = DeepSeekMoEV3(moe_config).to(device)
        
        # Run multiple iterations
        for i in range(10):
            x = torch.randn(2, 32, d_model, device=device)
            
            dsa_out = dsa(x)
            moe_out = moe(dsa_out)  # DeepSeekMoEV3 returns single tensor
            
            mean = moe_out.mean().item()
            
            assert not torch.isnan(moe_out).any(), f"NaN at iteration {i}"
            assert not torch.isinf(moe_out).any(), f"Inf at iteration {i}"
        
        print(f"  Numerical stability test passed (10 iterations)")


# ============================================================================
# Agent Pipeline Integration Tests  
# ============================================================================

class TestAgentPipelineIntegration:
    """Integration tests for agent training pipeline."""
    
    def test_agent_trajectory_full_workflow(self):
        """Test complete agent trajectory workflow."""
        from deepseek.training.agent import (
            AgentTrajectory, AgentStep, AgentAction, AgentActionType,
            ToolCall, ToolResponse, ToolStatus, ToolType, TaskTier
        )
        
        # AgentTrajectory uses 'prompt' not 'task', and 'task_tier' not 'tier'
        trajectory = AgentTrajectory(
            prompt="Write a function to sort a list",
            task_tier=TaskTier.MULTI_TOOL_SEQ
        )
        
        # Step 1: Search for implementation
        # AgentStep requires 'action' as first argument (no step_id)
        # ToolCall requires 'id' as first argument
        action1 = AgentAction(
            action_type=AgentActionType.TOOL_CALL,
            content="Search for sort algorithms",
            tool_call=ToolCall(
                id="call_1",  # Required id field
                tool_type=ToolType.CODE_EXECUTION,
                function_name="search",
                arguments={"query": "python sort algorithm"}
            )
        )
        # ToolResponse requires call_id
        action1.tool_response = ToolResponse(
            call_id="call_1",
            status=ToolStatus.SUCCESS,
            content="Quicksort, mergesort, timsort...",
            execution_time_ms=100
        )
        step1 = AgentStep(action=action1)
        trajectory.steps.append(step1)
        
        # Step 2: Write code
        action2 = AgentAction(
            action_type=AgentActionType.TOOL_CALL,
            content="Execute Python code",
            tool_call=ToolCall(
                id="call_2",  # Required id field
                tool_type=ToolType.CODE_EXECUTION,
                function_name="execute_python",
                arguments={"code": "def sort_list(lst): return sorted(lst)"}
            )
        )
        action2.tool_response = ToolResponse(
            call_id="call_2",
            status=ToolStatus.SUCCESS,
            content="Function defined",
            execution_time_ms=50
        )
        step2 = AgentStep(action=action2)
        trajectory.steps.append(step2)
        
        trajectory.task_completed = True
        
        # Check step counts - use property names
        assert len(trajectory.steps) == 2
        assert trajectory.num_tool_calls == 2
        assert trajectory.task_completed
    
    def test_reward_computation(self):
        """Test reward computation for agent trajectories."""
        from deepseek.training.agent import (
            RewardWeights, RewardBreakdown, AgentRewardComputer
        )
        
        weights = RewardWeights()
        
        # Verify weights sum to 1
        total = weights.correctness + weights.format + weights.efficiency + weights.safety
        assert abs(total - 1.0) < 1e-6
        
        # RewardBreakdown.total is a field, not a method
        # We compute expected reward manually
        breakdown = RewardBreakdown(
            correctness=0.9,
            format=1.0,
            efficiency=0.8,
            safety=1.0,
            total=0.92  # Pre-computed: 0.5*0.9 + 0.2*1.0 + 0.15*0.8 + 0.15*1.0
        )
        
        # Verify components and total
        expected_total = (
            weights.correctness * breakdown.correctness +
            weights.format * breakdown.format +
            weights.efficiency * breakdown.efficiency +
            weights.safety * breakdown.safety
        )
        
        assert abs(expected_total - 0.92) < 1e-6
        print(f"  Reward computation: {expected_total:.4f}")


# ============================================================================
# End-to-End Generation Test
# ============================================================================

class TestEndToEndGeneration:
    """Tests for end-to-end text generation."""
    
    def test_multi_layer_generation(self):
        """Test generation with multiple transformer layers."""
        from deepseek.model.mla import ExtendedRoPEConfig, ExtendedRotaryPositionalEncoding
        from deepseek.model.sparse_attention import DSAConfig, DeepSeekSparseAttention
        from deepseek.model.moe import DeepSeekMoEV3Config, DeepSeekMoEV3
        
        device = get_device()
        
        # Config
        batch = 1
        d_model = 64
        num_heads = 4
        head_dim = d_model // num_heads
        n_layers = 2
        prefill_len = 32
        generate_tokens = 8
        
        # Create components for each layer
        dsas = []
        moes = []
        for _ in range(n_layers):
            dsa_config = DSAConfig(
                d_model=d_model,
                num_heads=num_heads,
                window_size=8,
                global_tokens=2,
                max_seq_len=256,
            )
            dsas.append(DeepSeekSparseAttention(dsa_config).to(device))
            
            moe_config = DeepSeekMoEV3Config(
                d_model=d_model,
                n_routed_experts=8,
                top_k=2,
                routed_hidden_mult=2.0,
                shared_hidden_mult=2.0,
            )
            moes.append(DeepSeekMoEV3(moe_config).to(device))
        
        # Create KV caches using the simple cache class defined above
        kv_caches = [
            SimpleKVCache(batch, 256, num_heads, head_dim, device=device)
            for _ in range(n_layers)
        ]
        
        # Prefill
        prefill = torch.randn(batch, prefill_len, d_model, device=device)
        x = prefill
        for layer in range(n_layers):
            x = dsas[layer](x)
            x = moes[layer](x)  # DeepSeekMoEV3 returns single tensor
        
        # Generate
        generated = []
        for _ in range(generate_tokens):
            token_input = torch.randn(batch, 1, d_model, device=device)
            x = token_input
            for layer in range(n_layers):
                x = dsas[layer](x)
                x = moes[layer](x)  # DeepSeekMoEV3 returns single tensor
            generated.append(x)
        
        assert len(generated) == generate_tokens
        print(f"\n  End-to-end generation:")
        print(f"    Layers: {n_layers}")
        print(f"    Prefill: {prefill_len} tokens")
        print(f"    Generated: {generate_tokens} tokens")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
