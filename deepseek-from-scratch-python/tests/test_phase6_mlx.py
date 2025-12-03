"""
DeepSeek-V3.2 Phase 6: Production Hardening Tests for MLX

Apple Silicon optimized tests for:
- End-to-end inference benchmarks
- Speculative decoding with MTP heads
- KV cache optimization
- 128K context verification
- 256-expert routing validation
- Full pipeline stress tests
"""

import pytest
import time
import sys
import os
from typing import List, Tuple

# Add mlx_impl to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mlx_impl'))

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def skip_if_no_mlx():
    """Skip test if MLX is not available."""
    if not MLX_AVAILABLE:
        pytest.skip("MLX not available")


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
        if MLX_AVAILABLE:
            mx.eval(mx.zeros(1))  # Sync
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if MLX_AVAILABLE:
            mx.eval(mx.zeros(1))  # Sync
        self.duration = time.perf_counter() - self.start
    
    def ms(self) -> float:
        return self.duration * 1000


# ============================================================================
# 128K Context Tests
# ============================================================================

class TestLongContextMLX:
    """Tests for 128K context window support on MLX."""
    
    def test_rope_128k_initialization(self):
        """Test RoPE initialization for 128K context."""
        skip_if_no_mlx()
        from attention import YaRNRoPE, RoPEConfig
        
        config = RoPEConfig(
            d_head=64,
            max_seq_len=131072,
            scaling_type="yarn",
            yarn_original_max_position=4096,
        )
        rope = YaRNRoPE(config)
        
        assert config.max_seq_len == 131072
    
    def test_rope_all_scaling_types(self):
        """Test all RoPE scaling types on MLX."""
        skip_if_no_mlx()
        from attention import RoPEConfig, create_rope
        
        scaling_types = [
            ("none", {}),
            ("linear", {"linear_scale": 4.0}),
            ("ntk", {"ntk_alpha": 32.0}),
            ("yarn", {"yarn_original_max_position": 4096}),
        ]
        
        for scaling_type, kwargs in scaling_types:
            config = RoPEConfig(
                d_head=64,
                max_seq_len=8192,
                scaling_type=scaling_type,
                **kwargs
            )
            rope = create_rope(config)
            
            x = mx.random.normal((1, 4, 32, 64))
            out = rope(x)
            
            assert out.shape == x.shape
            print(f"  ✓ MLX RoPE {scaling_type} scaling passed")
    
    def test_dsa_long_context_mlx(self):
        """Test DSA with longer sequences on MLX."""
        skip_if_no_mlx()
        from attention import DSAConfig, DeepSeekSparseAttention
        
        config = DSAConfig(
            d_model=128,
            num_heads=4,
            window_size=256,
            num_global_tokens=32,
            max_seq_len=4096,
        )
        dsa = DeepSeekSparseAttention(config)
        
        seq_len = 2048
        x = mx.random.normal((1, seq_len, 128))
        
        with BenchmarkTimer("MLX DSA 2K seq") as timer:
            out = dsa(x)
            mx.eval(out)
        
        assert out.shape == x.shape
        print(f"  MLX DSA 2K sequence: {timer.ms():.2f}ms")


# ============================================================================
# 256-Expert MoE Tests
# ============================================================================

class TestLargeExpertMoEMLX:
    """Tests for 256-expert MoE on MLX."""
    
    def test_moe_v3_default_config(self):
        """Test MoE V3 default configuration on MLX."""
        skip_if_no_mlx()
        from moe import MoEV3Config
        
        config = MoEV3Config()
        
        assert config.n_routed_experts == 256
        assert config.top_k == 8
    
    def test_moe_256_expert_forward_mlx(self):
        """Test MoE with 256 experts on MLX."""
        skip_if_no_mlx()
        from moe import MoEV3Config, MoEV3
        
        config = MoEV3Config(
            d_model=64,
            n_routed_experts=256,
            n_shared_experts=2,
            top_k=8,
            routed_expert_hidden=128,
            shared_expert_hidden=128,
            n_expert_groups=8,
            top_k_groups=4,
        )
        moe = MoEV3(config)
        
        batch, seq = 4, 16
        x = mx.random.normal((batch, seq, 64))
        
        with BenchmarkTimer("MLX MoE 256 experts") as timer:
            out, aux_loss = moe(x)
            mx.eval(out)
        
        assert out.shape == x.shape
        print(f"  MLX MoE 256 experts forward: {timer.ms():.2f}ms")
    
    def test_moe_hierarchical_routing_mlx(self):
        """Test hierarchical routing on MLX."""
        skip_if_no_mlx()
        from moe import MoEV3Config, MoEV3
        
        config = MoEV3Config(
            d_model=64,
            n_routed_experts=64,
            top_k=8,
            n_expert_groups=8,
            top_k_groups=4,
            routed_expert_hidden=128,
            shared_expert_hidden=128,
        )
        
        assert config.experts_per_group == 8
        
        moe = MoEV3(config)
        x = mx.random.normal((2, 32, 64))
        out, _ = moe(x)
        
        assert out.shape == x.shape


# ============================================================================
# KV Cache Tests for MLX
# ============================================================================

class TestKVCacheMLX:
    """Tests for KV cache on Apple Silicon."""
    
    def test_kv_cache_creation_mlx(self):
        """Test KV cache initialization on MLX."""
        skip_if_no_mlx()
        from attention import KVCache
        
        cache = KVCache(
            batch_size=2,
            max_seq_len=2048,
            num_heads=8,
            head_dim=64,
        )
        
        assert cache.current_seq_len == 0
        assert cache.max_seq_len == 2048
    
    def test_kv_cache_incremental_update_mlx(self):
        """Test incremental KV cache updates on MLX."""
        skip_if_no_mlx()
        from attention import KVCache
        
        batch, heads, head_dim = 2, 8, 64
        
        cache = KVCache(
            batch_size=batch,
            max_seq_len=1024,
            num_heads=heads,
            head_dim=head_dim,
        )
        
        # Prefill
        k = mx.random.normal((batch, heads, 128, head_dim))
        v = mx.random.normal((batch, heads, 128, head_dim))
        k_out, v_out = cache.update(k, v)
        
        assert k_out.shape[2] == 128
        
        # Generate
        for i in range(10):
            k = mx.random.normal((batch, heads, 1, head_dim))
            v = mx.random.normal((batch, heads, 1, head_dim))
            k_out, v_out = cache.update(k, v)
        
        assert cache.current_seq_len == 138


# ============================================================================
# Speculative Decoding Tests for MLX
# ============================================================================

class TestSpeculativeDecodingMLX:
    """Tests for speculative decoding on Apple Silicon."""
    
    def test_mtp_model_forward_mlx(self):
        """Test MTP model forward pass on MLX."""
        skip_if_no_mlx()
        from mtp import MTPConfig, MTPModel
        
        config = MTPConfig(
            d_model=128,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            num_predict_tokens=4,
        )
        mtp = MTPModel(config)
        
        batch, seq = 2, 32
        input_ids = mx.random.randint(0, 1000, (batch, seq))
        
        outputs = mtp(input_ids)
        
        assert outputs.shape[0] == batch
        assert outputs.shape[1] == seq
    
    def test_speculative_decode_step_mlx(self):
        """Test speculative decoding step on MLX."""
        skip_if_no_mlx()
        from mtp import MTPConfig, MTPModel
        
        config = MTPConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            vocab_size=100,
            num_predict_tokens=3,
        )
        mtp = MTPModel(config)
        
        batch = 1
        input_ids = mx.random.randint(0, 100, (batch, 1))
        
        with BenchmarkTimer("MLX MTP speculative") as timer:
            outputs = mtp(input_ids)
            mx.eval(outputs)
        
        print(f"  MLX MTP speculative step: {timer.ms():.2f}ms")


# ============================================================================
# Full Pipeline Benchmarks for MLX
# ============================================================================

class TestFullPipelineBenchmarksMLX:
    """End-to-end benchmarks on Apple Silicon."""
    
    def test_dsa_scaling_benchmark_mlx(self):
        """Benchmark DSA scaling on MLX."""
        skip_if_no_mlx()
        from attention import DSAConfig, DeepSeekSparseAttention
        
        seq_lengths = [128, 256, 512, 1024]
        d_model = 64
        
        print("\n  MLX DSA Scaling Benchmark:")
        print(f"  {'Seq Len':>8} | {'Time (ms)':>12}")
        print(f"  {'-'*8}-+-{'-'*12}")
        
        for seq_len in seq_lengths:
            config = DSAConfig(
                d_model=d_model,
                num_heads=4,
                window_size=32,
                num_global_tokens=8,
                max_seq_len=seq_len * 2,
            )
            dsa = DeepSeekSparseAttention(config)
            
            x = mx.random.normal((1, seq_len, d_model))
            
            # Warmup
            _ = dsa(x)
            mx.eval(_)
            
            # Benchmark
            times = []
            for _ in range(10):
                with BenchmarkTimer("") as timer:
                    out = dsa(x)
                    mx.eval(out)
                times.append(timer.ms())
            
            avg_time = sum(times) / len(times)
            print(f"  {seq_len:>8} | {avg_time:>12.2f}")
    
    def test_moe_scaling_benchmark_mlx(self):
        """Benchmark MoE scaling on MLX."""
        skip_if_no_mlx()
        from moe import MoEV3Config, MoEV3
        
        configs = [
            (16, 2),
            (32, 4),
            (64, 8),
        ]
        
        d_model = 32
        batch, seq = 4, 32
        
        print("\n  MLX MoE Expert Scaling Benchmark:")
        print(f"  {'Experts':>8} | {'Active':>8} | {'Time (ms)':>12}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*12}")
        
        for n_experts, top_k in configs:
            config = MoEV3Config(
                d_model=d_model,
                n_routed_experts=n_experts,
                top_k=top_k,
                n_expert_groups=max(1, n_experts // 8),
                top_k_groups=max(1, top_k // 2),
                routed_expert_hidden=d_model * 2,
                shared_expert_hidden=d_model * 2,
            )
            moe = MoEV3(config)
            
            x = mx.random.normal((batch, seq, d_model))
            
            # Warmup
            _ = moe(x)
            mx.eval(_[0])
            
            # Benchmark
            times = []
            for _ in range(10):
                with BenchmarkTimer("") as timer:
                    out, _ = moe(x)
                    mx.eval(out)
                times.append(timer.ms())
            
            avg_time = sum(times) / len(times)
            print(f"  {n_experts:>8} | {top_k:>8} | {avg_time:>12.2f}")
    
    def test_full_v32_pipeline_mlx(self):
        """Test full V3.2 pipeline on MLX."""
        skip_if_no_mlx()
        from attention import RoPEConfig, create_rope, DSAConfig, DeepSeekSparseAttention
        from moe import MoEV3Config, MoEV3
        
        batch = 2
        seq = 64
        d_model = 128
        num_heads = 4
        
        # Create components
        rope_config = RoPEConfig(
            d_head=d_model // num_heads,
            max_seq_len=8192,
            scaling_type="yarn",
        )
        rope = create_rope(rope_config)
        
        dsa_config = DSAConfig(
            d_model=d_model,
            num_heads=num_heads,
            window_size=16,
            num_global_tokens=4,
            max_seq_len=1024,
        )
        dsa = DeepSeekSparseAttention(dsa_config)
        
        moe_config = MoEV3Config(
            d_model=d_model,
            n_routed_experts=16,
            top_k=2,
            n_expert_groups=4,
            top_k_groups=2,
            routed_expert_hidden=d_model * 2,
            shared_expert_hidden=d_model * 2,
        )
        moe = MoEV3(moe_config)
        
        # Input
        x = mx.random.normal((batch, seq, d_model))
        
        # Forward
        with BenchmarkTimer("DSA") as dsa_timer:
            dsa_out = dsa(x)
            mx.eval(dsa_out)
        
        with BenchmarkTimer("MoE") as moe_timer:
            moe_out, _ = moe(dsa_out)
            mx.eval(moe_out)
        
        assert moe_out.shape == x.shape
        
        print(f"\n  MLX Full V3.2 Pipeline Test:")
        print(f"    DSA time: {dsa_timer.ms():.2f}ms")
        print(f"    MoE time: {moe_timer.ms():.2f}ms")
        print(f"    Total: {dsa_timer.ms() + moe_timer.ms():.2f}ms")
    
    def test_numerical_stability_mlx(self):
        """Test numerical stability on MLX."""
        skip_if_no_mlx()
        from attention import DSAConfig, DeepSeekSparseAttention
        from moe import MoEV3Config, MoEV3
        
        d_model = 64
        
        dsa_config = DSAConfig(
            d_model=d_model,
            num_heads=4,
            window_size=8,
            num_global_tokens=2,
            max_seq_len=128,
        )
        dsa = DeepSeekSparseAttention(dsa_config)
        
        moe_config = MoEV3Config(
            d_model=d_model,
            n_routed_experts=16,
            top_k=2,
            routed_expert_hidden=d_model * 2,
            shared_expert_hidden=d_model * 2,
        )
        moe = MoEV3(moe_config)
        
        for i in range(10):
            x = mx.random.normal((2, 32, d_model))
            
            dsa_out = dsa(x)
            moe_out, _ = moe(dsa_out)
            mx.eval(moe_out)
            
            mean = moe_out.mean().item()
            assert not mx.isnan(moe_out).any().item(), f"NaN at iteration {i}"
        
        print(f"  MLX numerical stability test passed (10 iterations)")


# ============================================================================
# Agent Pipeline Integration Tests for MLX
# ============================================================================

class TestAgentPipelineMLX:
    """Agent pipeline tests on MLX."""
    
    def test_agent_trajectory_workflow_mlx(self):
        """Test agent trajectory on MLX."""
        skip_if_no_mlx()
        from agent import (
            AgentTrajectory, AgentStep, AgentAction, AgentActionType,
            ToolCall, ToolResponse, ToolStatus, ToolType, TaskTier
        )
        
        trajectory = AgentTrajectory(
            task="Write sorting function",
            tier=TaskTier.MULTI_TOOL_SEQUENTIAL
        )
        
        step1 = AgentStep(step_id=0)
        step1.action = AgentAction(
            action_type=AgentActionType.TOOL_CALL,
            tool_call=ToolCall(
                tool_type=ToolType.CODE_EXECUTION,
                function_name="search",
                arguments={"query": "sort algorithm"}
            )
        )
        step1.response = ToolResponse(
            status=ToolStatus.SUCCESS,
            content="Found algorithms",
            execution_time_ms=100
        )
        trajectory.add_step(step1)
        
        assert trajectory.num_steps() == 1
    
    def test_reward_computation_mlx(self):
        """Test reward computation on MLX."""
        skip_if_no_mlx()
        from agent import RewardWeights, RewardBreakdown
        
        weights = RewardWeights()
        
        breakdown = RewardBreakdown(
            correctness=0.9,
            format=1.0,
            efficiency=0.8,
            safety=1.0
        )
        
        reward = breakdown.total(weights)
        assert abs(reward - 0.92) < 1e-6


# ============================================================================
# End-to-End Generation on MLX
# ============================================================================

class TestEndToEndGenerationMLX:
    """End-to-end generation tests on MLX."""
    
    def test_multi_layer_generation_mlx(self):
        """Test generation with multiple layers on MLX."""
        skip_if_no_mlx()
        from attention import DSAConfig, DeepSeekSparseAttention, KVCache
        from moe import MoEV3Config, MoEV3
        
        batch = 1
        d_model = 64
        num_heads = 4
        head_dim = d_model // num_heads
        n_layers = 2
        prefill_len = 32
        generate_tokens = 8
        
        # Create layers
        dsas = []
        moes = []
        for _ in range(n_layers):
            dsa_config = DSAConfig(
                d_model=d_model,
                num_heads=num_heads,
                window_size=8,
                num_global_tokens=2,
                max_seq_len=256,
            )
            dsas.append(DeepSeekSparseAttention(dsa_config))
            
            moe_config = MoEV3Config(
                d_model=d_model,
                n_routed_experts=8,
                top_k=2,
                routed_expert_hidden=d_model * 2,
                shared_expert_hidden=d_model * 2,
            )
            moes.append(MoEV3(moe_config))
        
        # Prefill
        prefill = mx.random.normal((batch, prefill_len, d_model))
        x = prefill
        for layer in range(n_layers):
            x = dsas[layer](x)
            x, _ = moes[layer](x)
        mx.eval(x)
        
        # Generate
        generated = []
        for _ in range(generate_tokens):
            token_input = mx.random.normal((batch, 1, d_model))
            x = token_input
            for layer in range(n_layers):
                x = dsas[layer](x)
                x, _ = moes[layer](x)
            mx.eval(x)
            generated.append(x)
        
        assert len(generated) == generate_tokens
        print(f"\n  MLX End-to-end generation:")
        print(f"    Layers: {n_layers}")
        print(f"    Prefill: {prefill_len} tokens")
        print(f"    Generated: {generate_tokens} tokens")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
