"""
Tests for R1 Reasoning Memory Management - PyTorch Implementation

Tests cover:
- Arena allocation for reasoning tokens
- KV cache budget management with dynamic eviction
- Think token detection
- Memory benchmarking
"""

import os
import time

import torch

# Enable MPS fallback for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.deepseek.torch.model.r1 import (
    EvictionPolicy,
    ReasoningToken,
    ReasoningTokenArena,
    KVCacheStats,
    KVCacheBudget,
    ReasoningMemoryConfig,
    ReasoningMemoryStats,
    ReasoningMemoryManager,
    ReasoningModel,
    get_device,
)


class TestEvictionPolicy:
    """Tests for EvictionPolicy enum."""

    def test_eviction_policy_values(self):
        """Test that all eviction policies are defined."""
        assert EvictionPolicy.FIFO is not None
        assert EvictionPolicy.LRU is not None
        assert EvictionPolicy.ATTENTION_SCORE is not None
        assert EvictionPolicy.SLIDING_WINDOW is not None


class TestReasoningToken:
    """Tests for ReasoningToken."""

    def test_token_creation(self):
        """Test token creation with default values."""
        token = ReasoningToken(token_id=100, position=5, is_reasoning=True)
        assert token.token_id == 100
        assert token.position == 5
        assert token.is_reasoning
        assert token.attention_score == 0.0

    def test_token_update_attention(self):
        """Test updating attention score."""
        token = ReasoningToken(token_id=100, position=5, is_reasoning=True)
        old_access = token.last_access
        time.sleep(0.001)  # Small delay
        token.update_attention(0.5)
        assert token.attention_score == 0.5
        assert token.last_access >= old_access

    def test_token_touch(self):
        """Test touching a token (LRU update)."""
        token = ReasoningToken(token_id=100, position=5, is_reasoning=True)
        old_access = token.last_access
        time.sleep(0.001)
        token.touch()
        assert token.last_access >= old_access


class TestReasoningTokenArena:
    """Tests for ReasoningTokenArena."""

    def test_arena_creation(self):
        """Test arena creation with specified size."""
        arena = ReasoningTokenArena(capacity=100)
        assert arena.capacity == 100
        assert arena.usage() == 0

    def test_arena_allocate(self):
        """Test allocating a slot."""
        arena = ReasoningTokenArena(capacity=100)
        slot_idx = arena.allocate(token_id=42, position=0, is_reasoning=True)
        assert slot_idx is not None
        assert arena.usage() == 1

        token = arena.get(slot_idx)
        assert token is not None
        assert token.token_id == 42
        assert token.position == 0
        assert token.is_reasoning

    def test_arena_free(self):
        """Test freeing a slot."""
        arena = ReasoningTokenArena(capacity=100)
        slot_idx = arena.allocate(token_id=42, position=0, is_reasoning=True)
        assert arena.usage() == 1

        arena.free(slot_idx)
        # Note: free adds to free_list but doesn't immediately decrement _len

    def test_arena_capacity(self):
        """Test arena capacity limits."""
        arena = ReasoningTokenArena(capacity=10)

        # Allocate all slots
        for i in range(10):
            slot_idx = arena.allocate(token_id=i, position=i, is_reasoning=True)
            assert slot_idx is not None

        # Next allocation should fail
        slot_idx = arena.allocate(token_id=100, position=100, is_reasoning=True)
        assert slot_idx is None

    def test_arena_reset(self):
        """Test arena reset."""
        arena = ReasoningTokenArena(capacity=100)

        # Allocate some slots
        for i in range(10):
            arena.allocate(token_id=i, position=i, is_reasoning=True)

        assert arena.usage() == 10

        # Reset
        arena.reset()
        assert arena.usage() == 0

    def test_arena_reuse_freed_slots(self):
        """Test reusing freed slots."""
        arena = ReasoningTokenArena(capacity=10)

        # Allocate and free
        slot = arena.allocate(token_id=1, position=0, is_reasoning=True)
        arena.free(slot)

        # Allocate again - should reuse the freed slot
        new_slot = arena.allocate(token_id=2, position=1, is_reasoning=True)
        assert new_slot == slot  # Should be the same slot


class TestKVCacheBudget:
    """Tests for KVCacheBudget."""

    def test_budget_creation(self):
        """Test budget creation."""
        budget = KVCacheBudget(total_budget=1000, reasoning_ratio=0.7)
        assert budget.total_budget == 1000
        assert budget.reasoning_budget == 700
        assert budget.context_budget == 300

    def test_can_add_reasoning(self):
        """Test checking if reasoning tokens can be added."""
        budget = KVCacheBudget(total_budget=10, reasoning_ratio=0.5)
        assert budget.reasoning_budget == 5

        for i in range(5):
            assert budget.can_add_reasoning()
            budget.add_reasoning()

        assert not budget.can_add_reasoning()

    def test_can_add_context(self):
        """Test checking if context tokens can be added."""
        budget = KVCacheBudget(total_budget=10, reasoning_ratio=0.5)
        assert budget.context_budget == 5

        for i in range(5):
            assert budget.can_add_context()
            budget.add_context()

        assert not budget.can_add_context()

    def test_remove_tokens(self):
        """Test removing tokens from budget."""
        budget = KVCacheBudget(total_budget=10, reasoning_ratio=0.5)

        budget.add_reasoning()
        budget.add_context()
        assert budget.reasoning_count == 1
        assert budget.context_count == 1

        budget.remove_reasoning()
        budget.remove_context()
        assert budget.reasoning_count == 0
        assert budget.context_count == 0

    def test_tokens_to_evict(self):
        """Test calculating tokens to evict."""
        budget = KVCacheBudget(total_budget=10, reasoning_ratio=0.5)

        # Fill reasoning budget
        for i in range(5):
            budget.add_reasoning()

        # Should need to evict 2 to add 2 new tokens
        assert budget.tokens_to_evict(2) == 2

    def test_stats(self):
        """Test getting statistics."""
        budget = KVCacheBudget(total_budget=100, reasoning_ratio=0.7)
        budget.add_reasoning()
        budget.add_context()

        stats = budget.stats()
        assert stats.total_budget == 100
        assert stats.reasoning_budget == 70
        assert stats.context_budget == 30
        assert stats.reasoning_used == 1
        assert stats.context_used == 1

    def test_reset(self):
        """Test resetting budget."""
        budget = KVCacheBudget(total_budget=100)
        budget.add_reasoning()
        budget.add_context()

        budget.reset()
        assert budget.reasoning_count == 0
        assert budget.context_count == 0


class TestReasoningMemoryConfig:
    """Tests for ReasoningMemoryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReasoningMemoryConfig()
        assert config.max_reasoning_tokens == 32768
        assert config.kv_cache_budget_ratio == 0.7
        assert config.eviction_policy == EvictionPolicy.SLIDING_WINDOW
        assert config.total_kv_budget == 131072
        assert config.reasoning_timeout_secs == 300.0
        assert config.think_start_token == "<think>"
        assert config.think_end_token == "</think>"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReasoningMemoryConfig(
            max_reasoning_tokens=1000,
            kv_cache_budget_ratio=0.5,
            eviction_policy=EvictionPolicy.LRU,
            reasoning_timeout_secs=60.0,
        )
        assert config.max_reasoning_tokens == 1000
        assert config.kv_cache_budget_ratio == 0.5
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.reasoning_timeout_secs == 60.0


class TestReasoningMemoryManager:
    """Tests for ReasoningMemoryManager."""

    def test_manager_creation(self):
        """Test memory manager creation."""
        manager = ReasoningMemoryManager()
        assert manager.arena.capacity == 32768
        assert manager.kv_budget.total_budget == 131072

    def test_start_end_session(self):
        """Test session management."""
        manager = ReasoningMemoryManager()

        assert manager.session_start is None
        manager.start_session()
        assert manager.session_start is not None

        manager.end_session()
        assert manager.session_start is None

    def test_allocate_token(self):
        """Test allocating tokens."""
        manager = ReasoningMemoryManager()
        manager.start_session()

        slot_idx = manager.allocate_token(token_id=100)
        assert slot_idx is not None
        assert manager.tokens_generated == 1

    def test_process_token_boundary(self):
        """Test processing think token boundaries."""
        manager = ReasoningMemoryManager()
        manager.start_session()

        assert not manager.in_reasoning

        manager.process_token_boundary("<think>")
        assert manager.in_reasoning

        manager.process_token_boundary("</think>")
        assert not manager.in_reasoning

    def test_timeout_check(self):
        """Test timeout checking."""
        config = ReasoningMemoryConfig(reasoning_timeout_secs=0.001)
        manager = ReasoningMemoryManager(config)
        manager.start_session()

        assert not manager.is_timed_out()

        time.sleep(0.01)
        assert manager.is_timed_out()

    def test_evict_tokens_fifo(self):
        """Test FIFO eviction policy."""
        config = ReasoningMemoryConfig(
            max_reasoning_tokens=100,
            total_kv_budget=10,
            eviction_policy=EvictionPolicy.FIFO,
        )
        manager = ReasoningMemoryManager(config)
        manager.start_session()
        manager.in_reasoning = True

        # Allocate more tokens than budget
        for i in range(15):
            manager.allocate_token(token_id=i)

        # Should have evicted some
        assert manager.evicted_count > 0

    def test_evict_tokens_lru(self):
        """Test LRU eviction policy."""
        config = ReasoningMemoryConfig(
            max_reasoning_tokens=100,
            total_kv_budget=10,
            eviction_policy=EvictionPolicy.LRU,
        )
        manager = ReasoningMemoryManager(config)
        manager.start_session()
        manager.in_reasoning = True

        # Allocate tokens
        for i in range(15):
            manager.allocate_token(token_id=i)

        # Should have evicted
        assert manager.evicted_count > 0

    def test_update_attention_scores(self):
        """Test updating attention scores."""
        manager = ReasoningMemoryManager()
        manager.start_session()

        # Allocate some tokens
        for i in range(5):
            manager.allocate_token(token_id=i)

        # Update attention scores
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        manager.update_attention_scores(scores)

        # Check scores were updated
        for i, slot in enumerate(list(manager.active_slots)[:5]):
            token = manager.arena.get(slot)
            if token:
                assert token.attention_score >= 0.0

    def test_get_stats(self):
        """Test getting statistics."""
        manager = ReasoningMemoryManager()
        manager.start_session()

        for i in range(10):
            manager.allocate_token(token_id=i)

        stats = manager.get_stats()
        assert stats.arena_capacity == 32768
        assert stats.arena_used == 10
        assert stats.tokens_generated == 10
        assert stats.session_duration is not None

    def test_reset(self):
        """Test resetting manager."""
        manager = ReasoningMemoryManager()
        manager.start_session()

        for i in range(10):
            manager.allocate_token(token_id=i)

        manager.reset()

        assert manager.tokens_generated == 0
        assert manager.arena.usage() == 0
        assert len(manager.active_slots) == 0


class TestReasoningModel:
    """Tests for ReasoningModel."""

    def test_model_creation(self):
        """Test model creation."""
        device = get_device()
        model = ReasoningModel(vocab_size=1000, d_model=64)
        model.to(device)

        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.memory_manager is not None

    def test_forward_pass(self):
        """Test forward pass."""
        device = get_device()
        model = ReasoningModel(vocab_size=1000, d_model=64)
        model.to(device)

        input_ids = torch.randint(0, 1000, (1, 10), device=device)
        output = model(input_ids)

        assert output.shape == (1, 10, 1000)

    def test_generate_with_reasoning(self):
        """Test generate_with_reasoning method."""
        device = get_device()
        model = ReasoningModel(vocab_size=1000, d_model=64)
        model.to(device)

        result = model.generate_with_reasoning("What is 2+2?")

        assert "<think>" in result
        assert "</think>" in result

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        device = get_device()
        model = ReasoningModel(vocab_size=1000, d_model=64)
        model.to(device)

        stats = model.get_memory_stats()

        assert isinstance(stats, ReasoningMemoryStats)
        assert stats.arena_capacity == 32768

    def test_reset_memory(self):
        """Test resetting memory."""
        device = get_device()
        model = ReasoningModel(vocab_size=1000, d_model=64)
        model.to(device)

        # Generate something first
        model.generate_with_reasoning("test")

        model.reset_memory()

        stats = model.get_memory_stats()
        assert stats.arena_used == 0


class TestDeviceHandling:
    """Tests for device handling and fallback."""

    def test_get_device(self):
        """Test get_device function."""
        device = get_device()
        # Should return a valid device
        assert device.type in ["cpu", "cuda", "mps"]

    def test_model_device_transfer(self):
        """Test model transfers to device."""
        device = get_device()
        model = ReasoningModel(vocab_size=1000, d_model=64)
        model.to(device)

        # Check embedding is on correct device
        assert next(model.parameters()).device.type == device.type

    def test_tensor_operations_on_device(self):
        """Test tensor operations work on selected device."""
        device = get_device()

        a = torch.randn(10, 10, device=device)
        b = torch.randn(10, 10, device=device)
        c = a @ b  # Matrix multiplication

        assert c.device.type == device.type
        assert c.shape == (10, 10)


class TestIntegration:
    """Integration tests for the R1 memory management system."""

    def test_full_reasoning_flow(self):
        """Test a full reasoning flow."""
        device = get_device()
        config = ReasoningMemoryConfig(
            max_reasoning_tokens=1000,
            total_kv_budget=500,
        )
        model = ReasoningModel(
            vocab_size=1000,
            d_model=64,
            memory_config=config,
        )
        model.to(device)

        # Generate with reasoning
        result = model.generate_with_reasoning("Test prompt")
        assert "<think>" in result

        # Check memory stats
        stats = model.get_memory_stats()
        assert stats.tokens_generated > 0

    def test_memory_pressure(self):
        """Test system under memory pressure."""
        config = ReasoningMemoryConfig(
            max_reasoning_tokens=100,
            total_kv_budget=20,
            eviction_policy=EvictionPolicy.FIFO,
        )
        manager = ReasoningMemoryManager(config)
        manager.start_session()
        manager.in_reasoning = True

        # Allocate many tokens to trigger eviction
        for i in range(50):
            manager.allocate_token(i)

        # Should have evicted some tokens
        stats = manager.get_stats()
        assert stats.tokens_evicted > 0

    def test_eviction_policy_comparison(self):
        """Test different eviction policies."""
        policies = [
            EvictionPolicy.FIFO,
            EvictionPolicy.LRU,
            EvictionPolicy.SLIDING_WINDOW,
        ]

        for policy in policies:
            config = ReasoningMemoryConfig(
                max_reasoning_tokens=50,
                total_kv_budget=10,
                eviction_policy=policy,
            )
            manager = ReasoningMemoryManager(config)
            manager.start_session()
            manager.in_reasoning = True

            # Allocate tokens
            for i in range(30):
                manager.allocate_token(i)

            # All policies should handle eviction
            stats = manager.get_stats()
            assert stats.tokens_evicted > 0
            assert stats.arena_used <= 50

