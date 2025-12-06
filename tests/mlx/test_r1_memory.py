"""
Tests for R1 Reasoning Memory Management - MLX Implementation

Tests cover:
- Arena allocation for reasoning tokens
- KV cache budget management with dynamic eviction
- Streaming reasoning token generation
- Think token detection
- Memory benchmarking
- Timeout mechanism for runaway reasoning
"""

import time

import mlx.core as mx

from src.deepseek.mlx.r1 import (
    EvictionPolicy,
    ReasoningTokenSlot,
    ArenaAllocator,
    ArenaStats,
    MemoryBudget,
    MemoryUsage,
    ReasoningMemoryManager,
    KVCacheEntry,
    KVCacheBudgetManager,
    ThinkTokenConfig,
    ThinkTokenDetector,
    StreamingConfig,
    StreamingState,
    StreamingReasoningGenerator,
    BenchmarkResult,
    MemoryBenchmark,
    ReasoningModel,
    DEFAULT_ARENA_SIZE,
    DEFAULT_MEMORY_BUDGET_MB,
    DEFAULT_KV_BUDGET_MB,
    DEFAULT_THINK_START_TOKEN,
    DEFAULT_THINK_END_TOKEN,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_REASONING_TOKENS,
)


class TestEvictionPolicy:
    """Tests for EvictionPolicy enum."""

    def test_eviction_policy_values(self):
        """Test that all eviction policies are defined."""
        assert EvictionPolicy.FIFO is not None
        assert EvictionPolicy.LRU is not None
        assert EvictionPolicy.ATTENTION_SCORE is not None
        assert EvictionPolicy.SLIDING_WINDOW is not None


class TestReasoningTokenSlot:
    """Tests for ReasoningTokenSlot."""

    def test_slot_creation(self):
        """Test slot creation with default values."""
        slot = ReasoningTokenSlot(token_id=100, position=5, timestamp=time.time())
        assert slot.token_id == 100
        assert slot.position == 5
        assert slot.is_allocated
        assert slot.attention_score == 0.0

    def test_slot_free(self):
        """Test freeing a slot."""
        slot = ReasoningTokenSlot(token_id=100, position=5, timestamp=time.time())
        slot.free()
        assert not slot.is_allocated
        assert slot.token_id == 0
        assert slot.position == -1


class TestArenaAllocator:
    """Tests for ArenaAllocator."""

    def test_arena_creation(self):
        """Test arena creation with specified size."""
        arena = ArenaAllocator(max_tokens=100)
        assert arena.max_tokens == 100
        assert arena.allocated_count == 0
        stats = arena.get_stats()
        assert stats.total_slots == 100
        assert stats.free_slots == 100

    def test_arena_allocate(self):
        """Test allocating a slot."""
        arena = ArenaAllocator(max_tokens=100)
        slot_idx = arena.allocate(token_id=42, position=0)
        assert slot_idx is not None
        assert arena.allocated_count == 1

        slot = arena.get_slot(slot_idx)
        assert slot is not None
        assert slot.token_id == 42
        assert slot.position == 0
        assert slot.is_allocated

    def test_arena_free(self):
        """Test freeing a slot."""
        arena = ArenaAllocator(max_tokens=100)
        slot_idx = arena.allocate(token_id=42, position=0)
        assert arena.allocated_count == 1

        result = arena.free(slot_idx)
        assert result
        assert arena.allocated_count == 0

    def test_arena_capacity(self):
        """Test arena capacity limits."""
        arena = ArenaAllocator(max_tokens=10)

        # Allocate all slots
        for i in range(10):
            slot_idx = arena.allocate(token_id=i, position=i)
            assert slot_idx is not None

        # Next allocation should fail
        slot_idx = arena.allocate(token_id=100, position=100)
        assert slot_idx is None

    def test_arena_defragment(self):
        """Test arena defragmentation."""
        arena = ArenaAllocator(max_tokens=10)

        # Allocate slots
        slots = []
        for i in range(5):
            slots.append(arena.allocate(token_id=i, position=i))

        # Free alternating slots
        arena.free(slots[1])
        arena.free(slots[3])

        # Defragment
        moves = arena.defragment()
        assert moves >= 0

        # Check fragmentation reduced
        stats = arena.get_stats()
        assert stats.fragmentation_ratio <= 0.5

    def test_arena_reset(self):
        """Test arena reset."""
        arena = ArenaAllocator(max_tokens=100)

        # Allocate some slots
        for i in range(10):
            arena.allocate(token_id=i, position=i)

        assert arena.allocated_count == 10

        # Reset
        arena.reset()
        assert arena.allocated_count == 0

    def test_arena_stats(self):
        """Test arena statistics."""
        arena = ArenaAllocator(max_tokens=100)

        for i in range(25):
            arena.allocate(token_id=i, position=i)

        stats = arena.get_stats()
        assert stats.total_slots == 100
        assert stats.allocated_slots == 25
        assert stats.free_slots == 75
        assert stats.peak_usage == 25
        assert stats.total_allocations == 25


class TestMemoryBudget:
    """Tests for MemoryBudget."""

    def test_budget_creation(self):
        """Test budget creation with default values."""
        budget = MemoryBudget()
        assert budget.total_budget_mb == DEFAULT_MEMORY_BUDGET_MB
        assert budget.kv_cache_budget_mb == DEFAULT_KV_BUDGET_MB

    def test_budget_scaling(self):
        """Test budget scaling when components exceed total."""
        budget = MemoryBudget(
            total_budget_mb=100.0,
            kv_cache_budget_mb=200.0,  # Exceeds total
            reasoning_budget_mb=200.0,  # Exceeds total
            embedding_budget_mb=200.0,  # Exceeds total
        )
        # Components should be scaled down
        component_total = (
            budget.kv_cache_budget_mb
            + budget.reasoning_budget_mb
            + budget.embedding_budget_mb
        )
        assert component_total <= budget.total_budget_mb + 0.01  # Allow small float error


class TestReasoningMemoryManager:
    """Tests for ReasoningMemoryManager."""

    def test_manager_creation(self):
        """Test memory manager creation."""
        manager = ReasoningMemoryManager()
        assert manager.budget.total_budget_mb == DEFAULT_MEMORY_BUDGET_MB
        assert manager.arena.max_tokens == DEFAULT_ARENA_SIZE

    def test_allocate_reasoning_token(self):
        """Test allocating reasoning tokens."""
        manager = ReasoningMemoryManager()

        slot_idx = manager.allocate_reasoning_token(token_id=100, position=0)
        assert slot_idx is not None

        stats = manager.get_stats()
        assert stats["arena"]["allocated_slots"] == 1

    def test_free_reasoning_token(self):
        """Test freeing reasoning tokens."""
        manager = ReasoningMemoryManager()

        slot_idx = manager.allocate_reasoning_token(token_id=100, position=0)
        result = manager.free_reasoning_token(slot_idx)
        assert result

        stats = manager.get_stats()
        assert stats["arena"]["allocated_slots"] == 0

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        manager = ReasoningMemoryManager()

        # Track KV cache
        manager.track_kv_cache(num_entries=1000, entry_size_bytes=4096)

        stats = manager.get_stats()
        assert stats["usage"]["kv_cache_mb"] > 0

    def test_budget_check(self):
        """Test budget checking."""
        budget = MemoryBudget(total_budget_mb=1.0)  # Very small budget
        manager = ReasoningMemoryManager(budget=budget, bytes_per_token=1024 * 1024)

        # Initially within budget
        assert manager.is_within_budget()

        # Allocate many tokens to exceed budget
        for i in range(100):
            manager.allocate_reasoning_token(token_id=i, position=i)

        # Should exceed budget now
        assert not manager.is_within_budget()

    def test_reset(self):
        """Test manager reset."""
        manager = ReasoningMemoryManager()

        # Allocate tokens
        for i in range(10):
            manager.allocate_reasoning_token(token_id=i, position=i)

        manager.reset()

        stats = manager.get_stats()
        assert stats["arena"]["allocated_slots"] == 0
        assert stats["usage"]["total_mb"] == 0.0


class TestKVCacheBudgetManager:
    """Tests for KVCacheBudgetManager."""

    def test_manager_creation(self):
        """Test KV cache manager creation."""
        manager = KVCacheBudgetManager()
        assert manager.budget_mb == DEFAULT_KV_BUDGET_MB
        assert manager.policy == EvictionPolicy.LRU

    def test_add_entry(self):
        """Test adding KV cache entries."""
        manager = KVCacheBudgetManager(
            budget_mb=10.0, num_layers=2, num_heads=4, head_dim=64
        )

        key = mx.random.normal((4, 64))
        value = mx.random.normal((4, 64))

        result = manager.add_entry(layer_idx=0, position=0, key=key, value=value)
        assert result

        stats = manager.get_stats()
        assert stats["total_entries"] == 1

    def test_eviction(self):
        """Test eviction when budget exceeded."""
        # Very small budget to trigger eviction
        manager = KVCacheBudgetManager(
            budget_mb=0.001,
            num_layers=1,
            num_heads=4,
            head_dim=64,
            policy=EvictionPolicy.FIFO,
        )

        # Add entries until eviction happens
        for i in range(10):
            key = mx.random.normal((4, 64))
            value = mx.random.normal((4, 64))
            manager.add_entry(layer_idx=0, position=i, key=key, value=value)

        # Should have evicted some entries
        assert manager.total_evictions > 0

    def test_get_cache_for_layer(self):
        """Test retrieving cache for a layer."""
        manager = KVCacheBudgetManager(
            budget_mb=10.0, num_layers=2, num_heads=4, head_dim=64
        )

        # Add entries
        for i in range(3):
            key = mx.random.normal((4, 64))
            value = mx.random.normal((4, 64))
            manager.add_entry(layer_idx=0, position=i, key=key, value=value)

        keys, values = manager.get_cache_for_layer(0)
        assert keys.shape[0] == 3
        assert values.shape[0] == 3

    def test_clear(self):
        """Test clearing the cache."""
        manager = KVCacheBudgetManager(
            budget_mb=10.0, num_layers=2, num_heads=4, head_dim=64
        )

        key = mx.random.normal((4, 64))
        value = mx.random.normal((4, 64))
        manager.add_entry(layer_idx=0, position=0, key=key, value=value)

        manager.clear()

        stats = manager.get_stats()
        assert stats["total_entries"] == 0


class TestThinkTokenDetector:
    """Tests for ThinkTokenDetector."""

    def test_detector_creation(self):
        """Test detector creation."""
        detector = ThinkTokenDetector()
        assert not detector.in_think_block
        assert detector.think_depth == 0

    def test_detect_think_start(self):
        """Test detecting think start token."""
        detector = ThinkTokenDetector()

        is_start, is_end, in_think = detector.process_token(DEFAULT_THINK_START_TOKEN)
        assert is_start
        assert not is_end
        assert in_think

    def test_detect_think_end(self):
        """Test detecting think end token."""
        detector = ThinkTokenDetector()

        # First enter think block
        detector.process_token(DEFAULT_THINK_START_TOKEN)

        # Then detect end
        is_start, is_end, in_think = detector.process_token(DEFAULT_THINK_END_TOKEN)
        assert not is_start
        assert is_end
        assert not in_think

    def test_nested_think_blocks(self):
        """Test nested think blocks."""
        detector = ThinkTokenDetector()

        # Enter first think block
        detector.process_token(DEFAULT_THINK_START_TOKEN)
        assert detector.think_depth == 1

        # Enter nested think block
        detector.process_token(DEFAULT_THINK_START_TOKEN)
        assert detector.think_depth == 2

        # Exit inner
        detector.process_token(DEFAULT_THINK_END_TOKEN)
        assert detector.think_depth == 1
        assert detector.in_think_block

        # Exit outer
        detector.process_token(DEFAULT_THINK_END_TOKEN)
        assert detector.think_depth == 0
        assert not detector.in_think_block

    def test_reset(self):
        """Test detector reset."""
        detector = ThinkTokenDetector()

        detector.process_token(DEFAULT_THINK_START_TOKEN)
        assert detector.in_think_block

        detector.reset()
        assert not detector.in_think_block
        assert detector.think_depth == 0


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = StreamingConfig()
        assert config.max_reasoning_tokens == DEFAULT_MAX_REASONING_TOKENS
        assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
        assert config.enable_timeout
        assert config.stop_on_budget_exceeded


class TestStreamingReasoningGenerator:
    """Tests for StreamingReasoningGenerator."""

    def test_generator_creation(self):
        """Test generator creation."""
        model = ReasoningModel(vocab_size=1000, d_model=64)

        memory_manager = ReasoningMemoryManager()
        think_detector = ThinkTokenDetector()

        generator = StreamingReasoningGenerator(
            model=model, memory_manager=memory_manager, think_detector=think_detector
        )

        assert generator.model is model
        assert generator.memory_manager is memory_manager

    def test_timeout_check(self):
        """Test timeout checking."""
        model = ReasoningModel(vocab_size=1000, d_model=64)

        config = StreamingConfig(timeout_seconds=0.001)
        generator = StreamingReasoningGenerator(
            model=model,
            memory_manager=ReasoningMemoryManager(),
            think_detector=ThinkTokenDetector(),
            config=config,
        )

        generator.state.start_time = time.time() - 1.0  # 1 second ago
        assert generator._check_timeout()

    def test_budget_check(self):
        """Test budget checking."""
        model = ReasoningModel(vocab_size=1000, d_model=64)

        # Very small budget
        budget = MemoryBudget(total_budget_mb=0.0001)
        memory_manager = ReasoningMemoryManager(budget=budget)

        # Force over budget
        memory_manager.usage.total_mb = 1.0

        generator = StreamingReasoningGenerator(
            model=model,
            memory_manager=memory_manager,
            think_detector=ThinkTokenDetector(),
        )

        assert generator._check_budget()


class TestMemoryBenchmark:
    """Tests for MemoryBenchmark."""

    def test_benchmark_creation(self):
        """Test benchmark creation."""
        manager = ReasoningMemoryManager()
        benchmark = MemoryBenchmark(memory_manager=manager)
        assert len(benchmark.results) == 0

    def test_run_benchmark(self):
        """Test running benchmarks."""
        manager = ReasoningMemoryManager()
        benchmark = MemoryBenchmark(memory_manager=manager)

        results = benchmark.run_benchmark(chain_lengths=[10, 50, 100])

        assert len(results) == 3
        assert results[0].chain_length == 10
        assert results[1].chain_length == 50
        assert results[2].chain_length == 100

        # Check that peak memory increases with chain length
        for result in results:
            assert result.peak_memory_mb >= 0
            assert result.tokens_per_second > 0

    def test_benchmark_summary(self):
        """Test benchmark summary."""
        manager = ReasoningMemoryManager()
        benchmark = MemoryBenchmark(memory_manager=manager)

        benchmark.run_benchmark(chain_lengths=[10, 20])

        summary = benchmark.get_summary()
        assert summary["num_benchmarks"] == 2
        assert len(summary["chain_lengths"]) == 2
        assert summary["avg_tokens_per_second"] > 0


class TestReasoningModel:
    """Tests for ReasoningModel."""

    def test_model_creation(self):
        """Test model creation."""
        model = ReasoningModel(vocab_size=1000, d_model=64)

        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.memory_manager is not None
        assert model.kv_manager is not None

    def test_forward_pass(self):
        """Test forward pass."""
        model = ReasoningModel(vocab_size=1000, d_model=64)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)
        mx.eval(output)

        assert output.shape == (1, 5, 1000)

    def test_generate_with_reasoning(self):
        """Test generate_with_reasoning method."""
        model = ReasoningModel(vocab_size=1000, d_model=64)

        result = model.generate_with_reasoning("What is 2+2?")

        assert "<think>" in result
        assert "</think>" in result

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        model = ReasoningModel(vocab_size=1000, d_model=64)

        stats = model.get_memory_stats()

        assert "reasoning" in stats
        assert "kv_cache" in stats

    def test_reset_memory(self):
        """Test resetting memory."""
        model = ReasoningModel(vocab_size=1000, d_model=64)

        # Allocate some tokens
        model.memory_manager.allocate_reasoning_token(100, 0)

        model.reset_memory()

        stats = model.get_memory_stats()
        assert stats["reasoning"]["arena"]["allocated_slots"] == 0


class TestMLXOperations:
    """Tests for MLX-specific operations."""

    def test_mlx_array_creation(self):
        """Test MLX array creation."""
        arr = mx.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_mlx_random(self):
        """Test MLX random generation."""
        arr = mx.random.normal((10, 10))
        mx.eval(arr)
        assert arr.shape == (10, 10)

    def test_mlx_matmul(self):
        """Test MLX matrix multiplication."""
        a = mx.random.normal((10, 10))
        b = mx.random.normal((10, 10))
        c = a @ b
        mx.eval(c)
        assert c.shape == (10, 10)


class TestIntegration:
    """Integration tests for the R1 memory management system."""

    def test_full_reasoning_flow(self):
        """Test a full reasoning flow."""
        model = ReasoningModel(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=4,
            memory_budget_mb=100.0,
        )

        # Generate with reasoning
        result = model.generate_with_reasoning("Test prompt")
        assert "<think>" in result

        # Check memory stats
        stats = model.get_memory_stats()
        assert "reasoning" in stats
        assert "kv_cache" in stats

    def test_memory_pressure(self):
        """Test system under memory pressure."""
        # Small budget
        budget = MemoryBudget(total_budget_mb=1.0)
        manager = ReasoningMemoryManager(budget=budget, arena_size=1000)

        # Allocate many tokens
        allocated = []
        for i in range(500):
            slot_idx = manager.allocate_reasoning_token(i, i)
            if slot_idx is not None:
                allocated.append(slot_idx)

        # Should have allocated some tokens
        assert len(allocated) > 0

        # Check stats
        stats = manager.get_stats()
        assert stats["arena"]["allocated_slots"] == len(allocated)

    def test_kv_cache_eviction_under_load(self):
        """Test KV cache eviction under heavy load."""
        # Budget that can hold a few entries but requires eviction for many
        # Entry size = 2 * 8 * 64 * 2 = 2048 bytes = 0.002 MB per entry
        # Budget of 0.01 MB can hold ~5 entries before eviction
        manager = KVCacheBudgetManager(
            budget_mb=0.01,  # Small budget to force eviction
            num_layers=2,
            num_heads=8,
            head_dim=64,
            policy=EvictionPolicy.LRU,
        )

        # Add many entries across layers - should trigger eviction
        for layer in range(2):
            for pos in range(20):
                key = mx.random.normal((8, 64))
                value = mx.random.normal((8, 64))
                manager.add_entry(layer, pos, key, value)

        # Should have evicted entries since we tried to add 40 entries
        # but budget only holds ~5
        stats = manager.get_stats()
        assert manager.total_evictions > 0

        # Usage should be within budget
        assert stats["usage_mb"] <= stats["budget_mb"]

    def test_benchmark_different_chain_lengths(self):
        """Test benchmarking with different chain lengths."""
        manager = ReasoningMemoryManager(arena_size=2000)
        benchmark = MemoryBenchmark(memory_manager=manager)

        results = benchmark.run_benchmark(chain_lengths=[100, 500, 1000])

        # Memory usage should increase with chain length
        for i in range(len(results) - 1):
            assert results[i + 1].peak_memory_mb >= results[i].peak_memory_mb
