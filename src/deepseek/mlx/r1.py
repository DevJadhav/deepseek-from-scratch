"""
R1 Reasoning Model - MLX Implementation

This module provides comprehensive memory management for R1 reasoning tokens
including arena allocation, KV cache budget management, streaming generation,
and memory benchmarking optimized for Apple Silicon.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Iterator, Tuple, Dict, Any, Callable
import time

import mlx.core as mx
import mlx.nn as nn


# =============================================================================
# Constants
# =============================================================================

# Default token IDs for think markers (should be set from tokenizer)
DEFAULT_THINK_START_TOKEN: int = 32000  # <think>
DEFAULT_THINK_END_TOKEN: int = 32001    # </think>

# Memory budget defaults
DEFAULT_MEMORY_BUDGET_MB: float = 1024.0  # 1GB default
DEFAULT_KV_BUDGET_MB: float = 512.0
DEFAULT_ARENA_SIZE: int = 8192  # Maximum reasoning tokens

# Timeout defaults
DEFAULT_TIMEOUT_SECONDS: float = 60.0
DEFAULT_MAX_REASONING_TOKENS: int = 4096


class EvictionPolicy(Enum):
    """Policy for evicting KV cache entries when memory budget exceeded."""
    
    FIFO = auto()           # First-in-first-out
    LRU = auto()            # Least recently used
    ATTENTION_SCORE = auto() # Lowest attention scores
    SLIDING_WINDOW = auto()  # Fixed window sliding eviction


@dataclass
class ReasoningTokenSlot:
    """A slot in the reasoning token arena."""
    
    token_id: int
    position: int
    timestamp: float
    is_allocated: bool = True
    attention_score: float = 0.0
    
    def free(self) -> None:
        """Mark slot as free."""
        self.is_allocated = False
        self.token_id = 0
        self.position = -1


@dataclass
class ArenaStats:
    """Statistics for arena allocator."""
    
    total_slots: int
    allocated_slots: int
    free_slots: int
    fragmentation_ratio: float
    peak_usage: int
    total_allocations: int
    total_frees: int


class ArenaAllocator:
    """
    Arena allocator for reasoning tokens.
    
    Pre-allocates a fixed pool of token slots to avoid
    allocation overhead during reasoning generation.
    """
    
    def __init__(self, max_tokens: int = DEFAULT_ARENA_SIZE):
        self.max_tokens = max_tokens
        self.slots: List[ReasoningTokenSlot] = []
        self.free_list: List[int] = list(range(max_tokens))
        self.allocated_count = 0
        self.peak_usage = 0
        self.total_allocations = 0
        self.total_frees = 0
        
        # Pre-allocate slots
        for i in range(max_tokens):
            self.slots.append(ReasoningTokenSlot(
                token_id=0,
                position=-1,
                timestamp=0.0,
                is_allocated=False
            ))
    
    def allocate(self, token_id: int, position: int) -> Optional[int]:
        """
        Allocate a slot for a reasoning token.
        
        Returns slot index or None if arena is full.
        """
        if not self.free_list:
            return None
        
        slot_idx = self.free_list.pop()
        slot = self.slots[slot_idx]
        slot.token_id = token_id
        slot.position = position
        slot.timestamp = time.time()
        slot.is_allocated = True
        slot.attention_score = 0.0
        
        self.allocated_count += 1
        self.total_allocations += 1
        self.peak_usage = max(self.peak_usage, self.allocated_count)
        
        return slot_idx
    
    def free(self, slot_idx: int) -> bool:
        """
        Free a slot back to the arena.
        
        Returns True if successful.
        """
        if slot_idx < 0 or slot_idx >= self.max_tokens:
            return False
        
        slot = self.slots[slot_idx]
        if not slot.is_allocated:
            return False
        
        slot.free()
        self.free_list.append(slot_idx)
        self.allocated_count -= 1
        self.total_frees += 1
        
        return True
    
    def get_slot(self, slot_idx: int) -> Optional[ReasoningTokenSlot]:
        """Get a slot by index."""
        if slot_idx < 0 or slot_idx >= self.max_tokens:
            return None
        return self.slots[slot_idx]
    
    def defragment(self) -> int:
        """
        Defragment the arena by compacting allocated slots.
        
        Returns number of slots moved.
        """
        allocated_slots: List[Tuple[int, ReasoningTokenSlot]] = []
        
        for i, slot in enumerate(self.slots):
            if slot.is_allocated:
                allocated_slots.append((i, slot))
        
        # Sort by position
        allocated_slots.sort(key=lambda x: x[1].position)
        
        moves = 0
        new_free_list: List[int] = []
        
        for new_idx, (old_idx, slot) in enumerate(allocated_slots):
            if new_idx != old_idx:
                # Move slot
                self.slots[new_idx] = ReasoningTokenSlot(
                    token_id=slot.token_id,
                    position=slot.position,
                    timestamp=slot.timestamp,
                    is_allocated=True,
                    attention_score=slot.attention_score
                )
                moves += 1
        
        # Clear remaining slots and rebuild free list
        for i in range(len(allocated_slots), self.max_tokens):
            self.slots[i] = ReasoningTokenSlot(
                token_id=0,
                position=-1,
                timestamp=0.0,
                is_allocated=False
            )
            new_free_list.append(i)
        
        self.free_list = new_free_list
        
        return moves
    
    def get_stats(self) -> ArenaStats:
        """Get arena statistics."""
        free_count = len(self.free_list)
        fragmentation = 0.0
        
        if self.allocated_count > 0:
            # Calculate fragmentation as holes in allocated region
            allocated_positions = [
                i for i, s in enumerate(self.slots) if s.is_allocated
            ]
            if allocated_positions:
                span = max(allocated_positions) - min(allocated_positions) + 1
                fragmentation = 1.0 - (self.allocated_count / span) if span > 0 else 0.0
        
        return ArenaStats(
            total_slots=self.max_tokens,
            allocated_slots=self.allocated_count,
            free_slots=free_count,
            fragmentation_ratio=fragmentation,
            peak_usage=self.peak_usage,
            total_allocations=self.total_allocations,
            total_frees=self.total_frees
        )
    
    def reset(self) -> None:
        """Reset the arena to initial state."""
        for slot in self.slots:
            slot.free()
        self.free_list = list(range(self.max_tokens))
        self.allocated_count = 0


@dataclass 
class MemoryBudget:
    """Memory budget configuration."""
    
    total_budget_mb: float = DEFAULT_MEMORY_BUDGET_MB
    kv_cache_budget_mb: float = DEFAULT_KV_BUDGET_MB
    reasoning_budget_mb: float = 256.0
    embedding_budget_mb: float = 256.0
    
    def __post_init__(self) -> None:
        component_total = (
            self.kv_cache_budget_mb + 
            self.reasoning_budget_mb + 
            self.embedding_budget_mb
        )
        if component_total > self.total_budget_mb:
            # Scale down proportionally
            scale = self.total_budget_mb / component_total
            self.kv_cache_budget_mb *= scale
            self.reasoning_budget_mb *= scale
            self.embedding_budget_mb *= scale


@dataclass
class MemoryUsage:
    """Current memory usage statistics."""
    
    kv_cache_mb: float = 0.0
    reasoning_tokens_mb: float = 0.0
    embeddings_mb: float = 0.0
    total_mb: float = 0.0
    peak_mb: float = 0.0


class ReasoningMemoryManager:
    """
    Manages memory allocation for R1 reasoning generation on MLX.
    
    Features:
    - Arena allocation for reasoning tokens
    - KV cache budget management
    - Dynamic memory tracking (MLX unified memory)
    """
    
    def __init__(
        self,
        budget: Optional[MemoryBudget] = None,
        arena_size: int = DEFAULT_ARENA_SIZE,
        bytes_per_token: int = 4
    ):
        self.budget = budget or MemoryBudget()
        self.arena = ArenaAllocator(arena_size)
        self.bytes_per_token = bytes_per_token
        
        self.usage = MemoryUsage()
        self.kv_cache_entries: Dict[int, float] = {}  # position -> timestamp
        
    def allocate_reasoning_token(self, token_id: int, position: int) -> Optional[int]:
        """Allocate space for a reasoning token."""
        slot_idx = self.arena.allocate(token_id, position)
        if slot_idx is not None:
            self._update_reasoning_memory()
        return slot_idx
    
    def free_reasoning_token(self, slot_idx: int) -> bool:
        """Free a reasoning token slot."""
        result = self.arena.free(slot_idx)
        if result:
            self._update_reasoning_memory()
        return result
    
    def _update_reasoning_memory(self) -> None:
        """Update reasoning memory usage."""
        stats = self.arena.get_stats()
        self.usage.reasoning_tokens_mb = (
            stats.allocated_slots * self.bytes_per_token
        ) / (1024 * 1024)
        self._update_total()
    
    def _update_total(self) -> None:
        """Update total memory usage."""
        self.usage.total_mb = (
            self.usage.kv_cache_mb +
            self.usage.reasoning_tokens_mb +
            self.usage.embeddings_mb
        )
        self.usage.peak_mb = max(self.usage.peak_mb, self.usage.total_mb)
    
    def track_kv_cache(self, num_entries: int, entry_size_bytes: int) -> None:
        """Track KV cache memory usage."""
        self.usage.kv_cache_mb = (num_entries * entry_size_bytes) / (1024 * 1024)
        self._update_total()
    
    def track_embeddings(self, size_bytes: int) -> None:
        """Track embedding memory usage."""
        self.usage.embeddings_mb = size_bytes / (1024 * 1024)
        self._update_total()
    
    def is_within_budget(self) -> bool:
        """Check if current usage is within budget."""
        return self.usage.total_mb <= self.budget.total_budget_mb
    
    def get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        return max(0.0, self.budget.total_budget_mb - self.usage.total_mb)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        arena_stats = self.arena.get_stats()
        return {
            "usage": {
                "kv_cache_mb": self.usage.kv_cache_mb,
                "reasoning_tokens_mb": self.usage.reasoning_tokens_mb,
                "embeddings_mb": self.usage.embeddings_mb,
                "total_mb": self.usage.total_mb,
                "peak_mb": self.usage.peak_mb,
            },
            "budget": {
                "total_mb": self.budget.total_budget_mb,
                "kv_cache_mb": self.budget.kv_cache_budget_mb,
                "reasoning_mb": self.budget.reasoning_budget_mb,
                "available_mb": self.get_available_memory_mb(),
            },
            "arena": {
                "total_slots": arena_stats.total_slots,
                "allocated_slots": arena_stats.allocated_slots,
                "free_slots": arena_stats.free_slots,
                "fragmentation": arena_stats.fragmentation_ratio,
                "peak_usage": arena_stats.peak_usage,
            }
        }
    
    def reset(self) -> None:
        """Reset memory manager state."""
        self.arena.reset()
        self.usage = MemoryUsage()
        self.kv_cache_entries.clear()


@dataclass
class KVCacheEntry:
    """An entry in the KV cache."""
    
    position: int
    key: mx.array
    value: mx.array
    timestamp: float
    access_count: int = 0
    attention_score: float = 0.0


class KVCacheBudgetManager:
    """
    Manages KV cache memory with dynamic eviction.
    
    Supports multiple eviction policies for managing memory
    when budget is exceeded during long reasoning chains.
    """
    
    def __init__(
        self,
        budget_mb: float = DEFAULT_KV_BUDGET_MB,
        policy: EvictionPolicy = EvictionPolicy.LRU,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64
    ):
        self.budget_mb = budget_mb
        self.policy = policy
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Per-layer KV cache
        self.cache: Dict[int, List[KVCacheEntry]] = {
            i: [] for i in range(num_layers)
        }
        
        self.current_usage_bytes = 0
        self.total_evictions = 0
        
    def _entry_size_bytes(self) -> int:
        """Calculate size of one KV cache entry in bytes."""
        # key + value, both [num_heads, head_dim], float16 = 2 bytes
        return 2 * self.num_heads * self.head_dim * 2
    
    def _current_usage_mb(self) -> float:
        """Get current KV cache usage in MB."""
        return self.current_usage_bytes / (1024 * 1024)
    
    def add_entry(
        self,
        layer_idx: int,
        position: int,
        key: mx.array,
        value: mx.array,
        attention_score: float = 0.0
    ) -> bool:
        """
        Add a KV cache entry, evicting if necessary.
        
        Returns True if entry was added successfully.
        """
        entry_size = self._entry_size_bytes()
        
        # Check if we need to evict
        while (self.current_usage_bytes + entry_size) / (1024 * 1024) > self.budget_mb:
            if not self._evict_one():
                return False  # Cannot evict, budget exceeded
        
        entry = KVCacheEntry(
            position=position,
            key=key,
            value=value,
            timestamp=time.time(),
            attention_score=attention_score
        )
        
        self.cache[layer_idx].append(entry)
        self.current_usage_bytes += entry_size
        
        return True
    
    def _evict_one(self) -> bool:
        """
        Evict one entry according to policy.
        
        Returns True if an entry was evicted.
        """
        # Find layer with entries to evict
        candidate_layer = -1
        candidate_idx = -1
        candidate_score = float('inf')
        
        for layer_idx, entries in self.cache.items():
            if not entries:
                continue
                
            for entry_idx, entry in enumerate(entries):
                score = self._get_eviction_score(entry)
                if score < candidate_score:
                    candidate_score = score
                    candidate_layer = layer_idx
                    candidate_idx = entry_idx
        
        if candidate_layer < 0:
            return False
        
        # Evict the entry
        del self.cache[candidate_layer][candidate_idx]
        self.current_usage_bytes -= self._entry_size_bytes()
        self.total_evictions += 1
        
        return True
    
    def _get_eviction_score(self, entry: KVCacheEntry) -> float:
        """
        Get eviction score for an entry.
        Lower score = more likely to be evicted.
        """
        if self.policy == EvictionPolicy.FIFO:
            return entry.timestamp  # Earlier = lower score = evict first
        elif self.policy == EvictionPolicy.LRU:
            return entry.timestamp  # Least recent = lower score
        elif self.policy == EvictionPolicy.ATTENTION_SCORE:
            return entry.attention_score  # Lower attention = evict first
        elif self.policy == EvictionPolicy.SLIDING_WINDOW:
            return entry.position  # Earlier position = evict first
        else:
            return entry.timestamp
    
    def update_attention_scores(
        self,
        layer_idx: int,
        scores: Dict[int, float]
    ) -> None:
        """Update attention scores for entries in a layer."""
        for entry in self.cache[layer_idx]:
            if entry.position in scores:
                entry.attention_score = scores[entry.position]
                entry.timestamp = time.time()  # Update LRU timestamp
                entry.access_count += 1
    
    def get_cache_for_layer(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """
        Get concatenated KV cache for a layer.
        
        Returns (keys, values) tensors.
        """
        entries = self.cache[layer_idx]
        if not entries:
            # Return empty tensors
            return (
                mx.zeros((0, self.num_heads, self.head_dim)),
                mx.zeros((0, self.num_heads, self.head_dim))
            )
        
        # Sort by position
        entries.sort(key=lambda e: e.position)
        
        keys = mx.stack([e.key for e in entries])
        values = mx.stack([e.value for e in entries])
        
        return keys, values
    
    def get_stats(self) -> Dict[str, Any]:
        """Get KV cache statistics."""
        total_entries = sum(len(entries) for entries in self.cache.values())
        
        return {
            "budget_mb": self.budget_mb,
            "usage_mb": self._current_usage_mb(),
            "utilization": self._current_usage_mb() / self.budget_mb if self.budget_mb > 0 else 0,
            "total_entries": total_entries,
            "total_evictions": self.total_evictions,
            "policy": self.policy.name,
            "entries_per_layer": {
                layer: len(entries) for layer, entries in self.cache.items()
            }
        }
    
    def clear(self) -> None:
        """Clear all KV cache entries."""
        for layer_idx in self.cache:
            self.cache[layer_idx] = []
        self.current_usage_bytes = 0


@dataclass
class ThinkTokenConfig:
    """Configuration for think token detection."""
    
    think_start_token: int = DEFAULT_THINK_START_TOKEN
    think_end_token: int = DEFAULT_THINK_END_TOKEN
    # Multi-token patterns
    think_start_pattern: List[int] = field(default_factory=list)
    think_end_pattern: List[int] = field(default_factory=list)


class ThinkTokenDetector:
    """
    Detects <think> and </think> tokens in generation.
    
    Supports both single-token and multi-token patterns.
    """
    
    def __init__(self, config: Optional[ThinkTokenConfig] = None):
        self.config = config or ThinkTokenConfig()
        self.in_think_block = False
        self.token_buffer: List[int] = []
        self.think_depth = 0  # Support nested think blocks
        
    def process_token(self, token_id: int) -> Tuple[bool, bool, bool]:
        """
        Process a token and detect think boundaries.
        
        Returns (is_think_start, is_think_end, in_think_block).
        """
        is_start = False
        is_end = False
        
        # Check single-token patterns
        if token_id == self.config.think_start_token:
            is_start = True
            self.in_think_block = True
            self.think_depth += 1
        elif token_id == self.config.think_end_token:
            is_end = True
            self.think_depth = max(0, self.think_depth - 1)
            if self.think_depth == 0:
                self.in_think_block = False
        
        # Check multi-token patterns
        if not is_start and not is_end:
            self.token_buffer.append(token_id)
            # Limit buffer size
            max_pattern_len = max(
                len(self.config.think_start_pattern),
                len(self.config.think_end_pattern),
                1
            )
            if len(self.token_buffer) > max_pattern_len:
                self.token_buffer = self.token_buffer[-max_pattern_len:]
            
            # Check for pattern match
            if (self.config.think_start_pattern and 
                self.token_buffer[-len(self.config.think_start_pattern):] == 
                self.config.think_start_pattern):
                is_start = True
                self.in_think_block = True
                self.think_depth += 1
            elif (self.config.think_end_pattern and
                  self.token_buffer[-len(self.config.think_end_pattern):] ==
                  self.config.think_end_pattern):
                is_end = True
                self.think_depth = max(0, self.think_depth - 1)
                if self.think_depth == 0:
                    self.in_think_block = False
        
        return is_start, is_end, self.in_think_block
    
    def reset(self) -> None:
        """Reset detector state."""
        self.in_think_block = False
        self.token_buffer.clear()
        self.think_depth = 0


@dataclass
class StreamingConfig:
    """Configuration for streaming reasoning generation."""
    
    max_reasoning_tokens: int = DEFAULT_MAX_REASONING_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    yield_interval: int = 1  # Yield every N tokens
    enable_timeout: bool = True
    stop_on_budget_exceeded: bool = True


@dataclass
class StreamingState:
    """State for streaming generation."""
    
    tokens_generated: int = 0
    reasoning_tokens: int = 0
    in_reasoning: bool = False
    start_time: float = 0.0
    timed_out: bool = False
    budget_exceeded: bool = False
    stopped_reason: Optional[str] = None


class StreamingReasoningGenerator:
    """
    Generates reasoning tokens with streaming output and timeout.
    
    Features:
    - Token-by-token streaming
    - Configurable timeout for runaway reasoning
    - Memory budget enforcement
    - Think token boundary detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_manager: ReasoningMemoryManager,
        think_detector: ThinkTokenDetector,
        config: Optional[StreamingConfig] = None
    ):
        self.model = model
        self.memory_manager = memory_manager
        self.think_detector = think_detector
        self.config = config or StreamingConfig()
        
        self.state = StreamingState()
        
    def _check_timeout(self) -> bool:
        """Check if generation has timed out."""
        if not self.config.enable_timeout:
            return False
        elapsed = time.time() - self.state.start_time
        return elapsed > self.config.timeout_seconds
    
    def _check_budget(self) -> bool:
        """Check if memory budget is exceeded."""
        if not self.config.stop_on_budget_exceeded:
            return False
        return not self.memory_manager.is_within_budget()
    
    def generate_stream(
        self,
        input_ids: mx.array,
        sample_fn: Optional[Callable[[mx.array], int]] = None
    ) -> Iterator[Tuple[int, StreamingState]]:
        """
        Generate tokens with streaming output.
        
        Yields (token_id, state) tuples.
        """
        self.state = StreamingState(start_time=time.time())
        self.think_detector.reset()
        
        # Default sampling: argmax
        if sample_fn is None:
            def sample_fn(logits: mx.array) -> int:
                return int(mx.argmax(logits[-1]).item())
        
        current_ids = input_ids
        
        while True:
            # Check stopping conditions
            if self._check_timeout():
                self.state.timed_out = True
                self.state.stopped_reason = "timeout"
                break
            
            if self._check_budget():
                self.state.budget_exceeded = True
                self.state.stopped_reason = "budget_exceeded"
                break
            
            if self.state.reasoning_tokens >= self.config.max_reasoning_tokens:
                self.state.stopped_reason = "max_tokens"
                break
            
            # Generate next token
            logits = self.model(current_ids)
            mx.eval(logits)  # Force evaluation
            
            next_token = sample_fn(logits)
            
            # Detect think boundaries
            is_start, is_end, in_think = self.think_detector.process_token(next_token)
            
            self.state.in_reasoning = in_think
            self.state.tokens_generated += 1
            
            if in_think or is_start:
                self.state.reasoning_tokens += 1
                # Allocate in arena
                self.memory_manager.allocate_reasoning_token(
                    next_token, 
                    self.state.reasoning_tokens
                )
            
            # Yield token and state
            if self.state.tokens_generated % self.config.yield_interval == 0:
                yield next_token, StreamingState(
                    tokens_generated=self.state.tokens_generated,
                    reasoning_tokens=self.state.reasoning_tokens,
                    in_reasoning=self.state.in_reasoning,
                    start_time=self.state.start_time,
                    timed_out=self.state.timed_out,
                    budget_exceeded=self.state.budget_exceeded
                )
            
            # Check for end of reasoning
            if is_end and not in_think:
                self.state.stopped_reason = "think_end"
                break
            
            # Update input for next iteration
            current_ids = mx.array([[next_token]])
        
        # Final yield
        yield -1, self.state  # -1 indicates end of stream
    
    def get_state(self) -> StreamingState:
        """Get current streaming state."""
        return self.state


@dataclass
class BenchmarkResult:
    """Results from memory benchmark."""
    
    chain_length: int
    peak_memory_mb: float
    final_memory_mb: float
    duration_seconds: float
    tokens_per_second: float
    arena_stats: ArenaStats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_length": self.chain_length,
            "peak_memory_mb": self.peak_memory_mb,
            "final_memory_mb": self.final_memory_mb,
            "duration_seconds": self.duration_seconds,
            "tokens_per_second": self.tokens_per_second,
            "arena": {
                "total_slots": self.arena_stats.total_slots,
                "allocated_slots": self.arena_stats.allocated_slots,
                "peak_usage": self.arena_stats.peak_usage,
                "fragmentation": self.arena_stats.fragmentation_ratio,
            }
        }


class MemoryBenchmark:
    """
    Benchmarks memory usage for varying reasoning chain lengths.
    
    Measures:
    - Peak memory usage
    - Allocation/deallocation overhead
    - Fragmentation over time
    - Tokens per second throughput
    """
    
    def __init__(self, memory_manager: ReasoningMemoryManager):
        self.memory_manager = memory_manager
        self.results: List[BenchmarkResult] = []
        
    def run_benchmark(
        self,
        chain_lengths: List[int],
        bytes_per_token: int = 4
    ) -> List[BenchmarkResult]:
        """
        Run benchmark for multiple chain lengths.
        
        Returns list of benchmark results.
        """
        self.results.clear()
        
        for length in chain_lengths:
            result = self._benchmark_chain_length(length, bytes_per_token)
            self.results.append(result)
        
        return self.results
    
    def _benchmark_chain_length(
        self,
        length: int,
        bytes_per_token: int
    ) -> BenchmarkResult:
        """Benchmark a single chain length."""
        self.memory_manager.reset()
        
        start_time = time.time()
        
        # Allocate tokens
        for i in range(length):
            token_id = i % 32000  # Simulate token IDs
            self.memory_manager.allocate_reasoning_token(token_id, i)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get stats
        stats = self.memory_manager.get_stats()
        arena_stats = self.memory_manager.arena.get_stats()
        
        return BenchmarkResult(
            chain_length=length,
            peak_memory_mb=stats["usage"]["peak_mb"],
            final_memory_mb=stats["usage"]["total_mb"],
            duration_seconds=duration,
            tokens_per_second=length / duration if duration > 0 else 0,
            arena_stats=arena_stats
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if not self.results:
            return {}
        
        return {
            "num_benchmarks": len(self.results),
            "chain_lengths": [r.chain_length for r in self.results],
            "peak_memories_mb": [r.peak_memory_mb for r in self.results],
            "avg_tokens_per_second": sum(
                r.tokens_per_second for r in self.results
            ) / len(self.results),
            "results": [r.to_dict() for r in self.results]
        }


class ReasoningModel(nn.Module):
    """
    DeepSeek-R1 style reasoning model with memory management.
    
    Features:
    - Arena-based memory allocation for reasoning tokens
    - KV cache budget management
    - Streaming generation with timeout
    - <think> token boundary detection
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 12,
        num_heads: int = 8,
        memory_budget_mb: float = DEFAULT_MEMORY_BUDGET_MB
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Memory management
        self.memory_manager = ReasoningMemoryManager(
            budget=MemoryBudget(total_budget_mb=memory_budget_mb)
        )
        
        # KV cache manager
        self.kv_manager = KVCacheBudgetManager(
            budget_mb=memory_budget_mb / 2,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=d_model // num_heads
        )
        
        # Think token detector
        self.think_detector = ThinkTokenDetector()
        
    def generate_with_reasoning(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> str:
        """Simulate generation with reasoning trace."""
        reasoning_trace = (
            f"<think>\n"
            f"The user is asking about {prompt}. \n"
            f"1. I need to identify the core question.\n"
            f"2. I should recall relevant information about DeepSeek-R1.\n"
            f"3. I need to formulate a clear and concise answer.\n"
            f"</think>"
        )
        
        final_answer = (
            "\nHere is the answer based on my reasoning:\n"
            "DeepSeek-R1 is a reasoning model that uses Reinforcement Learning to generate "
            "chain-of-thought traces before answering. This improves performance on complex tasks."
        )
        
        return reasoning_trace + final_answer
    
    def stream_generate(
        self,
        input_ids: mx.array,
        config: Optional[StreamingConfig] = None
    ) -> Iterator[Tuple[int, StreamingState]]:
        """Stream generate tokens with reasoning."""
        generator = StreamingReasoningGenerator(
            model=self,
            memory_manager=self.memory_manager,
            think_detector=self.think_detector,
            config=config
        )
        return generator.generate_stream(input_ids)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get combined memory statistics."""
        return {
            "reasoning": self.memory_manager.get_stats(),
            "kv_cache": self.kv_manager.get_stats()
        }
    
    def reset_memory(self) -> None:
        """Reset all memory managers."""
        self.memory_manager.reset()
        self.kv_manager.clear()
        self.think_detector.reset()
        
    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass through embedding and LM head."""
        x = self.embed(input_ids)
        return self.lm_head(x)
