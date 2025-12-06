"""
DeepSeek-R1 Reasoning Model Implementation - PyTorch

R1 introduces a "Reasoning" phase where the model generates a "Chain of Thought" (CoT)
enclosed in <think> and </think> tags before producing the final answer.
This is trained via Reinforcement Learning (GRPO) to incentivize reasoning.

Memory Management:
- ReasoningMemoryManager: Handles dynamic CoT token allocation
- KVCacheBudget: Manages KV cache memory with configurable eviction policies
- ReasoningTokenArena: Arena-style allocator for efficient token buffer management
"""

import torch
import torch.nn as nn
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum, auto
from collections import deque
import time

# Enable MPS fallback for operations not supported on Metal
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> torch.device:
    """Get best available device with fallback support."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Eviction Policy for KV Cache Management
# ============================================================================

class EvictionPolicy(Enum):
    """Policy for evicting tokens from KV cache when budget is exceeded."""
    FIFO = auto()            # First-In-First-Out: evict oldest tokens first
    LRU = auto()             # Least Recently Used: evict least recently accessed tokens
    ATTENTION_SCORE = auto() # Attention Score based: evict tokens with lowest cumulative attention
    SLIDING_WINDOW = auto()  # Sliding Window: keep only the most recent N tokens


# ============================================================================
# Reasoning Token
# ============================================================================

@dataclass
class ReasoningToken:
    """A single reasoning token with metadata for cache management."""
    token_id: int
    position: int
    is_reasoning: bool
    attention_score: float = 0.0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    
    def update_attention(self, score: float) -> None:
        """Update attention score (for AttentionScore eviction policy)."""
        self.attention_score += score
        self.last_access = time.time()
    
    def touch(self) -> None:
        """Mark token as accessed (for LRU eviction policy)."""
        self.last_access = time.time()


# ============================================================================
# Reasoning Token Arena
# ============================================================================

class ReasoningTokenArena:
    """
    Arena-style allocator for reasoning tokens.
    
    Provides efficient allocation and deallocation of reasoning token buffers
    with automatic cleanup when reasoning chains are completed.
    """
    
    def __init__(self, capacity: int):
        self.buffer: List[Optional[ReasoningToken]] = [None] * capacity
        self.capacity = capacity
        self._len = 0
        self.free_list: deque = deque()
    
    def allocate(self, token_id: int, position: int, is_reasoning: bool) -> Optional[int]:
        """Allocate a new token in the arena. Returns slot index or None if full."""
        if self.free_list:
            # Reuse freed slot
            slot = self.free_list.popleft()
            self.buffer[slot] = ReasoningToken(
                token_id=token_id,
                position=position,
                is_reasoning=is_reasoning
            )
            return slot
        elif self._len < self.capacity:
            # Allocate new slot
            slot = self._len
            self.buffer[slot] = ReasoningToken(
                token_id=token_id,
                position=position,
                is_reasoning=is_reasoning
            )
            self._len += 1
            return slot
        else:
            return None  # Arena full
    
    def free(self, slot: int) -> None:
        """Free a token slot for reuse."""
        if 0 <= slot < self._len:
            self.free_list.append(slot)
    
    def get(self, slot: int) -> Optional[ReasoningToken]:
        """Get a reference to a token."""
        if 0 <= slot < self._len:
            return self.buffer[slot]
        return None
    
    def reset(self) -> None:
        """Reset the arena (free all tokens)."""
        self.buffer = [None] * self.capacity
        self._len = 0
        self.free_list.clear()
    
    def usage(self) -> int:
        """Get current usage."""
        return self._len - len(self.free_list)
    
    def is_full(self) -> bool:
        """Check if arena is full."""
        return self.usage() >= self.capacity


# ============================================================================
# KV Cache Budget Manager
# ============================================================================

@dataclass
class KVCacheStats:
    """Statistics for KV cache usage."""
    total_budget: int
    reasoning_budget: int
    context_budget: int
    reasoning_used: int
    context_used: int
    reasoning_utilization: float
    context_utilization: float


class KVCacheBudget:
    """Manages KV cache memory budget for reasoning tokens."""
    
    def __init__(
        self,
        total_budget: int,
        reasoning_ratio: float = 0.7,
        eviction_policy: EvictionPolicy = EvictionPolicy.SLIDING_WINDOW
    ):
        self.total_budget = total_budget
        self.reasoning_budget = int(total_budget * reasoning_ratio)
        self.context_budget = total_budget - self.reasoning_budget
        self.reasoning_count = 0
        self.context_count = 0
        self.eviction_policy = eviction_policy
    
    def can_add_reasoning(self) -> bool:
        """Check if we can add a reasoning token."""
        return self.reasoning_count < self.reasoning_budget
    
    def can_add_context(self) -> bool:
        """Check if we can add a context token."""
        return self.context_count < self.context_budget
    
    def add_reasoning(self) -> bool:
        """Add a reasoning token to the budget."""
        if self.can_add_reasoning():
            self.reasoning_count += 1
            return True
        return False
    
    def add_context(self) -> bool:
        """Add a context token to the budget."""
        if self.can_add_context():
            self.context_count += 1
            return True
        return False
    
    def remove_reasoning(self) -> None:
        """Remove a reasoning token from the budget."""
        self.reasoning_count = max(0, self.reasoning_count - 1)
    
    def remove_context(self) -> None:
        """Remove a context token from the budget."""
        self.context_count = max(0, self.context_count - 1)
    
    def tokens_to_evict(self, new_tokens: int) -> int:
        """Get number of tokens to evict to fit new reasoning tokens."""
        total_after = self.reasoning_count + new_tokens
        return max(0, total_after - self.reasoning_budget)
    
    def stats(self) -> KVCacheStats:
        """Get statistics."""
        return KVCacheStats(
            total_budget=self.total_budget,
            reasoning_budget=self.reasoning_budget,
            context_budget=self.context_budget,
            reasoning_used=self.reasoning_count,
            context_used=self.context_count,
            reasoning_utilization=self.reasoning_count / max(1, self.reasoning_budget),
            context_utilization=self.context_count / max(1, self.context_budget),
        )
    
    def reset(self) -> None:
        """Reset all counts."""
        self.reasoning_count = 0
        self.context_count = 0


# ============================================================================
# Reasoning Memory Configuration
# ============================================================================

@dataclass
class ReasoningMemoryConfig:
    """Configuration for the ReasoningMemoryManager."""
    max_reasoning_tokens: int = 32768       # 32K reasoning tokens
    kv_cache_budget_ratio: float = 0.7      # 70% for reasoning, 30% for context
    eviction_policy: EvictionPolicy = EvictionPolicy.SLIDING_WINDOW
    total_kv_budget: int = 131072           # 128K total KV cache
    reasoning_timeout_secs: float = 300.0   # 5 minute timeout
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"


# ============================================================================
# Reasoning Memory Statistics
# ============================================================================

@dataclass
class ReasoningMemoryStats:
    """Statistics for reasoning memory usage."""
    arena_capacity: int
    arena_used: int
    kv_cache_stats: KVCacheStats
    active_tokens: int
    tokens_generated: int
    tokens_evicted: int
    in_reasoning: bool
    session_duration: Optional[float]


# ============================================================================
# Reasoning Memory Manager
# ============================================================================

class ReasoningMemoryManager:
    """
    Main memory manager for R1 reasoning tokens.
    
    Handles:
    - Arena allocation for reasoning tokens
    - KV cache budget management
    - Streaming token generation
    - Timeout mechanism for runaway reasoning
    """
    
    def __init__(self, config: Optional[ReasoningMemoryConfig] = None):
        self.config = config or ReasoningMemoryConfig()
        self.arena = ReasoningTokenArena(self.config.max_reasoning_tokens)
        self.kv_budget = KVCacheBudget(
            self.config.total_kv_budget,
            self.config.kv_cache_budget_ratio,
            self.config.eviction_policy
        )
        self.active_slots: deque = deque()
        self.in_reasoning = False
        self.session_start: Optional[float] = None
        self.tokens_generated = 0
        self.evicted_count = 0
    
    def start_session(self) -> None:
        """Start a new reasoning session."""
        self.session_start = time.time()
        self.tokens_generated = 0
        self.in_reasoning = False
    
    def end_session(self) -> None:
        """End the current reasoning session."""
        self.session_start = None
        self.in_reasoning = False
    
    def is_timed_out(self) -> bool:
        """Check if reasoning has timed out."""
        if self.session_start is None:
            return False
        return (time.time() - self.session_start) > self.config.reasoning_timeout_secs
    
    def allocate_token(self, token_id: int) -> Optional[int]:
        """
        Allocate a new reasoning token.
        
        Returns the slot index if successful, or None if allocation failed.
        """
        # Check timeout
        if self.is_timed_out():
            return None
        
        # Check if we need to evict tokens
        if self.in_reasoning and not self.kv_budget.can_add_reasoning():
            self.evict_tokens(1)
        
        # Allocate in arena
        position = self.tokens_generated
        slot = self.arena.allocate(token_id, position, self.in_reasoning)
        
        if slot is not None:
            # Update KV budget
            if self.in_reasoning:
                self.kv_budget.add_reasoning()
            else:
                self.kv_budget.add_context()
            
            self.active_slots.append(slot)
            self.tokens_generated += 1
            return slot
        else:
            # Arena full, try evicting
            self.evict_tokens(1)
            return self.arena.allocate(token_id, position, self.in_reasoning)
    
    def process_token_boundary(self, token_text: str) -> None:
        """Process a token to detect <think> boundaries."""
        if self.config.think_start_token in token_text:
            self.in_reasoning = True
        elif self.config.think_end_token in token_text:
            self.in_reasoning = False
    
    def evict_tokens(self, count: int) -> None:
        """Evict tokens based on the configured policy."""
        to_evict = min(count, len(self.active_slots))
        
        if self.kv_budget.eviction_policy in (EvictionPolicy.FIFO, EvictionPolicy.SLIDING_WINDOW):
            # Evict from the front (oldest)
            for _ in range(to_evict):
                if self.active_slots:
                    slot = self.active_slots.popleft()
                    token = self.arena.get(slot)
                    if token:
                        if token.is_reasoning:
                            self.kv_budget.remove_reasoning()
                        else:
                            self.kv_budget.remove_context()
                    self.arena.free(slot)
                    self.evicted_count += 1
        
        elif self.kv_budget.eviction_policy == EvictionPolicy.LRU:
            # Find and evict least recently used
            slots_with_access = [
                (slot, self.arena.get(slot).last_access if self.arena.get(slot) else float('inf'))
                for slot in self.active_slots
            ]
            slots_with_access.sort(key=lambda x: x[1])
            
            for slot, _ in slots_with_access[:to_evict]:
                self.active_slots.remove(slot)
                token = self.arena.get(slot)
                if token:
                    if token.is_reasoning:
                        self.kv_budget.remove_reasoning()
                    else:
                        self.kv_budget.remove_context()
                self.arena.free(slot)
                self.evicted_count += 1
        
        elif self.kv_budget.eviction_policy == EvictionPolicy.ATTENTION_SCORE:
            # Find and evict lowest attention score
            slots_with_score = [
                (slot, self.arena.get(slot).attention_score if self.arena.get(slot) else float('inf'))
                for slot in self.active_slots
            ]
            slots_with_score.sort(key=lambda x: x[1])
            
            for slot, _ in slots_with_score[:to_evict]:
                self.active_slots.remove(slot)
                token = self.arena.get(slot)
                if token:
                    if token.is_reasoning:
                        self.kv_budget.remove_reasoning()
                    else:
                        self.kv_budget.remove_context()
                self.arena.free(slot)
                self.evicted_count += 1
    
    def update_attention_scores(self, scores: List[float]) -> None:
        """Update attention scores for tokens."""
        for i, score in enumerate(scores):
            if i < len(self.active_slots):
                slot = list(self.active_slots)[i]
                token = self.arena.get(slot)
                if token:
                    token.update_attention(score)
    
    def get_stats(self) -> ReasoningMemoryStats:
        """Get memory statistics."""
        session_duration = None
        if self.session_start is not None:
            session_duration = time.time() - self.session_start
        
        return ReasoningMemoryStats(
            arena_capacity=self.arena.capacity,
            arena_used=self.arena.usage(),
            kv_cache_stats=self.kv_budget.stats(),
            active_tokens=len(self.active_slots),
            tokens_generated=self.tokens_generated,
            tokens_evicted=self.evicted_count,
            in_reasoning=self.in_reasoning,
            session_duration=session_duration,
        )
    
    def reset(self) -> None:
        """Reset the manager for a new generation."""
        self.arena.reset()
        self.kv_budget.reset()
        self.active_slots.clear()
        self.in_reasoning = False
        self.session_start = None
        self.tokens_generated = 0
        self.evicted_count = 0
    
    def is_in_reasoning(self) -> bool:
        """Check if currently in reasoning mode."""
        return self.in_reasoning
    
    @property
    def timeout(self) -> float:
        """Get the configured timeout."""
        return self.config.reasoning_timeout_secs


# ============================================================================
# Reasoning Model with Memory Management
# ============================================================================

class ReasoningModel(nn.Module):
    """
    DeepSeek-R1 Reasoning Model with integrated memory management.
    
    Features:
    - Dynamic CoT token allocation with arena memory
    - KV cache budget management with configurable eviction
    - Timeout mechanism for runaway reasoning
    - Memory statistics and monitoring
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        memory_config: Optional[ReasoningMemoryConfig] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.memory_manager = ReasoningMemoryManager(memory_config)
    
    def generate_with_reasoning(self, prompt: str) -> str:
        """Generate with reasoning, using memory management."""
        # Start a reasoning session
        self.memory_manager.start_session()
        
        # Simulate reasoning trace
        reasoning_trace = (
            f"<think>\n"
            f"The user is asking about {prompt}. \n"
            f"1. I need to identify the core question.\n"
            f"2. I should recall relevant information about DeepSeek-R1.\n"
            f"3. I need to formulate a clear and concise answer.\n"
            f"</think>"
        )
        
        # Process think boundaries
        self.memory_manager.process_token_boundary("<think>")
        
        # Allocate tokens for the reasoning trace (simulated)
        for i, _ in enumerate(reasoning_trace):
            if self.memory_manager.is_timed_out():
                self.memory_manager.end_session()
                return "Error: Reasoning timed out"
            
            # Allocate token (using position as simulated token_id)
            self.memory_manager.allocate_token(i)
        
        self.memory_manager.process_token_boundary("</think>")
        
        final_answer = (
            "\nHere is the answer based on my reasoning:\n"
            "DeepSeek-R1 is a reasoning model that uses Reinforcement Learning to generate "
            "chain-of-thought traces before answering. This improves performance on complex tasks."
        )
        
        # End session
        self.memory_manager.end_session()
        
        return reasoning_trace + final_answer
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.embed(input_ids)
        return self.lm_head(x)
    
    def get_memory_stats(self) -> ReasoningMemoryStats:
        """Get memory statistics."""
        return self.memory_manager.get_stats()
    
    def reset_memory(self) -> None:
        """Reset memory manager."""
        self.memory_manager.reset()
