//! DeepSeek-R1 Reasoning Model Implementation
//!
//! R1 introduces a "Reasoning" phase where the model generates a "Chain of Thought" (CoT)
//! enclosed in <think> and </think> tags before producing the final answer.
//! This is trained via Reinforcement Learning (GRPO) to incentivize reasoning.
//!
//! # Memory Management
//!
//! The `ReasoningMemoryManager` handles:
//! - Dynamic CoT token allocation with arena-style memory
//! - KV cache budget management with configurable eviction policies
//! - Streaming token generation with memory limits
//! - Timeout mechanisms for runaway reasoning chains

use candle_core::{Result, Tensor, Device};
use candle_nn::{VarBuilder, Module};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ============================================================================
// Eviction Policy for KV Cache Management
// ============================================================================

/// Policy for evicting tokens from KV cache when budget is exceeded
#[derive(Clone, Debug, PartialEq)]
pub enum EvictionPolicy {
    /// First-In-First-Out: evict oldest tokens first
    FIFO,
    /// Least Recently Used: evict least recently accessed tokens
    LRU,
    /// Attention Score based: evict tokens with lowest cumulative attention
    AttentionScore,
    /// Sliding Window: keep only the most recent N tokens
    SlidingWindow,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        EvictionPolicy::SlidingWindow
    }
}

// ============================================================================
// Reasoning Token Buffer
// ============================================================================

/// A single reasoning token with metadata for cache management
#[derive(Clone, Debug)]
pub struct ReasoningToken {
    /// Token ID
    pub token_id: u32,
    /// Position in the reasoning sequence
    pub position: usize,
    /// Whether this token is inside <think> tags
    pub is_reasoning: bool,
    /// Cumulative attention score (for AttentionScore eviction)
    pub attention_score: f32,
    /// Last access time (for LRU eviction)
    pub last_access: Instant,
    /// Creation time
    pub created_at: Instant,
}

impl ReasoningToken {
    pub fn new(token_id: u32, position: usize, is_reasoning: bool) -> Self {
        let now = Instant::now();
        Self {
            token_id,
            position,
            is_reasoning,
            attention_score: 0.0,
            last_access: now,
            created_at: now,
        }
    }
    
    /// Update attention score (for AttentionScore eviction policy)
    pub fn update_attention(&mut self, score: f32) {
        self.attention_score += score;
        self.last_access = Instant::now();
    }
    
    /// Mark token as accessed (for LRU eviction policy)
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }
}

// ============================================================================
// Reasoning Token Arena
// ============================================================================

/// Arena-style allocator for reasoning tokens
/// 
/// Provides efficient allocation and deallocation of reasoning token buffers
/// with automatic cleanup when reasoning chains are completed.
#[derive(Debug)]
pub struct ReasoningTokenArena {
    /// Pre-allocated buffer for reasoning tokens
    buffer: Vec<ReasoningToken>,
    /// Maximum capacity
    capacity: usize,
    /// Current number of tokens
    len: usize,
    /// Free list for reusing slots
    free_list: VecDeque<usize>,
}

impl ReasoningTokenArena {
    /// Create a new arena with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            len: 0,
            free_list: VecDeque::new(),
        }
    }
    
    /// Allocate a new token in the arena
    pub fn allocate(&mut self, token_id: u32, position: usize, is_reasoning: bool) -> Option<usize> {
        if let Some(slot) = self.free_list.pop_front() {
            // Reuse freed slot
            self.buffer[slot] = ReasoningToken::new(token_id, position, is_reasoning);
            Some(slot)
        } else if self.len < self.capacity {
            // Allocate new slot
            let slot = self.len;
            self.buffer.push(ReasoningToken::new(token_id, position, is_reasoning));
            self.len += 1;
            Some(slot)
        } else {
            None // Arena full
        }
    }
    
    /// Free a token slot for reuse
    pub fn free(&mut self, slot: usize) {
        if slot < self.len {
            self.free_list.push_back(slot);
        }
    }
    
    /// Get a reference to a token
    pub fn get(&self, slot: usize) -> Option<&ReasoningToken> {
        self.buffer.get(slot)
    }
    
    /// Get a mutable reference to a token
    pub fn get_mut(&mut self, slot: usize) -> Option<&mut ReasoningToken> {
        self.buffer.get_mut(slot)
    }
    
    /// Reset the arena (free all tokens)
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.len = 0;
        self.free_list.clear();
    }
    
    /// Get current usage
    pub fn usage(&self) -> usize {
        self.len - self.free_list.len()
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Check if arena is full
    pub fn is_full(&self) -> bool {
        self.usage() >= self.capacity
    }
}

// ============================================================================
// KV Cache Budget Manager
// ============================================================================

/// Manages KV cache memory budget for reasoning tokens
#[derive(Debug)]
pub struct KVCacheBudget {
    /// Total budget in number of tokens
    total_budget: usize,
    /// Budget allocated for reasoning tokens (ratio * total)
    reasoning_budget: usize,
    /// Budget allocated for context tokens
    context_budget: usize,
    /// Current reasoning token count
    reasoning_count: usize,
    /// Current context token count
    context_count: usize,
    /// Eviction policy
    eviction_policy: EvictionPolicy,
}

impl KVCacheBudget {
    /// Create a new KV cache budget manager
    /// 
    /// # Arguments
    /// * `total_budget` - Total number of tokens in budget
    /// * `reasoning_ratio` - Fraction of budget for reasoning (0.0-1.0)
    /// * `eviction_policy` - Policy for evicting tokens when over budget
    pub fn new(total_budget: usize, reasoning_ratio: f32, eviction_policy: EvictionPolicy) -> Self {
        let reasoning_budget = (total_budget as f32 * reasoning_ratio) as usize;
        let context_budget = total_budget - reasoning_budget;
        
        Self {
            total_budget,
            reasoning_budget,
            context_budget,
            reasoning_count: 0,
            context_count: 0,
            eviction_policy,
        }
    }
    
    /// Check if we can add a reasoning token
    pub fn can_add_reasoning(&self) -> bool {
        self.reasoning_count < self.reasoning_budget
    }
    
    /// Check if we can add a context token
    pub fn can_add_context(&self) -> bool {
        self.context_count < self.context_budget
    }
    
    /// Add a reasoning token to the budget
    pub fn add_reasoning(&mut self) -> bool {
        if self.can_add_reasoning() {
            self.reasoning_count += 1;
            true
        } else {
            false
        }
    }
    
    /// Add a context token to the budget
    pub fn add_context(&mut self) -> bool {
        if self.can_add_context() {
            self.context_count += 1;
            true
        } else {
            false
        }
    }
    
    /// Remove a reasoning token from the budget
    pub fn remove_reasoning(&mut self) {
        self.reasoning_count = self.reasoning_count.saturating_sub(1);
    }
    
    /// Remove a context token from the budget
    pub fn remove_context(&mut self) {
        self.context_count = self.context_count.saturating_sub(1);
    }
    
    /// Get number of tokens to evict to fit new reasoning tokens
    pub fn tokens_to_evict(&self, new_tokens: usize) -> usize {
        let total_after = self.reasoning_count + new_tokens;
        if total_after > self.reasoning_budget {
            total_after - self.reasoning_budget
        } else {
            0
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> KVCacheStats {
        KVCacheStats {
            total_budget: self.total_budget,
            reasoning_budget: self.reasoning_budget,
            context_budget: self.context_budget,
            reasoning_used: self.reasoning_count,
            context_used: self.context_count,
            reasoning_utilization: self.reasoning_count as f32 / self.reasoning_budget.max(1) as f32,
            context_utilization: self.context_count as f32 / self.context_budget.max(1) as f32,
        }
    }
    
    /// Get eviction policy
    pub fn eviction_policy(&self) -> &EvictionPolicy {
        &self.eviction_policy
    }
    
    /// Reset all counts
    pub fn reset(&mut self) {
        self.reasoning_count = 0;
        self.context_count = 0;
    }
}

/// Statistics for KV cache usage
#[derive(Clone, Debug)]
pub struct KVCacheStats {
    pub total_budget: usize,
    pub reasoning_budget: usize,
    pub context_budget: usize,
    pub reasoning_used: usize,
    pub context_used: usize,
    pub reasoning_utilization: f32,
    pub context_utilization: f32,
}

// ============================================================================
// Reasoning Memory Manager
// ============================================================================

/// Configuration for the ReasoningMemoryManager
#[derive(Clone, Debug)]
pub struct ReasoningMemoryConfig {
    /// Maximum number of reasoning tokens
    pub max_reasoning_tokens: usize,
    /// KV cache budget ratio for reasoning (vs context)
    pub kv_cache_budget_ratio: f32,
    /// Eviction policy for KV cache
    pub eviction_policy: EvictionPolicy,
    /// Total KV cache budget (number of tokens)
    pub total_kv_budget: usize,
    /// Timeout for reasoning generation (in seconds)
    pub reasoning_timeout_secs: u64,
    /// Think start token (default: "<think>")
    pub think_start_token: String,
    /// Think end token (default: "</think>")
    pub think_end_token: String,
}

impl Default for ReasoningMemoryConfig {
    fn default() -> Self {
        Self {
            max_reasoning_tokens: 32768,  // 32K reasoning tokens
            kv_cache_budget_ratio: 0.7,   // 70% for reasoning, 30% for context
            eviction_policy: EvictionPolicy::SlidingWindow,
            total_kv_budget: 131072,      // 128K total KV cache
            reasoning_timeout_secs: 300,  // 5 minute timeout
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
        }
    }
}

/// Main memory manager for R1 reasoning tokens
/// 
/// Handles:
/// - Arena allocation for reasoning tokens
/// - KV cache budget management
/// - Streaming token generation
/// - Timeout mechanism for runaway reasoning
#[derive(Debug)]
pub struct ReasoningMemoryManager {
    /// Configuration
    config: ReasoningMemoryConfig,
    /// Arena allocator for reasoning tokens
    arena: ReasoningTokenArena,
    /// KV cache budget manager
    kv_budget: KVCacheBudget,
    /// Active token slots (in generation order)
    active_slots: VecDeque<usize>,
    /// Whether we're currently inside <think> tags
    in_reasoning: bool,
    /// Start time of current reasoning session
    session_start: Option<Instant>,
    /// Total tokens generated in current session
    tokens_generated: usize,
    /// Number of evicted tokens
    evicted_count: usize,
}

impl ReasoningMemoryManager {
    /// Create a new ReasoningMemoryManager
    pub fn new(config: ReasoningMemoryConfig) -> Self {
        let arena = ReasoningTokenArena::new(config.max_reasoning_tokens);
        let kv_budget = KVCacheBudget::new(
            config.total_kv_budget,
            config.kv_cache_budget_ratio,
            config.eviction_policy.clone(),
        );
        
        Self {
            config,
            arena,
            kv_budget,
            active_slots: VecDeque::new(),
            in_reasoning: false,
            session_start: None,
            tokens_generated: 0,
            evicted_count: 0,
        }
    }
    
    /// Start a new reasoning session
    pub fn start_session(&mut self) {
        self.session_start = Some(Instant::now());
        self.tokens_generated = 0;
        self.in_reasoning = false;
    }
    
    /// End the current reasoning session
    pub fn end_session(&mut self) {
        self.session_start = None;
        self.in_reasoning = false;
    }
    
    /// Check if reasoning has timed out
    pub fn is_timed_out(&self) -> bool {
        if let Some(start) = self.session_start {
            start.elapsed() > Duration::from_secs(self.config.reasoning_timeout_secs)
        } else {
            false
        }
    }
    
    /// Allocate a new reasoning token
    /// 
    /// Returns the slot index if successful, or None if allocation failed
    pub fn allocate_token(&mut self, token_id: u32) -> Option<usize> {
        // Check timeout
        if self.is_timed_out() {
            return None;
        }
        
        // Check if we need to evict tokens
        if self.in_reasoning && !self.kv_budget.can_add_reasoning() {
            self.evict_tokens(1);
        }
        
        // Allocate in arena
        let position = self.tokens_generated;
        if let Some(slot) = self.arena.allocate(token_id, position, self.in_reasoning) {
            // Update KV budget
            if self.in_reasoning {
                self.kv_budget.add_reasoning();
            } else {
                self.kv_budget.add_context();
            }
            
            self.active_slots.push_back(slot);
            self.tokens_generated += 1;
            Some(slot)
        } else {
            // Arena full, try evicting
            self.evict_tokens(1);
            self.arena.allocate(token_id, position, self.in_reasoning)
        }
    }
    
    /// Process a token to detect <think> boundaries
    pub fn process_token_boundary(&mut self, token_text: &str) {
        if token_text.contains(&self.config.think_start_token) {
            self.in_reasoning = true;
        } else if token_text.contains(&self.config.think_end_token) {
            self.in_reasoning = false;
        }
    }
    
    /// Evict tokens based on the configured policy
    pub fn evict_tokens(&mut self, count: usize) {
        let to_evict = count.min(self.active_slots.len());
        
        match self.kv_budget.eviction_policy() {
            EvictionPolicy::FIFO | EvictionPolicy::SlidingWindow => {
                // Evict from the front (oldest)
                for _ in 0..to_evict {
                    if let Some(slot) = self.active_slots.pop_front() {
                        if let Some(token) = self.arena.get(slot) {
                            if token.is_reasoning {
                                self.kv_budget.remove_reasoning();
                            } else {
                                self.kv_budget.remove_context();
                            }
                        }
                        self.arena.free(slot);
                        self.evicted_count += 1;
                    }
                }
            }
            EvictionPolicy::LRU => {
                // Find and evict least recently used
                let mut slots_by_access: Vec<_> = self.active_slots.iter()
                    .filter_map(|&slot| {
                        self.arena.get(slot).map(|t| (slot, t.last_access))
                    })
                    .collect();
                slots_by_access.sort_by_key(|(_, access)| *access);
                
                for (slot, _) in slots_by_access.into_iter().take(to_evict) {
                    self.active_slots.retain(|&s| s != slot);
                    if let Some(token) = self.arena.get(slot) {
                        if token.is_reasoning {
                            self.kv_budget.remove_reasoning();
                        } else {
                            self.kv_budget.remove_context();
                        }
                    }
                    self.arena.free(slot);
                    self.evicted_count += 1;
                }
            }
            EvictionPolicy::AttentionScore => {
                // Find and evict lowest attention score
                let mut slots_by_score: Vec<_> = self.active_slots.iter()
                    .filter_map(|&slot| {
                        self.arena.get(slot).map(|t| (slot, t.attention_score))
                    })
                    .collect();
                slots_by_score.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
                
                for (slot, _) in slots_by_score.into_iter().take(to_evict) {
                    self.active_slots.retain(|&s| s != slot);
                    if let Some(token) = self.arena.get(slot) {
                        if token.is_reasoning {
                            self.kv_budget.remove_reasoning();
                        } else {
                            self.kv_budget.remove_context();
                        }
                    }
                    self.arena.free(slot);
                    self.evicted_count += 1;
                }
            }
        }
    }
    
    /// Update attention scores for tokens
    pub fn update_attention_scores(&mut self, scores: &[f32]) {
        for (i, &score) in scores.iter().enumerate() {
            if i < self.active_slots.len() {
                if let Some(slot) = self.active_slots.get(i) {
                    if let Some(token) = self.arena.get_mut(*slot) {
                        token.update_attention(score);
                    }
                }
            }
        }
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> ReasoningMemoryStats {
        ReasoningMemoryStats {
            arena_capacity: self.arena.capacity(),
            arena_used: self.arena.usage(),
            kv_cache_stats: self.kv_budget.stats(),
            active_tokens: self.active_slots.len(),
            tokens_generated: self.tokens_generated,
            tokens_evicted: self.evicted_count,
            in_reasoning: self.in_reasoning,
            session_duration: self.session_start.map(|s| s.elapsed()),
        }
    }
    
    /// Reset the manager for a new generation
    pub fn reset(&mut self) {
        self.arena.reset();
        self.kv_budget.reset();
        self.active_slots.clear();
        self.in_reasoning = false;
        self.session_start = None;
        self.tokens_generated = 0;
        self.evicted_count = 0;
    }
    
    /// Check if currently in reasoning mode
    pub fn is_in_reasoning(&self) -> bool {
        self.in_reasoning
    }
    
    /// Get the configured timeout
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.config.reasoning_timeout_secs)
    }
}

/// Statistics for reasoning memory usage
#[derive(Clone, Debug)]
pub struct ReasoningMemoryStats {
    pub arena_capacity: usize,
    pub arena_used: usize,
    pub kv_cache_stats: KVCacheStats,
    pub active_tokens: usize,
    pub tokens_generated: usize,
    pub tokens_evicted: usize,
    pub in_reasoning: bool,
    pub session_duration: Option<Duration>,
}

// ============================================================================
// Reasoning Model with Memory Management
// ============================================================================

/// DeepSeek-R1 Reasoning Model with integrated memory management
pub struct ReasoningModel {
    vocab_size: usize,
    d_model: usize,
    embed: candle_nn::Embedding,
    lm_head: candle_nn::Linear,
    /// Memory manager for reasoning tokens
    memory_manager: ReasoningMemoryManager,
}

impl ReasoningModel {
    pub fn new(vocab_size: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let embed = candle_nn::embedding(vocab_size, d_model, vb.pp("embed"))?;
        let lm_head = candle_nn::linear(d_model, vocab_size, vb.pp("lm_head"))?;
        let memory_manager = ReasoningMemoryManager::new(ReasoningMemoryConfig::default());

        Ok(Self {
            vocab_size,
            d_model,
            embed,
            lm_head,
            memory_manager,
        })
    }
    
    /// Create with custom memory configuration
    pub fn with_memory_config(
        vocab_size: usize,
        d_model: usize,
        vb: VarBuilder,
        memory_config: ReasoningMemoryConfig,
    ) -> Result<Self> {
        let embed = candle_nn::embedding(vocab_size, d_model, vb.pp("embed"))?;
        let lm_head = candle_nn::linear(d_model, vocab_size, vb.pp("lm_head"))?;
        let memory_manager = ReasoningMemoryManager::new(memory_config);

        Ok(Self {
            vocab_size,
            d_model,
            embed,
            lm_head,
            memory_manager,
        })
    }
    
    /// Get memory manager for direct access
    pub fn memory_manager(&self) -> &ReasoningMemoryManager {
        &self.memory_manager
    }
    
    /// Get mutable memory manager
    pub fn memory_manager_mut(&mut self) -> &mut ReasoningMemoryManager {
        &mut self.memory_manager
    }

    /// Generate with reasoning, using memory management
    pub fn generate_with_reasoning(&mut self, prompt: &str, _device: &Device) -> Result<String> {
        // Start a reasoning session
        self.memory_manager.start_session();
        
        // Simulate reasoning trace (in production, this would be autoregressive generation)
        let reasoning_trace = format!(
            "<think>\nThe user is asking about {}. \n\
            1. I need to identify the core question.\n\
            2. I should recall relevant information about DeepSeek-R1.\n\
            3. I need to formulate a clear and concise answer.\n\
            </think>", 
            prompt
        );
        
        // Process think boundaries
        self.memory_manager.process_token_boundary("<think>");
        
        // Allocate tokens for the reasoning trace (simulated)
        for (i, _) in reasoning_trace.chars().enumerate() {
            if self.memory_manager.is_timed_out() {
                self.memory_manager.end_session();
                return Ok("Error: Reasoning timed out".to_string());
            }
            
            // Allocate token (using position as simulated token_id)
            let _ = self.memory_manager.allocate_token(i as u32);
        }
        
        self.memory_manager.process_token_boundary("</think>");
        
        let final_answer = format!(
            "\nHere is the answer based on my reasoning:\n\
            DeepSeek-R1 is a reasoning model that uses Reinforcement Learning to generate \
            chain-of-thought traces before answering. This improves performance on complex tasks."
        );

        // End session
        self.memory_manager.end_session();
        
        Ok(format!("{}{}", reasoning_trace, final_answer))
    }
    
    /// Standard forward pass for training/inference
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let x = self.embed.forward(input_ids)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }
    
    /// Get memory statistics
    pub fn get_memory_stats(&self) -> ReasoningMemoryStats {
        self.memory_manager.get_stats()
    }
    
    /// Reset memory manager
    pub fn reset_memory(&mut self) {
        self.memory_manager.reset();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reasoning_token_arena_allocation() {
        let mut arena = ReasoningTokenArena::new(100);
        
        // Allocate some tokens
        let slot1 = arena.allocate(1, 0, false).unwrap();
        let slot2 = arena.allocate(2, 1, true).unwrap();
        let slot3 = arena.allocate(3, 2, true).unwrap();
        
        assert_eq!(arena.usage(), 3);
        
        // Check token properties
        let token1 = arena.get(slot1).unwrap();
        assert_eq!(token1.token_id, 1);
        assert!(!token1.is_reasoning);
        
        let token2 = arena.get(slot2).unwrap();
        assert_eq!(token2.token_id, 2);
        assert!(token2.is_reasoning);
        
        // Free a slot
        arena.free(slot2);
        assert_eq!(arena.usage(), 2);
        
        // Reuse freed slot
        let slot4 = arena.allocate(4, 3, false).unwrap();
        assert_eq!(slot4, slot2); // Should reuse the freed slot
        assert_eq!(arena.usage(), 3);
    }
    
    #[test]
    fn test_reasoning_token_arena_capacity() {
        let mut arena = ReasoningTokenArena::new(3);
        
        arena.allocate(1, 0, false).unwrap();
        arena.allocate(2, 1, false).unwrap();
        arena.allocate(3, 2, false).unwrap();
        
        // Should be full now
        assert!(arena.is_full());
        assert!(arena.allocate(4, 3, false).is_none());
        
        // Free one and try again
        arena.free(0);
        assert!(!arena.is_full());
        assert!(arena.allocate(4, 3, false).is_some());
    }
    
    #[test]
    fn test_kv_cache_budget() {
        let mut budget = KVCacheBudget::new(100, 0.7, EvictionPolicy::FIFO);
        
        // Should have 70 for reasoning, 30 for context
        assert_eq!(budget.stats().reasoning_budget, 70);
        assert_eq!(budget.stats().context_budget, 30);
        
        // Add reasoning tokens
        for _ in 0..70 {
            assert!(budget.add_reasoning());
        }
        assert!(!budget.can_add_reasoning());
        
        // Add context tokens
        for _ in 0..30 {
            assert!(budget.add_context());
        }
        assert!(!budget.can_add_context());
        
        // Check utilization
        let stats = budget.stats();
        assert!((stats.reasoning_utilization - 1.0).abs() < 0.01);
        assert!((stats.context_utilization - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_reasoning_memory_manager_basic() {
        let config = ReasoningMemoryConfig {
            max_reasoning_tokens: 100,
            total_kv_budget: 50,
            kv_cache_budget_ratio: 0.8,
            ..Default::default()
        };
        
        let mut manager = ReasoningMemoryManager::new(config);
        manager.start_session();
        
        // Allocate tokens
        for i in 0..10 {
            let slot = manager.allocate_token(i);
            assert!(slot.is_some());
        }
        
        let stats = manager.get_stats();
        assert_eq!(stats.tokens_generated, 10);
        assert_eq!(stats.active_tokens, 10);
        
        manager.end_session();
    }
    
    #[test]
    fn test_reasoning_memory_manager_think_detection() {
        let mut manager = ReasoningMemoryManager::new(ReasoningMemoryConfig::default());
        manager.start_session();
        
        assert!(!manager.is_in_reasoning());
        
        manager.process_token_boundary("<think>");
        assert!(manager.is_in_reasoning());
        
        manager.process_token_boundary("some text");
        assert!(manager.is_in_reasoning());
        
        manager.process_token_boundary("</think>");
        assert!(!manager.is_in_reasoning());
        
        manager.end_session();
    }
    
    #[test]
    fn test_reasoning_memory_manager_eviction() {
        let config = ReasoningMemoryConfig {
            max_reasoning_tokens: 20,
            total_kv_budget: 10,
            kv_cache_budget_ratio: 1.0, // All for reasoning
            eviction_policy: EvictionPolicy::FIFO,
            ..Default::default()
        };
        
        let mut manager = ReasoningMemoryManager::new(config);
        manager.start_session();
        manager.process_token_boundary("<think>");
        
        // Fill up to budget
        for i in 0..15 {
            manager.allocate_token(i);
        }
        
        let stats = manager.get_stats();
        // Some tokens should have been evicted
        assert!(stats.tokens_evicted > 0);
        
        manager.end_session();
    }
    
    #[test]
    fn test_eviction_policy_lru() {
        let config = ReasoningMemoryConfig {
            max_reasoning_tokens: 10,
            total_kv_budget: 5,
            kv_cache_budget_ratio: 1.0,
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };
        
        let mut manager = ReasoningMemoryManager::new(config);
        manager.start_session();
        manager.process_token_boundary("<think>");
        
        // Allocate tokens
        for i in 0..5 {
            manager.allocate_token(i);
        }
        
        // Touch some tokens to make them "recently used"
        if let Some(&slot) = manager.active_slots.get(2) {
            if let Some(token) = manager.arena.get_mut(slot) {
                token.touch();
            }
        }
        
        // Force eviction by adding more
        manager.allocate_token(10);
        
        // The untouched tokens should be evicted first
        let stats = manager.get_stats();
        assert!(stats.tokens_evicted > 0);
        
        manager.end_session();
    }
    
    #[test]
    fn test_reasoning_memory_stats() {
        let config = ReasoningMemoryConfig {
            max_reasoning_tokens: 100,
            total_kv_budget: 50,
            kv_cache_budget_ratio: 0.6,
            ..Default::default()
        };
        
        let manager = ReasoningMemoryManager::new(config);
        let stats = manager.get_stats();
        
        assert_eq!(stats.arena_capacity, 100);
        assert_eq!(stats.arena_used, 0);
        assert_eq!(stats.kv_cache_stats.total_budget, 50);
        assert_eq!(stats.kv_cache_stats.reasoning_budget, 30);
        assert_eq!(stats.kv_cache_stats.context_budget, 20);
    }
}
