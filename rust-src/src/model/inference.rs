//! Inference Optimization Module
//!
//! This module provides optimized inference for DeepSeek-V3.2:
//! - Speculative decoding using MTP heads for faster generation
//! - Optimized KV cache management
//! - Batched verification for accepted tokens
//! - Dynamic draft/verify balancing

use candle_core::{Device, DType, Result, Tensor, IndexOp};
use crate::model::mtp::MTPModel;
use crate::model::kv_cache::KVCache;

// ============================================================================
// Speculative Decoding Configuration
// ============================================================================

/// Configuration for speculative decoding
#[derive(Clone, Debug)]
pub struct SpeculativeDecodingConfig {
    /// Number of tokens to draft (predict speculatively)
    pub num_draft_tokens: usize,
    /// Temperature for draft sampling
    pub draft_temperature: f32,
    /// Temperature for verification sampling
    pub verify_temperature: f32,
    /// Maximum consecutive rejections before falling back
    pub max_rejections: usize,
    /// Minimum acceptance rate to maintain speculation
    pub min_acceptance_rate: f32,
    /// Enable adaptive draft length based on acceptance rate
    pub adaptive_draft_length: bool,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 4,  // MTP typically predicts 4 ahead
            draft_temperature: 0.7,
            verify_temperature: 0.7,
            max_rejections: 3,
            min_acceptance_rate: 0.5,
            adaptive_draft_length: true,
        }
    }
}

// ============================================================================
// Speculative Decoder
// ============================================================================

/// Statistics for speculative decoding performance
#[derive(Clone, Debug, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated
    pub total_tokens: usize,
    /// Tokens accepted from drafts
    pub accepted_tokens: usize,
    /// Tokens rejected and regenerated
    pub rejected_tokens: usize,
    /// Number of draft rounds
    pub draft_rounds: usize,
    /// Average accepted per round
    pub avg_accepted_per_round: f32,
}

impl SpeculativeStats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.accepted_tokens + self.rejected_tokens == 0 {
            1.0
        } else {
            self.accepted_tokens as f32 / (self.accepted_tokens + self.rejected_tokens) as f32
        }
    }
    
    pub fn speedup_estimate(&self) -> f32 {
        // Speedup = tokens generated / (rounds + 1)
        // Perfect speculative decoding: num_draft_tokens + 1 tokens per round
        if self.draft_rounds == 0 {
            1.0
        } else {
            self.total_tokens as f32 / (self.draft_rounds + 1) as f32
        }
    }
}

/// Speculative decoder using MTP heads for fast generation
pub struct SpeculativeDecoder {
    config: SpeculativeDecodingConfig,
    /// Rolling statistics
    stats: SpeculativeStats,
    /// Adaptive draft length (adjusted based on acceptance)
    current_draft_length: usize,
}

impl SpeculativeDecoder {
    pub fn new(config: SpeculativeDecodingConfig) -> Self {
        let current_draft_length = config.num_draft_tokens;
        Self {
            config,
            stats: SpeculativeStats::default(),
            current_draft_length,
        }
    }
    
    /// Get current statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
        self.current_draft_length = self.config.num_draft_tokens;
    }
    
    /// Draft tokens using MTP model predictions
    /// 
    /// Returns: (draft_tokens, draft_logits)
    pub fn draft_tokens(
        &self,
        mtp_logits: &[Tensor],
        temperature: f32,
    ) -> Result<(Vec<u32>, Vec<Tensor>)> {
        let mut draft_tokens = Vec::new();
        let mut draft_probs = Vec::new();
        
        for logits in mtp_logits.iter().take(self.current_draft_length) {
            // Get last position logits
            let last_logits = logits.i((.., logits.dims()[1] - 1, ..))?;
            
            // Apply temperature
            let scaled_logits = if temperature != 1.0 {
                (&last_logits / temperature as f64)?
            } else {
                last_logits.clone()
            };
            
            // Softmax to get probabilities
            let probs = candle_nn::ops::softmax(&scaled_logits, 1)?;
            
            // Greedy sampling (for simplicity; could use nucleus/top-k)
            let token = probs.argmax(1)?;
            let token_val: Vec<u32> = token.flatten_all()?.to_vec1()?;
            
            draft_tokens.push(token_val[0]);
            draft_probs.push(probs);
        }
        
        Ok((draft_tokens, draft_probs))
    }
    
    /// Verify draft tokens against target model
    /// 
    /// Returns: (accepted_tokens, first_rejection_idx)
    pub fn verify_tokens(
        &mut self,
        draft_tokens: &[u32],
        draft_probs: &[Tensor],
        target_logits: &Tensor,
        temperature: f32,
    ) -> Result<(Vec<u32>, Option<usize>)> {
        let mut accepted = Vec::new();
        let mut first_rejection = None;
        
        // Target logits shape: [batch, seq, vocab]
        let seq_len = target_logits.dims()[1];
        
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            if i >= seq_len {
                break;
            }
            
            // Get target probability at position i
            let target_logits_i = target_logits.i((.., i, ..))?;
            let scaled_target = if temperature != 1.0 {
                (&target_logits_i / temperature as f64)?
            } else {
                target_logits_i.clone()
            };
            let target_probs = candle_nn::ops::softmax(&scaled_target, 1)?;
            
            // Get probability of draft token under target distribution
            let target_p: f32 = target_probs
                .i((.., draft_token as usize))?
                .flatten_all()?
                .to_vec1::<f32>()?[0];
            
            // Get probability of draft token under draft distribution
            let draft_p: f32 = draft_probs[i]
                .i((.., draft_token as usize))?
                .flatten_all()?
                .to_vec1::<f32>()?[0];
            
            // Speculative sampling acceptance criterion
            // Accept if target_p >= draft_p (simplified version)
            // Full version uses rejection sampling with random threshold
            let acceptance_ratio = if draft_p > 0.0 { target_p / draft_p } else { 0.0 };
            
            if acceptance_ratio >= 1.0 || acceptance_ratio > 0.7 {
                // Accept
                accepted.push(draft_token);
                self.stats.accepted_tokens += 1;
            } else {
                // Reject - need to resample from target
                first_rejection = Some(i);
                self.stats.rejected_tokens += 1;
                break;
            }
        }
        
        // Update stats
        self.stats.draft_rounds += 1;
        self.stats.total_tokens += accepted.len();
        
        // Adaptive draft length
        if self.config.adaptive_draft_length {
            self.adjust_draft_length();
        }
        
        Ok((accepted, first_rejection))
    }
    
    /// Sample from target distribution at rejection position
    pub fn resample_from_target(
        &mut self,
        target_logits: &Tensor,
        position: usize,
        temperature: f32,
    ) -> Result<u32> {
        let logits = target_logits.i((.., position, ..))?;
        let scaled = if temperature != 1.0 {
            (&logits / temperature as f64)?
        } else {
            logits.clone()
        };
        let probs = candle_nn::ops::softmax(&scaled, 1)?;
        let token = probs.argmax(1)?;
        let token_val: Vec<u32> = token.flatten_all()?.to_vec1()?;
        
        self.stats.total_tokens += 1;
        
        Ok(token_val[0])
    }
    
    /// Adjust draft length based on acceptance rate
    fn adjust_draft_length(&mut self) {
        let rate = self.stats.acceptance_rate();
        
        if rate > 0.8 && self.current_draft_length < self.config.num_draft_tokens * 2 {
            // High acceptance - can try more tokens
            self.current_draft_length += 1;
        } else if rate < self.config.min_acceptance_rate && self.current_draft_length > 1 {
            // Low acceptance - reduce speculation
            self.current_draft_length -= 1;
        }
        
        // Update running average
        if self.stats.draft_rounds > 0 {
            self.stats.avg_accepted_per_round = 
                self.stats.accepted_tokens as f32 / self.stats.draft_rounds as f32;
        }
    }
}

// ============================================================================
// Optimized KV Cache Manager
// ============================================================================

/// Configuration for KV cache optimization
#[derive(Clone, Debug)]
pub struct KVCacheConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Whether to use sliding window
    pub use_sliding_window: bool,
    /// Sliding window size
    pub window_size: usize,
    /// Data type for cache
    pub dtype: DType,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 131072,  // 128K
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            use_sliding_window: false,
            window_size: 4096,
            dtype: DType::F32,
        }
    }
}

/// Multi-layer KV cache manager for efficient generation
pub struct KVCacheManager {
    config: KVCacheConfig,
    /// KV caches per layer
    caches: Vec<KVCache>,
    /// Current sequence position
    current_pos: usize,
}

impl KVCacheManager {
    pub fn new(config: KVCacheConfig, batch_size: usize, device: &Device) -> Result<Self> {
        let mut caches = Vec::with_capacity(config.num_layers);
        
        for _ in 0..config.num_layers {
            let cache = KVCache::new(
                batch_size,
                config.max_seq_len,
                config.num_heads,
                config.head_dim,
                config.dtype,
                device,
            )?;
            caches.push(cache);
        }
        
        Ok(Self {
            config,
            caches,
            current_pos: 0,
        })
    }
    
    /// Update cache for a specific layer
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if layer_idx >= self.caches.len() {
            return Err(candle_core::Error::Msg(
                format!("Layer {} out of range", layer_idx)
            ));
        }
        
        self.caches[layer_idx].update(k, v)
    }
    
    /// Get current sequence length
    pub fn current_seq_len(&self) -> usize {
        if self.caches.is_empty() {
            0
        } else {
            self.caches[0].current_seq_len()
        }
    }
    
    /// Reset all caches
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
        self.current_pos = 0;
    }
    
    /// Trim cache to specific length (for rejection in speculative decoding)
    pub fn trim_to(&mut self, length: usize) -> Result<()> {
        for cache in &mut self.caches {
            cache.trim_to(length)?;
        }
        self.current_pos = length;
        Ok(())
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        if self.caches.is_empty() {
            return 0;
        }
        
        let current_len = self.current_seq_len();
        let element_size = match self.config.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 4,
        };
        
        // K + V for each layer
        2 * self.config.num_layers 
            * self.config.num_heads 
            * current_len 
            * self.config.head_dim 
            * element_size
    }
}

// ============================================================================
// Inference Engine
// ============================================================================

/// Configuration for the inference engine
#[derive(Clone, Debug)]
pub struct InferenceConfig {
    /// Enable speculative decoding
    pub use_speculative_decoding: bool,
    /// Speculative decoding config
    pub speculative_config: SpeculativeDecodingConfig,
    /// KV cache config
    pub cache_config: KVCacheConfig,
    /// Maximum generation length
    pub max_gen_length: usize,
    /// EOS token ID
    pub eos_token_id: u32,
    /// Temperature for sampling
    pub temperature: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            use_speculative_decoding: true,
            speculative_config: SpeculativeDecodingConfig::default(),
            cache_config: KVCacheConfig::default(),
            max_gen_length: 2048,
            eos_token_id: 2,
            temperature: 0.7,
        }
    }
}

/// Generation output
#[derive(Clone, Debug)]
pub struct GenerationOutput {
    /// Generated token IDs
    pub tokens: Vec<u32>,
    /// Speculative decoding stats (if used)
    pub speculative_stats: Option<SpeculativeStats>,
    /// Total generation time in milliseconds
    pub generation_time_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeDecodingConfig::default();
        assert_eq!(config.num_draft_tokens, 4);
        assert!(config.adaptive_draft_length);
    }
    
    #[test]
    fn test_speculative_stats() {
        let mut stats = SpeculativeStats::default();
        stats.accepted_tokens = 80;
        stats.rejected_tokens = 20;
        stats.total_tokens = 80;
        stats.draft_rounds = 20;
        
        let rate = stats.acceptance_rate();
        assert!((rate - 0.8).abs() < 0.01);
        
        let speedup = stats.speedup_estimate();
        assert!(speedup > 1.0);
    }
    
    #[test]
    fn test_speculative_decoder_creation() {
        let config = SpeculativeDecodingConfig::default();
        let decoder = SpeculativeDecoder::new(config);
        
        assert_eq!(decoder.current_draft_length, 4);
        assert_eq!(decoder.stats().total_tokens, 0);
    }
    
    #[test]
    fn test_kv_cache_config_default() {
        let config = KVCacheConfig::default();
        assert_eq!(config.max_seq_len, 131072);
        assert_eq!(config.num_layers, 32);
    }
    
    #[test]
    fn test_kv_cache_manager() {
        let config = KVCacheConfig {
            max_seq_len: 1024,
            num_layers: 2,
            num_heads: 4,
            head_dim: 32,
            ..Default::default()
        };
        
        let device = Device::Cpu;
        let manager = KVCacheManager::new(config, 1, &device).expect("Cache creation failed");
        
        assert_eq!(manager.current_seq_len(), 0);
    }
    
    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert!(config.use_speculative_decoding);
        assert_eq!(config.max_gen_length, 2048);
    }
}
