//! Gradient Checkpointing for DeepSeek Rust
//!
//! This module provides activation checkpointing (gradient checkpointing)
//! utilities to reduce memory usage during training at the cost of additional
//! computation.
//!
//! # Overview
//!
//! Gradient checkpointing works by:
//! 1. Not storing intermediate activations during forward pass
//! 2. Recomputing activations during backward pass when needed
//!
//! This trades compute for memory, typically reducing memory by 60-80%
//! while increasing compute by 20-30%.
//!
//! # Example
//! ```rust,ignore
//! use deepseek::training::checkpointing::{CheckpointConfig, CheckpointedModule};
//!
//! let config = CheckpointConfig::default()
//!     .with_checkpoint_every(2);  // Checkpoint every 2 layers
//!
//! let checkpointed = CheckpointedModule::new(transformer_layer, config);
//! ```

use candle_core::{Result, Tensor};
use std::cell::RefCell;

/// Configuration for gradient checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Enable checkpointing
    pub enabled: bool,
    /// Checkpoint every N layers (1 = every layer)
    pub checkpoint_every_n_layers: usize,
    /// Checkpoint MoE expert forward passes
    pub checkpoint_moe: bool,
    /// Checkpoint attention computation
    pub checkpoint_attention: bool,
    /// Checkpoint MLP/FFN computation
    pub checkpoint_mlp: bool,
    /// Use memory-efficient recomputation
    pub memory_efficient: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            checkpoint_every_n_layers: 1,
            checkpoint_moe: true,
            checkpoint_attention: true,
            checkpoint_mlp: true,
            memory_efficient: true,
        }
    }
}

impl CheckpointConfig {
    /// Create config that checkpoints every N layers
    pub fn with_checkpoint_every(mut self, n: usize) -> Self {
        self.checkpoint_every_n_layers = n;
        self
    }

    /// Enable/disable MoE checkpointing
    pub fn with_checkpoint_moe(mut self, enabled: bool) -> Self {
        self.checkpoint_moe = enabled;
        self
    }

    /// Enable/disable attention checkpointing
    pub fn with_checkpoint_attention(mut self, enabled: bool) -> Self {
        self.checkpoint_attention = enabled;
        self
    }

    /// Enable/disable MLP checkpointing
    pub fn with_checkpoint_mlp(mut self, enabled: bool) -> Self {
        self.checkpoint_mlp = enabled;
        self
    }

    /// Check if layer at index should be checkpointed
    pub fn should_checkpoint_layer(&self, layer_idx: usize) -> bool {
        self.enabled && (layer_idx % self.checkpoint_every_n_layers == 0)
    }
}

/// Activation storage for recomputation
#[derive(Debug)]
pub struct ActivationStore {
    /// Stored input tensors for recomputation
    inputs: RefCell<Vec<Tensor>>,
    /// Whether store is active
    active: bool,
}

impl Default for ActivationStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationStore {
    /// Create new activation store
    pub fn new() -> Self {
        Self {
            inputs: RefCell::new(Vec::new()),
            active: false,
        }
    }

    /// Start storing activations
    pub fn start(&mut self) {
        self.inputs.borrow_mut().clear();
        self.active = true;
    }

    /// Stop storing activations
    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Store an input tensor
    pub fn store(&self, tensor: &Tensor) {
        if self.active {
            self.inputs.borrow_mut().push(tensor.clone());
        }
    }

    /// Get stored inputs
    pub fn get_inputs(&self) -> Vec<Tensor> {
        self.inputs.borrow().clone()
    }

    /// Clear stored inputs
    pub fn clear(&self) {
        self.inputs.borrow_mut().clear();
    }

    /// Get memory usage estimate
    pub fn memory_estimate(&self) -> usize {
        self.inputs
            .borrow()
            .iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

/// Trait for modules that support checkpointing
pub trait Checkpointable {
    /// Forward pass that may use checkpointing
    fn forward_checkpointed(&self, x: &Tensor, checkpoint: bool) -> Result<Tensor>;
}

/// Statistics about checkpointing usage
#[derive(Debug, Clone, Default)]
pub struct CheckpointStats {
    /// Number of checkpointed layers
    pub checkpointed_layers: usize,
    /// Number of non-checkpointed layers
    pub non_checkpointed_layers: usize,
    /// Estimated memory saved (bytes)
    pub memory_saved_bytes: u64,
    /// Number of recomputations performed
    pub recomputation_count: u64,
}

impl CheckpointStats {
    /// Get memory saved in MB
    pub fn memory_saved_mb(&self) -> f64 {
        self.memory_saved_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get checkpoint ratio
    pub fn checkpoint_ratio(&self) -> f64 {
        let total = self.checkpointed_layers + self.non_checkpointed_layers;
        if total == 0 {
            0.0
        } else {
            self.checkpointed_layers as f64 / total as f64
        }
    }
}

/// Context for checkpointed forward pass
pub struct CheckpointContext {
    /// Configuration
    config: CheckpointConfig,
    /// Activation store
    store: ActivationStore,
    /// Statistics
    stats: RefCell<CheckpointStats>,
    /// Current layer index
    current_layer: RefCell<usize>,
}

impl CheckpointContext {
    /// Create new checkpoint context
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            store: ActivationStore::new(),
            stats: RefCell::new(CheckpointStats::default()),
            current_layer: RefCell::new(0),
        }
    }

    /// Check if current layer should be checkpointed
    pub fn should_checkpoint(&self) -> bool {
        self.config.should_checkpoint_layer(*self.current_layer.borrow())
    }

    /// Advance to next layer
    pub fn next_layer(&self) {
        *self.current_layer.borrow_mut() += 1;
    }

    /// Reset layer counter
    pub fn reset(&self) {
        *self.current_layer.borrow_mut() = 0;
    }

    /// Store activation if checkpointing
    pub fn maybe_store_activation(&self, tensor: &Tensor) {
        if self.should_checkpoint() {
            self.store.store(tensor);
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> CheckpointStats {
        self.stats.borrow().clone()
    }

    /// Update stats for checkpointed layer
    pub fn record_checkpointed(&self, activation_size_bytes: u64) {
        let mut stats = self.stats.borrow_mut();
        stats.checkpointed_layers += 1;
        stats.memory_saved_bytes += activation_size_bytes;
    }

    /// Update stats for non-checkpointed layer
    pub fn record_non_checkpointed(&self) {
        self.stats.borrow_mut().non_checkpointed_layers += 1;
    }

    /// Record recomputation
    pub fn record_recomputation(&self) {
        self.stats.borrow_mut().recomputation_count += 1;
    }
}

/// Selective recomputation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecomputeStrategy {
    /// Recompute all activations
    All,
    /// Recompute only attention
    AttentionOnly,
    /// Recompute only MLP/FFN
    MLPOnly,
    /// Recompute only MoE experts
    MoEOnly,
    /// Don't recompute (store all)
    None,
}

impl Default for RecomputeStrategy {
    fn default() -> Self {
        Self::All
    }
}

/// Get recommended checkpointing strategy based on model size and memory
pub fn recommend_checkpoint_strategy(
    num_params_millions: f64,
    available_memory_gb: f64,
) -> (CheckpointConfig, RecomputeStrategy) {
    // Rough heuristics based on model size and memory
    // Larger models need more aggressive checkpointing

    let params_per_gb = num_params_millions / available_memory_gb;

    let (checkpoint_every, checkpoint_moe, strategy) = if params_per_gb > 100.0 {
        // Very memory constrained: checkpoint everything
        (1, true, RecomputeStrategy::All)
    } else if params_per_gb > 50.0 {
        // Constrained: checkpoint every layer, skip MoE
        (1, false, RecomputeStrategy::AttentionOnly)
    } else if params_per_gb > 20.0 {
        // Moderate: checkpoint every 2 layers
        (2, true, RecomputeStrategy::All)
    } else if params_per_gb > 10.0 {
        // Comfortable: checkpoint every 3 layers
        (3, false, RecomputeStrategy::AttentionOnly)
    } else {
        // Plenty of memory: minimal checkpointing
        (4, false, RecomputeStrategy::None)
    };

    let config = CheckpointConfig {
        enabled: params_per_gb > 5.0, // Only enable if needed
        checkpoint_every_n_layers: checkpoint_every,
        checkpoint_moe,
        checkpoint_attention: true,
        checkpoint_mlp: true,
        memory_efficient: true,
    };

    (config, strategy)
}

/// Estimate memory savings from checkpointing
pub fn estimate_memory_savings(
    num_layers: usize,
    hidden_size: usize,
    seq_len: usize,
    batch_size: usize,
    config: &CheckpointConfig,
) -> (u64, u64) {
    // Estimate activation size per layer (rough approximation)
    // Main activations: hidden states after attention and FFN
    let bytes_per_element = 2; // Assuming FP16/BF16
    let activation_size_per_layer =
        2 * batch_size * seq_len * hidden_size * bytes_per_element;

    let checkpointed_layers = if config.enabled {
        num_layers / config.checkpoint_every_n_layers
    } else {
        0
    };

    let without_checkpointing = (num_layers * activation_size_per_layer) as u64;
    let with_checkpointing = ((num_layers - checkpointed_layers) * activation_size_per_layer
        + checkpointed_layers * batch_size * seq_len * hidden_size * bytes_per_element / 10)
        as u64; // Checkpointed layers store ~10% of activations

    (without_checkpointing, with_checkpointing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_config() {
        let config = CheckpointConfig::default();
        assert!(config.enabled);
        assert!(config.should_checkpoint_layer(0));
        assert!(config.should_checkpoint_layer(1));

        let config = config.with_checkpoint_every(2);
        assert!(config.should_checkpoint_layer(0));
        assert!(!config.should_checkpoint_layer(1));
        assert!(config.should_checkpoint_layer(2));
    }

    #[test]
    fn test_checkpoint_stats() {
        let stats = CheckpointStats {
            checkpointed_layers: 6,
            non_checkpointed_layers: 6,
            memory_saved_bytes: 1024 * 1024 * 500, // 500 MB
            recomputation_count: 100,
        };

        assert!((stats.checkpoint_ratio() - 0.5).abs() < 0.01);
        assert!((stats.memory_saved_mb() - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_recommend_strategy() {
        // Very constrained
        let (config, strategy) = recommend_checkpoint_strategy(7000.0, 24.0);
        assert!(config.enabled);
        assert_eq!(config.checkpoint_every_n_layers, 1);
        assert_eq!(strategy, RecomputeStrategy::All);

        // Plenty of memory
        let (config, _strategy) = recommend_checkpoint_strategy(100.0, 80.0);
        assert!(!config.enabled); // Less than 5 params per GB
    }

    #[test]
    fn test_memory_savings_estimate() {
        let config = CheckpointConfig::default();
        let (without, with) = estimate_memory_savings(12, 768, 512, 8, &config);
        assert!(with < without);
    }
}
