//! DeepSeek Training Pipeline
//!
//! This module implements the complete pre-training pipeline:
//! - Scaling law calculations
//! - Data mixing configuration
//! - WSD (Warmup-Stable-Decay) learning rate scheduler
//! - Curriculum learning
//! - Training configuration
//!
//! Based on: DeepSeek LLM paper (https://arxiv.org/abs/2401.02954)

use candle_core::Result;
use std::collections::HashMap;
use std::fs;

// =============================================================================
// Scaling Laws
// =============================================================================

/// Parameters for DeepSeek scaling laws
#[derive(Clone, Debug)]
pub struct ScalingLawParams {
    pub alpha: f64,    // Parameter scaling exponent
    pub beta: f64,     // Data scaling exponent  
    pub a: f64,        // Parameter coefficient
    pub b: f64,        // Data coefficient
    pub l_inf: f64,    // Irreducible loss
}

impl Default for ScalingLawParams {
    fn default() -> Self {
        Self {
            alpha: 0.34,
            beta: 0.28,
            a: 406.4,
            b: 410.7,
            l_inf: 1.69,
        }
    }
}

/// Scaling law calculator based on DeepSeek's findings.
///
/// L(N, D) = A/N^α + B/D^β + L∞
pub struct ScalingLaws {
    params: ScalingLawParams,
}

impl ScalingLaws {
    pub fn new(params: ScalingLawParams) -> Self {
        Self { params }
    }
    
    pub fn deepseek() -> Self {
        Self::new(ScalingLawParams::default())
    }
    
    /// Predict final training loss
    pub fn predict_loss(&self, num_params: f64, num_tokens: f64) -> f64 {
        let p = &self.params;
        p.a / num_params.powf(p.alpha) + p.b / num_tokens.powf(p.beta) + p.l_inf
    }
    
    /// Compute optimal model size and data amount for given compute budget
    /// Based on C ≈ 6ND
    pub fn optimal_config(&self, compute_flops: f64) -> (usize, usize) {
        // For Chinchilla-optimal: N ≈ D
        let optimal = (compute_flops / 6.0).sqrt();
        (optimal as usize, optimal as usize)
    }
    
    /// Get recommended learning rate based on model size
    /// η ∝ N^(-0.05)
    pub fn recommended_lr(&self, num_params: f64) -> f64 {
        let base_lr = 3e-4;  // For 1B model
        let base_size = 1e9;
        base_lr * (base_size / num_params).powf(0.05)
    }
    
    /// Calculate tokens needed to achieve target loss
    pub fn tokens_for_target_loss(&self, num_params: f64, target_loss: f64) -> Option<f64> {
        let p = &self.params;
        let param_contrib = p.a / num_params.powf(p.alpha);
        let remaining = target_loss - param_contrib - p.l_inf;
        
        if remaining <= 0.0 {
            None  // Cannot achieve with any amount of data
        } else {
            Some((p.b / remaining).powf(1.0 / p.beta))
        }
    }
}

// =============================================================================
// Data Mixing
// =============================================================================

/// Data mixing configuration for domain sampling
#[derive(Clone, Debug)]
pub struct DataMixingConfig {
    pub weights: HashMap<String, f64>,
    pub temperature: f64,
}

impl DataMixingConfig {
    pub fn new(weights: HashMap<String, f64>, temperature: f64) -> Self {
        Self { weights, temperature }
    }
    
    /// DeepSeek default mixing weights
    pub fn deepseek_default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("web".to_string(), 0.60);
        weights.insert("code".to_string(), 0.20);
        weights.insert("math".to_string(), 0.10);
        weights.insert("books".to_string(), 0.05);
        weights.insert("scientific".to_string(), 0.05);
        
        Self {
            weights,
            temperature: 1.0,
        }
    }
    
    /// Compute sampling probabilities: P(domain_i) = w_i^τ / Σ_j w_j^τ
    pub fn get_sampling_probs(&self) -> Vec<(String, f64)> {
        let total: f64 = self.weights.values()
            .map(|w| w.powf(self.temperature))
            .sum();
        
        self.weights.iter()
            .map(|(k, v)| (k.clone(), v.powf(self.temperature) / total))
            .collect()
    }
}

// =============================================================================
// Learning Rate Schedulers
// =============================================================================

/// WSD (Warmup-Stable-Decay) learning rate scheduler
/// 
/// Three phases:
/// 1. Warmup: Linear increase from 0 to peak_lr
/// 2. Stable: Constant at peak_lr
/// 3. Decay: Linear decrease to min_lr
pub struct WSDScheduler {
    warmup_steps: usize,
    stable_steps: usize,
    decay_steps: usize,
    peak_lr: f64,
    min_lr: f64,
    current_step: usize,
}

impl WSDScheduler {
    pub fn new(
        warmup_steps: usize,
        stable_steps: usize,
        decay_steps: usize,
        peak_lr: f64,
        min_lr: f64,
    ) -> Self {
        Self {
            warmup_steps,
            stable_steps,
            decay_steps,
            peak_lr,
            min_lr,
            current_step: 0,
        }
    }
    
    pub fn total_steps(&self) -> usize {
        self.warmup_steps + self.stable_steps + self.decay_steps
    }
    
    /// Get learning rate for a specific step
    pub fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.peak_lr * (step as f64) / (self.warmup_steps.max(1) as f64)
        } else if step < self.warmup_steps + self.stable_steps {
            // Stable phase
            self.peak_lr
        } else {
            // Linear decay
            let decay_step = step - self.warmup_steps - self.stable_steps;
            let progress = (decay_step as f64) / (self.decay_steps.max(1) as f64);
            self.min_lr + (self.peak_lr - self.min_lr) * (1.0 - progress).max(0.0)
        }
    }
    
    /// Get current learning rate
    pub fn current_lr(&self) -> f64 {
        self.get_lr(self.current_step)
    }
    
    /// Advance scheduler by one step
    pub fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }
    
    /// Reset to a specific step
    pub fn reset_to(&mut self, step: usize) {
        self.current_step = step;
    }
}

/// Cosine annealing with warmup
pub struct CosineWarmupScheduler {
    warmup_steps: usize,
    total_steps: usize,
    peak_lr: f64,
    min_lr: f64,
    current_step: usize,
}

impl CosineWarmupScheduler {
    pub fn new(
        warmup_steps: usize,
        total_steps: usize,
        peak_lr: f64,
        min_lr_ratio: f64,
    ) -> Self {
        Self {
            warmup_steps,
            total_steps,
            peak_lr,
            min_lr: peak_lr * min_lr_ratio,
            current_step: 0,
        }
    }
    
    pub fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            self.peak_lr * (step as f64) / (self.warmup_steps.max(1) as f64)
        } else {
            let progress = (step - self.warmup_steps) as f64 / 
                          (self.total_steps - self.warmup_steps).max(1) as f64;
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        }
    }
    
    pub fn step(&mut self) -> f64 {
        let lr = self.get_lr(self.current_step);
        self.current_step += 1;
        lr
    }
}

// =============================================================================
// Curriculum Learning
// =============================================================================

/// Curriculum learning scheduler for progressive training
pub struct CurriculumScheduler {
    start_seq_len: usize,
    end_seq_len: usize,
    seq_curriculum_steps: usize,
    difficulty_warmup_steps: usize,
}

impl CurriculumScheduler {
    pub fn new(
        start_seq_len: usize,
        end_seq_len: usize,
        seq_curriculum_steps: usize,
        difficulty_warmup_steps: usize,
    ) -> Self {
        Self {
            start_seq_len,
            end_seq_len,
            seq_curriculum_steps,
            difficulty_warmup_steps,
        }
    }
    
    /// Get sequence length for current step
    pub fn get_seq_length(&self, step: usize) -> usize {
        if step >= self.seq_curriculum_steps {
            return self.end_seq_len;
        }
        
        let progress = step as f64 / self.seq_curriculum_steps as f64;
        let seq_len = self.start_seq_len as f64 + 
            progress * (self.end_seq_len - self.start_seq_len) as f64;
        
        // Round to nearest power of 2 for efficiency
        let log2 = (seq_len.log2() + 0.5) as u32;
        2_usize.pow(log2)
    }
    
    /// Get weight for difficult examples (0 = easy only, 1 = all)
    pub fn get_difficulty_weight(&self, step: usize) -> f64 {
        if step >= self.difficulty_warmup_steps {
            1.0
        } else {
            step as f64 / self.difficulty_warmup_steps as f64
        }
    }
}

// =============================================================================
// Training Configuration
// =============================================================================

/// Complete training pipeline configuration
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    // Model
    pub model_size: usize,
    
    // Data
    pub seq_length: usize,
    pub batch_size: usize,
    
    // Training
    pub max_steps: usize,
    pub warmup_steps: usize,
    pub stable_ratio: f64,  // Portion of training in stable phase
    pub learning_rate: f64,
    pub min_lr_ratio: f64,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub gradient_accumulation_steps: usize,
    
    // Checkpointing
    pub save_every: usize,
    pub checkpoint_dir: String,
    
    // Logging
    pub log_every: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model_size: 7_000_000_000,  // 7B
            seq_length: 4096,
            batch_size: 8,
            max_steps: 100_000,
            warmup_steps: 2000,
            stable_ratio: 0.8,
            learning_rate: 2e-4,
            min_lr_ratio: 0.1,
            weight_decay: 0.1,
            max_grad_norm: 1.0,
            gradient_accumulation_steps: 4,
            save_every: 1000,
            checkpoint_dir: "./checkpoints".to_string(),
            log_every: 10,
        }
    }
}

impl PipelineConfig {
    /// Get recommended LR based on model size
    pub fn recommended_lr(model_size: usize) -> f64 {
        let base_lr = 3e-4;  // For 1B model
        let base_size = 1_000_000_000.0;
        base_lr * (base_size / model_size as f64).powf(0.05)
    }
    
    /// Create WSD scheduler from config
    pub fn create_wsd_scheduler(&self) -> WSDScheduler {
        let stable_steps = (self.max_steps as f64 * self.stable_ratio) as usize;
        let decay_steps = self.max_steps - self.warmup_steps - stable_steps;
        
        WSDScheduler::new(
            self.warmup_steps,
            stable_steps,
            decay_steps,
            self.learning_rate,
            self.learning_rate * self.min_lr_ratio,
        )
    }
    
    /// Create checkpoint directory
    pub fn create_checkpoint_dir(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.checkpoint_dir)
    }
    
    /// Compute effective batch size
    pub fn effective_batch_size(&self) -> usize {
        self.batch_size * self.gradient_accumulation_steps
    }
    
    /// Compute tokens per step
    pub fn tokens_per_step(&self) -> usize {
        self.effective_batch_size() * self.seq_length
    }
}

// =============================================================================
// Distributed Configuration
// =============================================================================

/// Configuration for distributed training
#[derive(Clone, Debug)]
pub struct DistributedConfig {
    pub world_size: usize,
    pub dp_size: usize,  // Data parallelism
    pub tp_size: usize,  // Tensor parallelism
    pub pp_size: usize,  // Pipeline parallelism
    pub zero_stage: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            dp_size: 1,
            tp_size: 1,
            pp_size: 1,
            zero_stage: 0,
        }
    }
}

impl DistributedConfig {
    pub fn single_gpu() -> Self {
        Self::default()
    }
    
    pub fn multi_gpu(num_gpus: usize, zero_stage: usize) -> Self {
        Self {
            world_size: num_gpus,
            dp_size: num_gpus,
            tp_size: 1,
            pp_size: 1,
            zero_stage,
        }
    }
    
    pub fn validate(&self) -> Result<()> {
        let computed = self.dp_size * self.tp_size * self.pp_size;
        if computed != self.world_size {
            return Err(candle_core::Error::Msg(format!(
                "DP({}) x TP({}) x PP({}) = {} != world_size({})",
                self.dp_size, self.tp_size, self.pp_size, computed, self.world_size
            )));
        }
        Ok(())
    }
}

// =============================================================================
// Activation Checkpointing (Phase 3)
// =============================================================================

/// Strategy for activation checkpointing to reduce memory usage.
#[derive(Clone, Debug, PartialEq)]
pub enum CheckpointingStrategy {
    /// No checkpointing (fastest, highest memory)
    None,
    /// Checkpoint every N layers
    EveryNLayers(usize),
    /// Checkpoint specific layer indices
    SelectiveLayers(Vec<usize>),
    /// Full checkpointing (slowest, lowest memory)
    Full,
}

impl Default for CheckpointingStrategy {
    fn default() -> Self {
        Self::None
    }
}

/// Activation checkpoint manager for gradient checkpointing.
/// 
/// Trades compute for memory by recomputing activations during backward pass
/// instead of storing them. Critical for training large models.
#[derive(Clone, Debug)]
pub struct ActivationCheckpointing {
    /// Checkpointing strategy
    pub strategy: CheckpointingStrategy,
    /// Number of layers in the model
    pub num_layers: usize,
    /// Layers that should be checkpointed
    checkpoint_layers: Vec<bool>,
    /// Memory saved estimate (bytes)
    pub estimated_memory_saved: usize,
}

impl ActivationCheckpointing {
    pub fn new(strategy: CheckpointingStrategy, num_layers: usize) -> Self {
        let checkpoint_layers = Self::compute_checkpoint_layers(&strategy, num_layers);
        
        Self {
            strategy,
            num_layers,
            checkpoint_layers,
            estimated_memory_saved: 0,
        }
    }
    
    fn compute_checkpoint_layers(strategy: &CheckpointingStrategy, num_layers: usize) -> Vec<bool> {
        match strategy {
            CheckpointingStrategy::None => vec![false; num_layers],
            CheckpointingStrategy::EveryNLayers(n) => {
                (0..num_layers).map(|i| i % n == 0).collect()
            }
            CheckpointingStrategy::SelectiveLayers(layers) => {
                (0..num_layers).map(|i| layers.contains(&i)).collect()
            }
            CheckpointingStrategy::Full => vec![true; num_layers],
        }
    }
    
    /// Check if a specific layer should be checkpointed.
    pub fn should_checkpoint(&self, layer_idx: usize) -> bool {
        self.checkpoint_layers.get(layer_idx).copied().unwrap_or(false)
    }
    
    /// Get count of checkpointed layers.
    pub fn num_checkpointed_layers(&self) -> usize {
        self.checkpoint_layers.iter().filter(|&&x| x).count()
    }
    
    /// Estimate memory savings (rough estimate based on layer count).
    /// Actual savings depend on model architecture.
    pub fn estimate_memory_savings(&self, hidden_dim: usize, seq_len: usize, batch_size: usize) -> usize {
        // Each checkpointed layer saves approximately:
        // 2 * hidden_dim * seq_len * batch_size * sizeof(f32) for activations
        let activation_size = 2 * hidden_dim * seq_len * batch_size * 4;  // 4 bytes for f32
        self.num_checkpointed_layers() * activation_size
    }
    
    /// Create strategy for optimal memory/compute tradeoff.
    /// Checkpoints every sqrt(num_layers) layers for optimal balance.
    pub fn optimal(num_layers: usize) -> Self {
        let n = (num_layers as f64).sqrt().ceil() as usize;
        Self::new(CheckpointingStrategy::EveryNLayers(n), num_layers)
    }
}

// =============================================================================
// ZeRO Optimizer Sharding (Phase 3)
// =============================================================================

/// ZeRO (Zero Redundancy Optimizer) stage configuration.
/// 
/// - Stage 0: No sharding (baseline)
/// - Stage 1: Optimizer state sharding (Adam moments)
/// - Stage 2: Stage 1 + Gradient sharding
/// - Stage 3: Stage 2 + Parameter sharding
#[derive(Clone, Debug, PartialEq)]
pub enum ZeROStage {
    /// No sharding
    Stage0,
    /// Optimizer state sharding (partitions Adam first/second moments)
    Stage1,
    /// Stage 1 + Gradient sharding
    Stage2,
    /// Stage 2 + Parameter sharding (most memory efficient)
    Stage3,
}

impl From<usize> for ZeROStage {
    fn from(stage: usize) -> Self {
        match stage {
            0 => ZeROStage::Stage0,
            1 => ZeROStage::Stage1,
            2 => ZeROStage::Stage2,
            3 => ZeROStage::Stage3,
            _ => ZeROStage::Stage0,
        }
    }
}

/// ZeRO optimizer state sharding manager.
/// 
/// Implements memory-efficient optimizer state distribution across data-parallel ranks.
#[derive(Clone, Debug)]
pub struct ZeROOptimizer {
    /// ZeRO stage
    pub stage: ZeROStage,
    /// Number of data-parallel ranks
    pub dp_size: usize,
    /// This rank in DP group
    pub dp_rank: usize,
    /// Partition size for each rank
    pub partition_size: usize,
}

impl ZeROOptimizer {
    pub fn new(stage: ZeROStage, dp_size: usize, dp_rank: usize, total_params: usize) -> Self {
        let partition_size = (total_params + dp_size - 1) / dp_size;
        
        Self {
            stage,
            dp_size,
            dp_rank,
            partition_size,
        }
    }
    
    /// Get parameter indices this rank is responsible for.
    pub fn get_param_partition(&self, total_params: usize) -> (usize, usize) {
        let start = self.dp_rank * self.partition_size;
        let end = (start + self.partition_size).min(total_params);
        (start, end)
    }
    
    /// Check if this rank owns a specific parameter index.
    pub fn owns_param(&self, param_idx: usize) -> bool {
        let (start, end) = self.get_param_partition(param_idx + 1);
        param_idx >= start && param_idx < end
    }
    
    /// Calculate memory savings from ZeRO sharding.
    /// 
    /// For Adam optimizer with fp32 params and fp32 moments:
    /// - Stage 0: 16 bytes per param (param + grad + m + v)
    /// - Stage 1: 4 + 4 + 8/dp_size bytes per param
    /// - Stage 2: 4 + 4/dp_size + 8/dp_size bytes per param
    /// - Stage 3: (4 + 4 + 8)/dp_size bytes per param
    pub fn memory_per_param(&self) -> f64 {
        let dp = self.dp_size as f64;
        match self.stage {
            ZeROStage::Stage0 => 16.0,
            ZeROStage::Stage1 => 8.0 + 8.0 / dp,  // params + grads + sharded moments
            ZeROStage::Stage2 => 4.0 + 12.0 / dp, // params + sharded (grads + moments)
            ZeROStage::Stage3 => 16.0 / dp,       // fully sharded
        }
    }
    
    /// Estimate total memory for optimizer state.
    pub fn estimate_memory(&self, total_params: usize) -> usize {
        (total_params as f64 * self.memory_per_param()) as usize
    }
    
    /// Check if gradients need to be reduced across ranks.
    pub fn needs_grad_reduce(&self) -> bool {
        matches!(self.stage, ZeROStage::Stage0 | ZeROStage::Stage1)
    }
    
    /// Check if gradients are sharded (Stage 2+).
    pub fn grads_sharded(&self) -> bool {
        matches!(self.stage, ZeROStage::Stage2 | ZeROStage::Stage3)
    }
    
    /// Check if parameters are sharded (Stage 3).
    pub fn params_sharded(&self) -> bool {
        matches!(self.stage, ZeROStage::Stage3)
    }
}

// =============================================================================
// Hierarchical All-to-All for Cross-Node Expert Parallelism (Phase 3)
// =============================================================================

/// Hierarchical communication topology for cross-node expert dispatch.
/// 
/// Uses two-level hierarchy:
/// 1. Intra-node: Fast NVLink/NVSwitch communication
/// 2. Inter-node: InfiniBand/Ethernet communication
#[derive(Clone, Debug)]
pub struct HierarchicalAllToAll {
    /// Total number of ranks
    pub world_size: usize,
    /// Ranks per node
    pub ranks_per_node: usize,
    /// Number of nodes
    pub num_nodes: usize,
    /// This rank's node ID
    pub node_id: usize,
    /// This rank's local rank within node
    pub local_rank: usize,
}

impl HierarchicalAllToAll {
    pub fn new(world_size: usize, ranks_per_node: usize, global_rank: usize) -> Self {
        let num_nodes = (world_size + ranks_per_node - 1) / ranks_per_node;
        let node_id = global_rank / ranks_per_node;
        let local_rank = global_rank % ranks_per_node;
        
        Self {
            world_size,
            ranks_per_node,
            num_nodes,
            node_id,
            local_rank,
        }
    }
    
    /// Get ranks in the same node (for intra-node communication).
    pub fn intra_node_ranks(&self) -> Vec<usize> {
        let start = self.node_id * self.ranks_per_node;
        let end = (start + self.ranks_per_node).min(self.world_size);
        (start..end).collect()
    }
    
    /// Get one rank from each node (for inter-node communication).
    pub fn inter_node_leader_ranks(&self) -> Vec<usize> {
        (0..self.num_nodes)
            .map(|n| n * self.ranks_per_node)
            .filter(|&r| r < self.world_size)
            .collect()
    }
    
    /// Check if this rank is the leader for its node.
    pub fn is_node_leader(&self) -> bool {
        self.local_rank == 0
    }
    
    /// Compute optimal send order for hierarchical all-to-all.
    /// 
    /// Returns (intra_node_targets, inter_node_targets)
    pub fn compute_send_order(&self, dest_rank: usize) -> (bool, bool) {
        let dest_node = dest_rank / self.ranks_per_node;
        let same_node = dest_node == self.node_id;
        (same_node, !same_node)
    }
    
    /// Estimate bandwidth utilization for hierarchical vs flat all-to-all.
    /// 
    /// Hierarchical is more efficient when inter-node bandwidth is limited.
    pub fn efficiency_ratio(&self, intra_bw_gbps: f64, inter_bw_gbps: f64) -> f64 {
        // Ratio of hierarchical vs flat all-to-all time
        // Hierarchical does: intra-node gather + inter-node exchange + intra-node scatter
        // Flat does: direct inter-node exchange for all pairs
        
        let n = self.world_size as f64;
        let r = self.ranks_per_node as f64;
        let nodes = self.num_nodes as f64;
        
        // Simplified model: flat = n * inter, hierarchical = r * intra + nodes * inter + r * intra
        let flat_time = n / inter_bw_gbps;
        let hier_time = 2.0 * r / intra_bw_gbps + nodes / inter_bw_gbps;
        
        flat_time / hier_time
    }
}

// =============================================================================
// Training Metrics
// =============================================================================

/// Training step metrics
#[derive(Clone, Debug, Default)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub lr: f64,
    pub grad_norm: f64,
    pub tokens_per_sec: f64,
    pub seq_length: usize,
}

impl TrainingMetrics {
    pub fn new(loss: f64, lr: f64, grad_norm: f64) -> Self {
        Self {
            loss,
            lr,
            grad_norm,
            tokens_per_sec: 0.0,
            seq_length: 0,
        }
    }
}

/// Training monitor for logging and health checks
pub struct TrainingMonitor {
    log_interval: usize,
    metrics_buffer: Vec<TrainingMetrics>,
}

impl TrainingMonitor {
    pub fn new(log_interval: usize) -> Self {
        Self {
            log_interval,
            metrics_buffer: Vec::new(),
        }
    }
    
    pub fn log(&mut self, step: usize, metrics: TrainingMetrics) {
        self.metrics_buffer.push(metrics);
        
        if step % self.log_interval == 0 && step > 0 {
            self.flush(step);
        }
    }
    
    fn flush(&mut self, step: usize) {
        if self.metrics_buffer.is_empty() {
            return;
        }
        
        let n = self.metrics_buffer.len() as f64;
        let avg_loss: f64 = self.metrics_buffer.iter().map(|m| m.loss).sum::<f64>() / n;
        let avg_lr: f64 = self.metrics_buffer.iter().map(|m| m.lr).sum::<f64>() / n;
        let avg_grad: f64 = self.metrics_buffer.iter().map(|m| m.grad_norm).sum::<f64>() / n;
        
        println!(
            "Step {:6} | Loss: {:.4} | LR: {:.2e} | Grad: {:.2}",
            step, avg_loss, avg_lr, avg_grad
        );
        
        // Health checks
        if avg_loss.is_nan() {
            println!("  ⚠️ NaN loss detected!");
        } else if avg_loss > 10.0 {
            println!("  ⚠️ Loss is very high - possible instability");
        }
        
        if avg_grad > 10.0 {
            println!("  ⚠️ High gradient norm - consider lower LR");
        }
        
        self.metrics_buffer.clear();
    }
}

// =============================================================================
// Demo Function
// =============================================================================



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scaling_laws() {
        let scaling = ScalingLaws::deepseek();
        
        // Test loss prediction
        let loss = scaling.predict_loss(7e9, 2e12);
        assert!(loss > 1.0 && loss < 3.0);
        
        // Test optimal config
        let (n, d) = scaling.optimal_config(1e23);
        assert!(n > 1e9 as usize);
        assert!(d > 1e9 as usize);
    }
    
    #[test]
    fn test_wsd_scheduler() {
        let scheduler = WSDScheduler::new(100, 800, 100, 1e-4, 1e-5);
        
        // Warmup
        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-10);
        assert!((scheduler.get_lr(50) - 0.5e-4).abs() < 1e-10);
        
        // Stable
        assert!((scheduler.get_lr(100) - 1e-4).abs() < 1e-10);
        assert!((scheduler.get_lr(500) - 1e-4).abs() < 1e-10);
        
        // Decay
        let lr_end = scheduler.get_lr(1000);
        assert!((lr_end - 1e-5).abs() < 1e-10);
    }
    
    #[test]
    fn test_curriculum_scheduler() {
        let curriculum = CurriculumScheduler::new(512, 4096, 10000, 5000);
        
        assert_eq!(curriculum.get_seq_length(0), 512);
        assert_eq!(curriculum.get_seq_length(10000), 4096);
        assert_eq!(curriculum.get_seq_length(20000), 4096);
        
        assert!((curriculum.get_difficulty_weight(0) - 0.0).abs() < 1e-10);
        assert!((curriculum.get_difficulty_weight(5000) - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_distributed_config() {
        let config = DistributedConfig::multi_gpu(8, 2);
        assert!(config.validate().is_ok());
        
        let invalid = DistributedConfig {
            world_size: 8,
            dp_size: 4,
            tp_size: 1,
            pp_size: 1,
            zero_stage: 0,
        };
        assert!(invalid.validate().is_err());
    }
    
    #[test]
    fn test_activation_checkpointing() {
        // Test no checkpointing
        let ac = ActivationCheckpointing::new(CheckpointingStrategy::None, 32);
        assert_eq!(ac.num_checkpointed_layers(), 0);
        assert!(!ac.should_checkpoint(0));
        
        // Test every N layers
        let ac = ActivationCheckpointing::new(CheckpointingStrategy::EveryNLayers(4), 32);
        assert_eq!(ac.num_checkpointed_layers(), 8);
        assert!(ac.should_checkpoint(0));
        assert!(!ac.should_checkpoint(1));
        assert!(ac.should_checkpoint(4));
        
        // Test selective layers
        let ac = ActivationCheckpointing::new(
            CheckpointingStrategy::SelectiveLayers(vec![0, 15, 31]),
            32
        );
        assert_eq!(ac.num_checkpointed_layers(), 3);
        assert!(ac.should_checkpoint(0));
        assert!(!ac.should_checkpoint(1));
        assert!(ac.should_checkpoint(15));
        
        // Test optimal
        let ac = ActivationCheckpointing::optimal(64);
        assert!(ac.num_checkpointed_layers() > 0);
        assert!(ac.num_checkpointed_layers() <= 64);
    }
    
    #[test]
    fn test_zero_optimizer() {
        let zero = ZeROOptimizer::new(ZeROStage::Stage1, 8, 0, 1000000);
        
        assert!(!zero.grads_sharded());
        assert!(!zero.params_sharded());
        assert!(zero.needs_grad_reduce());
        
        let (start, end) = zero.get_param_partition(1000000);
        assert_eq!(start, 0);
        assert!(end > 0);
        assert!(zero.owns_param(0));
        
        // Test Stage 3
        let zero3 = ZeROOptimizer::new(ZeROStage::Stage3, 8, 0, 1000000);
        assert!(zero3.grads_sharded());
        assert!(zero3.params_sharded());
        
        // Memory savings check
        let mem0 = ZeROOptimizer::new(ZeROStage::Stage0, 8, 0, 1000000).memory_per_param();
        let mem3 = ZeROOptimizer::new(ZeROStage::Stage3, 8, 0, 1000000).memory_per_param();
        assert!(mem3 < mem0);  // Stage 3 uses less memory
    }
    
    #[test]
    fn test_hierarchical_all_to_all() {
        // 16 ranks across 2 nodes (8 GPUs per node)
        let hier = HierarchicalAllToAll::new(16, 8, 0);
        
        assert_eq!(hier.num_nodes, 2);
        assert_eq!(hier.node_id, 0);
        assert_eq!(hier.local_rank, 0);
        assert!(hier.is_node_leader());
        
        let intra = hier.intra_node_ranks();
        assert_eq!(intra.len(), 8);
        assert_eq!(intra[0], 0);
        assert_eq!(intra[7], 7);
        
        let leaders = hier.inter_node_leader_ranks();
        assert_eq!(leaders.len(), 2);
        assert_eq!(leaders[0], 0);
        assert_eq!(leaders[1], 8);
        
        // Test send order
        let (intra, inter) = hier.compute_send_order(2);  // Same node
        assert!(intra);
        assert!(!inter);
        
        let (intra, inter) = hier.compute_send_order(10);  // Different node
        assert!(!intra);
        assert!(inter);
    }
}
