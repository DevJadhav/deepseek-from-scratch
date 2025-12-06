#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::needless_question_mark)]
#![allow(dead_code)]
#![allow(unused_parens)]

use candle_core::{Result, Tensor, DType, Module, Device};
use candle_nn::{Linear, VarBuilder, ops};
#[allow(unused_imports)]
use crate::distributed::expert::{ExpertParallelDispatch, ExpertParallelConfig, DispatchInfo};

// ============================================================================
// DeepSeek-V3 MoE Configuration
// ============================================================================

/// Configuration for DeepSeek-V3 style MoE with 256 experts
#[derive(Clone, Debug)]
pub struct DeepSeekMoEV3Config {
    /// Model dimension
    pub d_model: usize,
    /// Total number of routed experts
    pub n_routed_experts: usize,
    /// Number of shared experts
    pub n_shared_experts: usize,
    /// Number of experts activated per token
    pub top_k: usize,
    /// Hidden dimension for routed experts
    pub routed_expert_hidden: usize,
    /// Hidden dimension for shared experts
    pub shared_expert_hidden: usize,
    /// Number of expert groups for hierarchical routing
    pub n_expert_groups: usize,
    /// Top-k groups to select in first routing stage
    pub top_k_groups: usize,
    /// Expert capacity factor (1.0 = exact, >1.0 = slack)
    pub capacity_factor: f32,
    /// Enable auxiliary-loss-free load balancing
    pub aux_loss_free: bool,
    /// Bias learning rate for load balancing
    pub bias_lr: f64,
    /// EMA decay for load balancing
    pub ema_decay: f32,
    /// Minimum tokens per expert (for capacity)
    pub min_tokens_per_expert: usize,
    /// Enable expert dropout during training
    pub expert_dropout: f32,
    /// Routing strategy: "softmax" or "sigmoid"
    pub routing_strategy: String,
    /// Bias clamp value to prevent extreme biases
    pub bias_clamp: f32,
    /// Drop tokens when expert capacity is exceeded (vs route to shared expert)
    pub drop_tokens_on_overflow: bool,
    /// Track dropped tokens for metrics
    pub track_capacity_metrics: bool,
}

impl Default for DeepSeekMoEV3Config {
    fn default() -> Self {
        Self {
            d_model: 4096,
            n_routed_experts: 256,
            n_shared_experts: 2,
            top_k: 8,
            routed_expert_hidden: 1024,
            shared_expert_hidden: 4096,
            n_expert_groups: 8,      // 256 / 8 = 32 experts per group
            top_k_groups: 4,         // Select 4 groups, then 2 experts per group
            capacity_factor: 1.25,
            aux_loss_free: true,
            bias_lr: 0.001,
            ema_decay: 0.99,
            min_tokens_per_expert: 1,
            expert_dropout: 0.0,
            routing_strategy: "sigmoid".to_string(),
            bias_clamp: 2.0,
            drop_tokens_on_overflow: true,
            track_capacity_metrics: true,
        }
    }
}

impl DeepSeekMoEV3Config {
    /// Create config for DeepSeek-V3.2 (256 experts, 8 active)
    pub fn v3_256_8() -> Self {
        Self::default()
    }
    
    /// Create smaller config for testing (16 experts, 2 active)  
    pub fn small_16_2() -> Self {
        Self {
            n_routed_experts: 16,
            n_shared_experts: 2,
            top_k: 2,
            n_expert_groups: 4,
            top_k_groups: 2,
            ..Default::default()
        }
    }
    
    /// Experts per group
    pub fn experts_per_group(&self) -> usize {
        self.n_routed_experts / self.n_expert_groups
    }
    
    /// Experts to select per group
    pub fn experts_per_selected_group(&self) -> usize {
        self.top_k / self.top_k_groups
    }
}

// ============================================================================
// Load Balancing State (Auxiliary-Loss-Free)
// ============================================================================

/// State for auxiliary-loss-free load balancing
/// 
/// Uses bias-based adjustment with EMA updates per DeepSeek-V3 paper
pub struct LoadBalancingState {
    /// Per-expert bias terms for routing adjustment
    bias: Tensor,
    /// EMA of expert selection counts
    ema_counts: Vec<f32>,
    /// Configuration
    config: DeepSeekMoEV3Config,
    /// Current step for tracking
    step: usize,
    /// History of bias values (for visualization/analysis)
    bias_history: Vec<Vec<f32>>,
    /// History of load values (for visualization/analysis)
    load_history: Vec<Vec<f32>>,
    /// Maximum history size
    max_history_size: usize,
}

impl LoadBalancingState {
    pub fn new(config: &DeepSeekMoEV3Config, device: &Device) -> Result<Self> {
        let bias = Tensor::zeros((config.n_routed_experts,), DType::F32, device)?;
        let ema_counts = vec![1.0 / config.n_routed_experts as f32; config.n_routed_experts];
        
        Ok(Self {
            bias,
            ema_counts,
            config: config.clone(),
            step: 0,
            bias_history: Vec::new(),
            load_history: Vec::new(),
            max_history_size: 1000,  // Keep last 1000 steps
        })
    }
    
    /// Create with custom history size
    pub fn with_history_size(config: &DeepSeekMoEV3Config, device: &Device, max_history: usize) -> Result<Self> {
        let mut state = Self::new(config, device)?;
        state.max_history_size = max_history;
        Ok(state)
    }
    
    /// Get current bias tensor
    pub fn get_bias(&self) -> &Tensor {
        &self.bias
    }
    
    /// Get bias history for visualization
    pub fn get_bias_history(&self) -> &[Vec<f32>] {
        &self.bias_history
    }
    
    /// Get load history for visualization
    pub fn get_load_history(&self) -> &[Vec<f32>] {
        &self.load_history
    }
    
    /// Update bias based on observed expert selections
    /// 
    /// This is the auxiliary-loss-free load balancing from DeepSeek-V3:
    /// Instead of adding an auxiliary loss, we adjust routing bias terms
    /// to encourage underutilized experts and discourage overutilized ones.
    pub fn update(&mut self, expert_counts: &[f32], device: &Device) -> Result<()> {
        let n_experts = self.config.n_routed_experts;
        let decay = self.config.ema_decay;
        
        // Update EMA counts
        for i in 0..n_experts {
            self.ema_counts[i] = decay * self.ema_counts[i] + (1.0 - decay) * expert_counts[i];
        }
        
        // Record load history
        if self.load_history.len() < self.max_history_size {
            self.load_history.push(self.ema_counts.clone());
        } else if self.max_history_size > 0 {
            // Shift history and add new
            self.load_history.remove(0);
            self.load_history.push(self.ema_counts.clone());
        }
        
        // Compute target (uniform distribution)
        let total_count: f32 = self.ema_counts.iter().sum();
        let target = total_count / n_experts as f32;
        
        // Update bias: bias_i += lr * tanh((target - count_i) / (target + eps))
        let mut bias_vec = self.bias.to_vec1::<f32>()?;
        let lr = self.config.bias_lr as f32;
        let clamp_val = self.config.bias_clamp;
        
        for i in 0..n_experts {
            let count = self.ema_counts[i];
            let violation = (target - count) / (target + 1e-6);
            let adjustment = lr * violation.tanh();
            bias_vec[i] += adjustment;
            
            // Clamp to prevent extreme biases
            bias_vec[i] = bias_vec[i].clamp(-clamp_val, clamp_val);
        }
        
        // Record bias history
        if self.bias_history.len() < self.max_history_size {
            self.bias_history.push(bias_vec.clone());
        } else if self.max_history_size > 0 {
            // Shift history and add new
            self.bias_history.remove(0);
            self.bias_history.push(bias_vec.clone());
        }
        
        self.bias = Tensor::from_vec(bias_vec, (n_experts,), device)?;
        self.step += 1;
        
        Ok(())
    }
    
    /// Get load balancing statistics
    pub fn get_stats(&self) -> (f32, f32, f32) {
        let counts = &self.ema_counts;
        let mean = counts.iter().sum::<f32>() / counts.len() as f32;
        let max = counts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = counts.iter().cloned().fold(f32::INFINITY, f32::min);
        
        // Imbalance ratio: max/min
        let imbalance = if min > 0.0 { max / min } else { f32::INFINITY };
        
        (mean, imbalance, self.step as f32)
    }
    
    /// Get detailed statistics for logging
    pub fn get_detailed_stats(&self) -> LoadBalanceStats {
        let counts = &self.ema_counts;
        let mean = counts.iter().sum::<f32>() / counts.len() as f32;
        let std = (counts.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / counts.len() as f32).sqrt();
        let bias_vec: Vec<f32> = self.bias.to_vec1().unwrap_or_default();
        
        LoadBalanceStats {
            mean_count: mean,
            std_count: std,
            max_count: counts.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            min_count: counts.iter().cloned().fold(f32::INFINITY, f32::min),
            load_balance_cv: std / (mean + 1e-6),
            mean_bias: bias_vec.iter().sum::<f32>() / bias_vec.len() as f32,
            max_bias: bias_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            min_bias: bias_vec.iter().cloned().fold(f32::INFINITY, f32::min),
            step: self.step,
        }
    }
}

/// Detailed load balance statistics
#[derive(Clone, Debug, Default)]
pub struct LoadBalanceStats {
    pub mean_count: f32,
    pub std_count: f32,
    pub max_count: f32,
    pub min_count: f32,
    pub load_balance_cv: f32,
    pub mean_bias: f32,
    pub max_bias: f32,
    pub min_bias: f32,
    pub step: usize,
}

// ============================================================================
// RouterBiasController (DeepSeek-V3 Auxiliary-Loss-Free Load Balancing)
// ============================================================================

/// High-level controller for auxiliary-loss-free load balancing per DeepSeek-V3.
///
/// This controller wraps LoadBalancingState and provides a clean API for:
/// 1. Updating router biases AFTER each batch (not during backward pass)
/// 2. Disabling auxiliary loss when bias-based balancing is active
/// 3. Providing the `bias_update_alpha` hyperparameter (recommended 0.001)
///
/// Key difference from traditional auxiliary loss:
/// - Traditional: Add loss term during backward pass that competes with main loss
/// - Bias-based: Update biases directly after batch, no gradient interference
///
/// Usage:
/// ```rust,ignore
/// let controller = RouterBiasController::new(&config, &device)?;
/// 
/// // During forward pass:
/// let biased_logits = logits + controller.get_bias();
///
/// // AFTER backward pass (not during):
/// controller.update_after_batch(&expert_counts, &device)?;
/// ```
pub struct RouterBiasController {
    /// Internal load balancing state
    state: LoadBalancingState,
    /// Whether auxiliary loss should be disabled (always true when using this controller)
    aux_loss_disabled: bool,
}

impl RouterBiasController {
    /// Create a new RouterBiasController
    ///
    /// Args:
    ///     config: MoE configuration with bias_lr (alias: bias_update_alpha)
    ///     device: Device to create tensors on
    pub fn new(config: &DeepSeekMoEV3Config, device: &Device) -> Result<Self> {
        let state = LoadBalancingState::new(config, device)?;
        Ok(Self {
            state,
            aux_loss_disabled: true,  // Always disable aux loss when using bias-based
        })
    }

    /// Create with custom history size for visualization
    pub fn with_history_size(config: &DeepSeekMoEV3Config, device: &Device, max_history: usize) -> Result<Self> {
        let state = LoadBalancingState::with_history_size(config, device, max_history)?;
        Ok(Self {
            state,
            aux_loss_disabled: true,
        })
    }

    /// Get current bias tensor to add to routing logits
    pub fn get_bias(&self) -> &Tensor {
        self.state.get_bias()
    }

    /// Update biases after batch completion (NOT during backward pass)
    ///
    /// This should be called AFTER optimizer.step() and zero_grad(),
    /// not during the backward pass. This ensures no interference with gradients.
    ///
    /// Args:
    ///     expert_counts: Count of tokens routed to each expert in this batch
    ///     device: Device for tensor operations
    pub fn update_after_batch(&mut self, expert_counts: &[f32], device: &Device) -> Result<()> {
        self.state.update(expert_counts, device)
    }

    /// Check if auxiliary loss should be used
    /// 
    /// Returns false when using RouterBiasController (bias-based balancing)
    /// This prevents the competing auxiliary loss from interfering
    pub fn use_auxiliary_loss(&self) -> bool {
        !self.aux_loss_disabled
    }

    /// Get load balancing statistics
    pub fn get_stats(&self) -> (f32, f32, f32) {
        self.state.get_stats()
    }

    /// Get detailed statistics for logging
    pub fn get_detailed_stats(&self) -> LoadBalanceStats {
        self.state.get_detailed_stats()
    }

    /// Get bias history for visualization
    pub fn get_bias_history(&self) -> &[Vec<f32>] {
        self.state.get_bias_history()
    }

    /// Get load history for visualization  
    pub fn get_load_history(&self) -> &[Vec<f32>] {
        self.state.get_load_history()
    }

    /// Get the current step count
    pub fn step(&self) -> usize {
        self.state.step
    }
}

/// Alias for bias learning rate (recommended value: 0.001)
/// 
/// In DeepSeekMoEV3Config, use either:
/// - `bias_lr: 0.001` (original name)
/// - This constant as reference for recommended value
pub const BIAS_UPDATE_ALPHA_RECOMMENDED: f64 = 0.001;

// ============================================================================
// Expert Frequency Tracker (for Specialization Analysis)
// ============================================================================

/// Tracks expert usage frequency for specialization analysis
#[derive(Clone, Debug)]
pub struct ExpertFrequencyTracker {
    /// Total selections per expert across all time
    pub total_counts: Vec<u64>,
    /// Selections per expert in current window
    pub window_counts: Vec<u64>,
    /// Window size for moving statistics
    pub window_size: usize,
    /// Current position in window
    pub window_pos: usize,
    /// History of expert selections for analysis
    pub history: Vec<Vec<u32>>,
    /// Maximum history entries to keep
    pub max_history: usize,
}

impl ExpertFrequencyTracker {
    pub fn new(n_experts: usize, window_size: usize, max_history: usize) -> Self {
        Self {
            total_counts: vec![0; n_experts],
            window_counts: vec![0; n_experts],
            window_size,
            window_pos: 0,
            history: Vec::new(),
            max_history,
        }
    }
    
    /// Record expert selections from a batch
    pub fn record_batch(&mut self, expert_indices: &[u32]) {
        for &expert_id in expert_indices {
            let idx = expert_id as usize;
            if idx < self.total_counts.len() {
                self.total_counts[idx] += 1;
                self.window_counts[idx] += 1;
            }
        }
        
        // Record history
        if self.history.len() < self.max_history {
            self.history.push(expert_indices.to_vec());
        }
        
        self.window_pos += 1;
        if self.window_pos >= self.window_size {
            self.window_pos = 0;
            // Reset window counts for new window
            self.window_counts.fill(0);
        }
    }
    
    /// Get most frequently used experts
    pub fn top_experts(&self, k: usize) -> Vec<(usize, u64)> {
        let mut indexed: Vec<(usize, u64)> = self.total_counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.into_iter().take(k).collect()
    }
    
    /// Get least frequently used experts
    pub fn bottom_experts(&self, k: usize) -> Vec<(usize, u64)> {
        let mut indexed: Vec<(usize, u64)> = self.total_counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        indexed.sort_by(|a, b| a.1.cmp(&b.1));
        indexed.into_iter().take(k).collect()
    }
    
    /// Get expert utilization statistics
    pub fn utilization_stats(&self) -> (f32, f32, f32) {
        let total: u64 = self.total_counts.iter().sum();
        if total == 0 {
            return (0.0, 0.0, 0.0);
        }
        let n = self.total_counts.len() as f32;
        let mean = total as f32 / n;
        let variance = self.total_counts
            .iter()
            .map(|&c| ((c as f32) - mean).powi(2))
            .sum::<f32>() / n;
        let std = variance.sqrt();
        let cv = std / (mean + 1e-6);
        (mean, std, cv)
    }
    
    /// Reset tracker
    pub fn reset(&mut self) {
        self.total_counts.fill(0);
        self.window_counts.fill(0);
        self.window_pos = 0;
        self.history.clear();
    }
}

// ============================================================================
// Expert Specialization Tracker (Comprehensive Analysis)
// ============================================================================

/// Comprehensive tracker for expert specialization analysis
/// Matches Python implementation with full analysis capabilities
pub struct ExpertSpecializationTracker {
    /// Number of experts
    pub n_experts: usize,
    /// Token-to-expert routing counts
    pub routing_counts: Vec<Vec<u64>>,  // [n_tokens_types, n_experts]
    /// Total selections per expert
    pub total_selections: Vec<u64>,
    /// Co-occurrence matrix (which experts are selected together)
    pub co_occurrence: Vec<Vec<u64>>,  // [n_experts, n_experts]
    /// Activation patterns for different input categories
    pub category_activations: std::collections::HashMap<String, Vec<u64>>,
    /// Entropy of expert selections (measures specialization)
    pub selection_entropy: Vec<f64>,
    /// Window for moving statistics
    pub window_size: usize,
    /// Current window data
    pub window_data: Vec<Vec<u32>>,
}

impl ExpertSpecializationTracker {
    pub fn new(n_experts: usize, window_size: usize) -> Self {
        Self {
            n_experts,
            routing_counts: vec![vec![0; n_experts]; 10],  // 10 token categories
            total_selections: vec![0; n_experts],
            co_occurrence: vec![vec![0; n_experts]; n_experts],
            category_activations: std::collections::HashMap::new(),
            selection_entropy: vec![0.0; n_experts],
            window_size,
            window_data: Vec::new(),
        }
    }
    
    /// Record a batch of expert selections
    pub fn record_selections(&mut self, expert_indices: &[u32], category: Option<&str>) {
        // Update total selections
        for &idx in expert_indices {
            let i = idx as usize;
            if i < self.n_experts {
                self.total_selections[i] += 1;
            }
        }
        
        // Update co-occurrence matrix
        for i in 0..expert_indices.len() {
            for j in (i + 1)..expert_indices.len() {
                let e1 = expert_indices[i] as usize;
                let e2 = expert_indices[j] as usize;
                if e1 < self.n_experts && e2 < self.n_experts {
                    self.co_occurrence[e1][e2] += 1;
                    self.co_occurrence[e2][e1] += 1;
                }
            }
        }
        
        // Update category activations
        if let Some(cat) = category {
            let entry = self.category_activations.entry(cat.to_string())
                .or_insert_with(|| vec![0; self.n_experts]);
            for &idx in expert_indices {
                let i = idx as usize;
                if i < self.n_experts {
                    entry[i] += 1;
                }
            }
        }
        
        // Update window data
        self.window_data.push(expert_indices.to_vec());
        if self.window_data.len() > self.window_size {
            self.window_data.remove(0);
        }
    }
    
    /// Compute specialization entropy for each expert
    /// Lower entropy = more specialized (selected for specific categories)
    /// Higher entropy = more general (selected uniformly across categories)
    pub fn compute_specialization_entropy(&mut self) {
        let total: u64 = self.total_selections.iter().sum();
        if total == 0 {
            return;
        }
        
        for i in 0..self.n_experts {
            let expert_total = self.total_selections[i];
            if expert_total == 0 {
                self.selection_entropy[i] = 0.0;
                continue;
            }
            
            // Compute entropy based on co-occurrence distribution
            let mut entropy = 0.0;
            let row_sum: u64 = self.co_occurrence[i].iter().sum();
            
            if row_sum > 0 {
                for &count in &self.co_occurrence[i] {
                    if count > 0 {
                        let p = count as f64 / row_sum as f64;
                        entropy -= p * p.ln();
                    }
                }
            }
            
            self.selection_entropy[i] = entropy;
        }
    }
    
    /// Get most specialized experts (lowest entropy)
    pub fn most_specialized(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self.selection_entropy
            .iter()
            .enumerate()
            .map(|(i, &e)| (i, e))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).collect()
    }
    
    /// Get most general experts (highest entropy)  
    pub fn most_general(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self.selection_entropy
            .iter()
            .enumerate()
            .map(|(i, &e)| (i, e))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).collect()
    }
    
    /// Get experts that frequently co-occur
    pub fn get_expert_clusters(&self, threshold: u64) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.n_experts];
        let mut clusters = Vec::new();
        
        for i in 0..self.n_experts {
            if visited[i] {
                continue;
            }
            
            let mut cluster = vec![i];
            visited[i] = true;
            
            for j in (i + 1)..self.n_experts {
                if !visited[j] && self.co_occurrence[i][j] >= threshold {
                    cluster.push(j);
                    visited[j] = true;
                }
            }
            
            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }
        
        clusters
    }
    
    /// Get category-specific expert rankings
    pub fn get_category_rankings(&self, category: &str, k: usize) -> Vec<(usize, u64)> {
        if let Some(activations) = self.category_activations.get(category) {
            let mut indexed: Vec<(usize, u64)> = activations
                .iter()
                .enumerate()
                .map(|(i, &c)| (i, c))
                .collect();
            indexed.sort_by(|a, b| b.1.cmp(&a.1));
            indexed.into_iter().take(k).collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get comprehensive analysis report
    pub fn get_analysis_report(&mut self) -> ExpertAnalysisReport {
        self.compute_specialization_entropy();
        
        let total: u64 = self.total_selections.iter().sum();
        let mean = if self.n_experts > 0 { total as f64 / self.n_experts as f64 } else { 0.0 };
        let variance = self.total_selections
            .iter()
            .map(|&c| ((c as f64) - mean).powi(2))
            .sum::<f64>() / self.n_experts.max(1) as f64;
        let std = variance.sqrt();
        
        ExpertAnalysisReport {
            total_selections: total,
            mean_selections: mean,
            std_selections: std,
            utilization_cv: std / (mean + 1e-6),
            most_used: self.total_selections
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, &c)| (i, c))
                .unwrap_or((0, 0)),
            least_used: self.total_selections
                .iter()
                .enumerate()
                .min_by_key(|(_, &c)| c)
                .map(|(i, &c)| (i, c))
                .unwrap_or((0, 0)),
            most_specialized: self.most_specialized(5),
            most_general: self.most_general(5),
            n_categories: self.category_activations.len(),
            n_clusters: self.get_expert_clusters(10).len(),
        }
    }
    
    /// Reset tracker
    pub fn reset(&mut self) {
        self.total_selections.fill(0);
        for row in &mut self.routing_counts {
            row.fill(0);
        }
        for row in &mut self.co_occurrence {
            row.fill(0);
        }
        self.category_activations.clear();
        self.selection_entropy.fill(0.0);
        self.window_data.clear();
    }
}

/// Expert analysis report
#[derive(Clone, Debug)]
pub struct ExpertAnalysisReport {
    pub total_selections: u64,
    pub mean_selections: f64,
    pub std_selections: f64,
    pub utilization_cv: f64,
    pub most_used: (usize, u64),
    pub least_used: (usize, u64),
    pub most_specialized: Vec<(usize, f64)>,
    pub most_general: Vec<(usize, f64)>,
    pub n_categories: usize,
    pub n_clusters: usize,
}

// ============================================================================
// Expert Capacity Metrics
// ============================================================================

/// Metrics for tracking expert capacity and token dropping
#[derive(Clone, Debug, Default)]
pub struct CapacityMetrics {
    /// Total tokens processed
    pub total_tokens: usize,
    /// Tokens dropped due to capacity overflow
    pub dropped_tokens: usize,
    /// Per-expert overflow counts
    pub expert_overflow: Vec<usize>,
    /// Per-expert utilization (tokens / capacity)
    pub expert_utilization: Vec<f32>,
}

impl CapacityMetrics {
    pub fn new(n_experts: usize) -> Self {
        Self {
            total_tokens: 0,
            dropped_tokens: 0,
            expert_overflow: vec![0; n_experts],
            expert_utilization: vec![0.0; n_experts],
        }
    }
    
    /// Record token dispatch results
    pub fn record_dispatch(
        &mut self,
        expert_id: usize,
        tokens_routed: usize,
        capacity: usize,
    ) {
        self.total_tokens += tokens_routed.min(capacity);
        
        if tokens_routed > capacity {
            let overflow = tokens_routed - capacity;
            self.dropped_tokens += overflow;
            self.expert_overflow[expert_id] += overflow;
        }
        
        self.expert_utilization[expert_id] = tokens_routed as f32 / capacity.max(1) as f32;
    }
    
    /// Get drop rate (fraction of tokens dropped)
    pub fn drop_rate(&self) -> f32 {
        if self.total_tokens + self.dropped_tokens == 0 {
            0.0
        } else {
            self.dropped_tokens as f32 / (self.total_tokens + self.dropped_tokens) as f32
        }
    }
    
    /// Get average utilization across experts
    pub fn avg_utilization(&self) -> f32 {
        if self.expert_utilization.is_empty() {
            0.0
        } else {
            self.expert_utilization.iter().sum::<f32>() / self.expert_utilization.len() as f32
        }
    }
    
    /// Get most overloaded expert
    pub fn most_overloaded_expert(&self) -> (usize, usize) {
        self.expert_overflow
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, &count)| (idx, count))
            .unwrap_or((0, 0))
    }
    
    /// Reset metrics
    pub fn reset(&mut self) {
        self.total_tokens = 0;
        self.dropped_tokens = 0;
        self.expert_overflow.fill(0);
        self.expert_utilization.fill(0.0);
    }
}

// --- Helper Functions ---

fn gelu(x: &Tensor) -> Result<Tensor> {
    // Approx GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c1 = (2.0f64 / std::f64::consts::PI).sqrt();
    let c2 = 0.044715;
    let x3 = x.powf(3.0)?;
    let inner = ((x + (x3 * c2)?)? * c1)?;
    let tanh = inner.tanh()?;
    let res = ((x * 0.5)? * (tanh + 1.0)?)?;
    Ok(res)
}

// --- Expert Module ---

pub struct Expert {
    fc1: Linear,
    fc2: Linear,
}

impl Expert {
    pub fn new(d_model: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear_no_bias(d_model, hidden, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear_no_bias(hidden, d_model, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = gelu(&x)?;
        self.fc2.forward(&x)
    }
}

// --- DeepSeek MoE ---

pub struct DeepSeekMoE {
    d_model: usize,
    n_routed: usize,
    n_shared: usize,
    top_k: usize,
    routed_experts: Vec<Expert>,
    shared_experts: Vec<Expert>,
    centroids: Tensor, // Parameter
    bias: Tensor,      // Buffer (we'll treat as tensor for now, manual update)
    bias_lr: f64,
    /// Expert capacity factor for overflow protection
    capacity_factor: f32,
    /// Whether to drop tokens or route to shared on overflow
    drop_tokens_on_overflow: bool,
    /// Capacity metrics tracker
    capacity_metrics: CapacityMetrics,
}

impl DeepSeekMoE {
    pub fn new(
        d_model: usize,
        n_routed: usize,
        n_shared: usize,
        top_k: usize,
        routed_hidden: usize,
        shared_hidden: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::with_config(d_model, n_routed, n_shared, top_k, routed_hidden, shared_hidden, 1.25, true, vb)
    }
    
    /// Create MoE with capacity configuration
    pub fn with_config(
        d_model: usize,
        n_routed: usize,
        n_shared: usize,
        top_k: usize,
        routed_hidden: usize,
        shared_hidden: usize,
        capacity_factor: f32,
        drop_tokens_on_overflow: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut routed_experts = Vec::new();
        for i in 0..n_routed {
            routed_experts.push(Expert::new(d_model, routed_hidden, vb.pp(&format!("routed.{}", i)))?);
        }

        let mut shared_experts = Vec::new();
        for i in 0..n_shared {
            shared_experts.push(Expert::new(d_model, shared_hidden, vb.pp(&format!("shared.{}", i)))?);
        }

        // Centroids: (n_routed, d_model)
        let centroids = vb.get((n_routed, d_model), "centroids")?;
        
        // Bias: (n_routed) - initialized to zeros
        let bias = Tensor::zeros((n_routed,), DType::F32, vb.device())?;
        
        // Initialize capacity metrics tracker
        let capacity_metrics = CapacityMetrics::new(n_routed);

        Ok(Self {
            d_model,
            n_routed,
            n_shared,
            top_k,
            routed_experts,
            shared_experts,
            centroids,
            bias,
            bias_lr: 0.01,
            capacity_factor,
            drop_tokens_on_overflow,
            capacity_metrics,
        })
    }
    
    /// Compute expert capacity based on batch size and config
    fn compute_capacity(&self, n_tokens: usize) -> usize {
        let tokens_per_expert = (n_tokens * self.top_k) / self.n_routed;
        let capacity = ((tokens_per_expert as f32) * self.capacity_factor).ceil() as usize;
        capacity.max(1)  // At least 1 token capacity
    }
    
    /// Get capacity metrics for monitoring
    pub fn get_capacity_metrics(&self) -> &CapacityMetrics {
        &self.capacity_metrics
    }
    
    /// Reset capacity metrics
    pub fn reset_capacity_metrics(&mut self) {
        self.capacity_metrics.reset();
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_capacity(x, true)
    }
    
    /// Forward pass with capacity limiting
    /// 
    /// # Arguments
    /// * `x` - Input tensor (batch, seq_len, d_model)
    /// * `enforce_capacity` - Whether to enforce capacity limits (set false during inference if needed)
    pub fn forward_with_capacity(&self, x: &Tensor, enforce_capacity: bool) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?; // (N, D)
        let n_tokens = b * s;

        // 1. Shared Path
        let mut shared_out = Tensor::zeros_like(&x_flat)?;
        for exp in &self.shared_experts {
            shared_out = (shared_out + exp.forward(&x_flat)?)?;
        }

        // 2. Router
        // logits: (N, n_routed) = x_flat @ centroids.T + bias
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;

        let logits = logits.contiguous()?;
        // Top-K
        // topk_vals: (N, k), topk_idx: (N, k)
        // Workaround for missing top_k: arg_sort and gather
        let topk_idx = logits.arg_sort_last_dim(true)?.narrow(1, 0, self.top_k)?.contiguous()?;
        let topk_vals = logits.gather(&topk_idx, 1)?;

        // Softmax over top-k
        let gate = ops::softmax(&topk_vals, 1)?; // (N, k)

        // 3. Compute capacity per expert
        let capacity = if enforce_capacity {
            self.compute_capacity(n_tokens)
        } else {
            n_tokens  // No limit
        };

        // 4. Dispatch with capacity limiting
        let mut routed_out = Tensor::zeros_like(&x_flat)?;
        let mut expert_counts = vec![0usize; self.n_routed];
        let mut dropped_count = 0usize;
        
        // Pre-compute topk indices to avoid repeated conversion
        let topk_idx_vec = topk_idx.flatten_all()?.to_vec1::<u32>()?;
        
        // Build per-expert token lists with capacity limiting
        let mut expert_tokens: Vec<Vec<(u32, usize)>> = vec![Vec::new(); self.n_routed];
        
        for (flat_idx, &exp_id) in topk_idx_vec.iter().enumerate() {
            let expert_idx = exp_id as usize;
            if expert_idx >= self.n_routed {
                continue;
            }
            
            let row = flat_idx / self.top_k;
            let col = flat_idx % self.top_k;
            
            // Check capacity
            if expert_counts[expert_idx] < capacity {
                expert_tokens[expert_idx].push((row as u32, col));
                expert_counts[expert_idx] += 1;
            } else {
                // Capacity exceeded - drop token or route to shared
                dropped_count += 1;
            }
        }
        
        // Process each expert
        for i in 0..self.n_routed {
            let tokens = &expert_tokens[i];
            if tokens.is_empty() {
                continue;
            }
            
            let indices: Vec<u32> = tokens.iter().map(|(row, _)| *row).collect();
            let gate_indices: Vec<(usize, usize)> = tokens.iter()
                .map(|(row, col)| (*row as usize, *col))
                .collect();
            
            let indices_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), x.device())?;
            let exp_in = x_flat.index_select(&indices_tensor, 0)?;
            
            let out = self.routed_experts[i].forward(&exp_in)?;
            
            // Get gates
            // We need to gather specific values from `gate` (N, k).
            // gate_indices has (row, col).
            // We can flatten `gate` and select.
            let gate_flat = gate.flatten_all()?;
            let gate_select_indices: Vec<u32> = gate_indices.iter().map(|(r, c)| (r * self.top_k + c) as u32).collect();
            let gate_select_indices_len = gate_select_indices.len();
            let gate_select_indices_tensor = Tensor::from_vec(gate_select_indices, (gate_select_indices_len,), x.device())?;
            let w = gate_flat.index_select(&gate_select_indices_tensor, 0)?.reshape((indices.len(), 1))?;
            
            let weighted_out = out.broadcast_mul(&w)?;
            
            routed_out = routed_out.index_add(&indices_tensor, &weighted_out, 0)?;
        }
        
        // Log capacity metrics (could be made optional via feature flag)
        if dropped_count > 0 {
            tracing::debug!(
                "MoE capacity: {} tokens dropped out of {} ({}%)",
                dropped_count,
                n_tokens * self.top_k,
                (dropped_count as f32 / (n_tokens * self.top_k) as f32) * 100.0
            );
        }

        let routed_out = routed_out.reshape((b, s, d))?;
        let shared_out = shared_out.reshape((b, s, d))?;
        
        Ok((x + shared_out + routed_out)?)
    }
    
    /// Optimized forward pass with sorting for memory coalescing
    /// 
    /// This method sorts tokens by expert assignment before processing,
    /// which improves memory access patterns for large number of experts (256+).
    /// 
    /// # Arguments
    /// * `x` - Input tensor (batch, seq_len, d_model)
    pub fn forward_optimized(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        let n_tokens = b * s;

        // 1. Shared expert path (always active)
        let mut shared_out = Tensor::zeros_like(&x_flat)?;
        for exp in &self.shared_experts {
            shared_out = (shared_out + exp.forward(&x_flat)?)?;
        }

        // 2. Router computation
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;
        let logits = logits.contiguous()?;
        
        // Top-K selection
        let topk_idx = logits.arg_sort_last_dim(true)?.narrow(1, 0, self.top_k)?.contiguous()?;
        let topk_vals = logits.gather(&topk_idx, 1)?;
        let gate = ops::softmax(&topk_vals, 1)?;  // (N, k)

        // 3. Flatten routing for sorted dispatch
        // Expand token indices to match top-k: each token appears k times
        let topk_idx_flat = topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let gate_flat = gate.flatten_all()?.to_vec1::<f32>()?;
        
        // Build (token_id, expert_id, gate_weight) tuples
        let mut routing_info: Vec<(usize, usize, f32)> = Vec::with_capacity(n_tokens * self.top_k);
        for flat_idx in 0..(n_tokens * self.top_k) {
            let token_id = flat_idx / self.top_k;
            let expert_id = topk_idx_flat[flat_idx] as usize;
            let gate_weight = gate_flat[flat_idx];
            if expert_id < self.n_routed {
                routing_info.push((token_id, expert_id, gate_weight));
            }
        }
        
        // Sort by expert ID for memory coalescing
        routing_info.sort_by_key(|&(_, expert_id, _)| expert_id);
        
        // Compute expert boundaries
        let mut expert_counts = vec![0usize; self.n_routed];
        for &(_, expert_id, _) in &routing_info {
            expert_counts[expert_id] += 1;
        }
        
        let capacity = self.compute_capacity(n_tokens);
        let mut boundaries = vec![0usize; self.n_routed + 1];
        for e in 0..self.n_routed {
            boundaries[e + 1] = boundaries[e] + expert_counts[e].min(capacity);
        }
        
        // 4. Process experts with sorted tokens
        let mut output_parts: Vec<(usize, Tensor)> = Vec::new();  // (start_pos, output)
        let mut token_positions: Vec<(usize, usize)> = Vec::new();  // (original_token_id, output_idx)
        let mut current_output_idx = 0;
        
        let mut current_pos = 0;
        for e in 0..self.n_routed {
            let count = expert_counts[e].min(capacity);
            if count == 0 {
                current_pos += expert_counts[e];  // Skip this expert's tokens
                continue;
            }
            
            // Gather tokens for this expert
            let expert_tokens: Vec<_> = routing_info[current_pos..current_pos + count].to_vec();
            let token_ids: Vec<u32> = expert_tokens.iter().map(|&(t, _, _)| t as u32).collect();
            let weights: Vec<f32> = expert_tokens.iter().map(|&(_, _, w)| w).collect();
            
            // Create tensors
            let token_ids_tensor = Tensor::from_vec(token_ids.clone(), (count,), x.device())?;
            let weights_tensor = Tensor::from_vec(weights, (count, 1), x.device())?;
            
            // Gather input tokens
            let expert_input = x_flat.index_select(&token_ids_tensor, 0)?;
            
            // Process through expert
            let expert_out = self.routed_experts[e].forward(&expert_input)?;
            let weighted_out = expert_out.broadcast_mul(&weights_tensor)?;
            
            // Record output positions for scatter
            for (i, &tid) in token_ids.iter().enumerate() {
                token_positions.push((tid as usize, current_output_idx + i));
            }
            
            output_parts.push((current_output_idx, weighted_out));
            current_output_idx += count;
            current_pos += expert_counts[e];  // Move past all tokens for this expert
        }
        
        // 5. Scatter outputs back to original positions
        let mut routed_out = Tensor::zeros_like(&x_flat)?;
        
        for (start_idx, output_tensor) in output_parts {
            let output_len = output_tensor.dim(0)?;
            for i in 0..output_len {
                let global_idx = start_idx + i;
                if let Some(&(original_token_id, _)) = token_positions
                    .iter()
                    .find(|&&(_, out_idx)| out_idx == global_idx)
                {
                    let single_output = output_tensor.narrow(0, i, 1)?;
                    let indices = Tensor::from_vec(vec![original_token_id as u32], (1,), x.device())?;
                    routed_out = routed_out.index_add(&indices, &single_output, 0)?;
                }
            }
        }
        
        // Combine paths
        let routed_out = routed_out.reshape((b, s, d))?;
        let shared_out = shared_out.reshape((b, s, d))?;
        
        Ok((x + shared_out + routed_out)?)
    }
    
    // Manual bias update function (simplified)
    pub fn update_bias(&mut self, x: &Tensor) -> Result<()> {
        // x: (B, S, D)
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        
        // logits = x @ centroids.T + bias
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;
        let logits = logits.contiguous()?;
        
        // topk
        let topk_idx = logits.arg_sort_last_dim(true)?.narrow(1, 0, self.top_k)?.contiguous()?; // (N, k)
        
        // Count selections
        let topk_idx_vec = topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let mut counts = vec![0f32; self.n_routed];
        for &idx in &topk_idx_vec {
            if (idx as usize) < self.n_routed {
                counts[idx as usize] += 1.0;
            }
        }
        
        let avg = counts.iter().sum::<f32>() / (self.n_routed as f32).max(1.0);
        
        // Violation = (avg - count) / (avg + 1e-6)
        // Update bias += lr * tanh(violation)
        
        let mut bias_vec = self.bias.to_vec1::<f32>()?;
        for i in 0..self.n_routed {
            let count = counts[i];
            let violation = (avg - count) / (avg + 1e-6);
            bias_vec[i] += (self.bias_lr as f32) * violation.tanh();
        }
        
        self.bias = Tensor::from_vec(bias_vec, (self.n_routed,), x.device())?;
        
        Ok(())
    }
}

// --- Standard MoE ---

pub struct StandardMoE {
    n_routed: usize,
    top_k: usize,
    experts: Vec<Expert>,
    router: Linear,
}

impl StandardMoE {
    pub fn new(
        d_model: usize,
        n_routed: usize,
        top_k: usize,
        hidden_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut experts = Vec::new();
        for i in 0..n_routed {
            experts.push(Expert::new(d_model, hidden_dim, vb.pp(&format!("experts.{}", i)))?);
        }
        
        let router = candle_nn::linear_no_bias(d_model, n_routed, vb.pp("router"))?;
        
        Ok(Self {
            n_routed,
            top_k,
            experts,
            router,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, f32)> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        let n_tokens = b * s;

        let logits = self.router.forward(&x_flat)?;
        let probs = ops::softmax(&logits, 1)?.contiguous()?;
        
        let topk_idx = probs.arg_sort_last_dim(true)?.narrow(1, 0, self.top_k)?.contiguous()?;
        let topk_vals = probs.gather(&topk_idx, 1)?;
        
        // Normalize gates
        let topk_sum = topk_vals.sum_keepdim(1)?;
        let gates = topk_vals.broadcast_div(&topk_sum)?; // (N, k)
        
        // Aux Loss Calculation
        // f_i = fraction of tokens dispatched to expert i
        // p_i = fraction of router probability allocated to expert i
        
        // p_i: mean of probs over batch
        let p_i = probs.mean(0)?; // (n_routed)
        
        // f_i: count selections / N
        let topk_idx_vec = topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let mut counts = vec![0f32; self.n_routed];
        for &idx in &topk_idx_vec {
            if (idx as usize) < self.n_routed {
                counts[idx as usize] += 1.0;
            }
        }
        let f_i_vec: Vec<f32> = counts.iter().map(|&c| c / (n_tokens as f32)).collect();
        let f_i = Tensor::from_vec(f_i_vec, (self.n_routed,), x.device())?;
        
        let aux_loss = (p_i.mul(&f_i)?.sum_all()?.to_scalar::<f32>()?) * (self.n_routed as f32) * 0.01;

        // Dispatch (Same logic as DeepSeekMoE)
        let mut final_out = Tensor::zeros_like(&x_flat)?;
        
        for i in 0..self.n_routed {
            let mut indices = Vec::new();
            let mut gate_indices = Vec::new();
            
            for (flat_idx, &exp_id) in topk_idx_vec.iter().enumerate() {
                if exp_id as usize == i {
                    let row = flat_idx / self.top_k;
                    let col = flat_idx % self.top_k;
                    indices.push(row as u32);
                    gate_indices.push((row, col));
                }
            }
            
            if indices.is_empty() {
                continue;
            }
            
            let indices_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), x.device())?;
            let exp_in = x_flat.index_select(&indices_tensor, 0)?;
            
            let out = self.experts[i].forward(&exp_in)?;
            
            let gate_flat = gates.flatten_all()?;
            let gate_select_indices: Vec<u32> = gate_indices.iter().map(|(r, c)| (r * self.top_k + c) as u32).collect();
            let gate_select_indices_len = gate_select_indices.len();
            let gate_select_indices_tensor = Tensor::from_vec(gate_select_indices, (gate_select_indices_len,), x.device())?;
            let w = gate_flat.index_select(&gate_select_indices_tensor, 0)?.reshape((indices.len(), 1))?;
            
            let weighted_out = out.broadcast_mul(&w)?;
            
            final_out = final_out.index_add(&indices_tensor, &weighted_out, 0)?;
        }
        
        let final_out = final_out.reshape((b, s, d))?;
        
        Ok((final_out, aux_loss))
    }
}

// --- Expert Parallel DeepSeek MoE ---

/// DeepSeekMoE with Expert Parallelism support.
/// 
/// This variant distributes experts across EP ranks and uses
/// all-to-all communication to route tokens to the correct experts.
pub struct ExpertParallelMoE {
    d_model: usize,
    n_routed: usize,
    n_shared: usize,
    top_k: usize,
    /// Local routed experts (only experts assigned to this rank)
    local_experts: Vec<Expert>,
    /// Shared experts (replicated across all ranks)
    shared_experts: Vec<Expert>,
    centroids: Tensor,
    bias: Tensor,
    bias_lr: f64,
    /// EP dispatcher for token routing
    ep_dispatcher: ExpertParallelDispatch,
    /// EP configuration
    _ep_config: ExpertParallelConfig,
}

impl ExpertParallelMoE {
    /// Create a new EP-enabled MoE layer.
    ///
    /// Args:
    ///   d_model: Model dimension
    ///   n_routed: Total number of routed experts across all ranks
    ///   n_shared: Number of shared experts (replicated on each rank)
    ///   top_k: Number of experts to route each token to
    ///   routed_hidden: Hidden dim for routed experts
    ///   shared_hidden: Hidden dim for shared experts
    ///   ep_dispatcher: Expert parallel dispatcher (None for single-rank mode)
    ///   vb: Variable builder
    pub fn new(
        d_model: usize,
        n_routed: usize,
        n_shared: usize,
        top_k: usize,
        routed_hidden: usize,
        shared_hidden: usize,
        ep_dispatcher: Option<ExpertParallelDispatch>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ep_dispatcher = ep_dispatcher.unwrap_or_else(|| ExpertParallelDispatch::new(n_routed));
        let ep_config = ep_dispatcher.config().clone();
        let local_expert_ids = ep_config.local_expert_ids();
        
        // Only initialize local experts
        let mut local_experts = Vec::new();
        for &i in &local_expert_ids {
            local_experts.push(Expert::new(
                d_model,
                routed_hidden,
                vb.pp(&format!("routed.{}", i))
            )?);
        }

        // Shared experts are replicated
        let mut shared_experts = Vec::new();
        for i in 0..n_shared {
            shared_experts.push(Expert::new(
                d_model,
                shared_hidden,
                vb.pp(&format!("shared.{}", i))
            )?);
        }

        // Centroids: (n_routed, d_model) - full routing table
        let centroids = vb.get((n_routed, d_model), "centroids")?;
        
        // Bias: (n_routed)
        let bias = Tensor::zeros((n_routed,), DType::F32, vb.device())?;

        Ok(Self {
            d_model,
            n_routed,
            n_shared,
            top_k,
            local_experts,
            shared_experts,
            centroids,
            bias,
            bias_lr: 0.01,
            ep_dispatcher,
            _ep_config: ep_config,
        })
    }
    
    /// Forward pass with expert parallelism.
    ///
    /// 1. Compute routing for all tokens
    /// 2. Dispatch tokens to experts using all-to-all
    /// 3. Process tokens with local experts
    /// 4. Combine results using all-to-all
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        let _n_tokens = b * s;

        // 1. Shared Path (replicated computation)
        let mut shared_out = Tensor::zeros_like(&x_flat)?;
        for exp in &self.shared_experts {
            shared_out = (shared_out + exp.forward(&x_flat)?)?;
        }

        // 2. Router - compute routing decisions
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;
        let logits = logits.contiguous()?;
        
        // Top-K routing
        let topk_idx = logits.arg_sort_last_dim(true)?
            .narrow(1, 0, self.top_k)?
            .contiguous()?;
        let topk_vals = logits.gather(&topk_idx, 1)?;
        let gate = ops::softmax(&topk_vals, 1)?;

        // 3. Flatten routing for dispatch
        // For EP, we need to send each token to its target expert's rank
        // We'll process each top-k slot separately
        
        let ep_config = self.ep_dispatcher.config();
        let num_local_experts = ep_config.num_local_experts;
        
        let mut routed_out = Tensor::zeros_like(&x_flat)?;
        
        // For each top-k slot
        for k in 0..self.top_k {
            // Get expert indices for this slot
            let expert_indices = topk_idx.narrow(1, k, 1)?.squeeze(1)?;
            let gate_weights = gate.narrow(1, k, 1)?; // (N, 1)
            
            // Dispatch tokens to appropriate EP rank
            let (dispatched, dispatch_info) = self.ep_dispatcher.dispatch(&x_flat, &expert_indices)?;
            
            // Get expert indices for dispatched tokens (need to convert to local expert id)
            let dispatched_expert_indices = if ep_config.ep_size > 1 {
                // Remap global expert ID to local expert ID
                let dispatched_indices_vec = expert_indices.to_vec1::<u32>()?;
                let local_indices: Vec<u32> = dispatched_indices_vec
                    .iter()
                    .map(|&exp_id| (exp_id as usize % num_local_experts) as u32)
                    .collect();
                Tensor::from_vec(local_indices, expert_indices.shape(), x.device())?
            } else {
                expert_indices.clone()
            };
            
            // Process with local experts
            let local_out = self.process_local_experts(&dispatched, &dispatched_expert_indices)?;
            
            // Combine results back
            let combined = self.ep_dispatcher.combine(&local_out, &dispatch_info)?;
            
            // Weight by gate values
            let weighted = combined.broadcast_mul(&gate_weights)?;
            routed_out = (routed_out + weighted)?;
        }

        let routed_out = routed_out.reshape((b, s, d))?;
        let shared_out = shared_out.reshape((b, s, d))?;
        
        // Residual connection
        Ok((x + shared_out + routed_out)?)
    }
    
    /// Process tokens with local experts.
    fn process_local_experts(&self, x: &Tensor, local_expert_indices: &Tensor) -> Result<Tensor> {
        let _shape = x.dims2()?; // Validate 2D shape
        let indices_vec = local_expert_indices.to_vec1::<u32>()?;
        
        let mut output = Tensor::zeros_like(x)?;
        
        for (local_idx, expert) in self.local_experts.iter().enumerate() {
            // Find tokens routed to this local expert
            let mut token_indices = Vec::new();
            for (i, &exp_id) in indices_vec.iter().enumerate() {
                if exp_id as usize == local_idx {
                    token_indices.push(i as u32);
                }
            }
            
            if token_indices.is_empty() {
                continue;
            }
            
            let indices_tensor = Tensor::from_vec(
                token_indices.clone(),
                (token_indices.len(),),
                x.device()
            )?;
            let exp_in = x.index_select(&indices_tensor, 0)?;
            let exp_out = expert.forward(&exp_in)?;
            
            output = output.index_add(&indices_tensor, &exp_out, 0)?;
        }
        
        Ok(output)
    }
    
    /// Update bias for load balancing.
    pub fn update_bias(&mut self, x: &Tensor) -> Result<()> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;
        let logits = logits.contiguous()?;
        
        let topk_idx = logits.arg_sort_last_dim(true)?
            .narrow(1, 0, self.top_k)?
            .contiguous()?;
        
        let topk_idx_vec = topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let mut counts = vec![0f32; self.n_routed];
        for &idx in &topk_idx_vec {
            if (idx as usize) < self.n_routed {
                counts[idx as usize] += 1.0;
            }
        }
        
        let avg = counts.iter().sum::<f32>() / (self.n_routed as f32).max(1.0);
        
        let mut bias_vec = self.bias.to_vec1::<f32>()?;
        for i in 0..self.n_routed {
            let count = counts[i];
            let violation = (avg - count) / (avg + 1e-6);
            bias_vec[i] += (self.bias_lr as f32) * violation.tanh();
        }
        
        self.bias = Tensor::from_vec(bias_vec, (self.n_routed,), x.device())?;
        
        Ok(())
    }
}

#[cfg(test)]
mod ep_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    
    #[test]
    fn test_expert_parallel_moe_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let d_model = 32;
        let n_routed = 4;
        let n_shared = 1;
        let top_k = 2;
        let routed_hidden = 64;
        let shared_hidden = 64;
        
        let moe = ExpertParallelMoE::new(
            d_model,
            n_routed,
            n_shared,
            top_k,
            routed_hidden,
            shared_hidden,
            None, // Single-rank mode
            vb,
        )?;
        
        let batch_size = 2;
        let seq_len = 8;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
        
        let output = moe.forward(&x)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, d_model]);
        
        Ok(())
    }
}

// ============================================================================
// DeepSeek-V3 MoE (256 Experts, 8 Active, Hierarchical Routing)
// ============================================================================

/// DeepSeek-V3 style MoE layer with:
/// - 256 routed experts with 8 active per token
/// - Hierarchical routing (group selection → expert selection)
/// - Auxiliary-loss-free load balancing via bias adjustment
/// - Expert capacity for efficient batching
pub struct DeepSeekMoEV3 {
    config: DeepSeekMoEV3Config,
    d_model: usize,
    
    /// Routed experts
    routed_experts: Vec<Expert>,
    /// Shared experts (always active)
    shared_experts: Vec<Expert>,
    
    /// Group centroids for first-stage routing: (n_groups, d_model)
    group_centroids: Tensor,
    /// Expert centroids within groups: (n_routed, d_model)
    expert_centroids: Tensor,
    
    /// Load balancing state
    load_balance: LoadBalancingState,
    
    /// Capacity metrics for tracking dropped tokens
    capacity_metrics: CapacityMetrics,
    
    /// Training mode flag
    training: bool,
}

impl DeepSeekMoEV3 {
    /// Create a new DeepSeek-V3 style MoE layer
    pub fn new(config: DeepSeekMoEV3Config, vb: VarBuilder) -> Result<Self> {
        let d_model = config.d_model;
        let n_routed = config.n_routed_experts;
        let n_shared = config.n_shared_experts;
        let n_groups = config.n_expert_groups;
        
        // Create routed experts
        let mut routed_experts = Vec::with_capacity(n_routed);
        for i in 0..n_routed {
            routed_experts.push(Expert::new(
                d_model,
                config.routed_expert_hidden,
                vb.pp(&format!("routed.{}", i))
            )?);
        }
        
        // Create shared experts
        let mut shared_experts = Vec::with_capacity(n_shared);
        for i in 0..n_shared {
            shared_experts.push(Expert::new(
                d_model,
                config.shared_expert_hidden,
                vb.pp(&format!("shared.{}", i))
            )?);
        }
        
        // Group centroids for hierarchical routing
        let group_centroids = vb.get((n_groups, d_model), "group_centroids")?;
        
        // Expert centroids
        let expert_centroids = vb.get((n_routed, d_model), "expert_centroids")?;
        
        // Load balancing state
        let load_balance = LoadBalancingState::new(&config, vb.device())?;
        
        // Capacity metrics
        let capacity_metrics = CapacityMetrics::new(n_routed);
        
        Ok(Self {
            config,
            d_model,
            routed_experts,
            shared_experts,
            group_centroids,
            expert_centroids,
            load_balance,
            capacity_metrics,
            training: true,
        })
    }
    
    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    
    /// Forward pass with hierarchical routing
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        let _n_tokens = b * s;  // May be used for metrics/debugging
        
        // 1. Shared expert path (always active)
        let mut shared_out = Tensor::zeros_like(&x_flat)?;
        for exp in &self.shared_experts {
            shared_out = (shared_out + exp.forward(&x_flat)?)?;
        }
        
        // 2. Hierarchical routing
        let (expert_indices, gates, expert_counts) = self.hierarchical_route(&x_flat)?;
        
        // 3. Dispatch and compute routed experts
        let routed_out = self.dispatch_and_compute(&x_flat, &expert_indices, &gates)?;
        
        // 4. Update load balancing (training only)
        if self.training && self.config.aux_loss_free {
            self.load_balance.update(&expert_counts, x.device())?;
        }
        
        // 5. Combine outputs
        let shared_out = shared_out.reshape((b, s, d))?;
        let routed_out = routed_out.reshape((b, s, d))?;
        
        Ok((x + shared_out + routed_out)?)
    }
    
    /// Hierarchical routing: Group selection → Expert selection within groups
    fn hierarchical_route(&self, x: &Tensor) -> Result<(Tensor, Tensor, Vec<f32>)> {
        let _n_tokens = x.dim(0)?;
        let top_k = self.config.top_k;
        let _n_groups = self.config.n_expert_groups;
        let top_k_groups = self.config.top_k_groups;
        let experts_per_group = self.config.experts_per_group();
        let _experts_per_selected_group = self.config.experts_per_selected_group();
        
        // Stage 1: Group selection
        // Compute group affinities: (N, n_groups)
        let group_logits = x.matmul(&self.group_centroids.transpose(0, 1)?)?;
        
        // Select top-k groups
        let group_topk_idx = group_logits
            .arg_sort_last_dim(true)?
            .narrow(1, 0, top_k_groups)?
            .contiguous()?; // (N, top_k_groups)
        
        // Stage 2: Expert selection within selected groups
        // Compute expert affinities: (N, n_routed)
        let mut expert_logits = x.matmul(&self.expert_centroids.transpose(0, 1)?)?;
        
        // Add load balancing bias
        expert_logits = expert_logits.broadcast_add(self.load_balance.get_bias())?;
        
        // Mask experts not in selected groups
        let masked_logits = self.mask_experts_by_group(
            &expert_logits,
            &group_topk_idx,
            experts_per_group
        )?;
        
        // Select top experts from unmasked
        let expert_topk_idx = masked_logits
            .arg_sort_last_dim(true)?
            .narrow(1, 0, top_k)?
            .contiguous()?; // (N, top_k)
        
        let expert_topk_vals = expert_logits.gather(&expert_topk_idx, 1)?;
        
        // Compute gates via softmax over selected experts
        let gates = ops::softmax(&expert_topk_vals, 1)?; // (N, top_k)
        
        // Count expert selections for load balancing
        let expert_counts = self.count_expert_selections(&expert_topk_idx)?;
        
        Ok((expert_topk_idx, gates, expert_counts))
    }
    
    /// Mask experts not in selected groups
    fn mask_experts_by_group(
        &self,
        expert_logits: &Tensor,
        group_topk_idx: &Tensor,
        experts_per_group: usize,
    ) -> Result<Tensor> {
        let n_tokens = expert_logits.dim(0)?;
        let n_routed = self.config.n_routed_experts;
        
        // Create mask: (N, n_routed)
        let group_idx_vec = group_topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let top_k_groups = self.config.top_k_groups;
        
        let mut mask_data = vec![f32::NEG_INFINITY; n_tokens * n_routed];
        
        for token_idx in 0..n_tokens {
            for k in 0..top_k_groups {
                let group_id = group_idx_vec[token_idx * top_k_groups + k] as usize;
                let expert_start = group_id * experts_per_group;
                let expert_end = expert_start + experts_per_group;
                
                for expert_id in expert_start..expert_end {
                    if expert_id < n_routed {
                        mask_data[token_idx * n_routed + expert_id] = 0.0;
                    }
                }
            }
        }
        
        let mask = Tensor::from_vec(mask_data, (n_tokens, n_routed), expert_logits.device())?;
        
        expert_logits + mask
    }
    
    /// Count expert selections for load balancing updates
    fn count_expert_selections(&self, expert_topk_idx: &Tensor) -> Result<Vec<f32>> {
        let idx_vec = expert_topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let n_routed = self.config.n_routed_experts;
        
        let mut counts = vec![0.0f32; n_routed];
        for &idx in &idx_vec {
            if (idx as usize) < n_routed {
                counts[idx as usize] += 1.0;
            }
        }
        
        Ok(counts)
    }
    
    /// Dispatch tokens to experts and compute outputs
    fn dispatch_and_compute(
        &mut self,
        x: &Tensor,
        expert_indices: &Tensor,
        gates: &Tensor,
    ) -> Result<Tensor> {
        let n_tokens = x.dim(0)?;
        let d = self.d_model;
        let top_k = self.config.top_k;
        let n_routed = self.config.n_routed_experts;
        
        // Flatten indices and gates for processing
        let idx_vec = expert_indices.flatten_all()?.to_vec1::<u32>()?;
        let gate_flat = gates.flatten_all()?;
        
        let mut routed_out = Tensor::zeros((n_tokens, d), DType::F32, x.device())?;
        
        // Reset capacity metrics for this forward pass
        self.capacity_metrics.reset();
        
        // Process each expert
        for expert_id in 0..n_routed {
            // Find tokens routed to this expert
            let mut token_indices = Vec::new();
            let mut gate_indices = Vec::new();
            
            for (flat_idx, &exp_id) in idx_vec.iter().enumerate() {
                if exp_id as usize == expert_id {
                    let token_idx = flat_idx / top_k;
                    token_indices.push(token_idx as u32);
                    gate_indices.push(flat_idx as u32);
                }
            }
            
            if token_indices.is_empty() {
                continue;
            }
            
            // Apply capacity constraint
            let capacity = ((n_tokens as f32 / n_routed as f32) * top_k as f32 * self.config.capacity_factor) as usize;
            let capacity = capacity.max(self.config.min_tokens_per_expert);
            
            // Record capacity metrics before truncation
            let tokens_routed = token_indices.len();
            self.capacity_metrics.record_dispatch(expert_id, tokens_routed, capacity);
            
            if token_indices.len() > capacity {
                token_indices.truncate(capacity);
                gate_indices.truncate(capacity);
            }
            
            // Gather inputs
            let indices_tensor = Tensor::from_vec(token_indices.clone(), (token_indices.len(),), x.device())?;
            let exp_in = x.index_select(&indices_tensor, 0)?;
            
            // Process through expert
            let exp_out = self.routed_experts[expert_id].forward(&exp_in)?;
            
            // Gather gates
            let gate_indices_tensor = Tensor::from_vec(gate_indices.clone(), (gate_indices.len(),), x.device())?;
            let token_gates = gate_flat.index_select(&gate_indices_tensor, 0)?.reshape((token_indices.len(), 1))?;
            
            // Weight by gates
            let weighted_out = exp_out.broadcast_mul(&token_gates)?;
            
            // Scatter add to output
            routed_out = routed_out.index_add(&indices_tensor, &weighted_out, 0)?;
        }
        
        Ok(routed_out)
    }
    
    /// Get load balancing statistics
    pub fn get_load_balance_stats(&self) -> (f32, f32, f32) {
        self.load_balance.get_stats()
    }
    
    /// Get capacity metrics
    pub fn get_capacity_metrics(&self) -> &CapacityMetrics {
        &self.capacity_metrics
    }
    
    /// Reset capacity metrics
    pub fn reset_capacity_metrics(&mut self) {
        self.capacity_metrics.reset();
    }
    
    /// Get configuration
    pub fn config(&self) -> &DeepSeekMoEV3Config {
        &self.config
    }
}

#[cfg(test)]
mod v3_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    
    #[test]
    fn test_deepseek_moe_v3_small() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let mut config = DeepSeekMoEV3Config::small_16_2();
        config.d_model = 32;
        config.routed_expert_hidden = 64;
        config.shared_expert_hidden = 64;
        
        let mut moe = DeepSeekMoEV3::new(config, vb)?;
        
        let batch_size = 2;
        let seq_len = 8;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, 32), &device)?;
        
        let output = moe.forward(&x)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, 32]);
        
        Ok(())
    }
    
    #[test]
    fn test_load_balancing_state() -> Result<()> {
        let device = Device::Cpu;
        let config = DeepSeekMoEV3Config::small_16_2();
        
        let mut state = LoadBalancingState::new(&config, &device)?;
        
        // Simulate uneven expert selection (16 experts total)
        // Total = 100, target per expert = 100/16 = 6.25
        let mut counts = vec![6.25f32; 16]; // Start with balanced counts
        counts[0] = 20.0; // Expert 0 overused (20 > 6.25)
        counts[1] = 0.5;  // Expert 1 underused (0.5 < 6.25)
        
        state.update(&counts, &device)?;
        
        let bias = state.get_bias();
        let bias_vec = bias.to_vec1::<f32>()?;
        
        // Expert 0 should have negative bias (discourage) - count > target
        // Expert 1 should have positive bias (encourage) - count < target
        assert!(bias_vec[0] < 0.0, "Overused expert should have negative bias, got {}", bias_vec[0]);
        assert!(bias_vec[1] > 0.0, "Underused expert should have positive bias, got {}", bias_vec[1]);
        
        Ok(())
    }
    
    #[test]
    fn test_hierarchical_routing() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let mut config = DeepSeekMoEV3Config::small_16_2();
        config.d_model = 32;
        config.routed_expert_hidden = 64;
        config.shared_expert_hidden = 64;
        
        let moe = DeepSeekMoEV3::new(config.clone(), vb)?;
        
        let x = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        
        let (indices, gates, counts) = moe.hierarchical_route(&x)?;
        
        // Check shapes
        assert_eq!(indices.dims(), &[4, config.top_k]);
        assert_eq!(gates.dims(), &[4, config.top_k]);
        assert_eq!(counts.len(), config.n_routed_experts);
        
        // Gates should sum to 1
        let gate_sums = gates.sum(1)?;
        let gate_sums_vec = gate_sums.to_vec1::<f32>()?;
        for sum in gate_sums_vec {
            assert!((sum - 1.0).abs() < 1e-5, "Gates should sum to 1, got {}", sum);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_load_balancing_history() -> Result<()> {
        let device = Device::Cpu;
        let config = DeepSeekMoEV3Config::small_16_2();
        
        let mut state = LoadBalancingState::with_history_size(&config, &device, 100)?;
        
        // Perform multiple updates
        for i in 0..10 {
            let counts = vec![(i as f32 + 1.0); 16];
            state.update(&counts, &device)?;
        }
        
        // Check history was recorded
        let bias_history = state.get_bias_history();
        let load_history = state.get_load_history();
        
        assert_eq!(bias_history.len(), 10, "Should have 10 bias history entries");
        assert_eq!(load_history.len(), 10, "Should have 10 load history entries");
        
        // Check each history entry has correct size
        for entry in bias_history {
            assert_eq!(entry.len(), 16, "Each bias entry should have 16 values");
        }
        
        for entry in load_history {
            assert_eq!(entry.len(), 16, "Each load entry should have 16 values");
        }
        
        Ok(())
    }
    
    #[test]
    fn test_load_balancing_history_limit() -> Result<()> {
        let device = Device::Cpu;
        let config = DeepSeekMoEV3Config::small_16_2();
        
        // Create state with small history limit
        let mut state = LoadBalancingState::with_history_size(&config, &device, 5)?;
        
        // Perform more updates than history size
        for i in 0..20 {
            let counts = vec![(i as f32 + 1.0); 16];
            state.update(&counts, &device)?;
        }
        
        // History should be capped at max size
        let bias_history = state.get_bias_history();
        let load_history = state.get_load_history();
        
        assert_eq!(bias_history.len(), 5, "History should be capped at 5");
        assert_eq!(load_history.len(), 5, "History should be capped at 5");
        
        Ok(())
    }
    
    #[test]
    fn test_load_balance_detailed_stats() -> Result<()> {
        let device = Device::Cpu;
        let config = DeepSeekMoEV3Config::small_16_2();
        
        let mut state = LoadBalancingState::new(&config, &device)?;
        
        // Update with varied counts
        let counts = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
                          8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        state.update(&counts, &device)?;
        
        let stats = state.get_detailed_stats();
        
        // Check all fields are populated
        assert!(stats.mean_count > 0.0);
        assert!(stats.std_count > 0.0);
        assert!(stats.load_balance_cv > 0.0);
        assert_eq!(stats.step, 1);
        
        Ok(())
    }
    
    #[test]
    fn test_expert_specialization_tracker() {
        let mut tracker = ExpertSpecializationTracker::new(16, 100);
        
        // Record selections from different categories
        tracker.record_selections(&[0, 1, 2], Some("code"));
        tracker.record_selections(&[0, 3, 4], Some("code"));
        tracker.record_selections(&[5, 6, 7], Some("math"));
        tracker.record_selections(&[5, 8, 9], Some("math"));
        
        // Check total selections
        assert_eq!(tracker.total_selections[0], 2); // Expert 0 selected twice
        assert_eq!(tracker.total_selections[5], 2); // Expert 5 selected twice
        
        // Check category activations
        let code_rankings = tracker.get_category_rankings("code", 5);
        assert!(!code_rankings.is_empty());
        
        // Expert 0 should be top for "code" category
        assert_eq!(code_rankings[0].0, 0);
    }
    
    #[test]
    fn test_expert_specialization_entropy() {
        let mut tracker = ExpertSpecializationTracker::new(8, 100);
        
        // Create specialized pattern - expert 0 always with experts 1,2
        for _ in 0..10 {
            tracker.record_selections(&[0, 1, 2], None);
        }
        
        // Create general pattern - expert 3 with different experts
        tracker.record_selections(&[3, 4, 5], None);
        tracker.record_selections(&[3, 5, 6], None);
        tracker.record_selections(&[3, 6, 7], None);
        
        tracker.compute_specialization_entropy();
        
        // Expert 0 should have lower entropy (more specialized)
        // Expert 3 should have higher entropy (more general)
        let most_specialized = tracker.most_specialized(3);
        let most_general = tracker.most_general(3);
        
        assert!(!most_specialized.is_empty());
        assert!(!most_general.is_empty());
    }
    
    #[test]
    fn test_expert_specialization_clusters() {
        let mut tracker = ExpertSpecializationTracker::new(8, 100);
        
        // Create cluster of experts 0, 1, 2 (always selected together)
        for _ in 0..20 {
            tracker.record_selections(&[0, 1, 2], None);
        }
        
        // Create another cluster of experts 5, 6, 7
        for _ in 0..20 {
            tracker.record_selections(&[5, 6, 7], None);
        }
        
        // Find clusters with threshold 10 (should find both clusters)
        let clusters = tracker.get_expert_clusters(10);
        
        // Should find at least one cluster
        assert!(!clusters.is_empty());
    }
    
    #[test]
    fn test_expert_analysis_report() {
        let mut tracker = ExpertSpecializationTracker::new(8, 100);
        
        // Add some data
        for i in 0..100 {
            tracker.record_selections(&[(i % 8) as u32, ((i + 1) % 8) as u32], Some("test"));
        }
        
        let report = tracker.get_analysis_report();
        
        assert!(report.total_selections > 0);
        assert!(report.mean_selections > 0.0);
        assert_eq!(report.n_categories, 1);
        assert!(!report.most_specialized.is_empty());
        assert!(!report.most_general.is_empty());
    }
    
    #[test]
    fn test_expert_frequency_tracker() {
        let mut tracker = ExpertFrequencyTracker::new(16, 100, 50);
        
        // Record batches
        tracker.record_batch(&[0, 1, 2, 3]);
        tracker.record_batch(&[0, 0, 1, 1]);
        
        // Check counts
        assert_eq!(tracker.total_counts[0], 3); // Expert 0 selected 3 times
        assert_eq!(tracker.total_counts[1], 3); // Expert 1 selected 3 times
        assert_eq!(tracker.total_counts[2], 1);
        assert_eq!(tracker.total_counts[3], 1);
        
        // Check top experts
        let top = tracker.top_experts(2);
        assert!(top[0].0 == 0 || top[0].0 == 1); // Either 0 or 1 is top
        
        // Check utilization stats
        let (mean, std, cv) = tracker.utilization_stats();
        assert!(mean > 0.0);
        assert!(std >= 0.0);
        assert!(cv >= 0.0);
    }
    
    #[test]
    fn test_capacity_metrics() {
        let mut metrics = CapacityMetrics::new(8);
        
        // Record dispatches - some with overflow
        metrics.record_dispatch(0, 100, 50);  // Overflow: 50 dropped
        metrics.record_dispatch(1, 30, 50);   // No overflow
        metrics.record_dispatch(2, 80, 50);   // Overflow: 30 dropped
        
        // Check drop rate
        let drop_rate = metrics.drop_rate();
        assert!(drop_rate > 0.0, "Should have non-zero drop rate");
        
        // Check most overloaded
        let (expert, overflow) = metrics.most_overloaded_expert();
        assert_eq!(expert, 0, "Expert 0 should be most overloaded");
        assert_eq!(overflow, 50, "Expert 0 dropped 50 tokens");
        
        // Check average utilization
        let avg_util = metrics.avg_utilization();
        assert!(avg_util > 0.0);
    }
    
    #[test]
    fn test_deepseek_moe_capacity_limiting() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let d_model = 32;
        let n_routed = 4;
        let n_shared = 1;
        let top_k = 2;
        let routed_hidden = 64;
        let shared_hidden = 64;
        let capacity_factor = 1.5;  // 50% extra capacity
        
        // Create MoE with capacity limiting
        let moe = DeepSeekMoE::with_config(
            d_model,
            n_routed,
            n_shared,
            top_k,
            routed_hidden,
            shared_hidden,
            capacity_factor,
            true,  // drop_tokens_on_overflow
            vb,
        )?;
        
        // Test that capacity is computed correctly
        let n_tokens = 100;
        let expected_capacity = ((n_tokens * top_k / n_routed) as f32 * capacity_factor).ceil() as usize;
        let actual_capacity = moe.compute_capacity(n_tokens);
        assert_eq!(actual_capacity, expected_capacity, 
            "Capacity should be {} but got {}", expected_capacity, actual_capacity);
        
        // Run forward pass with capacity enforcement
        let batch_size = 4;
        let seq_len = 8;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
        
        let output = moe.forward_with_capacity(&x, true)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, d_model]);
        
        // Also test without capacity enforcement
        let output_no_cap = moe.forward_with_capacity(&x, false)?;
        assert_eq!(output_no_cap.dims(), &[batch_size, seq_len, d_model]);
        
        Ok(())
    }
    
    #[test]
    fn test_deepseek_moe_basic_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let moe = DeepSeekMoE::new(32, 4, 1, 2, 64, 64, vb)?;
        
        let x = Tensor::randn(0f32, 1f32, (2, 8, 32), &device)?;
        let output = moe.forward(&x)?;
        
        assert_eq!(output.dims(), &[2, 8, 32]);
        Ok(())
    }
}
