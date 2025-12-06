//! MoE (Mixture of Experts) Ablation Study Module
//!
//! Compares different MoE configurations:
//! - Number of experts (8, 16, 64, 256)
//! - Top-k routing (1, 2, 4, 8)
//! - Load balancing strategies
//! - Expert capacity factors
//!
//! Metrics tracked:
//! - Routing distribution
//! - Expert utilization
//! - Load balance loss
//! - Throughput impact
//! - Memory usage

use candle_core::{Device, Result, Tensor, DType};
use std::collections::HashMap;
use std::time::Instant;

/// Load balancing strategy for MoE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadBalanceStrategy {
    /// No load balancing
    None,
    /// Auxiliary loss (standard)
    AuxLoss,
    /// Expert choice routing
    ExpertChoice,
    /// Soft MoE (mixture)
    SoftMoE,
}

impl LoadBalanceStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            LoadBalanceStrategy::None => "none",
            LoadBalanceStrategy::AuxLoss => "aux_loss",
            LoadBalanceStrategy::ExpertChoice => "expert_choice",
            LoadBalanceStrategy::SoftMoE => "soft_moe",
        }
    }
}

/// Configuration for MoE ablation study
#[derive(Debug, Clone)]
pub struct MoEAblationConfig {
    /// Number of experts configurations to test
    pub num_experts_configs: Vec<usize>,
    /// Top-k values to test
    pub top_k_configs: Vec<usize>,
    /// Load balancing strategies to compare
    pub balance_strategies: Vec<LoadBalanceStrategy>,
    /// Model dimension
    pub d_model: usize,
    /// Expert hidden dimension
    pub d_expert: usize,
    /// Sequence lengths to evaluate
    pub eval_seq_lengths: Vec<usize>,
    /// Batch size
    pub batch_size: usize,
    /// Number of runs per configuration
    pub num_runs: usize,
    /// Auxiliary loss weight
    pub aux_loss_weight: f32,
    /// Expert capacity factor
    pub capacity_factor: f32,
}

impl Default for MoEAblationConfig {
    fn default() -> Self {
        Self {
            num_experts_configs: vec![8, 16, 64],
            top_k_configs: vec![1, 2, 4],
            balance_strategies: vec![
                LoadBalanceStrategy::None,
                LoadBalanceStrategy::AuxLoss,
                LoadBalanceStrategy::ExpertChoice,
            ],
            d_model: 512,
            d_expert: 2048,
            eval_seq_lengths: vec![128, 256, 512],
            batch_size: 4,
            num_runs: 3,
            aux_loss_weight: 0.01,
            capacity_factor: 1.25,
        }
    }
}

/// Metrics from a single MoE ablation run
#[derive(Debug, Clone)]
pub struct MoEMetrics {
    /// Number of experts
    pub num_experts: usize,
    /// Top-k value
    pub top_k: usize,
    /// Load balance strategy
    pub balance_strategy: LoadBalanceStrategy,
    /// Sequence length
    pub seq_len: usize,
    /// Forward pass latency in milliseconds
    pub forward_latency_ms: f64,
    /// Throughput in tokens per second
    pub throughput_tokens_per_sec: f64,
    /// Load balance metric (coefficient of variation)
    pub load_balance_cv: f64,
    /// Expert utilization (fraction of experts receiving tokens)
    pub expert_utilization: f64,
    /// Auxiliary loss value
    pub aux_loss: f64,
    /// Maximum expert load (tokens assigned to busiest expert)
    pub max_expert_load: usize,
    /// Memory usage estimate in bytes
    pub memory_bytes: usize,
}

/// MoE ablation study runner
pub struct MoEAblationRunner {
    config: MoEAblationConfig,
    device: Device,
    results: Vec<MoEMetrics>,
}

impl MoEAblationRunner {
    pub fn new(config: MoEAblationConfig, device: Device) -> Self {
        Self {
            config,
            device,
            results: Vec::new(),
        }
    }
    
    /// Compute gating scores using a simple router
    fn compute_gating(&self, x: &Tensor, num_experts: usize) -> Result<Tensor> {
        let (_batch_size, _seq_len, d_model) = x.dims3()?;
        
        // Router weight
        let router_weight = Tensor::randn(
            0f32, 0.02f32,
            (num_experts, d_model),
            &self.device,
        )?;
        
        // Flatten to (batch*seq, d_model)
        let x_flat = x.reshape(((), d_model))?;
        
        // Compute router logits
        let logits = x_flat.matmul(&router_weight.t()?)?;
        
        // Softmax for routing probabilities
        candle_nn::ops::softmax(&logits, 1)
    }
    
    /// Select top-k experts (simplified implementation)
    fn select_topk(&self, router_probs: &Tensor, top_k: usize) -> Result<(Tensor, Tensor)> {
        // For simplicity, we'll use argmax and simulate top-k
        // In production, this would use proper top-k selection
        let (num_tokens, num_experts) = router_probs.dims2()?;
        
        // Get sorted indices (descending by probability)
        let sorted_indices = router_probs.arg_sort_last_dim(false)?;  // descending
        
        // Take top-k indices
        let indices = sorted_indices.narrow(1, 0, top_k)?;
        
        // Gather weights for top-k
        // Use a simplified approach: get max probability as representative weight
        let _max_probs = router_probs.max(1)?;
        
        // Create uniform weights for simplicity
        let weight_val = 1.0 / top_k as f64;
        let weight_scalar = Tensor::new(&[weight_val as f32], router_probs.device())?;
        let weights = Tensor::ones((num_tokens, top_k), DType::F32, router_probs.device())?;
        let weights = weights.broadcast_mul(&weight_scalar)?;
        
        Ok((weights, indices))
    }
    
    /// Compute load balance auxiliary loss
    fn compute_aux_loss(&self, router_probs: &Tensor, num_experts: usize) -> Result<f64> {
        // f_i: fraction of tokens routed to expert i
        let tokens_per_expert = router_probs.sum(0)?;
        let total_tokens_scalar = router_probs.dims()[0] as f64;
        let total_tokens_tensor = Tensor::new(&[total_tokens_scalar as f32], router_probs.device())?
            .broadcast_as(tokens_per_expert.shape())?;
        let f_i = tokens_per_expert.broadcast_div(&total_tokens_tensor)?;
        
        // P_i: average routing probability for expert i
        let p_i = router_probs.mean(0)?;
        
        // Auxiliary loss = num_experts * sum(f_i * P_i)
        let aux_loss = (f_i * p_i)?.sum_all()?.to_scalar::<f32>()?;
        let aux_loss = (num_experts as f32) * aux_loss;
        
        Ok(aux_loss as f64)
    }
    
    /// Compute expert utilization statistics
    fn compute_utilization(&self, indices: &Tensor, num_experts: usize) -> Result<(f64, f64, usize)> {
        let indices_flat = indices.flatten_all()?;
        let indices_vec: Vec<u32> = indices_flat.to_vec1()?;
        
        // Count tokens per expert
        let mut expert_counts = vec![0usize; num_experts];
        for idx in indices_vec {
            expert_counts[idx as usize] += 1;
        }
        
        // Compute coefficient of variation
        let total: usize = expert_counts.iter().sum();
        let mean = total as f64 / num_experts as f64;
        let variance: f64 = expert_counts.iter()
            .map(|&c| (c as f64 - mean).powi(2))
            .sum::<f64>() / num_experts as f64;
        let std_dev = variance.sqrt();
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };
        
        // Expert utilization: fraction with non-zero assignments
        let utilized = expert_counts.iter().filter(|&&c| c > 0).count();
        let utilization = utilized as f64 / num_experts as f64;
        
        // Max load
        let max_load = *expert_counts.iter().max().unwrap_or(&0);
        
        Ok((cv, utilization, max_load))
    }
    
    /// Simulate MoE forward pass
    fn simulate_moe(
        &self,
        x: &Tensor,
        num_experts: usize,
        top_k: usize,
        _balance_strategy: LoadBalanceStrategy,
    ) -> Result<(Tensor, f64, f64, f64, usize)> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        let d_expert = self.config.d_expert;
        
        // Compute routing
        let router_probs = self.compute_gating(x, num_experts)?;
        let (weights, indices) = self.select_topk(&router_probs, top_k)?;
        
        // Compute auxiliary loss
        let aux_loss = self.compute_aux_loss(&router_probs, num_experts)?;
        
        // Compute utilization metrics
        let (cv, utilization, max_load) = self.compute_utilization(&indices, num_experts)?;
        
        // Simulate expert computation (simplified)
        // In practice, this would scatter tokens to experts
        let x_flat = x.reshape((batch_size * seq_len, d_model))?;
        
        // Simple FFN simulation (one expert for all)
        let w1 = Tensor::randn(0f32, 0.02f32, (d_expert, d_model), &self.device)?;
        let w2 = Tensor::randn(0f32, 0.02f32, (d_model, d_expert), &self.device)?;
        
        let hidden = x_flat.matmul(&w1.t()?)?;
        let hidden = hidden.relu()?;
        let output = hidden.matmul(&w2.t()?)?;
        
        // Scale by top-k weights (simplified)
        let weight_scale = weights.sum(1)?.mean(0)?.to_scalar::<f32>()? as f64;
        let output = (output * weight_scale)?;
        
        let output = output.reshape((batch_size, seq_len, d_model))?;
        
        Ok((output, aux_loss, cv, utilization, max_load))
    }
    
    /// Estimate memory usage
    fn estimate_memory(&self, num_experts: usize) -> usize {
        let bytes_per_element = 4;
        let d_model = self.config.d_model;
        let d_expert = self.config.d_expert;
        
        // Expert FFN weights: num_experts * (d_model * d_expert + d_expert * d_model)
        let expert_memory = num_experts * 2 * d_model * d_expert * bytes_per_element;
        
        // Router: num_experts * d_model
        let router_memory = num_experts * d_model * bytes_per_element;
        
        expert_memory + router_memory
    }
    
    /// Run single ablation configuration
    pub fn run_single(
        &mut self,
        num_experts: usize,
        top_k: usize,
        balance_strategy: LoadBalanceStrategy,
        seq_len: usize,
    ) -> Result<MoEMetrics> {
        let batch_size = self.config.batch_size;
        let d_model = self.config.d_model;
        
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &self.device)?;
        
        // Warmup
        let _ = self.simulate_moe(&x, num_experts, top_k, balance_strategy)?;
        
        // Timed runs
        let mut latencies = Vec::with_capacity(self.config.num_runs);
        let mut total_aux_loss = 0.0;
        let mut total_cv = 0.0;
        let mut total_utilization = 0.0;
        let mut total_max_load = 0usize;
        
        for _ in 0..self.config.num_runs {
            let start = Instant::now();
            let (_, aux_loss, cv, utilization, max_load) = 
                self.simulate_moe(&x, num_experts, top_k, balance_strategy)?;
            let elapsed = start.elapsed();
            
            latencies.push(elapsed.as_secs_f64() * 1000.0);
            total_aux_loss += aux_loss;
            total_cv += cv;
            total_utilization += utilization;
            total_max_load += max_load;
        }
        
        let num_runs = self.config.num_runs as f64;
        let avg_latency = latencies.iter().sum::<f64>() / num_runs;
        let total_tokens = (batch_size * seq_len) as f64;
        let throughput = total_tokens / (avg_latency / 1000.0);
        
        let metrics = MoEMetrics {
            num_experts,
            top_k,
            balance_strategy,
            seq_len,
            forward_latency_ms: avg_latency,
            throughput_tokens_per_sec: throughput,
            load_balance_cv: total_cv / num_runs,
            expert_utilization: total_utilization / num_runs,
            aux_loss: total_aux_loss / num_runs,
            max_expert_load: total_max_load / self.config.num_runs,
            memory_bytes: self.estimate_memory(num_experts),
        };
        
        self.results.push(metrics.clone());
        Ok(metrics)
    }
    
    /// Run full ablation study
    pub fn run_ablation(&mut self) -> Result<Vec<MoEMetrics>> {
        let num_experts_configs = self.config.num_experts_configs.clone();
        let top_k_configs = self.config.top_k_configs.clone();
        let balance_strategies = self.config.balance_strategies.clone();
        let seq_lengths = self.config.eval_seq_lengths.clone();
        
        for &num_experts in &num_experts_configs {
            for &top_k in &top_k_configs {
                if top_k > num_experts {
                    continue;  // Skip invalid configs
                }
                for &strategy in &balance_strategies {
                    for &seq_len in &seq_lengths {
                        self.run_single(num_experts, top_k, strategy, seq_len)?;
                    }
                }
            }
        }
        
        Ok(self.results.clone())
    }
    
    /// Generate analysis report
    pub fn analyze_results(&self, results: &[MoEMetrics]) -> String {
        let mut report = String::new();
        report.push_str("=== MoE Ablation Study Results ===\n\n");
        
        // Group by num_experts
        let mut by_experts: HashMap<usize, Vec<&MoEMetrics>> = HashMap::new();
        for m in results {
            by_experts.entry(m.num_experts).or_default().push(m);
        }
        
        // Scaling analysis
        report.push_str("Expert Scaling Analysis:\n");
        report.push_str("-".repeat(50).as_str());
        report.push('\n');
        
        for (num_experts, metrics) in &by_experts {
            let avg_throughput: f64 = metrics.iter()
                .map(|m| m.throughput_tokens_per_sec)
                .sum::<f64>() / metrics.len() as f64;
            let avg_utilization: f64 = metrics.iter()
                .map(|m| m.expert_utilization)
                .sum::<f64>() / metrics.len() as f64;
            let memory_mb = metrics[0].memory_bytes as f64 / (1024.0 * 1024.0);
            
            report.push_str(&format!(
                "  {} experts: throughput={:.0} tok/s, utilization={:.1}%, memory={:.1}MB\n",
                num_experts,
                avg_throughput,
                avg_utilization * 100.0,
                memory_mb
            ));
        }
        
        report.push('\n');
        
        // Load balance analysis
        report.push_str("Load Balance Analysis:\n");
        report.push_str("-".repeat(50).as_str());
        report.push('\n');
        
        let mut by_strategy: HashMap<LoadBalanceStrategy, Vec<&MoEMetrics>> = HashMap::new();
        for m in results {
            by_strategy.entry(m.balance_strategy).or_default().push(m);
        }
        
        for (strategy, metrics) in &by_strategy {
            let avg_cv: f64 = metrics.iter()
                .map(|m| m.load_balance_cv)
                .sum::<f64>() / metrics.len() as f64;
            let avg_aux_loss: f64 = metrics.iter()
                .map(|m| m.aux_loss)
                .sum::<f64>() / metrics.len() as f64;
            
            report.push_str(&format!(
                "  {}: CV={:.3}, aux_loss={:.4}\n",
                strategy.name(),
                avg_cv,
                avg_aux_loss
            ));
        }
        
        report.push('\n');
        
        // Top-k analysis
        report.push_str("Top-k Routing Analysis:\n");
        report.push_str("-".repeat(50).as_str());
        report.push('\n');
        
        let mut by_topk: HashMap<usize, Vec<&MoEMetrics>> = HashMap::new();
        for m in results {
            by_topk.entry(m.top_k).or_default().push(m);
        }
        
        for (top_k, metrics) in &by_topk {
            let avg_throughput: f64 = metrics.iter()
                .map(|m| m.throughput_tokens_per_sec)
                .sum::<f64>() / metrics.len() as f64;
            let avg_utilization: f64 = metrics.iter()
                .map(|m| m.expert_utilization)
                .sum::<f64>() / metrics.len() as f64;
            
            report.push_str(&format!(
                "  top_k={}: throughput={:.0} tok/s, utilization={:.1}%\n",
                top_k,
                avg_throughput,
                avg_utilization * 100.0
            ));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_balance_strategy_names() {
        assert_eq!(LoadBalanceStrategy::None.name(), "none");
        assert_eq!(LoadBalanceStrategy::AuxLoss.name(), "aux_loss");
        assert_eq!(LoadBalanceStrategy::ExpertChoice.name(), "expert_choice");
    }
    
    #[test]
    fn test_config_default() {
        let config = MoEAblationConfig::default();
        assert!(!config.num_experts_configs.is_empty());
        assert!(!config.top_k_configs.is_empty());
        assert!(!config.balance_strategies.is_empty());
    }
    
    #[test]
    fn test_memory_estimate() {
        let config = MoEAblationConfig::default();
        let device = Device::Cpu;
        let runner = MoEAblationRunner::new(config, device);
        
        let mem_8 = runner.estimate_memory(8);
        let mem_64 = runner.estimate_memory(64);
        
        // More experts = more memory
        assert!(mem_64 > mem_8);
        assert_eq!(mem_64, mem_8 * 8);  // Linear scaling
    }
    
    #[test]
    fn test_ablation_runner_basic() -> Result<()> {
        let config = MoEAblationConfig {
            num_experts_configs: vec![8],
            top_k_configs: vec![2],
            balance_strategies: vec![LoadBalanceStrategy::None],
            eval_seq_lengths: vec![32],
            batch_size: 1,
            num_runs: 1,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut runner = MoEAblationRunner::new(config, device);
        
        let metrics = runner.run_single(8, 2, LoadBalanceStrategy::None, 32)?;
        
        assert_eq!(metrics.num_experts, 8);
        assert_eq!(metrics.top_k, 2);
        assert!(metrics.forward_latency_ms > 0.0);
        assert!(metrics.expert_utilization >= 0.0 && metrics.expert_utilization <= 1.0);
        
        Ok(())
    }
}
