//! RoPE Scaling Strategy Ablation Study Hooks
//!
//! This module provides hooks for comparing different RoPE scaling strategies:
//! - Standard (no scaling)
//! - Linear interpolation
//! - NTK-aware scaling
//! - YaRN (Yet another RoPE extensioN)
//! - Dynamic NTK
//!
//! Reference: DeepSeek-V3 architecture specification for 128K+ context support.

use candle_core::{Device, IndexOp, Result, Tensor};
use std::collections::HashMap;
use std::time::Instant;

/// Metrics collected during ablation study
#[derive(Debug, Clone)]
pub struct AblationMetrics {
    /// Name of the scaling strategy
    pub strategy_name: String,
    /// Perplexity at different sequence lengths
    pub perplexity_by_length: HashMap<usize, f64>,
    /// Attention entropy (measure of attention spread)
    pub attention_entropy: Vec<f64>,
    /// Position embedding similarity decay
    pub position_similarity_decay: Vec<f64>,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Forward pass time (milliseconds)
    pub forward_time_ms: f64,
    /// Effective context utilization (how well long-range dependencies are captured)
    pub context_utilization: Vec<f64>,
}

impl AblationMetrics {
    pub fn new(strategy_name: &str) -> Self {
        Self {
            strategy_name: strategy_name.to_string(),
            perplexity_by_length: HashMap::new(),
            attention_entropy: Vec::new(),
            position_similarity_decay: Vec::new(),
            memory_usage: 0,
            forward_time_ms: 0.0,
            context_utilization: Vec::new(),
        }
    }
}

/// Configuration for ablation study
#[derive(Debug, Clone)]
pub struct AblationConfig {
    /// Sequence lengths to evaluate
    pub eval_seq_lengths: Vec<usize>,
    /// Number of samples per sequence length
    pub samples_per_length: usize,
    /// Whether to measure attention entropy
    pub measure_attention_entropy: bool,
    /// Whether to measure position similarity decay
    pub measure_position_decay: bool,
    /// Whether to log intermediate results
    pub verbose: bool,
    /// Strategies to compare
    pub strategies: Vec<RoPEStrategy>,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            eval_seq_lengths: vec![1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
            samples_per_length: 100,
            measure_attention_entropy: true,
            measure_position_decay: true,
            verbose: true,
            strategies: vec![
                RoPEStrategy::Standard,
                RoPEStrategy::Linear { scale: 4.0 },
                RoPEStrategy::NTKAware { alpha: 2.0 },
                RoPEStrategy::YaRN {
                    scale: 4.0,
                    original_max_seq_len: 4096,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    attention_factor: 0.1,
                },
                RoPEStrategy::DynamicNTK { max_position_embeddings: 4096 },
            ],
        }
    }
}

/// RoPE scaling strategy variants for ablation
#[derive(Debug, Clone)]
pub enum RoPEStrategy {
    Standard,
    Linear { scale: f32 },
    NTKAware { alpha: f32 },
    YaRN {
        scale: f32,
        original_max_seq_len: usize,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
    },
    DynamicNTK { max_position_embeddings: usize },
}

impl RoPEStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            RoPEStrategy::Standard => "standard",
            RoPEStrategy::Linear { .. } => "linear",
            RoPEStrategy::NTKAware { .. } => "ntk_aware",
            RoPEStrategy::YaRN { .. } => "yarn",
            RoPEStrategy::DynamicNTK { .. } => "dynamic_ntk",
        }
    }
}

/// Ablation study runner for RoPE scaling strategies
pub struct RoPEAblationStudy {
    config: AblationConfig,
    results: HashMap<String, AblationMetrics>,
}

impl RoPEAblationStudy {
    pub fn new(config: AblationConfig) -> Self {
        Self {
            config,
            results: HashMap::new(),
        }
    }

    /// Run ablation study comparing all configured strategies
    pub fn run(&mut self, device: &Device) -> Result<()> {
        for strategy in &self.config.strategies.clone() {
            let metrics = self.evaluate_strategy(strategy, device)?;
            self.results.insert(strategy.name().to_string(), metrics);
        }
        Ok(())
    }

    /// Evaluate a single RoPE scaling strategy
    fn evaluate_strategy(
        &self,
        strategy: &RoPEStrategy,
        device: &Device,
    ) -> Result<AblationMetrics> {
        let mut metrics = AblationMetrics::new(strategy.name());

        for &seq_len in &self.config.eval_seq_lengths {
            if self.config.verbose {
                println!("Evaluating {} at seq_len={}", strategy.name(), seq_len);
            }

            // Generate test embeddings
            let d_head = 64;
            let batch_size = 1;
            let num_heads = 8;

            // Compute RoPE embeddings for this strategy
            let start = Instant::now();
            let inv_freq = self.compute_inv_freq(strategy, d_head, seq_len, device)?;
            
            // Generate positions
            let positions = Tensor::arange(0f32, seq_len as f32, device)?;
            
            // Compute frequencies
            let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
            let cos = freqs.cos()?;
            let sin = freqs.sin()?;
            
            // Measure forward time
            metrics.forward_time_ms += start.elapsed().as_secs_f64() * 1000.0;

            // Measure attention entropy if enabled
            if self.config.measure_attention_entropy {
                let entropy = self.compute_position_entropy(&cos, &sin)?;
                metrics.attention_entropy.push(entropy);
            }

            // Measure position similarity decay if enabled
            if self.config.measure_position_decay {
                let decay = self.compute_position_decay(&cos, &sin, seq_len)?;
                metrics.position_similarity_decay.extend(decay);
            }

            // Compute context utilization metric
            let utilization = self.compute_context_utilization(&cos, &sin, seq_len)?;
            metrics.context_utilization.push(utilization);
        }

        metrics.forward_time_ms /= self.config.eval_seq_lengths.len() as f64;

        Ok(metrics)
    }

    /// Compute inverse frequencies for a given strategy
    fn compute_inv_freq(
        &self,
        strategy: &RoPEStrategy,
        d_head: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let base = 10000f32;
        let half_dim = d_head / 2;

        let inv_freq: Vec<f32> = match strategy {
            RoPEStrategy::Standard => {
                (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / base.powf(i as f32 / d_head as f32))
                    .collect()
            }

            RoPEStrategy::Linear { scale } => {
                (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / (scale * base.powf(i as f32 / d_head as f32)))
                    .collect()
            }

            RoPEStrategy::NTKAware { alpha } => {
                let new_base = base * alpha.powf(d_head as f32 / (d_head as f32 - 2.0));
                (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / new_base.powf(i as f32 / d_head as f32))
                    .collect()
            }

            RoPEStrategy::YaRN {
                scale,
                original_max_seq_len,
                beta_fast,
                beta_slow,
                ..
            } => {
                let mut inv_freq = Vec::with_capacity(half_dim);
                for i in (0..d_head).step_by(2) {
                    let dim_idx = i as f32 / d_head as f32;
                    let base_freq = 1.0 / base.powf(dim_idx);
                    let wavelength = 2.0 * std::f32::consts::PI / base_freq;

                    let low_freq_wavelen = (*original_max_seq_len as f32) / beta_slow;
                    let high_freq_wavelen = (*original_max_seq_len as f32) / beta_fast;

                    let gamma = if wavelength < high_freq_wavelen {
                        0.0
                    } else if wavelength > low_freq_wavelen {
                        1.0
                    } else {
                        (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
                    };

                    let scaled_freq = base_freq / scale;
                    let final_freq = (1.0 - gamma) * base_freq + gamma * scaled_freq;
                    inv_freq.push(final_freq);
                }
                inv_freq
            }

            RoPEStrategy::DynamicNTK { max_position_embeddings } => {
                let alpha = (max_seq_len as f32 / *max_position_embeddings as f32).max(1.0);
                let new_base = base * alpha.powf(d_head as f32 / (d_head as f32 - 2.0));
                (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / new_base.powf(i as f32 / d_head as f32))
                    .collect()
            }
        };

        Tensor::from_vec(inv_freq, (half_dim,), device)
    }

    /// Compute entropy of position embeddings (measure of information content)
    fn compute_position_entropy(&self, cos: &Tensor, sin: &Tensor) -> Result<f64> {
        // Use variance as a proxy for entropy
        let cos_var = cos.var(0)?.mean_all()?.to_scalar::<f32>()? as f64;
        let sin_var = sin.var(0)?.mean_all()?.to_scalar::<f32>()? as f64;
        Ok((cos_var + sin_var) / 2.0)
    }

    /// Compute position similarity decay curve
    fn compute_position_decay(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        seq_len: usize,
    ) -> Result<Vec<f64>> {
        // Compute similarity between position 0 and positions at increasing distances
        let mut decay = Vec::new();
        let sample_distances = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

        let cos_0 = cos.i(0)?;
        let sin_0 = sin.i(0)?;

        for &dist in &sample_distances {
            if dist >= seq_len {
                break;
            }
            let cos_d = cos.i(dist)?;
            let sin_d = sin.i(dist)?;

            // Cosine similarity between position embeddings
            let cos_sim = (&cos_0 * &cos_d)?.sum_all()?.to_scalar::<f32>()? as f64;
            let sin_sim = (&sin_0 * &sin_d)?.sum_all()?.to_scalar::<f32>()? as f64;
            
            let similarity = (cos_sim + sin_sim) / 2.0;
            decay.push(similarity);
        }

        Ok(decay)
    }

    /// Compute context utilization metric
    fn compute_context_utilization(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        seq_len: usize,
    ) -> Result<f64> {
        // Context utilization: how distinguishable are far positions?
        // Higher is better for long-range dependencies
        
        if seq_len < 100 {
            return Ok(1.0);
        }

        // Compare embeddings at beginning, middle, and end
        let positions = [0, seq_len / 4, seq_len / 2, 3 * seq_len / 4, seq_len - 1];
        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                if positions[j] >= seq_len {
                    continue;
                }
                let cos_i = cos.i(positions[i])?;
                let cos_j = cos.i(positions[j])?;
                let sin_i = sin.i(positions[i])?;
                let sin_j = sin.i(positions[j])?;

                // L2 distance between embeddings
                let cos_diff = (&cos_i - &cos_j)?;
                let sin_diff = (&sin_i - &sin_j)?;
                let sum_sq = (cos_diff.sqr()?.sum_all()? + sin_diff.sqr()?.sum_all()?)?;
                let dist = sum_sq.sqrt()?.to_scalar::<f32>()? as f64;
                
                total_distance += dist;
                count += 1;
            }
        }

        Ok(if count > 0 { total_distance / count as f64 } else { 0.0 })
    }

    /// Get ablation study results
    pub fn get_results(&self) -> &HashMap<String, AblationMetrics> {
        &self.results
    }

    /// Generate summary report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# RoPE Scaling Strategy Ablation Study Report\n\n");

        for (name, metrics) in &self.results {
            report.push_str(&format!("## {}\n\n", name));
            report.push_str(&format!("- Average forward time: {:.2} ms\n", metrics.forward_time_ms));
            
            if !metrics.attention_entropy.is_empty() {
                let avg_entropy: f64 = metrics.attention_entropy.iter().sum::<f64>() 
                    / metrics.attention_entropy.len() as f64;
                report.push_str(&format!("- Average attention entropy: {:.4}\n", avg_entropy));
            }
            
            if !metrics.context_utilization.is_empty() {
                let avg_util: f64 = metrics.context_utilization.iter().sum::<f64>() 
                    / metrics.context_utilization.len() as f64;
                report.push_str(&format!("- Average context utilization: {:.4}\n", avg_util));
            }
            
            report.push('\n');
        }

        report
    }
}

/// Hook for logging ablation study metrics during training
pub struct AblationTrainingHook {
    metrics_log: Vec<(usize, String, f64)>, // (step, metric_name, value)
    log_interval: usize,
}

impl AblationTrainingHook {
    pub fn new(log_interval: usize) -> Self {
        Self {
            metrics_log: Vec::new(),
            log_interval,
        }
    }

    /// Log a metric at the current training step
    pub fn log_metric(&mut self, step: usize, name: &str, value: f64) {
        if step % self.log_interval == 0 {
            self.metrics_log.push((step, name.to_string(), value));
        }
    }

    /// Get all logged metrics
    pub fn get_metrics(&self) -> &[(usize, String, f64)] {
        &self.metrics_log
    }

    /// Export metrics to CSV format
    pub fn export_csv(&self) -> String {
        let mut csv = String::from("step,metric,value\n");
        for (step, name, value) in &self.metrics_log {
            csv.push_str(&format!("{},{},{}\n", step, name, value));
        }
        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ablation_config_default() {
        let config = AblationConfig::default();
        assert!(!config.strategies.is_empty());
        assert!(!config.eval_seq_lengths.is_empty());
    }

    #[test]
    fn test_rope_strategy_names() {
        assert_eq!(RoPEStrategy::Standard.name(), "standard");
        assert_eq!(RoPEStrategy::Linear { scale: 4.0 }.name(), "linear");
        assert_eq!(RoPEStrategy::NTKAware { alpha: 2.0 }.name(), "ntk_aware");
    }

    #[test]
    fn test_ablation_study_basic() -> Result<()> {
        let config = AblationConfig {
            eval_seq_lengths: vec![128, 256],
            samples_per_length: 1,
            strategies: vec![RoPEStrategy::Standard],
            ..Default::default()
        };
        
        let mut study = RoPEAblationStudy::new(config);
        study.run(&Device::Cpu)?;
        
        let results = study.get_results();
        assert!(results.contains_key("standard"));
        
        Ok(())
    }

    #[test]
    fn test_training_hook() {
        let mut hook = AblationTrainingHook::new(10);
        
        for step in 0..100 {
            hook.log_metric(step, "loss", 1.0 / (step as f64 + 1.0));
        }
        
        let metrics = hook.get_metrics();
        assert_eq!(metrics.len(), 10); // Logged at steps 0, 10, 20, ..., 90
    }
}
