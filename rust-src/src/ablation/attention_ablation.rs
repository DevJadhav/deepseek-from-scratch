//! Attention Ablation Study Module
//!
//! Compares different attention mechanisms:
//! - Multi-Latent Attention (MLA) - DeepSeek's innovation
//! - Grouped-Query Attention (GQA) - Llama 2 approach  
//! - Multi-Head Attention (MHA) - Standard transformer attention
//!
//! Metrics tracked:
//! - Perplexity
//! - Throughput (tokens/second)
//! - Memory usage
//! - KV cache size
//! - Training loss convergence

use candle_core::{Device, Result, Tensor, DType};
use std::collections::HashMap;
use std::time::Instant;

/// Attention mechanism type for ablation study
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionType {
    /// Multi-Head Attention (standard transformer)
    MHA,
    /// Grouped-Query Attention (Llama 2 style)
    GQA,
    /// Multi-Latent Attention (DeepSeek innovation)
    MLA,
}

impl AttentionType {
    pub fn name(&self) -> &'static str {
        match self {
            AttentionType::MHA => "mha",
            AttentionType::GQA => "gqa",
            AttentionType::MLA => "mla",
        }
    }
    
    pub fn description(&self) -> &'static str {
        match self {
            AttentionType::MHA => "Multi-Head Attention (standard)",
            AttentionType::GQA => "Grouped-Query Attention (Llama 2)",
            AttentionType::MLA => "Multi-Latent Attention (DeepSeek)",
        }
    }
}

/// Configuration for attention ablation study
#[derive(Debug, Clone)]
pub struct AttentionAblationConfig {
    /// Attention types to compare
    pub attention_types: Vec<AttentionType>,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Latent dimension for MLA
    pub d_latent: usize,
    /// Sequence lengths to evaluate
    pub eval_seq_lengths: Vec<usize>,
    /// Batch size for evaluation
    pub batch_size: usize,
    /// Number of evaluation runs per config
    pub num_runs: usize,
}

impl Default for AttentionAblationConfig {
    fn default() -> Self {
        Self {
            attention_types: vec![
                AttentionType::MHA,
                AttentionType::GQA,
                AttentionType::MLA,
            ],
            d_model: 512,
            num_heads: 8,
            num_kv_heads: 2,  // For GQA: 4 query groups
            head_dim: 64,
            d_latent: 64,  // For MLA: 8x compression
            eval_seq_lengths: vec![128, 256, 512, 1024],
            batch_size: 4,
            num_runs: 3,
        }
    }
}

/// Metrics from a single attention ablation run
#[derive(Debug, Clone)]
pub struct AttentionMetrics {
    /// Attention type
    pub attention_type: AttentionType,
    /// Sequence length
    pub seq_len: usize,
    /// Forward pass latency in milliseconds
    pub forward_latency_ms: f64,
    /// Throughput in tokens per second
    pub throughput_tokens_per_sec: f64,
    /// Memory for KV cache in bytes
    pub kv_cache_memory_bytes: usize,
    /// Peak memory usage estimate in bytes
    pub peak_memory_bytes: usize,
    /// Numerical precision metric (Frobenius norm of output)
    pub output_norm: f64,
}

/// Attention ablation study runner
pub struct AttentionAblationRunner {
    config: AttentionAblationConfig,
    device: Device,
    results: Vec<AttentionMetrics>,
}

impl AttentionAblationRunner {
    pub fn new(config: AttentionAblationConfig, device: Device) -> Self {
        Self {
            config,
            device,
            results: Vec::new(),
        }
    }
    
    /// Estimate KV cache memory for different attention types
    fn estimate_kv_cache_memory(&self, attn_type: AttentionType, seq_len: usize) -> usize {
        let batch_size = self.config.batch_size;
        let bytes_per_element = 4; // f32
        
        match attn_type {
            AttentionType::MHA => {
                // Standard: 2 (K+V) * batch * heads * seq_len * head_dim
                2 * batch_size * self.config.num_heads * seq_len * self.config.head_dim * bytes_per_element
            }
            AttentionType::GQA => {
                // GQA: 2 (K+V) * batch * kv_heads * seq_len * head_dim
                2 * batch_size * self.config.num_kv_heads * seq_len * self.config.head_dim * bytes_per_element
            }
            AttentionType::MLA => {
                // MLA: batch * seq_len * d_latent (compressed representation)
                batch_size * seq_len * self.config.d_latent * bytes_per_element
            }
        }
    }
    
    /// Create dummy input tensors for benchmarking
    fn create_dummy_inputs(&self, seq_len: usize) -> Result<Tensor> {
        Tensor::randn(
            0f32,
            1f32,
            (self.config.batch_size, seq_len, self.config.d_model),
            &self.device,
        )
    }
    
    /// Simulate MHA forward pass
    fn simulate_mha(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        
        // Flatten batch and seq for matmul
        let x_flat = x.reshape((batch_size * seq_len, d_model))?;
        
        // Simulated Q, K, V projections
        let qkv_weight = Tensor::randn(
            0f32, 0.02f32,
            (3 * num_heads * head_dim, d_model),
            &self.device,
        )?;
        
        let qkv = x_flat.matmul(&qkv_weight.t()?)?;
        let qkv = qkv.reshape((batch_size, seq_len, 3 * num_heads * head_dim))?;
        
        // Split into Q, K, V and reshape
        let qkv_size = num_heads * head_dim;
        let q = qkv.narrow(2, 0, qkv_size)?
            .reshape((batch_size, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv.narrow(2, qkv_size, qkv_size)?
            .reshape((batch_size, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?;
        let v = qkv.narrow(2, 2 * qkv_size, qkv_size)?
            .reshape((batch_size, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        
        // Attention scores
        let scale = (head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?;
        let scores = (scores / scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
        
        // Apply attention
        let v_c = v.contiguous()?;
        let attn_output = attn_weights.matmul(&v_c)?;
        let attn_output = attn_output.transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size * seq_len, num_heads * head_dim))?;
        
        // Output projection
        let out_weight = Tensor::randn(
            0f32, 0.02f32,
            (d_model, num_heads * head_dim),
            &self.device,
        )?;
        
        let output = attn_output.matmul(&out_weight.t()?)?;
        output.reshape((batch_size, seq_len, d_model))
    }
    
    /// Simulate GQA forward pass
    fn simulate_gqa(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let num_groups = num_heads / num_kv_heads;
        
        // Flatten for matmul
        let x_flat = x.reshape((batch_size * seq_len, d_model))?;
        
        // Q projection (full heads)
        let q_weight = Tensor::randn(
            0f32, 0.02f32,
            (num_heads * head_dim, d_model),
            &self.device,
        )?;
        
        // KV projection (reduced heads)
        let kv_weight = Tensor::randn(
            0f32, 0.02f32,
            (2 * num_kv_heads * head_dim, d_model),
            &self.device,
        )?;
        
        let q = x_flat.matmul(&q_weight.t()?)?
            .reshape((batch_size, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let kv = x_flat.matmul(&kv_weight.t()?)?
            .reshape((batch_size, seq_len, 2 * num_kv_heads * head_dim))?;
        let kv_size = num_kv_heads * head_dim;
        
        let k = kv.narrow(2, 0, kv_size)?
            .reshape((batch_size, seq_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?;
        let v = kv.narrow(2, kv_size, kv_size)?
            .reshape((batch_size, seq_len, num_kv_heads, head_dim))?
            .transpose(1, 2)?;
        
        // Expand K, V for grouped attention
        let k_expanded = k.unsqueeze(2)?
            .broadcast_as((batch_size, num_kv_heads, num_groups, seq_len, head_dim))?
            .reshape((batch_size, num_heads, seq_len, head_dim))?
            .contiguous()?;
        let v_expanded = v.unsqueeze(2)?
            .broadcast_as((batch_size, num_kv_heads, num_groups, seq_len, head_dim))?
            .reshape((batch_size, num_heads, seq_len, head_dim))?
            .contiguous()?;
        
        // Attention
        let scale = (head_dim as f64).sqrt();
        let q_c = q.contiguous()?;
        let k_t = k_expanded.transpose(2, 3)?.contiguous()?;
        let scores = q_c.matmul(&k_t)?;
        let scores = (scores / scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
        let attn_output = attn_weights.matmul(&v_expanded)?;
        
        let attn_output = attn_output.transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size * seq_len, num_heads * head_dim))?;
        
        // Output projection
        let out_weight = Tensor::randn(
            0f32, 0.02f32,
            (d_model, num_heads * head_dim),
            &self.device,
        )?;
        
        let output = attn_output.matmul(&out_weight.t()?)?;
        output.reshape((batch_size, seq_len, d_model))
    }
    
    /// Simulate MLA forward pass
    fn simulate_mla(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let d_latent = self.config.d_latent;
        
        // Flatten for matmul
        let x_flat = x.reshape((batch_size * seq_len, d_model))?;
        
        // Q projection (full)
        let q_weight = Tensor::randn(
            0f32, 0.02f32,
            (num_heads * head_dim, d_model),
            &self.device,
        )?;
        
        // KV down projection (compress to latent)
        let kv_down_weight = Tensor::randn(
            0f32, 0.02f32,
            (d_latent, d_model),
            &self.device,
        )?;
        
        // KV up projections (expand from latent)
        let k_up_weight = Tensor::randn(
            0f32, 0.02f32,
            (num_heads * head_dim, d_latent),
            &self.device,
        )?;
        let v_up_weight = Tensor::randn(
            0f32, 0.02f32,
            (num_heads * head_dim, d_latent),
            &self.device,
        )?;
        
        // Q projection
        let q = x_flat.matmul(&q_weight.t()?)?
            .reshape((batch_size, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        
        // KV compression and expansion
        let c_kv = x_flat.matmul(&kv_down_weight.t()?)?;  // Latent representation [B*S, d_latent]
        let k = c_kv.matmul(&k_up_weight.t()?)?  // [B*S, num_heads*head_dim]
            .reshape((batch_size, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = c_kv.matmul(&v_up_weight.t()?)?
            .reshape((batch_size, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        
        // Attention
        let scale = (head_dim as f64).sqrt();
        let scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = (scores / scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
        let attn_output = attn_weights.contiguous()?.matmul(&v.contiguous()?)?;
        
        let attn_output = attn_output.transpose(1, 2)?
            .reshape((batch_size * seq_len, num_heads * head_dim))?;
        
        // Output projection
        let out_weight = Tensor::randn(
            0f32, 0.02f32,
            (d_model, num_heads * head_dim),
            &self.device,
        )?;
        
        let output = attn_output.matmul(&out_weight.t()?)?;
        output.reshape((batch_size, seq_len, d_model))
    }
    
    /// Run attention forward pass based on type
    fn run_attention(&self, attn_type: AttentionType, x: &Tensor) -> Result<Tensor> {
        match attn_type {
            AttentionType::MHA => self.simulate_mha(x),
            AttentionType::GQA => self.simulate_gqa(x),
            AttentionType::MLA => self.simulate_mla(x),
        }
    }
    
    /// Run ablation study for a single configuration
    pub fn run_single(&mut self, attn_type: AttentionType, seq_len: usize) -> Result<AttentionMetrics> {
        let x = self.create_dummy_inputs(seq_len)?;
        
        // Warmup
        let _ = self.run_attention(attn_type, &x)?;
        
        // Timed runs
        let mut latencies = Vec::with_capacity(self.config.num_runs);
        let mut output_norm = 0.0;
        
        for _ in 0..self.config.num_runs {
            let start = Instant::now();
            let output = self.run_attention(attn_type, &x)?;
            let elapsed = start.elapsed();
            latencies.push(elapsed.as_secs_f64() * 1000.0);
            
            // Compute output norm for numerical verification
            let flat = output.flatten_all()?;
            let squared_sum = flat.sqr()?.sum_all()?.to_scalar::<f32>()?;
            output_norm = (squared_sum as f64).sqrt();
        }
        
        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let total_tokens = (self.config.batch_size * seq_len) as f64;
        let throughput = total_tokens / (avg_latency / 1000.0);
        
        let kv_cache_memory = self.estimate_kv_cache_memory(attn_type, seq_len);
        
        // Rough peak memory estimate (input + weights + KV cache + output)
        let bytes_per_element = 4;
        let input_memory = self.config.batch_size * seq_len * self.config.d_model * bytes_per_element;
        let peak_memory = input_memory + kv_cache_memory + input_memory;  // Simplified
        
        let metrics = AttentionMetrics {
            attention_type: attn_type,
            seq_len,
            forward_latency_ms: avg_latency,
            throughput_tokens_per_sec: throughput,
            kv_cache_memory_bytes: kv_cache_memory,
            peak_memory_bytes: peak_memory,
            output_norm,
        };
        
        self.results.push(metrics.clone());
        Ok(metrics)
    }
    
    /// Run full ablation study
    pub fn run_ablation(&mut self) -> Result<Vec<AttentionMetrics>> {
        let attention_types = self.config.attention_types.clone();
        let seq_lengths = self.config.eval_seq_lengths.clone();
        
        for attn_type in &attention_types {
            for &seq_len in &seq_lengths {
                self.run_single(*attn_type, seq_len)?;
            }
        }
        
        Ok(self.results.clone())
    }
    
    /// Generate analysis report
    pub fn analyze_results(&self, results: &[AttentionMetrics]) -> String {
        let mut report = String::new();
        report.push_str("=== Attention Ablation Study Results ===\n\n");
        
        // Group by attention type
        let mut by_type: HashMap<AttentionType, Vec<&AttentionMetrics>> = HashMap::new();
        for m in results {
            by_type.entry(m.attention_type).or_default().push(m);
        }
        
        // Memory efficiency comparison
        report.push_str("Memory Efficiency (KV Cache):\n");
        report.push_str("-".repeat(50).as_str());
        report.push('\n');
        
        for (attn_type, metrics) in &by_type {
            let avg_memory: f64 = metrics.iter()
                .map(|m| m.kv_cache_memory_bytes as f64)
                .sum::<f64>() / metrics.len() as f64;
            report.push_str(&format!(
                "  {}: {:.2} KB average\n",
                attn_type.name(),
                avg_memory / 1024.0
            ));
        }
        
        // MLA memory savings vs MHA
        if let (Some(mla_metrics), Some(mha_metrics)) = (
            by_type.get(&AttentionType::MLA),
            by_type.get(&AttentionType::MHA),
        ) {
            let mla_avg: f64 = mla_metrics.iter()
                .map(|m| m.kv_cache_memory_bytes as f64)
                .sum::<f64>() / mla_metrics.len() as f64;
            let mha_avg: f64 = mha_metrics.iter()
                .map(|m| m.kv_cache_memory_bytes as f64)
                .sum::<f64>() / mha_metrics.len() as f64;
            let reduction = (1.0 - mla_avg / mha_avg) * 100.0;
            report.push_str(&format!(
                "\nMLA achieves {:.1}% memory reduction vs MHA\n",
                reduction
            ));
        }
        
        report.push('\n');
        
        // Throughput comparison
        report.push_str("Throughput (tokens/sec):\n");
        report.push_str("-".repeat(50).as_str());
        report.push('\n');
        
        for (attn_type, metrics) in &by_type {
            let avg_throughput: f64 = metrics.iter()
                .map(|m| m.throughput_tokens_per_sec)
                .sum::<f64>() / metrics.len() as f64;
            report.push_str(&format!(
                "  {}: {:.0} tokens/sec\n",
                attn_type.name(),
                avg_throughput
            ));
        }
        
        report.push('\n');
        
        // Per sequence length breakdown
        report.push_str("Per-Sequence-Length Analysis:\n");
        report.push_str("-".repeat(50).as_str());
        report.push('\n');
        
        for m in results {
            report.push_str(&format!(
                "  {} @ seq_len={}: latency={:.2}ms, throughput={:.0} tok/s, kv_cache={:.1}KB\n",
                m.attention_type.name(),
                m.seq_len,
                m.forward_latency_ms,
                m.throughput_tokens_per_sec,
                m.kv_cache_memory_bytes as f64 / 1024.0,
            ));
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_type_names() {
        assert_eq!(AttentionType::MHA.name(), "mha");
        assert_eq!(AttentionType::GQA.name(), "gqa");
        assert_eq!(AttentionType::MLA.name(), "mla");
    }
    
    #[test]
    fn test_config_default() {
        let config = AttentionAblationConfig::default();
        assert_eq!(config.attention_types.len(), 3);
        assert!(!config.eval_seq_lengths.is_empty());
    }
    
    #[test]
    fn test_kv_cache_memory_estimates() {
        let config = AttentionAblationConfig::default();
        let device = Device::Cpu;
        let runner = AttentionAblationRunner::new(config, device);
        
        let seq_len = 512;
        let mha_memory = runner.estimate_kv_cache_memory(AttentionType::MHA, seq_len);
        let gqa_memory = runner.estimate_kv_cache_memory(AttentionType::GQA, seq_len);
        let mla_memory = runner.estimate_kv_cache_memory(AttentionType::MLA, seq_len);
        
        // MHA should use most memory
        assert!(mha_memory > gqa_memory);
        // MLA should be most efficient
        assert!(mla_memory < gqa_memory);
    }
    
    #[test]
    fn test_ablation_runner_basic() -> Result<()> {
        let config = AttentionAblationConfig {
            attention_types: vec![AttentionType::MHA],
            eval_seq_lengths: vec![32],
            batch_size: 1,
            num_runs: 1,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut runner = AttentionAblationRunner::new(config, device);
        
        let metrics = runner.run_single(AttentionType::MHA, 32)?;
        
        assert_eq!(metrics.attention_type, AttentionType::MHA);
        assert_eq!(metrics.seq_len, 32);
        assert!(metrics.forward_latency_ms > 0.0);
        
        Ok(())
    }
}
