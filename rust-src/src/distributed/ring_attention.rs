//! Ring Attention and Sequence Parallelism for Distributed Training.
//!
//! This module provides comprehensive sequence parallelism capabilities:
//! - Ring attention for extreme sequence lengths (128K+)
//! - All-gather/reduce-scatter for attention patterns
//! - Sequence-parallel LayerNorm/RMSNorm
//! - RoPE with position offset for split sequences
//! - Memory-efficient attention patterns
//!
//! Reference: DeepSeek-V3 sequence parallelism implementation.

use candle_core::{Result, Tensor, DType, Device};
use super::{get_sp_size, get_sp_rank, get_sp_group};
use std::f32::consts::PI;

// ============================================================================
// Sequence Parallelism Configuration
// ============================================================================

/// Configuration for sequence parallelism.
#[derive(Clone, Debug)]
pub struct SequenceParallelConfig {
    /// Number of sequence parallel ranks
    pub sp_size: usize,
    /// This rank in the SP group
    pub sp_rank: usize,
    /// Enable ring attention for long sequences
    pub ring_attention: bool,
    /// Enable distributed layer norm
    pub distributed_layernorm: bool,
    /// Enable RoPE position offset
    pub rope_offset: bool,
    /// Maximum sequence length per rank
    pub max_seq_len_per_rank: usize,
}

impl Default for SequenceParallelConfig {
    fn default() -> Self {
        Self {
            sp_size: get_sp_size(),
            sp_rank: get_sp_rank(),
            ring_attention: true,
            distributed_layernorm: true,
            rope_offset: true,
            max_seq_len_per_rank: 8192,
        }
    }
}

impl SequenceParallelConfig {
    pub fn new(sp_size: usize, sp_rank: usize) -> Self {
        Self {
            sp_size,
            sp_rank,
            ..Default::default()
        }
    }
    
    /// Get global position offset for this rank.
    pub fn position_offset(&self, seq_len: usize) -> usize {
        self.sp_rank * seq_len
    }
    
    /// Calculate total sequence length across all ranks.
    pub fn total_seq_len(&self, local_seq_len: usize) -> usize {
        local_seq_len * self.sp_size
    }
}

// ============================================================================
// All-Gather / Reduce-Scatter Operations
// ============================================================================

/// All-gather operation for sequence parallelism.
/// Gathers sequence chunks from all SP ranks.
pub fn all_gather_sequence(tensor: &Tensor) -> Result<Tensor> {
    let sp_size = get_sp_size();
    
    if sp_size <= 1 {
        return Ok(tensor.clone());
    }
    
    if let Some(group) = get_sp_group() {
        // Gather from all ranks
        let gathered = group.communicator.all_gather(tensor)?;
        
        // Concatenate along sequence dimension (dim 1 for [B, S, H])
        // The all_gather returns [B, S, H] from each rank
        // We need to concatenate to get [B, S*sp_size, H]
        Ok(gathered)
    } else {
        Ok(tensor.clone())
    }
}

/// Reduce-scatter operation for sequence parallelism.
/// Reduces and scatters to each SP rank's local sequence chunk.
pub fn reduce_scatter_sequence(tensor: &Tensor) -> Result<Tensor> {
    let sp_size = get_sp_size();
    
    if sp_size <= 1 {
        return Ok(tensor.clone());
    }
    
    if let Some(group) = get_sp_group() {
        group.communicator.reduce_scatter(tensor)
    } else {
        Ok(tensor.clone())
    }
}

/// All-reduce operation with mean reduction.
pub fn all_reduce_mean(tensor: &Tensor) -> Result<Tensor> {
    let sp_size = get_sp_size();
    
    if sp_size <= 1 {
        return Ok(tensor.clone());
    }
    
    if let Some(group) = get_sp_group() {
        let sum = group.communicator.all_reduce(tensor)?;
        sum.affine(1.0 / sp_size as f64, 0.0)
    } else {
        Ok(tensor.clone())
    }
}

// ============================================================================
// RoPE with Position Offset
// ============================================================================

/// Rotary Position Embedding with sequence parallel offset.
pub struct RoPEWithOffset {
    /// Dimension for RoPE
    dim: usize,
    /// Base frequency
    base: f32,
    /// Sequence parallel config
    config: SequenceParallelConfig,
    /// Cached cos values
    cos_cache: Option<Tensor>,
    /// Cached sin values
    sin_cache: Option<Tensor>,
    /// Maximum cached sequence length
    cached_seq_len: usize,
}

impl RoPEWithOffset {
    pub fn new(dim: usize, base: f32, config: SequenceParallelConfig) -> Self {
        Self {
            dim,
            base,
            config,
            cos_cache: None,
            sin_cache: None,
            cached_seq_len: 0,
        }
    }
    
    /// Get or compute cos/sin for given sequence length.
    fn get_cos_sin(&mut self, seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let offset = self.config.position_offset(seq_len);
        let total_len = seq_len + offset;
        
        if self.cached_seq_len < total_len || self.cos_cache.is_none() {
            self.compute_cache(total_len, device)?;
        }
        
        // Extract the portion for this rank with offset
        let cos = self.cos_cache.as_ref().unwrap()
            .narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.as_ref().unwrap()
            .narrow(0, offset, seq_len)?;
        
        Ok((cos, sin))
    }
    
    fn compute_cache(&mut self, max_seq_len: usize, device: &Device) -> Result<()> {
        let half_dim = self.dim / 2;
        
        // Compute frequencies: 1 / (base^(2i/dim)) for i in 0..half_dim
        let mut freqs = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq = 1.0 / self.base.powf(2.0 * i as f32 / self.dim as f32);
            freqs.push(freq);
        }
        let freqs = Tensor::from_vec(freqs, (1, half_dim), device)?;
        
        // Compute positions
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;
        
        // Compute angles: positions * freqs -> (max_seq_len, half_dim)
        let angles = positions.matmul(&freqs)?;
        
        // Compute cos and sin
        // Since candle doesn't have direct cos/sin, we use the formula
        // cos(x) = (e^(ix) + e^(-ix)) / 2, sin(x) = (e^(ix) - e^(-ix)) / (2i)
        // But for simplicity, we'll compute it element-wise
        let angles_vec: Vec<f32> = angles.to_vec2()?
            .into_iter()
            .flatten()
            .collect();
        
        let cos_vals: Vec<f32> = angles_vec.iter().map(|&x| x.cos()).collect();
        let sin_vals: Vec<f32> = angles_vec.iter().map(|&x| x.sin()).collect();
        
        self.cos_cache = Some(Tensor::from_vec(cos_vals, (max_seq_len, half_dim), device)?);
        self.sin_cache = Some(Tensor::from_vec(sin_vals, (max_seq_len, half_dim), device)?);
        self.cached_seq_len = max_seq_len;
        
        Ok(())
    }
    
    /// Apply RoPE to query and key tensors.
    /// Input shape: (batch, heads, seq_len, head_dim)
    pub fn apply(&mut self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        let device = q.device();
        
        let (cos, sin) = self.get_cos_sin(seq_len, device)?;
        
        // Reshape cos/sin to broadcast: (1, 1, seq_len, head_dim/2)
        let cos = cos.reshape((1, 1, seq_len, head_dim / 2))?;
        let sin = sin.reshape((1, 1, seq_len, head_dim / 2))?;
        
        let q_rotated = self.rotate_half(q, &cos, &sin)?;
        let k_rotated = self.rotate_half(k, &cos, &sin)?;
        
        Ok((q_rotated, k_rotated))
    }
    
    fn rotate_half(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (batch, heads, seq_len, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;
        
        // Split into two halves
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;
        
        // Rotate: [x1, x2] -> [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        let rotated_x1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
        let rotated_x2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;
        
        // Concatenate back
        Tensor::cat(&[&rotated_x1, &rotated_x2], 3)
    }
}

// ============================================================================
// Ring Attention Configuration and Implementation
// ============================================================================

/// Ring attention configuration.
#[derive(Clone, Debug)]
pub struct RingAttentionConfig {
    /// Number of sequence parallel ranks
    pub sp_size: usize,
    /// This rank in the SP group
    pub sp_rank: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout probability (0.0 = no dropout)
    pub dropout: f64,
    /// Scale factor for attention (usually 1/sqrt(d_k))
    pub scale: Option<f32>,
}

impl RingAttentionConfig {
    pub fn new(causal: bool, dropout: f64, scale: Option<f32>) -> Self {
        Self {
            sp_size: get_sp_size(),
            sp_rank: get_sp_rank(),
            causal,
            dropout,
            scale,
        }
    }
}

/// Pass K and V to the next rank in the ring, receive from previous.
fn ring_pass(k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
    let sp_size = get_sp_size();
    
    if sp_size <= 1 {
        return Ok((k.clone(), v.clone()));
    }
    
    let sp_rank = get_sp_rank();
    let next_rank = (sp_rank + 1) % sp_size;
    let prev_rank = (sp_rank + sp_size - 1) % sp_size;
    
    if let Some(group) = get_sp_group() {
        // Send to next, receive from previous
        // In a real implementation, these would be async and overlapped
        group.communicator.send(k, next_rank)?;
        group.communicator.send(v, next_rank)?;
        
        let k_recv = group.communicator.recv(k.dims(), k.device(), prev_rank)?;
        let v_recv = group.communicator.recv(v.dims(), v.device(), prev_rank)?;
        
        Ok((k_recv, v_recv))
    } else {
        Ok((k.clone(), v.clone()))
    }
}

/// Compute attention scores with optional causal masking.
fn compute_attention_scores(
    q: &Tensor,
    k: &Tensor,
    scale: f32,
    causal: bool,
    q_offset: usize,
    k_offset: usize,
) -> Result<Tensor> {
    // scores = Q @ K^T * scale
    let scores = (q.matmul(&k.transpose(2, 3)?)? * scale as f64)?;
    
    if causal {
        // Apply causal mask based on global positions
        let (_, _, seq_q, _) = q.dims4()?;
        let (_, _, seq_k, _) = k.dims4()?;
        
        // Create causal mask: position i can only attend to positions <= i
        // Global positions: q_offset..q_offset+seq_q for Q
        //                  k_offset..k_offset+seq_k for K
        let mut mask_vec = vec![0f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let q_pos = q_offset + i;
                let k_pos = k_offset + j;
                if k_pos > q_pos {
                    mask_vec[i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }
        
        let mask = Tensor::from_vec(mask_vec, (1, 1, seq_q, seq_k), q.device())?;
        let scores = scores.broadcast_add(&mask)?;
        
        Ok(scores)
    } else {
        Ok(scores)
    }
}

/// Ring Attention module.
/// 
/// Each SP rank holds Q, K, V for a portion of the sequence.
/// Attention is computed by iterating through the ring, accumulating
/// attention outputs using the online softmax trick for numerical stability.
pub struct RingAttention {
    config: RingAttentionConfig,
}

impl RingAttention {
    pub fn new(config: RingAttentionConfig) -> Self {
        Self { config }
    }
    
    /// Compute ring attention.
    /// 
    /// Args:
    ///   q: Query tensor (batch, heads, seq_local, d_k)
    ///   k: Key tensor (batch, heads, seq_local, d_k)
    ///   v: Value tensor (batch, heads, seq_local, d_v)
    /// 
    /// Returns:
    ///   attention output (batch, heads, seq_local, d_v)
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let sp_size = self.config.sp_size;
        
        if sp_size <= 1 {
            // Standard attention
            return self.standard_attention(q, k, v);
        }
        
        let (batch, heads, seq_local, d_k) = q.dims4()?;
        let (_, _, _, d_v) = v.dims4()?;
        
        let scale = self.config.scale.unwrap_or(1.0 / (d_k as f32).sqrt());
        
        // Initialize accumulators for online softmax
        // out_acc: Accumulated weighted values
        // max_acc: Running maximum for numerical stability
        // sum_acc: Running sum of exp(scores - max) for normalization
        let mut out_acc = Tensor::zeros((batch, heads, seq_local, d_v), DType::F32, q.device())?;
        let mut max_acc = Tensor::full(f32::NEG_INFINITY, (batch, heads, seq_local, 1), q.device())?;
        let mut sum_acc = Tensor::zeros((batch, heads, seq_local, 1), DType::F32, q.device())?;
        
        // Current K, V (will be passed around the ring)
        let mut k_curr = k.clone();
        let mut v_curr = v.clone();
        
        // Global offset for causal masking
        let q_offset = self.config.sp_rank * seq_local;
        
        for ring_step in 0..sp_size {
            // Compute which rank's K/V we currently have
            let k_rank = (self.config.sp_rank + sp_size - ring_step) % sp_size;
            let k_offset = k_rank * seq_local;
            
            // Compute attention scores for this chunk
            let scores = compute_attention_scores(
                q, &k_curr, scale, self.config.causal, q_offset, k_offset
            )?;
            
            // Online softmax update
            // new_max = max(max_acc, max(scores))
            let chunk_max = scores.max_keepdim(3)?;  // (B, H, seq_q, 1)
            let new_max = max_acc.maximum(&chunk_max)?;
            
            // Rescale previous accumulator
            let scale_old = ((&max_acc - &new_max)?.exp())?;
            out_acc = out_acc.broadcast_mul(&scale_old)?;
            sum_acc = sum_acc.broadcast_mul(&scale_old)?;
            
            // Compute attention weights for this chunk
            // Broadcast new_max to match scores shape for subtraction
            let new_max_broadcast = new_max.broadcast_as(scores.shape())?;
            let scores_exp = (&scores - &new_max_broadcast)?.exp()?;
            let chunk_sum = scores_exp.sum_keepdim(3)?;
            
            // Accumulate
            // out_acc += scores_exp @ v_curr
            let chunk_out = scores_exp.matmul(&v_curr)?;
            out_acc = (out_acc + chunk_out)?;
            sum_acc = (sum_acc + chunk_sum)?;
            max_acc = new_max;
            
            // Pass K, V to next rank (except on last iteration)
            if ring_step < sp_size - 1 {
                let (k_new, v_new) = ring_pass(&k_curr, &v_curr)?;
                k_curr = k_new;
                v_curr = v_new;
            }
        }
        
        // Normalize by sum
        let output = out_acc.broadcast_div(&sum_acc)?;
        
        Ok(output)
    }
    
    /// Standard (non-ring) attention for single rank.
    fn standard_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_, _, _, d_k) = q.dims4()?;
        let scale = self.config.scale.unwrap_or(1.0 / (d_k as f32).sqrt());
        
        // scores = Q @ K^T * scale
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale as f64)?;
        
        // Apply causal mask if needed
        let scores = if self.config.causal {
            let (_, _, seq_q, seq_k) = scores.dims4()?;
            let mut mask_vec = vec![0f32; seq_q * seq_k];
            for i in 0..seq_q {
                for j in 0..seq_k {
                    if j > i {
                        mask_vec[i * seq_k + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = Tensor::from_vec(mask_vec, (1, 1, seq_q, seq_k), q.device())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };
        
        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
        
        // Output = weights @ V
        let output = attn_weights.matmul(v)?;
        
        Ok(output)
    }
}

/// Distributed LayerNorm for sequence parallelism.
/// 
/// When sequence is split across ranks, LayerNorm needs to
/// gather statistics across all chunks for correct normalization.
pub struct DistributedLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    normalized_shape: Vec<usize>,
}

impl DistributedLayerNorm {
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let size: usize = normalized_shape.iter().product();
        let weight = Tensor::ones(size, DType::F32, device)?;
        let bias = Tensor::zeros(size, DType::F32, device)?;
        
        Ok(Self {
            weight,
            bias,
            eps,
            normalized_shape,
        })
    }
    
    /// Forward pass with distributed statistics gathering.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sp_size = get_sp_size();
        
        if sp_size <= 1 {
            // Standard LayerNorm
            return self.standard_forward(x);
        }
        
        // For distributed LayerNorm:
        // 1. Compute local sum and sum of squares
        // 2. All-reduce to get global statistics
        // 3. Normalize using global mean and variance
        
        let dims = x.dims();
        let normalized_dims = self.normalized_shape.len();
        let reduce_dims: Vec<usize> = (dims.len() - normalized_dims..dims.len()).collect();
        
        // Local statistics
        let local_sum = x.sum_keepdim(&reduce_dims[..])?;
        let local_sq_sum = x.sqr()?.sum_keepdim(&reduce_dims[..])?;
        
        // All-reduce statistics
        let (global_sum, global_sq_sum) = if let Some(group) = get_sp_group() {
            (group.communicator.all_reduce(&local_sum)?, group.communicator.all_reduce(&local_sq_sum)?)
        } else {
            (local_sum, local_sq_sum)
        };
        
        // Compute mean and variance from global stats
        let n_elements: usize = self.normalized_shape.iter().product();
        let n_total = (n_elements * sp_size) as f64;
        
        let mean = (&global_sum / n_total)?;
        let variance = ((&global_sq_sum / n_total)? - mean.sqr()?)?;
        
        // Normalize
        let x_centered = x.broadcast_sub(&mean)?;
        let std = (variance + self.eps)?.sqrt()?;
        let x_norm = x_centered.broadcast_div(&std)?;
        
        // Apply affine transformation
        let output = x_norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)?;
        
        Ok(output)
    }
    
    fn standard_forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let normalized_dims = self.normalized_shape.len();
        let reduce_dims: Vec<usize> = (dims.len() - normalized_dims..dims.len()).collect();
        
        let mean = x.mean_keepdim(&reduce_dims[..])?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(&reduce_dims[..])?;
        let std = (variance + self.eps)?.sqrt()?;
        let x_norm = x_centered.broadcast_div(&std)?;
        
        let output = x_norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)?;
        
        Ok(output)
    }
}

// ============================================================================
// Distributed RMSNorm for Sequence Parallelism
// ============================================================================

/// Distributed RMSNorm for sequence parallelism.
/// 
/// RMSNorm normalizes by root mean square, without centering.
/// When sequence is split across ranks, we need to compute global RMS.
pub struct DistributedRMSNorm {
    weight: Tensor,
    eps: f64,
    dim: usize,
}

impl DistributedRMSNorm {
    pub fn new(dim: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::ones(dim, DType::F32, device)?;
        
        Ok(Self {
            weight,
            eps,
            dim,
        })
    }
    
    /// Forward pass with distributed RMS computation.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sp_size = get_sp_size();
        
        if sp_size <= 1 {
            return self.standard_forward(x);
        }
        
        // Compute local sum of squares
        let local_sq_sum = x.sqr()?.sum_keepdim(x.dims().len() - 1)?;
        
        // All-reduce sum of squares
        let global_sq_sum = if let Some(group) = get_sp_group() {
            group.communicator.all_reduce(&local_sq_sum)?
        } else {
            local_sq_sum
        };
        
        // Compute global RMS
        let n_total = (self.dim * sp_size) as f64;
        let mean_sq = (&global_sq_sum / n_total)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        
        // Normalize
        let x_norm = x.broadcast_div(&rms)?;
        
        // Apply weight
        x_norm.broadcast_mul(&self.weight)
    }
    
    fn standard_forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len() - 1;
        let mean_sq = x.sqr()?.mean_keepdim(last_dim)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let x_norm = x.broadcast_div(&rms)?;
        x_norm.broadcast_mul(&self.weight)
    }
}

// ============================================================================
// Sequence Parallel Attention Wrapper
// ============================================================================

/// Complete sequence parallel attention module.
/// Combines RoPE, ring attention, and proper gather/scatter operations.
pub struct SequenceParallelAttention {
    config: SequenceParallelConfig,
    ring_attention: RingAttention,
    rope: Option<RoPEWithOffset>,
}

impl SequenceParallelAttention {
    pub fn new(
        config: SequenceParallelConfig,
        head_dim: usize,
        rope_base: Option<f32>,
    ) -> Self {
        let ring_config = RingAttentionConfig {
            sp_size: config.sp_size,
            sp_rank: config.sp_rank,
            causal: true,
            dropout: 0.0,
            scale: None,
        };
        
        let rope = if config.rope_offset {
            rope_base.map(|base| RoPEWithOffset::new(head_dim, base, config.clone()))
        } else {
            None
        };
        
        Self {
            config,
            ring_attention: RingAttention::new(ring_config),
            rope,
        }
    }
    
    /// Forward pass with sequence parallelism.
    /// 
    /// Input: (batch, seq_local, hidden_dim) - local sequence chunk
    /// Output: (batch, seq_local, hidden_dim) - processed local chunk
    pub fn forward(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor> {
        // Apply RoPE if configured
        let (q, k) = if let Some(ref mut rope) = self.rope {
            rope.apply(q, k)?
        } else {
            (q.clone(), k.clone())
        };
        
        // Compute attention using ring pattern
        let output = self.ring_attention.forward(&q, &k, v)?;
        
        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sequence_parallel_config() {
        let config = SequenceParallelConfig::new(4, 2);
        assert_eq!(config.sp_size, 4);
        assert_eq!(config.sp_rank, 2);
        assert_eq!(config.position_offset(100), 200);
        assert_eq!(config.total_seq_len(100), 400);
    }
    
    #[test]
    fn test_ring_attention_config() {
        let config = RingAttentionConfig::new(true, 0.0, None);
        assert_eq!(config.sp_size, 1);  // Default single rank
        assert!(config.causal);
    }
    
    #[test]
    fn test_ring_attention_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let config = RingAttentionConfig::new(true, 0.0, None);
        let ring_attn = RingAttention::new(config);
        
        let batch = 2;
        let heads = 4;
        let seq = 8;
        let d_k = 16;
        
        let q = Tensor::randn(0f32, 1f32, (batch, heads, seq, d_k), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, d_k), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, d_k), &device)?;
        
        let output = ring_attn.forward(&q, &k, &v)?;
        
        assert_eq!(output.dims(), &[batch, heads, seq, d_k]);
        Ok(())
    }
    
    #[test]
    fn test_distributed_layer_norm_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let ln = DistributedLayerNorm::new(vec![64], 1e-5, &device)?;
        
        let x = Tensor::randn(0f32, 1f32, (2, 10, 64), &device)?;
        let output = ln.forward(&x)?;
        
        assert_eq!(output.dims(), x.dims());
        Ok(())
    }
    
    #[test]
    fn test_distributed_rms_norm_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let rms_norm = DistributedRMSNorm::new(64, 1e-5, &device)?;
        
        let x = Tensor::randn(0f32, 1f32, (2, 10, 64), &device)?;
        let output = rms_norm.forward(&x)?;
        
        assert_eq!(output.dims(), x.dims());
        Ok(())
    }
    
    #[test]
    fn test_rope_with_offset_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let config = SequenceParallelConfig::new(1, 0);
        let mut rope = RoPEWithOffset::new(32, 10000.0, config);
        
        let batch = 2;
        let heads = 4;
        let seq = 8;
        let head_dim = 32;
        
        let q = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, head_dim), &device)?;
        
        let (q_rot, k_rot) = rope.apply(&q, &k)?;
        
        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
        Ok(())
    }
    
    #[test]
    fn test_rope_position_offset() {
        // Test that position offset is calculated correctly
        let config = SequenceParallelConfig::new(4, 2);
        let rope = RoPEWithOffset::new(32, 10000.0, config.clone());
        
        // Rank 2 with seq_len 100 should start at position 200
        assert_eq!(config.position_offset(100), 200);
    }
    
    #[test]
    fn test_sequence_parallel_attention_creation() -> Result<()> {
        let config = SequenceParallelConfig::new(1, 0);
        let attn = SequenceParallelAttention::new(config, 64, Some(10000.0));
        
        assert!(attn.rope.is_some());
        Ok(())
    }
    
    #[test]
    fn test_ring_attention_non_causal() -> Result<()> {
        let device = Device::Cpu;
        let config = RingAttentionConfig::new(false, 0.0, Some(0.125)); // Non-causal
        let ring_attn = RingAttention::new(config);
        
        let batch = 2;
        let heads = 4;
        let seq = 8;
        let d_k = 16;
        
        let q = Tensor::randn(0f32, 1f32, (batch, heads, seq, d_k), &device)?;
        let k = Tensor::randn(0f32, 1f32, (batch, heads, seq, d_k), &device)?;
        let v = Tensor::randn(0f32, 1f32, (batch, heads, seq, d_k), &device)?;
        
        let output = ring_attn.forward(&q, &k, &v)?;
        
        assert_eq!(output.dims(), &[batch, heads, seq, d_k]);
        Ok(())
    }
    
    #[test]
    fn test_all_gather_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?;
        
        // Single rank should return same tensor
        let gathered = all_gather_sequence(&tensor)?;
        assert_eq!(gathered.dims(), tensor.dims());
        Ok(())
    }
    
    #[test]
    fn test_reduce_scatter_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, (2, 4, 8), &device)?;
        
        // Single rank should return same tensor
        let scattered = reduce_scatter_sequence(&tensor)?;
        assert_eq!(scattered.dims(), tensor.dims());
        Ok(())
    }
}
