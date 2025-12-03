//! Ring Attention for Sequence Parallelism.
//!
//! Implements ring attention where each rank holds a portion of the sequence
//! and attention is computed by passing K/V chunks around a ring.

use candle_core::{Result, Tensor, DType, Device};
use super::{get_sp_size, get_sp_rank, get_sp_group};

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

#[cfg(test)]
mod tests {
    use super::*;
    
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
}
