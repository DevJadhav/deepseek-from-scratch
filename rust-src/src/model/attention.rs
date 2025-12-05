use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, ops};

/// Configuration for attention computation backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackend {
    /// Standard attention (always available)
    Standard,
    /// Memory-efficient attention (chunked computation)
    MemoryEfficient,
    /// Flash attention (if available via Candle CUDA)
    Flash,
}

impl Default for AttentionBackend {
    fn default() -> Self {
        Self::Standard
    }
}

/// Configuration for attention computation
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Attention backend to use
    pub backend: AttentionBackend,
    /// Whether to use causal masking
    pub is_causal: bool,
    /// Dropout probability (0.0 during inference)
    pub dropout_p: f32,
    /// Chunk size for memory-efficient attention (None = no chunking)
    pub chunk_size: Option<usize>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            backend: AttentionBackend::Standard,
            is_causal: true,
            dropout_p: 0.0,
            chunk_size: None,
        }
    }
}

/// Detect optimal attention backend for current device
pub fn detect_optimal_backend(device: &Device) -> AttentionBackend {
    match device {
        Device::Cuda(_) => {
            // Check if flash attention is available
            // For now, default to memory-efficient on CUDA
            AttentionBackend::MemoryEfficient
        }
        Device::Metal(_) => {
            // Metal backend uses optimized attention
            AttentionBackend::MemoryEfficient
        }
        Device::Cpu => AttentionBackend::Standard,
    }
}

/// Memory-efficient attention with chunked computation
/// 
/// Splits the sequence into chunks to reduce peak memory usage.
/// This is particularly useful for long sequences.
pub fn chunked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    chunk_size: usize,
    is_causal: bool,
) -> Result<Tensor> {
    let (batch_size, num_heads, seq_len, d_head) = q.dims4()?;
    let scale = 1.0 / (d_head as f64).sqrt();
    
    if seq_len <= chunk_size {
        return standard_attention(q, k, v, is_causal);
    }
    
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    let mut outputs = Vec::with_capacity(num_chunks);
    
    for i in 0..num_chunks {
        let start_idx = i * chunk_size;
        let end_idx = ((i + 1) * chunk_size).min(seq_len);
        
        // Extract Q chunk
        let q_chunk = q.narrow(2, start_idx, end_idx - start_idx)?;
        
        // For causal attention, K/V only include up to current position
        let (k_chunk, v_chunk) = if is_causal {
            (
                k.narrow(2, 0, end_idx)?,
                v.narrow(2, 0, end_idx)?,
            )
        } else {
            (k.clone(), v.clone())
        };
        
        // Compute attention for this chunk
        let attn_scores = (q_chunk.matmul(&k_chunk.transpose(2, 3)?)? * scale)?;
        
        // Apply causal mask if needed (only for first chunk needs full mask)
        let attn_scores = if is_causal && i == 0 {
            apply_causal_mask(&attn_scores)?
        } else if is_causal {
            // For subsequent chunks, mask positions after the chunk
            apply_causal_mask_for_chunk(&attn_scores, start_idx)?
        } else {
            attn_scores
        };
        
        let attn_weights = ops::softmax(&attn_scores, 3)?;
        let chunk_output = attn_weights.matmul(&v_chunk)?;
        outputs.push(chunk_output);
    }
    
    // Concatenate outputs along sequence dimension
    Tensor::cat(&outputs, 2)
}

/// Apply causal mask to attention scores
fn apply_causal_mask(attn_scores: &Tensor) -> Result<Tensor> {
    let (batch_size, num_heads, q_len, kv_len) = attn_scores.dims4()?;
    
    let mask: Vec<u8> = (0..q_len)
        .flat_map(|i| (0..kv_len).map(move |j| if j <= i { 1 } else { 0 }))
        .collect();
    let mask = Tensor::from_vec(mask, (q_len, kv_len), attn_scores.device())?;
    let mask = mask.broadcast_as((batch_size, num_heads, q_len, kv_len))?;
    
    let neg_inf = Tensor::new(f32::NEG_INFINITY, attn_scores.device())?
        .broadcast_as(attn_scores.shape())?;
    mask.where_cond(attn_scores, &neg_inf)
}

/// Apply causal mask for a specific chunk
fn apply_causal_mask_for_chunk(attn_scores: &Tensor, query_offset: usize) -> Result<Tensor> {
    let (batch_size, num_heads, q_len, kv_len) = attn_scores.dims4()?;
    
    let mask: Vec<u8> = (0..q_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                if j <= i + query_offset { 1 } else { 0 }
            })
        })
        .collect();
    let mask = Tensor::from_vec(mask, (q_len, kv_len), attn_scores.device())?;
    let mask = mask.broadcast_as((batch_size, num_heads, q_len, kv_len))?;
    
    let neg_inf = Tensor::new(f32::NEG_INFINITY, attn_scores.device())?
        .broadcast_as(attn_scores.shape())?;
    mask.where_cond(attn_scores, &neg_inf)
}

/// Standard scaled dot-product attention
pub fn standard_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    is_causal: bool,
) -> Result<Tensor> {
    let (_, _, _, d_head) = q.dims4()?;
    let scale = 1.0 / (d_head as f64).sqrt();
    
    let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    
    let attn_scores = if is_causal {
        apply_causal_mask(&attn_scores)?
    } else {
        attn_scores
    };
    
    let attn_weights = ops::softmax(&attn_scores, 3)?;
    attn_weights.matmul(v)
}

/// Scaled dot-product attention with configurable backend
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    config: &AttentionConfig,
) -> Result<Tensor> {
    match config.backend {
        AttentionBackend::Standard => {
            standard_attention(q, k, v, config.is_causal)
        }
        AttentionBackend::MemoryEfficient => {
            let chunk_size = config.chunk_size.unwrap_or(1024);
            chunked_attention(q, k, v, chunk_size, config.is_causal)
        }
        AttentionBackend::Flash => {
            // Flash attention not directly available in Candle
            // Fall back to memory-efficient
            let chunk_size = config.chunk_size.unwrap_or(1024);
            chunked_attention(q, k, v, chunk_size, config.is_causal)
        }
    }
}

pub struct MultiQueryAttention {
    d_model: usize,
    num_heads: usize,
    d_head: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    config: AttentionConfig,
}

impl MultiQueryAttention {
    pub fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_config(d_model, num_heads, AttentionConfig::default(), vb)
    }
    
    pub fn with_config(
        d_model: usize,
        num_heads: usize,
        config: AttentionConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            candle_core::bail!("d_model must be divisible by num_heads");
        }
        let d_head = d_model / num_heads;
        
        let w_q = candle_nn::linear(d_model, d_model, vb.pp("w_q"))?;
        let w_k = candle_nn::linear(d_model, d_head, vb.pp("w_k"))?; // Single projection for K
        let w_v = candle_nn::linear(d_model, d_head, vb.pp("w_v"))?; // Single projection for V
        let w_o = candle_nn::linear(d_model, d_model, vb.pp("w_o"))?;

        Ok(Self {
            d_model,
            num_heads,
            d_head,
            w_q,
            w_k,
            w_v,
            w_o,
            config,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Query: (B, seq_len, num_heads, d_head) -> (B, num_heads, seq_len, d_head)
        let q = self.w_q.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // Key & Value: (B, seq_len, 1, d_head) -> (B, 1, seq_len, d_head)
        let k = self.w_k.forward(x)?
            .reshape((batch_size, seq_len, 1, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let v = self.w_v.forward(x)?
            .reshape((batch_size, seq_len, 1, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // Broadcast K and V to match num_heads
        let k = k.broadcast_as((batch_size, self.num_heads, seq_len, self.d_head))?.contiguous()?;
        let v = v.broadcast_as((batch_size, self.num_heads, seq_len, self.d_head))?.contiguous()?;

        // Use configurable attention backend
        let context = scaled_dot_product_attention(&q, &k, &v, &self.config)?;

        // (B, num_heads, seq_len, d_head) -> (B, seq_len, d_model)
        let context = context.transpose(1, 2)?
            .reshape((batch_size, seq_len, self.d_model))?;

        self.w_o.forward(&context)
    }
}

pub struct GroupedQueryAttention {
    d_model: usize,
    num_heads: usize,
    num_groups: usize,
    d_head: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    config: AttentionConfig,
}

impl GroupedQueryAttention {
    pub fn new(d_model: usize, num_heads: usize, num_groups: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_config(d_model, num_heads, num_groups, AttentionConfig::default(), vb)
    }
    
    pub fn with_config(
        d_model: usize,
        num_heads: usize,
        num_groups: usize,
        config: AttentionConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            candle_core::bail!("d_model must be divisible by num_heads");
        }
        if num_heads % num_groups != 0 {
            candle_core::bail!("num_heads must be divisible by num_groups");
        }
        
        let d_head = d_model / num_heads;
        
        let w_q = candle_nn::linear(d_model, d_model, vb.pp("w_q"))?;
        let w_k = candle_nn::linear(d_model, num_groups * d_head, vb.pp("w_k"))?;
        let w_v = candle_nn::linear(d_model, num_groups * d_head, vb.pp("w_v"))?;
        let w_o = candle_nn::linear(d_model, d_model, vb.pp("w_o"))?;

        Ok(Self {
            d_model,
            num_heads,
            num_groups,
            d_head,
            w_q,
            w_k,
            w_v,
            w_o,
            config,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Query: (B, seq_len, num_heads, d_head) -> (B, num_heads, seq_len, d_head)
        let q = self.w_q.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // Key & Value: (B, seq_len, num_groups, d_head) -> (B, num_groups, seq_len, d_head)
        let k = self.w_k.forward(x)?
            .reshape((batch_size, seq_len, self.num_groups, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let v = self.w_v.forward(x)?
            .reshape((batch_size, seq_len, self.num_groups, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // Repeat K and V to match num_heads
        let heads_per_group = self.num_heads / self.num_groups;
        
        // (B, num_groups, 1, seq_len, d_head)
        let k = k.unsqueeze(2)?;
        let v = v.unsqueeze(2)?;
        
        // (B, num_groups, heads_per_group, seq_len, d_head)
        let k = k.broadcast_as((batch_size, self.num_groups, heads_per_group, seq_len, self.d_head))?;
        let v = v.broadcast_as((batch_size, self.num_groups, heads_per_group, seq_len, self.d_head))?;
        
        // Flatten to (B, num_heads, seq_len, d_head)
        let k = k.flatten(1, 2)?;
        let v = v.flatten(1, 2)?;

        // Use configurable attention backend
        let context = scaled_dot_product_attention(&q, &k, &v, &self.config)?;

        let context = context.transpose(1, 2)?
            .reshape((batch_size, seq_len, self.d_model))?;

        self.w_o.forward(&context)
    }
}

