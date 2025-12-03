use candle_core::{Device, Result, Tensor, IndexOp};
use candle_nn::{Linear, Module, VarBuilder, ops};
use crate::distributed::ring_attention::{RingAttention, RingAttentionConfig};
use crate::distributed::{get_sp_size, get_sp_rank};

/// Configuration for extended RoPE supporting 128K+ context
#[derive(Clone, Debug)]
pub struct RoPEConfig {
    /// Head dimension
    pub d_head: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Base frequency (default: 10000.0)
    pub base: f32,
    /// RoPE scaling type
    pub scaling_type: RoPEScalingType,
    /// Original trained context length (for scaling)
    pub original_max_seq_len: usize,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            d_head: 64,
            max_seq_len: 131072, // 128K
            base: 10000.0,
            scaling_type: RoPEScalingType::None,
            original_max_seq_len: 4096,
        }
    }
}

impl RoPEConfig {
    /// Create config for 128K context with NTK-aware scaling
    pub fn for_128k_ntk_aware(d_head: usize) -> Self {
        Self {
            d_head,
            max_seq_len: 131072,
            base: 10000.0,
            scaling_type: RoPEScalingType::NTKAware { alpha: 32.0 },
            original_max_seq_len: 4096,
        }
    }

    /// Create config for 128K context with YaRN scaling
    pub fn for_128k_yarn(d_head: usize) -> Self {
        Self {
            d_head,
            max_seq_len: 131072,
            base: 10000.0,
            scaling_type: RoPEScalingType::YaRN {
                scale: 32.0,
                original_max_seq_len: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: 0.1,
            },
            original_max_seq_len: 4096,
        }
    }
}

/// RoPE scaling types for long context
#[derive(Clone, Debug)]
pub enum RoPEScalingType {
    /// No scaling (original RoPE)
    None,
    /// Linear interpolation
    Linear { scale: f32 },
    /// NTK-aware scaling (modifies base frequency)
    NTKAware { alpha: f32 },
    /// YaRN: Yet another RoPE extensioN
    YaRN {
        scale: f32,
        original_max_seq_len: usize,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
    },
    /// Dynamic NTK (compute alpha based on sequence length)
    DynamicNTK { max_position_embeddings: usize },
}

pub struct RotaryPositionalEncoding {
    theta: Tensor,
}

impl RotaryPositionalEncoding {
    pub fn new(d_head: usize, _max_seq_len: usize, device: &Device) -> Result<Self> {
        let theta: Vec<f32> = (0..d_head)
            .step_by(2)
            .map(|i| 1.0 / 10000f32.powf(i as f32 / d_head as f32))
            .collect();
        let theta = Tensor::from_vec(theta, (d_head / 2,), device)?;
        Ok(Self { theta })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, num_heads, seq_len, d_head)
        let (batch, num_heads, seq_len, d_head) = x.dims4()?;
        
        // Compute frequencies
        // positions: (seq_len)
        let positions = Tensor::arange(0f32, seq_len as f32, x.device())?;
        
        // freqs: (seq_len, d_head / 2)
        // outer product of positions and theta
        let freqs = positions.unsqueeze(1)?.matmul(&self.theta.unsqueeze(0)?)?;

        // cos and sin: (seq_len, d_head / 2)
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        // Reshape x to separate even and odd indices
        // x_reshaped: (batch, num_heads, seq_len, d_head/2, 2)
        let x_reshaped = x.reshape((batch, num_heads, seq_len, d_head / 2, 2))?;
        
        // Split into real and imag parts
        // real: (batch, num_heads, seq_len, d_head/2)
        // imag: (batch, num_heads, seq_len, d_head/2)
        let real = x_reshaped.i((.., .., .., .., 0))?;
        let imag = x_reshaped.i((.., .., .., .., 1))?;

        // Broadcast cos and sin to match batch and num_heads
        // (1, 1, seq_len, d_head/2)
        let cos = cos.reshape((1, 1, seq_len, d_head / 2))?;
        let sin = sin.reshape((1, 1, seq_len, d_head / 2))?;
        
        let cos = cos.broadcast_as((batch, num_heads, seq_len, d_head / 2))?;
        let sin = sin.broadcast_as((batch, num_heads, seq_len, d_head / 2))?;

        // Apply rotation
        // x_out_real = real * cos - imag * sin
        // x_out_imag = real * sin + imag * cos
        let out_real = (real.mul(&cos)? - imag.mul(&sin)?)?;
        let out_imag = (real.mul(&sin)? + imag.mul(&cos)?)?;

        // Stack back together
        // (batch, num_heads, seq_len, d_head/2, 2)
        let out = Tensor::stack(&[&out_real, &out_imag], 4)?;
        
        // Flatten back to (batch, num_heads, seq_len, d_head)
        out.flatten_from(3)
    }
}

/// Extended Rotary Positional Encoding supporting 128K+ context
/// 
/// Implements multiple scaling strategies:
/// - Linear interpolation
/// - NTK-aware scaling (modifies base frequency)
/// - YaRN (Yet another RoPE extensioN)
/// - Dynamic NTK
pub struct ExtendedRotaryPositionalEncoding {
    config: RoPEConfig,
    /// Precomputed inverse frequencies (may be scaled)
    inv_freq: Tensor,
    /// YaRN-specific: attention scaling factor
    mscale: f32,
    device: Device,
}

impl ExtendedRotaryPositionalEncoding {
    pub fn new(config: RoPEConfig, device: &Device) -> Result<Self> {
        let inv_freq = Self::compute_inv_freq(&config, device)?;
        let mscale = Self::compute_mscale(&config);
        
        Ok(Self {
            config,
            inv_freq,
            mscale,
            device: device.clone(),
        })
    }

    /// Compute inverse frequencies based on scaling type
    fn compute_inv_freq(config: &RoPEConfig, device: &Device) -> Result<Tensor> {
        let d_head = config.d_head;
        
        match &config.scaling_type {
            RoPEScalingType::None => {
                // Standard RoPE
                let inv_freq: Vec<f32> = (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / config.base.powf(i as f32 / d_head as f32))
                    .collect();
                Tensor::from_vec(inv_freq, (d_head / 2,), device)
            }
            
            RoPEScalingType::Linear { scale } => {
                // Linear scaling: divide frequencies by scale
                let inv_freq: Vec<f32> = (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / (scale * config.base.powf(i as f32 / d_head as f32)))
                    .collect();
                Tensor::from_vec(inv_freq, (d_head / 2,), device)
            }
            
            RoPEScalingType::NTKAware { alpha } => {
                // NTK-aware: scale the base frequency
                // new_base = base * alpha^(d/(d-2))
                let new_base = config.base * alpha.powf(d_head as f32 / (d_head as f32 - 2.0));
                let inv_freq: Vec<f32> = (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / new_base.powf(i as f32 / d_head as f32))
                    .collect();
                Tensor::from_vec(inv_freq, (d_head / 2,), device)
            }
            
            RoPEScalingType::YaRN { 
                scale, 
                original_max_seq_len, 
                beta_fast, 
                beta_slow,
                .. 
            } => {
                // YaRN: interpolate between scaled and unscaled based on frequency
                let half_dim = d_head / 2;
                let mut inv_freq = Vec::with_capacity(half_dim);
                
                for i in (0..d_head).step_by(2) {
                    let dim_idx = i as f32 / d_head as f32;
                    let base_freq = 1.0 / config.base.powf(dim_idx);
                    
                    // Compute wavelength
                    let wavelength = 2.0 * std::f32::consts::PI / base_freq;
                    
                    // Compute interpolation factor
                    let low_freq_wavelen = (*original_max_seq_len as f32) / *beta_slow;
                    let high_freq_wavelen = (*original_max_seq_len as f32) / *beta_fast;
                    
                    let gamma = if wavelength < high_freq_wavelen {
                        // High frequency: no interpolation
                        0.0
                    } else if wavelength > low_freq_wavelen {
                        // Low frequency: full interpolation
                        1.0
                    } else {
                        // Middle: smooth interpolation
                        let ratio = (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen);
                        ratio
                    };
                    
                    // Interpolate: (1-gamma)*original + gamma*scaled
                    let scaled_freq = base_freq / scale;
                    let final_freq = (1.0 - gamma) * base_freq + gamma * scaled_freq;
                    
                    inv_freq.push(final_freq);
                }
                
                Tensor::from_vec(inv_freq, (half_dim,), device)
            }
            
            RoPEScalingType::DynamicNTK { max_position_embeddings } => {
                // Dynamic NTK: same as NTK-aware but alpha computed from ratio
                let alpha = (config.max_seq_len as f32 / *max_position_embeddings as f32).max(1.0);
                let new_base = config.base * alpha.powf(d_head as f32 / (d_head as f32 - 2.0));
                let inv_freq: Vec<f32> = (0..d_head)
                    .step_by(2)
                    .map(|i| 1.0 / new_base.powf(i as f32 / d_head as f32))
                    .collect();
                Tensor::from_vec(inv_freq, (d_head / 2,), device)
            }
        }
    }

    /// Compute attention scaling factor for YaRN
    fn compute_mscale(config: &RoPEConfig) -> f32 {
        match &config.scaling_type {
            RoPEScalingType::YaRN { scale, attention_factor, .. } => {
                // mscale = 0.1 * ln(scale) + 1.0
                let mscale = attention_factor * scale.ln() + 1.0;
                mscale
            }
            _ => 1.0,
        }
    }

    /// Forward pass applying RoPE
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_offset(x, 0)
    }

    /// Forward pass with position offset (for sequence parallelism or KV cache)
    pub fn forward_with_offset(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch, num_heads, seq_len, d_head) = x.dims4()?;
        
        // Compute positions with offset
        let positions = Tensor::arange(
            offset as f32,
            (offset + seq_len) as f32,
            x.device()
        )?;
        
        // Compute frequencies: (seq_len, d_head/2)
        let freqs = positions.unsqueeze(1)?.matmul(&self.inv_freq.unsqueeze(0)?)?;
        
        // Compute cos and sin with optional mscale
        let cos = (freqs.cos()? * self.mscale as f64)?;
        let sin = (freqs.sin()? * self.mscale as f64)?;
        
        // Reshape x to separate even and odd dimensions
        let x_reshaped = x.reshape((batch, num_heads, seq_len, d_head / 2, 2))?;
        
        let real = x_reshaped.i((.., .., .., .., 0))?;
        let imag = x_reshaped.i((.., .., .., .., 1))?;
        
        // Broadcast cos/sin
        let cos = cos.reshape((1, 1, seq_len, d_head / 2))?;
        let sin = sin.reshape((1, 1, seq_len, d_head / 2))?;
        let cos = cos.broadcast_as((batch, num_heads, seq_len, d_head / 2))?;
        let sin = sin.broadcast_as((batch, num_heads, seq_len, d_head / 2))?;
        
        // Apply rotation
        let out_real = (real.mul(&cos)? - imag.mul(&sin)?)?;
        let out_imag = (real.mul(&sin)? + imag.mul(&cos)?)?;
        
        // Stack and flatten
        let out = Tensor::stack(&[&out_real, &out_imag], 4)?;
        out.flatten_from(3)
    }

    /// Get the mscale factor (for attention score scaling in YaRN)
    pub fn get_mscale(&self) -> f32 {
        self.mscale
    }

    /// Get the configuration
    pub fn config(&self) -> &RoPEConfig {
        &self.config
    }
}

pub struct MultiHeadLatentAttention {
    d_model: usize,
    num_heads: usize,
    d_head: usize,
    d_latent: usize,
    w_q: Linear,
    w_dkv: Linear,
    w_uk: Linear,
    w_uv: Linear,
    w_o: Linear,
}

impl MultiHeadLatentAttention {
    pub fn new(d_model: usize, num_heads: usize, d_latent: usize, vb: VarBuilder) -> Result<Self> {
        if d_model % num_heads != 0 {
            candle_core::bail!("d_model must be divisible by num_heads");
        }
        let d_head = d_model / num_heads;

        let w_q = candle_nn::linear(d_model, d_model, vb.pp("w_q"))?;
        let w_dkv = candle_nn::linear(d_model, d_latent, vb.pp("w_dkv"))?;
        let w_uk = candle_nn::linear(d_latent, d_model, vb.pp("w_uk"))?;
        let w_uv = candle_nn::linear(d_latent, d_model, vb.pp("w_uv"))?;
        let w_o = candle_nn::linear(d_model, d_model, vb.pp("w_o"))?;

        Ok(Self {
            d_model,
            num_heads,
            d_head,
            d_latent,
            w_q,
            w_dkv,
            w_uk,
            w_uv,
            w_o,
        })
    }

    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut crate::model::kv_cache::KVCache>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // 1. Query Path
        let q = self.w_q.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // 2. Key/Value Path (MLA)
        // Down-project to latent space
        let c_kv = self.w_dkv.forward(x)?; // (batch, seq_len, d_latent)

        // Up-project to full K and V
        let k = self.w_uk.forward(&c_kv)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let v = self.w_uv.forward(&c_kv)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k, v)
        };

        // 3. Standard Attention
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Causal mask
        // If using cache, we need to handle masking correctly.
        // The query length is seq_len (usually 1 during generation), key length is total_seq_len.
        let total_seq_len = k.dim(2)?;
        
        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..total_seq_len).map(move |j| {
                // For the i-th query (relative to current chunk), it attends to j-th key.
                // If we are generating, the current query is at position (total_seq_len - seq_len + i).
                // It should attend to all keys <= its position.
                let current_pos = total_seq_len - seq_len + i;
                if j <= current_pos { 1 } else { 0 }
            }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, total_seq_len), x.device())?;
        let mask = mask.broadcast_as((batch_size, self.num_heads, seq_len, total_seq_len))?;
        
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(attn_scores.shape())?;
        let attn_scores = mask.where_cond(&attn_scores, &neg_inf)?;

        let attn_weights = ops::softmax(&attn_scores, 3)?;
        
        let context = attn_weights.matmul(&v)?;

        let context = context.transpose(1, 2)?
            .reshape((batch_size, seq_len, self.d_model))?;

        self.w_o.forward(&context)
    }
}

// --- Sequence Parallel MLA ---

/// Multi-Head Latent Attention with Sequence Parallelism.
/// 
/// This variant uses Ring Attention to distribute the sequence dimension
/// across multiple ranks, enabling training on longer sequences.
pub struct SequenceParallelMLA {
    d_model: usize,
    num_heads: usize,
    d_head: usize,
    _d_latent: usize,
    w_q: Linear,
    w_dkv: Linear,
    w_uk: Linear,
    w_uv: Linear,
    w_o: Linear,
    /// Ring attention for sequence parallelism
    ring_attention: RingAttention,
}

impl SequenceParallelMLA {
    /// Create a new sequence-parallel MLA layer.
    ///
    /// Args:
    ///   d_model: Model dimension
    ///   num_heads: Number of attention heads
    ///   d_latent: Latent dimension for K/V compression
    ///   causal: Whether to use causal masking
    ///   vb: Variable builder for weight initialization
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_latent: usize,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            candle_core::bail!("d_model must be divisible by num_heads");
        }
        let d_head = d_model / num_heads;

        let w_q = candle_nn::linear(d_model, d_model, vb.pp("w_q"))?;
        let w_dkv = candle_nn::linear(d_model, d_latent, vb.pp("w_dkv"))?;
        let w_uk = candle_nn::linear(d_latent, d_model, vb.pp("w_uk"))?;
        let w_uv = candle_nn::linear(d_latent, d_model, vb.pp("w_uv"))?;
        let w_o = candle_nn::linear(d_model, d_model, vb.pp("w_o"))?;

        // Ring attention configuration
        let ring_config = RingAttentionConfig::new(causal, 0.0, Some(1.0 / (d_head as f32).sqrt()));
        let ring_attention = RingAttention::new(ring_config);

        Ok(Self {
            d_model,
            num_heads,
            d_head,
            _d_latent: d_latent,
            w_q,
            w_dkv,
            w_uk,
            w_uv,
            w_o,
            ring_attention,
        })
    }

    /// Forward pass with sequence parallelism.
    ///
    /// The input x should already be partitioned across SP ranks (each rank
    /// holds seq_len/sp_size positions). The ring attention mechanism will
    /// communicate K/V chunks to compute the full attention.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, local_seq_len, _) = x.dims3()?;

        // 1. Query Path (local computation)
        let q = self.w_q.forward(x)?
            .reshape((batch_size, local_seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?  // (batch, heads, local_seq, d_head)
            .contiguous()?;

        // 2. Key/Value Path with MLA compression (local computation)
        let c_kv = self.w_dkv.forward(x)?; // (batch, local_seq, d_latent)

        let k = self.w_uk.forward(&c_kv)?
            .reshape((batch_size, local_seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let v = self.w_uv.forward(&c_kv)?
            .reshape((batch_size, local_seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // 3. Ring Attention - handles communication and distributed attention
        let context = self.ring_attention.forward(&q, &k, &v)?;

        // 4. Output projection
        let context = context.transpose(1, 2)?
            .reshape((batch_size, local_seq_len, self.d_model))?;

        self.w_o.forward(&context)
    }
    
    /// Shard input sequence across SP ranks.
    /// 
    /// Takes full sequence input and returns the local portion for this rank.
    pub fn shard_input(&self, x: &Tensor) -> Result<Tensor> {
        let sp_size = get_sp_size();
        let sp_rank = get_sp_rank();
        
        if sp_size <= 1 {
            return Ok(x.clone());
        }
        
        let (_batch, seq_len, _d_model) = x.dims3()?;
        let local_seq_len = seq_len / sp_size;
        let start = sp_rank * local_seq_len;
        
        x.narrow(1, start, local_seq_len)
    }
    
    /// Gather output from all SP ranks.
    /// 
    /// After forward pass, each rank has output for its local sequence portion.
    /// This function gathers all portions to reconstruct the full sequence.
    pub fn gather_output(&self, x: &Tensor) -> Result<Tensor> {
        let sp_size = get_sp_size();
        
        if sp_size <= 1 {
            return Ok(x.clone());
        }
        
        // Use all_gather to collect from all ranks
        if let Some(group) = crate::distributed::get_sp_group() {
            group.communicator.all_gather(x)
        } else {
            Ok(x.clone())
        }
    }
}

/// DeepSeek Attention with Sequence Parallelism.
/// 
/// Extends DeepSeekAttention with Ring Attention for sequence parallelism.
pub struct SequenceParallelDeepSeekAttention {
    d_model: usize,
    num_heads: usize,
    d_head: usize,
    _d_latent: usize,
    d_rope: usize,
    w_q_content: Linear,
    w_dkv_content: Linear,
    w_uk_content: Linear,
    w_uv_content: Linear,
    w_k_pos: Linear,
    w_q_pos: Linear,
    w_o: Linear,
    rope: RotaryPositionalEncoding,
    ring_attention: RingAttention,
}

impl SequenceParallelDeepSeekAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_latent: usize,
        d_rope: usize,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            candle_core::bail!("d_model must be divisible by num_heads");
        }
        let d_head = d_model / num_heads;

        // Content Path
        let w_q_content = candle_nn::linear(d_model, d_model, vb.pp("w_q_content"))?;
        let w_dkv_content = candle_nn::linear(d_model, d_latent, vb.pp("w_dkv_content"))?;
        let w_uk_content = candle_nn::linear(d_latent, d_model, vb.pp("w_uk_content"))?;
        let w_uv_content = candle_nn::linear(d_latent, d_model, vb.pp("w_uv_content"))?;

        // Position Path
        let w_k_pos = candle_nn::linear(d_model, d_rope * num_heads, vb.pp("w_k_pos"))?;
        let w_q_pos = candle_nn::linear(d_model, d_rope * num_heads, vb.pp("w_q_pos"))?;

        let rope = RotaryPositionalEncoding::new(d_rope, 2048, vb.device())?;

        let w_o = candle_nn::linear(d_model, d_model, vb.pp("w_o"))?;

        // Ring attention for sequence parallelism
        let ring_config = RingAttentionConfig::new(causal, 0.0, Some(1.0 / (d_head as f32).sqrt()));
        let ring_attention = RingAttention::new(ring_config);

        Ok(Self {
            d_model,
            num_heads,
            d_head,
            _d_latent: d_latent,
            d_rope,
            w_q_content,
            w_dkv_content,
            w_uk_content,
            w_uv_content,
            w_k_pos,
            w_q_pos,
            w_o,
            rope,
            ring_attention,
        })
    }

    /// Forward pass with sequence parallelism.
    ///
    /// Input should be the local sequence portion for this rank.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, local_seq_len, _) = x.dims3()?;

        // A: Content Path
        let q_c = self.w_q_content.forward(x)?
            .reshape((batch_size, local_seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let c_kv = self.w_dkv_content.forward(x)?;
        
        let k_c = self.w_uk_content.forward(&c_kv)?
            .reshape((batch_size, local_seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let v_c = self.w_uv_content.forward(&c_kv)?
            .reshape((batch_size, local_seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // B: Position Path
        // Note: For proper SP, we need to adjust positions based on SP rank
        let sp_rank = get_sp_rank();
        let global_offset = sp_rank * local_seq_len;
        
        let q_r_unrotated = self.w_q_pos.forward(x)?
            .reshape((batch_size, local_seq_len, self.num_heads, self.d_rope))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let k_r_unrotated = self.w_k_pos.forward(x)?
            .reshape((batch_size, local_seq_len, self.num_heads, self.d_rope))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE with offset for global positions
        let q_r = self.rope.forward_with_offset(&q_r_unrotated, global_offset)?;
        let k_r = self.rope.forward_with_offset(&k_r_unrotated, global_offset)?;

        // C: Combine content and position info
        // We'll use ring attention for the content path primarily
        // Position scores are computed locally per chunk
        
        // Ring attention for content-based attention
        let context_c = self.ring_attention.forward(&q_c, &k_c, &v_c)?;
        
        // Compute position scores locally (since RoPE encodes position)
        // For full ring attention with position, we'd need to extend the protocol
        // Here we use a simplified approach: position attention is local
        let scale_rope = 1.0 / (self.d_rope as f64).sqrt();
        let position_scores = (q_r.matmul(&k_r.transpose(2, 3)?)? * scale_rope)?;
        let position_weights = ops::softmax(&position_scores, 3)?;
        
        // The position-weighted context uses the same V as content
        // This is a simplification; full implementation would integrate better
        let context_p = position_weights.matmul(&v_c)?;
        
        // Average content and position contexts
        let context = ((context_c + context_p)? * 0.5)?;

        let context = context.transpose(1, 2)?
            .reshape((batch_size, local_seq_len, self.d_model))?;

        self.w_o.forward(&context)
    }
}

// Extend RotaryPositionalEncoding with offset support
impl RotaryPositionalEncoding {
    /// Apply RoPE with a position offset (for sequence parallelism).
    pub fn forward_with_offset(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch, num_heads, seq_len, d_head) = x.dims4()?;
        
        // Compute frequencies with offset
        let positions = Tensor::arange(
            offset as f32,
            (offset + seq_len) as f32,
            x.device()
        )?;
        
        let freqs = positions.unsqueeze(1)?.matmul(&self.theta.unsqueeze(0)?)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let x_reshaped = x.reshape((batch, num_heads, seq_len, d_head / 2, 2))?;
        
        let real = x_reshaped.i((.., .., .., .., 0))?;
        let imag = x_reshaped.i((.., .., .., .., 1))?;

        let cos = cos.reshape((1, 1, seq_len, d_head / 2))?;
        let sin = sin.reshape((1, 1, seq_len, d_head / 2))?;
        
        let cos = cos.broadcast_as((batch, num_heads, seq_len, d_head / 2))?;
        let sin = sin.broadcast_as((batch, num_heads, seq_len, d_head / 2))?;

        let out_real = (real.mul(&cos)? - imag.mul(&sin)?)?;
        let out_imag = (real.mul(&sin)? + imag.mul(&cos)?)?;

        let out = Tensor::stack(&[&out_real, &out_imag], 4)?;
        
        out.flatten_from(3)
    }
}

pub struct DeepSeekAttention {
    d_model: usize,
    num_heads: usize,
    d_head: usize,
    d_latent: usize,
    d_rope: usize,
    w_q_content: Linear,
    w_dkv_content: Linear,
    w_uk_content: Linear,
    w_uv_content: Linear,
    w_k_pos: Linear,
    w_q_pos: Linear,
    w_o: Linear,
    rope: RotaryPositionalEncoding,
}

impl DeepSeekAttention {
    pub fn new(d_model: usize, num_heads: usize, d_latent: usize, d_rope: usize, vb: VarBuilder) -> Result<Self> {
        if d_model % num_heads != 0 {
            candle_core::bail!("d_model must be divisible by num_heads");
        }
        let d_head = d_model / num_heads;

        // Content Path
        let w_q_content = candle_nn::linear(d_model, d_model, vb.pp("w_q_content"))?;
        let w_dkv_content = candle_nn::linear(d_model, d_latent, vb.pp("w_dkv_content"))?;
        let w_uk_content = candle_nn::linear(d_latent, d_model, vb.pp("w_uk_content"))?;
        let w_uv_content = candle_nn::linear(d_latent, d_model, vb.pp("w_uv_content"))?;

        // Position Path
        let w_k_pos = candle_nn::linear(d_model, d_rope * num_heads, vb.pp("w_k_pos"))?;
        let w_q_pos = candle_nn::linear(d_model, d_rope * num_heads, vb.pp("w_q_pos"))?;

        let rope = RotaryPositionalEncoding::new(d_rope, 2048, vb.device())?;

        let w_o = candle_nn::linear(d_model, d_model, vb.pp("w_o"))?;

        Ok(Self {
            d_model,
            num_heads,
            d_head,
            d_latent,
            d_rope,
            w_q_content,
            w_dkv_content,
            w_uk_content,
            w_uv_content,
            w_k_pos,
            w_q_pos,
            w_o,
            rope,
        })
    }

    pub fn forward(&self, x: &Tensor, _kv_cache: Option<&mut crate::model::kv_cache::KVCache>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // A: Content Path
        let q_c = self.w_q_content.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let c_kv = self.w_dkv_content.forward(x)?;
        
        let k_c = self.w_uk_content.forward(&c_kv)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let v_c = self.w_uv_content.forward(&c_kv)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // B: Position Path
        let q_r_unrotated = self.w_q_pos.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_rope))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let k_r_unrotated = self.w_k_pos.forward(x)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_rope))?
            .transpose(1, 2)?
            .contiguous()?;

        let q_r = self.rope.forward(&q_r_unrotated)?;
        let k_r = self.rope.forward(&k_r_unrotated)?;

        // C: Combine Scores
        let scale_head = 1.0 / (self.d_head as f64).sqrt();
        let content_scores = (q_c.matmul(&k_c.transpose(2, 3)?)? * scale_head)?;
        
        let scale_rope = 1.0 / (self.d_rope as f64).sqrt();
        let position_scores = (q_r.matmul(&k_r.transpose(2, 3)?)? * scale_rope)?;

        let attn_scores = (content_scores + position_scores)?;

        // D: Masking and Output
        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 1 } else { 0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), x.device())?;
        let mask = mask.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        
        
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(attn_scores.shape())?;
        let attn_scores = mask.where_cond(&attn_scores, &neg_inf)?;

        let attn_weights = ops::softmax(&attn_scores, 3)?;

        let context = attn_weights.matmul(&v_c)?;

        let context = context.transpose(1, 2)?
            .reshape((batch_size, seq_len, self.d_model))?;

        self.w_o.forward(&context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    use candle_core::DType;
    
    #[test]
    fn test_sequence_parallel_mla_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let d_model = 64;
        let num_heads = 4;
        let d_latent = 32;
        let causal = true;
        
        let mla = SequenceParallelMLA::new(d_model, num_heads, d_latent, causal, vb)?;
        
        let batch_size = 2;
        let local_seq_len = 16;
        let x = Tensor::randn(0f32, 1f32, (batch_size, local_seq_len, d_model), &device)?;
        
        let output = mla.forward(&x)?;
        assert_eq!(output.dims(), &[batch_size, local_seq_len, d_model]);
        
        Ok(())
    }
    
    #[test]
    fn test_sequence_parallel_deepseek_attention() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let d_model = 64;
        let num_heads = 4;
        let d_latent = 32;
        let d_rope = 16;
        let causal = true;
        
        let attn = SequenceParallelDeepSeekAttention::new(
            d_model, num_heads, d_latent, d_rope, causal, vb
        )?;
        
        let batch_size = 2;
        let local_seq_len = 16;
        let x = Tensor::randn(0f32, 1f32, (batch_size, local_seq_len, d_model), &device)?;
        
        let output = attn.forward(&x)?;
        assert_eq!(output.dims(), &[batch_size, local_seq_len, d_model]);
        
        Ok(())
    }
    
    #[test]
    fn test_rope_with_offset() -> Result<()> {
        let device = Device::Cpu;
        let d_head = 16;
        
        let rope = RotaryPositionalEncoding::new(d_head, 1024, &device)?;
        
        let batch = 2;
        let num_heads = 4;
        let seq_len = 8;
        
        let x = Tensor::randn(0f32, 1f32, (batch, num_heads, seq_len, d_head), &device)?;
        
        // Test with offset
        let out_offset = rope.forward_with_offset(&x, 100)?;
        assert_eq!(out_offset.dims(), x.dims());
        
        // Output should be different from no-offset version
        let out_no_offset = rope.forward(&x)?;
        let diff = (out_offset - out_no_offset)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff > 0.0, "Offset should produce different output");
        
        Ok(())
    }

    #[test]
    fn test_extended_rope_128k_ntk() -> Result<()> {
        let device = Device::Cpu;
        let d_head = 64;
        
        let config = RoPEConfig::for_128k_ntk_aware(d_head);
        assert_eq!(config.max_seq_len, 131072);
        
        let rope = ExtendedRotaryPositionalEncoding::new(config, &device)?;
        
        let batch = 2;
        let num_heads = 4;
        let seq_len = 128; // Test with reasonable length
        
        let x = Tensor::randn(0f32, 1f32, (batch, num_heads, seq_len, d_head), &device)?;
        
        let output = rope.forward(&x)?;
        assert_eq!(output.dims(), &[batch, num_heads, seq_len, d_head]);
        
        // Verify mscale is 1.0 for NTK (no attention scaling)
        assert!((rope.get_mscale() - 1.0).abs() < 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_extended_rope_128k_yarn() -> Result<()> {
        let device = Device::Cpu;
        let d_head = 64;
        
        let config = RoPEConfig::for_128k_yarn(d_head);
        assert_eq!(config.max_seq_len, 131072);
        
        let rope = ExtendedRotaryPositionalEncoding::new(config, &device)?;
        
        let batch = 2;
        let num_heads = 4;
        let seq_len = 128;
        
        let x = Tensor::randn(0f32, 1f32, (batch, num_heads, seq_len, d_head), &device)?;
        
        let output = rope.forward(&x)?;
        assert_eq!(output.dims(), &[batch, num_heads, seq_len, d_head]);
        
        // YaRN should have mscale > 1.0 for scale > 1
        assert!(rope.get_mscale() > 1.0);
        
        Ok(())
    }

    #[test]
    fn test_extended_rope_with_offset() -> Result<()> {
        let device = Device::Cpu;
        let d_head = 32;
        
        let config = RoPEConfig {
            d_head,
            max_seq_len: 8192,
            scaling_type: RoPEScalingType::Linear { scale: 2.0 },
            ..Default::default()
        };
        
        let rope = ExtendedRotaryPositionalEncoding::new(config, &device)?;
        
        let batch = 1;
        let num_heads = 2;
        let seq_len = 16;
        
        let x = Tensor::randn(0f32, 1f32, (batch, num_heads, seq_len, d_head), &device)?;
        
        // Test with different offsets
        let out_offset_0 = rope.forward_with_offset(&x, 0)?;
        let out_offset_100 = rope.forward_with_offset(&x, 100)?;
        
        // Outputs should be different due to position offset
        let diff = (&out_offset_0 - &out_offset_100)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff > 0.0, "Offset should produce different output");
        
        Ok(())
    }

    #[test]
    fn test_rope_scaling_types() -> Result<()> {
        let device = Device::Cpu;
        let d_head = 32;
        let batch = 1;
        let num_heads = 2;
        let seq_len = 16;
        
        let x = Tensor::randn(0f32, 1f32, (batch, num_heads, seq_len, d_head), &device)?;
        
        // Test all scaling types produce valid output
        let scaling_types = vec![
            RoPEScalingType::None,
            RoPEScalingType::Linear { scale: 4.0 },
            RoPEScalingType::NTKAware { alpha: 8.0 },
            RoPEScalingType::DynamicNTK { max_position_embeddings: 4096 },
        ];
        
        for scaling_type in scaling_types {
            let config = RoPEConfig {
                d_head,
                max_seq_len: 32768,
                scaling_type,
                ..Default::default()
            };
            
            let rope = ExtendedRotaryPositionalEncoding::new(config, &device)?;
            let output = rope.forward(&x)?;
            
            assert_eq!(output.dims(), x.dims());
            
            // Verify no NaN or Inf values - flatten to vec1 for 4D tensor
            let output_flat = output.flatten_all()?;
            let has_nan = output_flat.to_vec1::<f32>()?.iter()
                .any(|v| v.is_nan() || v.is_infinite());
            assert!(!has_nan, "Output should not contain NaN or Inf");
        }
        
        Ok(())
    }
}

