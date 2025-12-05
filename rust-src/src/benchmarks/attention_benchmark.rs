use candle_core::{Device, Result, Tensor, DType, Module, IndexOp};
use candle_nn::{Linear, VarBuilder, ops};
use crate::model::attention::{MultiQueryAttention, GroupedQueryAttention};
use crate::model::mla::{MultiHeadLatentAttention, DeepSeekAttention};

// Configuration struct
#[derive(Clone, Copy)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub block_size: usize,
    pub d_latent: usize, // For MLA
    pub d_rope: usize,   // For MLA/DeepSeek
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_layer: 6,
            n_head: 8,
            n_embd: 512,
            vocab_size: 50257,
            block_size: 256,
            d_latent: 128,
            d_rope: 64,
        }
    }
}

// Standard Multi-Head Attention (MHA)
pub struct MultiHeadAttention {
    n_head: usize,
    d_head: usize,
    c_attn: Linear,
    c_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_head = cfg.n_head;
        let n_embd = cfg.n_embd;
        let d_head = n_embd / n_head;
        
        let c_attn = candle_nn::linear(n_embd, 3 * n_embd, vb.pp("c_attn"))?;
        let c_proj = candle_nn::linear(n_embd, n_embd, vb.pp("c_proj"))?;

        Ok(Self {
            n_head,
            d_head,
            c_attn,
            c_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        
        // (B, T, 3*C)
        let qkv = self.c_attn.forward(x)?;
        
        // Split into Q, K, V
        let qkv = qkv.reshape((b, t, 3, self.n_head, self.d_head))?;
        let q = qkv.i((.., .., 0, .., ..))?; // (B, T, H, D)
        let k = qkv.i((.., .., 1, .., ..))?;
        let v = qkv.i((.., .., 2, .., ..))?;

        // Transpose for attention: (B, H, T, D)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Causal mask
        let mask: Vec<u8> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j <= i { 1 } else { 0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (t, t), x.device())?;
        let mask = mask.broadcast_as((b, self.n_head, t, t))?;
        
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(attn_scores.shape())?;
        let attn_scores = mask.where_cond(&attn_scores, &neg_inf)?;

        let attn_weights = ops::softmax(&attn_scores, 3)?;
        
        let y = attn_weights.matmul(&v)?; // (B, H, T, D)
        
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        
        self.c_proj.forward(&y)
    }
}

// Enum to hold different attention types
pub enum AttentionVariant {
    MHA(MultiHeadAttention),
    MQA(MultiQueryAttention),
    GQA(GroupedQueryAttention),
    MLA(MultiHeadLatentAttention),
    DeepSeek(DeepSeekAttention),
}

impl AttentionVariant {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::MHA(m) => m.forward(x),
            Self::MQA(m) => m.forward(x),
            Self::GQA(m) => m.forward(x),
            Self::MLA(m) => m.forward(x, None),
            Self::DeepSeek(m) => m.forward(x, None),
        }
    }
}

// Simplified Block
pub struct Block {
    ln1: candle_nn::LayerNorm,
    attn: AttentionVariant,
    ln2: candle_nn::LayerNorm,
    mlp_fc: Linear,
    mlp_proj: Linear,
}

impl Block {
    pub fn new(cfg: &Config, attn_type: &str, vb: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln2"))?;
        
        let attn = match attn_type {
            "MHA" => AttentionVariant::MHA(MultiHeadAttention::new(cfg, vb.pp("attn"))?),
            "MQA" => AttentionVariant::MQA(MultiQueryAttention::new(cfg.n_embd, cfg.n_head, vb.pp("attn"))?),
            "GQA" => AttentionVariant::GQA(GroupedQueryAttention::new(cfg.n_embd, cfg.n_head, 2, vb.pp("attn"))?), // 2 groups for demo
            "MLA" => AttentionVariant::MLA(MultiHeadLatentAttention::new(cfg.n_embd, cfg.n_head, cfg.d_latent, vb.pp("attn"))?),
            "DeepSeek" => AttentionVariant::DeepSeek(DeepSeekAttention::new(cfg.n_embd, cfg.n_head, cfg.d_latent, cfg.d_rope, vb.pp("attn"))?),
            _ => candle_core::bail!("Unknown attention type"),
        };

        let mlp_fc = candle_nn::linear(cfg.n_embd, 4 * cfg.n_embd, vb.pp("mlp_fc"))?;
        let mlp_proj = candle_nn::linear(4 * cfg.n_embd, cfg.n_embd, vb.pp("mlp_proj"))?;

        Ok(Self {
            ln1,
            attn,
            ln2,
            mlp_fc,
            mlp_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.ln1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (x + residual)?;
        
        let residual = &x;
        let x = self.ln2.forward(&x)?;
        let x = self.mlp_fc.forward(&x)?;
        let x = x.gelu()?;
        let x = self.mlp_proj.forward(&x)?;
        let x = (x + residual)?;
        
        Ok(x)
    }
}

// Simplified GPT
pub struct GPT {
    wte: candle_nn::Embedding,
    blocks: Vec<Block>,
    ln_f: candle_nn::LayerNorm,
    lm_head: Linear,
}

impl GPT {
    pub fn new(cfg: &Config, attn_type: &str, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let mut blocks = Vec::new();
        for i in 0..cfg.n_layer {
            blocks.push(Block::new(cfg, attn_type, vb.pp(&format!("h.{}", i)))?);
        }
        let ln_f = candle_nn::layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_f"))?;
        let lm_head = candle_nn::linear(cfg.n_embd, cfg.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let mut x = self.wte.forward(idx)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

pub fn run_benchmark() -> Result<()> {
    println!("\n=== Chapter 2 Bonus: Attention Benchmarking ===");
    let device = if candle_core::utils::metal_is_available() {
        println!("Using Metal GPU");
        Device::new_metal(0)?
    } else {
        println!("Using CPU");
        Device::Cpu
    };
    let cfg = Config::default();
    let attn_types = ["MHA", "MQA", "GQA", "MLA", "DeepSeek"];
    let batch_size = 4;
    let seq_len = 64;
    let warmup_iters = 5;
    let measure_iters = 20;
    let input = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
    
    println!("Config: batch={}, seq_len={}, d_model={}, warmup={}, measure={} iterations", 
        batch_size, seq_len, cfg.n_embd, warmup_iters, measure_iters);

    for attn_type in attn_types {
        println!("\nBenchmarking {}...", attn_type);
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GPT::new(&cfg, attn_type, vb)?;
        
        // Warmup
        for _ in 0..warmup_iters {
            let _ = model.forward(&input)?;
        }

        let start = std::time::Instant::now();
        for _ in 0..measure_iters {
            let _ = model.forward(&input)?;
        }
        let duration = start.elapsed();
        println!("{} Average Time: {:.2?} per forward pass", attn_type, duration / measure_iters as u32);
    }

    Ok(())
}
