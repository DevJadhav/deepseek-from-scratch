use candle_core::{Result, Tensor, Module};
use candle_nn::{Linear, VarBuilder, ops};

/// Transformer Block with Multi-Head Attention and FFN
pub struct TransformerBlock {
    ln1: candle_nn::LayerNorm,
    ln2: candle_nn::LayerNorm,
    // Attention
    n_head: usize,
    d_head: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    // FFN
    ffn_up: Linear,
    ffn_down: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let d_head = d_model / n_head;
        
        let ln1 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("ln2"))?;
        
        // Multi-head attention projections
        let w_q = candle_nn::linear(d_model, d_model, vb.pp("w_q"))?;
        let w_k = candle_nn::linear(d_model, d_model, vb.pp("w_k"))?;
        let w_v = candle_nn::linear(d_model, d_model, vb.pp("w_v"))?;
        let w_o = candle_nn::linear(d_model, d_model, vb.pp("w_o"))?;
        
        // FFN (4x expansion)
        let ffn_up = candle_nn::linear(d_model, d_model * 4, vb.pp("ffn_up"))?;
        let ffn_down = candle_nn::linear(d_model * 4, d_model, vb.pp("ffn_down"))?;
        
        Ok(Self {
            ln1,
            ln2,
            n_head,
            d_head,
            w_q,
            w_k,
            w_v,
            w_o,
            ffn_up,
            ffn_down,
        })
    }
    
    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        
        // Q, K, V projections
        let q = self.w_q.forward(x)?
            .reshape((batch_size, seq_len, self.n_head, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let k = self.w_k.forward(x)?
            .reshape((batch_size, seq_len, self.n_head, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        let v = self.w_v.forward(x)?
            .reshape((batch_size, seq_len, self.n_head, self.d_head))?
            .transpose(1, 2)?
            .contiguous()?;
        
        // Scaled dot-product attention
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        
        // Causal mask
        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 1 } else { 0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), x.device())?;
        let mask = mask.broadcast_as((batch_size, self.n_head, seq_len, seq_len))?;
        
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(attn_scores.shape())?;
        let attn_scores = mask.where_cond(&attn_scores, &neg_inf)?;
        
        let attn_weights = ops::softmax(&attn_scores, 3)?;
        let context = attn_weights.matmul(&v)?;
        
        // Reshape back
        let context = context.transpose(1, 2)?
            .reshape((batch_size, seq_len, d_model))?;
        
        self.w_o.forward(&context)
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm attention
        let residual = x;
        let x = self.ln1.forward(x)?;
        let x = self.attention(&x)?;
        let x = (x + residual)?;
        
        // Pre-norm FFN
        let residual = &x;
        let h = self.ln2.forward(&x)?;
        let h = self.ffn_up.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ffn_down.forward(&h)?;
        let x = (h + residual)?;
        
        Ok(x)
    }
}

// MTP Module: A lightweight transformer block for predicting future tokens
pub struct MTPModule {
    block: TransformerBlock,
    proj_out: Linear,
}

impl MTPModule {
    pub fn new(d_model: usize, n_head: usize, vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let block = TransformerBlock::new(d_model, n_head, vb.pp("block"))?;
        let proj_out = candle_nn::linear(d_model, vocab_size, vb.pp("proj_out"))?;

        Ok(Self {
            block,
            proj_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Returns (logits, hidden_state)
        let hidden = self.block.forward(x)?;
        let logits = self.proj_out.forward(&hidden)?;
        Ok((logits, hidden))
    }
}

// MTP Model: Full transformer base model with MTP prediction heads
pub struct MTPModel {
    embed: candle_nn::Embedding,
    base_blocks: Vec<TransformerBlock>,
    ln_f: candle_nn::LayerNorm,
    lm_head: Linear,
    mtp_modules: Vec<MTPModule>,
}

impl MTPModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        k_predictions: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n_head = 8; // Default number of attention heads
        
        let embed = candle_nn::embedding(vocab_size, d_model, vb.pp("embed"))?;
        
        // Base transformer blocks with full attention
        let mut base_blocks = Vec::new();
        for i in 0..n_layers {
            base_blocks.push(TransformerBlock::new(d_model, n_head, vb.pp(format!("block_{}", i)))?);
        }
        
        // Final layer norm and LM head for main prediction
        let ln_f = candle_nn::layer_norm(d_model, 1e-5, vb.pp("ln_f"))?;
        let lm_head = candle_nn::linear(d_model, vocab_size, vb.pp("lm_head"))?;
        
        // MTP modules for predicting future tokens
        let mut mtp_modules = Vec::new();
        for i in 0..k_predictions {
            mtp_modules.push(MTPModule::new(d_model, n_head, vocab_size, vb.pp(format!("mtp_{}", i)))?);
        }

        Ok(Self {
            embed,
            base_blocks,
            ln_f,
            lm_head,
            mtp_modules,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        // 1. Embedding
        let mut x = self.embed.forward(input_ids)?;
        
        // 2. Base transformer forward pass
        for block in &self.base_blocks {
            x = block.forward(&x)?;
        }
        
        // 3. Main prediction (next token t+1)
        let hidden = self.ln_f.forward(&x)?;
        let main_logits = self.lm_head.forward(&hidden)?;

        // 4. MTP Forward (Sequential)
        // Each MTP module takes the previous hidden state and predicts the next token
        // MTP[0] predicts t+2, MTP[1] predicts t+3, etc.
        let mut mtp_logits = Vec::new();
        let mut current_hidden = x;

        for module in &self.mtp_modules {
            let (logits, new_hidden) = module.forward(&current_hidden)?;
            mtp_logits.push(logits);
            current_hidden = new_hidden;
        }

        Ok((main_logits, mtp_logits))
    }
}
