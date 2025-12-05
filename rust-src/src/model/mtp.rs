use candle_core::{Result, Tensor, Module, DType};
use candle_nn::{Linear, VarBuilder, ops};

// ============================================================================
// MTP Configuration
// ============================================================================

/// Configuration for Multi-Token Prediction
#[derive(Clone, Debug)]
pub struct MTPConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model dimension
    pub d_model: usize,
    /// Embedding dimension (can differ from d_model)
    pub d_embed: usize,
    /// Number of transformer layers in base model
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of future tokens to predict (D parameter)
    pub prediction_depth: usize,
    /// Base weight for MTP losses
    pub mtp_loss_weight: f32,
    /// Decay factor per depth (weight = base * decay^depth)
    pub depth_decay: f32,
    /// MTP dropout rate (randomly skip MTP during training)
    pub mtp_dropout: f32,
    /// Enable speculative decoding for inference
    pub enable_speculative: bool,
    /// Confidence threshold for speculation acceptance
    pub speculative_threshold: f32,
    /// Use shared embedding with main LM head
    pub use_shared_embedding: bool,
    /// Use sequential refinement (each MTP refines previous)
    pub use_sequential_refinement: bool,
}

impl Default for MTPConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 4096,
            d_embed: 4096,
            num_layers: 24,
            num_heads: 32,
            prediction_depth: 3,
            mtp_loss_weight: 1.0,
            depth_decay: 0.8,
            mtp_dropout: 0.0,
            enable_speculative: true,
            speculative_threshold: 0.9,
            use_shared_embedding: true,
            use_sequential_refinement: true,
        }
    }
}

impl MTPConfig {
    /// Get loss weights for each prediction depth
    pub fn get_loss_weights(&self) -> Vec<f32> {
        (0..self.prediction_depth)
            .map(|d| self.mtp_loss_weight * self.depth_decay.powi(d as i32))
            .collect()
    }
    
    /// Small config for testing
    pub fn small() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 256,
            d_embed: 256,
            num_layers: 4,
            num_heads: 4,
            prediction_depth: 2,
            ..Default::default()
        }
    }
    
    /// Medium config
    pub fn medium() -> Self {
        Self {
            vocab_size: 16000,
            d_model: 1024,
            d_embed: 1024,
            num_layers: 12,
            num_heads: 16,
            prediction_depth: 3,
            ..Default::default()
        }
    }
}

// ============================================================================
// MTP Metrics
// ============================================================================

/// Metrics for tracking MTP performance
#[derive(Clone, Debug, Default)]
pub struct MTPMetrics {
    /// Accuracy at each prediction depth
    pub depth_accuracy: Vec<f32>,
    /// Loss at each prediction depth  
    pub depth_loss: Vec<f32>,
    /// Total samples processed
    pub total_samples: usize,
    /// Speculative decoding acceptance rate
    pub speculation_acceptance_rate: f32,
}

impl MTPMetrics {
    pub fn new(depth: usize) -> Self {
        Self {
            depth_accuracy: vec![0.0; depth],
            depth_loss: vec![0.0; depth],
            total_samples: 0,
            speculation_acceptance_rate: 0.0,
        }
    }
    
    /// Update metrics with new batch results
    pub fn update(&mut self, depth_losses: &[f32], depth_correct: &[usize], batch_size: usize) {
        for (i, &loss) in depth_losses.iter().enumerate() {
            if i < self.depth_loss.len() {
                // Running average
                let old_weight = self.total_samples as f32 / (self.total_samples + batch_size) as f32;
                let new_weight = batch_size as f32 / (self.total_samples + batch_size) as f32;
                self.depth_loss[i] = old_weight * self.depth_loss[i] + new_weight * loss;
            }
        }
        
        for (i, &correct) in depth_correct.iter().enumerate() {
            if i < self.depth_accuracy.len() {
                let acc = correct as f32 / batch_size as f32;
                let old_weight = self.total_samples as f32 / (self.total_samples + batch_size) as f32;
                let new_weight = batch_size as f32 / (self.total_samples + batch_size) as f32;
                self.depth_accuracy[i] = old_weight * self.depth_accuracy[i] + new_weight * acc;
            }
        }
        
        self.total_samples += batch_size;
    }
    
    /// Reset metrics
    pub fn reset(&mut self) {
        self.depth_accuracy.fill(0.0);
        self.depth_loss.fill(0.0);
        self.total_samples = 0;
        self.speculation_acceptance_rate = 0.0;
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

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

// ============================================================================
// MTP Model V2: Configurable with MTPConfig
// ============================================================================

/// MTP Model with full configuration support
pub struct MTPModelV2 {
    config: MTPConfig,
    embed: candle_nn::Embedding,
    base_blocks: Vec<TransformerBlock>,
    ln_f: candle_nn::LayerNorm,
    lm_head: Linear,
    mtp_modules: Vec<MTPModule>,
    /// Loss weights for each depth
    loss_weights: Vec<f32>,
    /// Metrics tracker
    metrics: MTPMetrics,
    /// Training mode
    training: bool,
}

impl MTPModelV2 {
    /// Create new MTP model from config
    pub fn new(config: MTPConfig, vb: VarBuilder) -> Result<Self> {
        let n_head = config.num_heads;
        let d_model = config.d_model;
        let vocab_size = config.vocab_size;
        
        let embed = candle_nn::embedding(vocab_size, d_model, vb.pp("embed"))?;
        
        // Base transformer blocks
        let mut base_blocks = Vec::new();
        for i in 0..config.num_layers {
            base_blocks.push(TransformerBlock::new(d_model, n_head, vb.pp(format!("block_{}", i)))?);
        }
        
        // Final layer norm and LM head
        let ln_f = candle_nn::layer_norm(d_model, 1e-5, vb.pp("ln_f"))?;
        let lm_head = candle_nn::linear(d_model, vocab_size, vb.pp("lm_head"))?;
        
        // MTP modules
        let mut mtp_modules = Vec::new();
        let mtp_heads = if config.num_heads >= 4 { config.num_heads / 4 } else { 1 };
        for i in 0..config.prediction_depth {
            mtp_modules.push(MTPModule::new(d_model, mtp_heads, vocab_size, vb.pp(format!("mtp_{}", i)))?);
        }
        
        let loss_weights = config.get_loss_weights();
        let metrics = MTPMetrics::new(config.prediction_depth);
        
        Ok(Self {
            config,
            embed,
            base_blocks,
            ln_f,
            lm_head,
            mtp_modules,
            loss_weights,
            metrics,
            training: true,
        })
    }
    
    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    
    /// Get loss weights
    pub fn get_loss_weights(&self) -> &[f32] {
        &self.loss_weights
    }
    
    /// Get config
    pub fn config(&self) -> &MTPConfig {
        &self.config
    }
    
    /// Get metrics
    pub fn metrics(&self) -> &MTPMetrics {
        &self.metrics
    }
    
    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
    
    /// Forward pass with optional MTP dropout
    pub fn forward(&self, input_ids: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        // Embedding
        let mut x = self.embed.forward(input_ids)?;
        
        // Base transformer
        for block in &self.base_blocks {
            x = block.forward(&x)?;
        }
        
        // Main prediction (t+1)
        let hidden = self.ln_f.forward(&x)?;
        let main_logits = self.lm_head.forward(&hidden)?;
        
        // MTP Forward (Sequential)
        let mut mtp_logits = Vec::new();
        let mut current_hidden = x;
        
        for module in &self.mtp_modules {
            let (logits, new_hidden) = module.forward(&current_hidden)?;
            mtp_logits.push(logits);
            current_hidden = new_hidden;
        }
        
        Ok((main_logits, mtp_logits))
    }
    
    /// Forward pass for speculative decoding inference
    /// Returns main prediction and speculative candidates
    pub fn forward_speculative(&self, input_ids: &Tensor) -> Result<(Tensor, Vec<(Tensor, f32)>)> {
        let (main_logits, mtp_logits) = self.forward(input_ids)?;
        
        // For each MTP head, compute top-1 prediction and its confidence
        let mut speculative_candidates = Vec::new();
        
        for logits in mtp_logits {
            // Get probabilities
            let probs = ops::softmax(&logits, 2)?; // (B, S, V)
            
            // Get max probability (confidence)
            let max_probs = probs.max(2)?; // This would need to return (values, indices)
            
            // Store prediction with confidence
            speculative_candidates.push((logits, 0.0)); // Simplified - would need proper confidence extraction
        }
        
        Ok((main_logits, speculative_candidates))
    }
    
    /// Compute MTP training loss with weighted cross-entropy
    /// 
    /// This computes the combined loss across all MTP heads:
    /// L = L_main + sum_{d=1}^{D} w_d * L_d
    /// 
    /// Where:
    /// - L_main is the cross-entropy loss for next token prediction (t+1)
    /// - L_d is the cross-entropy loss for MTP head d (predicting t+1+d)
    /// - w_d is the depth-dependent weight (typically decaying with depth)
    pub fn compute_mtp_loss(
        &self, 
        input_ids: &Tensor, 
        targets: &Tensor,
        mtp_targets: Option<&[Tensor]>,
    ) -> Result<(Tensor, MTPLossDetails)> {
        let (main_logits, mtp_logits) = self.forward(input_ids)?;
        
        // Compute main loss (next token prediction)
        let main_loss = cross_entropy_loss(&main_logits, targets)?;
        
        // Get loss weights from config
        let weights = self.config.get_loss_weights();
        
        // Compute MTP losses
        let mut mtp_losses = Vec::new();
        let mut total_mtp_loss = Tensor::zeros((), DType::F32, main_loss.device())?;
        
        if let Some(targets_list) = mtp_targets {
            for (i, (logits, target)) in mtp_logits.iter().zip(targets_list.iter()).enumerate() {
                let loss = cross_entropy_loss(logits, target)?;
                let weighted_loss = (&loss * weights[i] as f64)?;
                total_mtp_loss = (&total_mtp_loss + &weighted_loss)?;
                mtp_losses.push(loss.to_vec0::<f32>()?);
            }
        } else {
            // If no explicit MTP targets, shift main targets for each depth
            // For depth d, target is tokens at position t+d+1
            for (i, logits) in mtp_logits.iter().enumerate() {
                // Simplified: use main targets shifted (in practice need proper target construction)
                let loss = cross_entropy_loss(logits, targets)?;
                let weighted_loss = (&loss * weights[i] as f64)?;
                total_mtp_loss = (&total_mtp_loss + &weighted_loss)?;
                mtp_losses.push(loss.to_vec0::<f32>()?);
            }
        }
        
        // Combine losses
        let combined_loss = (&main_loss + &total_mtp_loss)?;
        
        // Create loss details
        let details = MTPLossDetails {
            main_loss: main_loss.to_vec0::<f32>()?,
            mtp_losses,
            combined_loss: combined_loss.to_vec0::<f32>()?,
            weights: weights.clone(),
        };
        
        Ok((combined_loss, details))
    }
}

/// Detailed MTP loss breakdown for logging/analysis
#[derive(Clone, Debug)]
pub struct MTPLossDetails {
    /// Main (t+1) prediction loss
    pub main_loss: f32,
    /// Loss for each MTP head (t+2, t+3, ...)
    pub mtp_losses: Vec<f32>,
    /// Combined weighted loss
    pub combined_loss: f32,
    /// Weights used for each MTP head
    pub weights: Vec<f32>,
}

impl MTPLossDetails {
    /// Get average MTP loss
    pub fn avg_mtp_loss(&self) -> f32 {
        if self.mtp_losses.is_empty() {
            0.0
        } else {
            self.mtp_losses.iter().sum::<f32>() / self.mtp_losses.len() as f32
        }
    }
    
    /// Get loss summary string
    pub fn summary(&self) -> String {
        format!(
            "main={:.4}, mtp_avg={:.4}, combined={:.4}",
            self.main_loss,
            self.avg_mtp_loss(),
            self.combined_loss
        )
    }
}

/// Cross-entropy loss for language modeling
fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // logits: (B, S, V)
    // targets: (B, S)
    let dims = logits.dims();
    let batch_size = dims[0];
    let seq_len = dims[1];
    let vocab_size = dims[2];
    
    // Reshape logits to (B*S, V) for efficient computation
    let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
    let targets_flat = targets.flatten_all()?.to_dtype(DType::U32)?;
    
    // Compute softmax
    let log_probs = ops::log_softmax(&logits_flat, 1)?;
    
    // Gather target probabilities
    // Simplified: compute mean of gathered log probs
    let target_indices = targets_flat.unsqueeze(1)?;
    let gathered = log_probs.gather(&target_indices, 1)?;
    
    // Negative mean log probability = cross entropy
    let loss = gathered.mean_all()?.neg()?;
    
    Ok(loss)
}

#[cfg(test)]
mod mtp_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    
    #[test]
    fn test_mtp_config() {
        let config = MTPConfig::default();
        assert_eq!(config.prediction_depth, 3);
        
        let weights = config.get_loss_weights();
        assert_eq!(weights.len(), 3);
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert!((weights[1] - 0.8).abs() < 1e-6);
        assert!((weights[2] - 0.64).abs() < 1e-6);
    }
    
    #[test]
    fn test_mtp_model_v2() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let config = MTPConfig::small();
        let model = MTPModelV2::new(config.clone(), vb)?;
        
        let batch_size = 2;
        let seq_len = 8;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        
        let (main_logits, mtp_logits) = model.forward(&input_ids)?;
        
        assert_eq!(main_logits.dims(), &[batch_size, seq_len, config.vocab_size]);
        assert_eq!(mtp_logits.len(), config.prediction_depth);
        
        for logits in &mtp_logits {
            assert_eq!(logits.dims(), &[batch_size, seq_len, config.vocab_size]);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_mtp_loss_computation() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let config = MTPConfig::small();
        let model = MTPModelV2::new(config.clone(), vb)?;
        
        let batch_size = 2;
        let seq_len = 8;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        let targets = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        
        let (loss, details) = model.compute_mtp_loss(&input_ids, &targets, None)?;
        
        // Loss should be non-negative
        let loss_val: f32 = loss.to_vec0()?;
        assert!(loss_val >= 0.0, "Loss should be non-negative");
        
        // Should have details for each MTP head
        assert_eq!(details.mtp_losses.len(), config.prediction_depth);
        assert_eq!(details.weights.len(), config.prediction_depth);
        
        // Combined loss should be non-negative
        assert!(details.combined_loss >= 0.0);
        
        Ok(())
    }
    
    #[test]
    fn test_mtp_metrics() {
        let mut metrics = MTPMetrics::new(3);
        
        metrics.update(&[0.5, 0.6, 0.7], &[8, 6, 4], 10);
        
        assert!(metrics.depth_loss[0] > 0.0);
        assert!(metrics.depth_accuracy[0] > 0.0);
        assert_eq!(metrics.total_samples, 10);
    }
    
    #[test]
    fn test_mtp_loss_details() {
        let details = MTPLossDetails {
            main_loss: 2.5,
            mtp_losses: vec![2.8, 3.0, 3.2],
            combined_loss: 5.0,
            weights: vec![1.0, 0.8, 0.64],
        };
        
        let avg = details.avg_mtp_loss();
        assert!((avg - 3.0).abs() < 1e-6);
        
        let summary = details.summary();
        assert!(summary.contains("main="));
        assert!(summary.contains("mtp_avg="));
    }
}
