use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, ops};

pub struct MultiQueryAttention {
    d_model: usize,
    num_heads: usize,
    d_head: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
}

impl MultiQueryAttention {
    pub fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
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
        // (B, num_heads, seq_len, d_head)
        let k = k.broadcast_as((batch_size, self.num_heads, seq_len, self.d_head))?.contiguous()?;
        let v = v.broadcast_as((batch_size, self.num_heads, seq_len, self.d_head))?.contiguous()?;

        // Scaled Dot-Product Attention
        // (B, num_heads, seq_len, d_head) @ (B, num_heads, d_head, seq_len) -> (B, num_heads, seq_len, seq_len)
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Apply causal mask (simplified for demonstration)
        // In a real implementation, we'd pass a mask or generate one.
        // For now, let's skip the mask to keep it simple as per the Python example logic structure
        // (The python example does have a mask, let's see if we can easily add one)
        
        // Simple causal mask
        // Simple causal mask
        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 1 } else { 0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), x.device())?;
        let mask = mask.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        
        // We need to fill with -inf where mask is 0. 
        // Candle doesn't have masked_fill in the same way, but we can use `where_cond`
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(attn_scores.shape())?;
        let attn_scores = mask.where_cond(&attn_scores, &neg_inf)?;

        let attn_weights = ops::softmax(&attn_scores, 3)?;
        
        // (B, num_heads, seq_len, seq_len) @ (B, num_heads, seq_len, d_head) -> (B, num_heads, seq_len, d_head)
        let context = attn_weights.matmul(&v)?;

        // (B, num_heads, seq_len, d_head) -> (B, seq_len, num_heads, d_head) -> (B, seq_len, d_model)
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
}

impl GroupedQueryAttention {
    pub fn new(d_model: usize, num_heads: usize, num_groups: usize, vb: VarBuilder) -> Result<Self> {
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
        // We need to repeat each group `heads_per_group` times.
        // k: (B, num_groups, seq_len, d_head)
        // target: (B, num_heads, seq_len, d_head)
        let heads_per_group = self.num_heads / self.num_groups;
        
        // (B, num_groups, 1, seq_len, d_head)
        let k = k.unsqueeze(2)?;
        let v = v.unsqueeze(2)?;
        
        // (B, num_groups, heads_per_group, seq_len, d_head)
        let k = k.broadcast_as((batch_size, self.num_groups, heads_per_group, seq_len, self.d_head))?;
        let v = v.broadcast_as((batch_size, self.num_groups, heads_per_group, seq_len, self.d_head))?;
        
        // Flatten num_groups and heads_per_group
        // (B, num_heads, seq_len, d_head)
        let k = k.flatten(1, 2)?;
        let v = v.flatten(1, 2)?;

        // Scaled Dot-Product Attention
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Simple causal mask
        // Simple causal mask
        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 1 } else { 0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), x.device())?;
        let mask = mask.broadcast_as((batch_size, self.num_heads, seq_len, seq_len))?;
        
        let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(attn_scores.shape())?;
        let attn_scores = mask.where_cond(&attn_scores, &neg_inf)?;

        let attn_weights = ops::softmax(&attn_scores, 3)?;
        
        let context = attn_weights.matmul(&v)?;

        let context = context.transpose(1, 2)?
            .reshape((batch_size, seq_len, self.d_model))?;

        self.w_o.forward(&context)
    }
}

