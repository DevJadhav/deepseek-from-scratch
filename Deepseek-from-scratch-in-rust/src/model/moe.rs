use candle_core::{Result, Tensor, DType, Module, Device};
use candle_nn::{Linear, VarBuilder, ops};
#[allow(unused_imports)]
use crate::distributed::expert::{ExpertParallelDispatch, ExpertParallelConfig, DispatchInfo};

// ============================================================================
// DeepSeek-V3 MoE Configuration
// ============================================================================

/// Configuration for DeepSeek-V3 style MoE with 256 experts
#[derive(Clone, Debug)]
pub struct DeepSeekMoEV3Config {
    /// Model dimension
    pub d_model: usize,
    /// Total number of routed experts
    pub n_routed_experts: usize,
    /// Number of shared experts
    pub n_shared_experts: usize,
    /// Number of experts activated per token
    pub top_k: usize,
    /// Hidden dimension for routed experts
    pub routed_expert_hidden: usize,
    /// Hidden dimension for shared experts
    pub shared_expert_hidden: usize,
    /// Number of expert groups for hierarchical routing
    pub n_expert_groups: usize,
    /// Top-k groups to select in first routing stage
    pub top_k_groups: usize,
    /// Expert capacity factor (1.0 = exact, >1.0 = slack)
    pub capacity_factor: f32,
    /// Enable auxiliary-loss-free load balancing
    pub aux_loss_free: bool,
    /// Bias learning rate for load balancing
    pub bias_lr: f64,
    /// EMA decay for load balancing
    pub ema_decay: f32,
    /// Minimum tokens per expert (for capacity)
    pub min_tokens_per_expert: usize,
    /// Enable expert dropout during training
    pub expert_dropout: f32,
}

impl Default for DeepSeekMoEV3Config {
    fn default() -> Self {
        Self {
            d_model: 4096,
            n_routed_experts: 256,
            n_shared_experts: 2,
            top_k: 8,
            routed_expert_hidden: 1024,
            shared_expert_hidden: 4096,
            n_expert_groups: 8,      // 256 / 8 = 32 experts per group
            top_k_groups: 4,         // Select 4 groups, then 2 experts per group
            capacity_factor: 1.25,
            aux_loss_free: true,
            bias_lr: 0.001,
            ema_decay: 0.99,
            min_tokens_per_expert: 1,
            expert_dropout: 0.0,
        }
    }
}

impl DeepSeekMoEV3Config {
    /// Create config for DeepSeek-V3.2 (256 experts, 8 active)
    pub fn v3_256_8() -> Self {
        Self::default()
    }
    
    /// Create smaller config for testing (16 experts, 2 active)  
    pub fn small_16_2() -> Self {
        Self {
            n_routed_experts: 16,
            n_shared_experts: 2,
            top_k: 2,
            n_expert_groups: 4,
            top_k_groups: 2,
            ..Default::default()
        }
    }
    
    /// Experts per group
    pub fn experts_per_group(&self) -> usize {
        self.n_routed_experts / self.n_expert_groups
    }
    
    /// Experts to select per group
    pub fn experts_per_selected_group(&self) -> usize {
        self.top_k / self.top_k_groups
    }
}

// ============================================================================
// Load Balancing State (Auxiliary-Loss-Free)
// ============================================================================

/// State for auxiliary-loss-free load balancing
/// 
/// Uses bias-based adjustment with EMA updates per DeepSeek-V3 paper
pub struct LoadBalancingState {
    /// Per-expert bias terms for routing adjustment
    bias: Tensor,
    /// EMA of expert selection counts
    ema_counts: Vec<f32>,
    /// Configuration
    config: DeepSeekMoEV3Config,
    /// Current step for tracking
    step: usize,
}

impl LoadBalancingState {
    pub fn new(config: &DeepSeekMoEV3Config, device: &Device) -> Result<Self> {
        let bias = Tensor::zeros((config.n_routed_experts,), DType::F32, device)?;
        let ema_counts = vec![1.0 / config.n_routed_experts as f32; config.n_routed_experts];
        
        Ok(Self {
            bias,
            ema_counts,
            config: config.clone(),
            step: 0,
        })
    }
    
    /// Get current bias tensor
    pub fn get_bias(&self) -> &Tensor {
        &self.bias
    }
    
    /// Update bias based on observed expert selections
    /// 
    /// This is the auxiliary-loss-free load balancing from DeepSeek-V3:
    /// Instead of adding an auxiliary loss, we adjust routing bias terms
    /// to encourage underutilized experts and discourage overutilized ones.
    pub fn update(&mut self, expert_counts: &[f32], device: &Device) -> Result<()> {
        let n_experts = self.config.n_routed_experts;
        let decay = self.config.ema_decay;
        
        // Update EMA counts
        for i in 0..n_experts {
            self.ema_counts[i] = decay * self.ema_counts[i] + (1.0 - decay) * expert_counts[i];
        }
        
        // Compute target (uniform distribution)
        let total_count: f32 = self.ema_counts.iter().sum();
        let target = total_count / n_experts as f32;
        
        // Update bias: bias_i += lr * tanh((target - count_i) / (target + eps))
        let mut bias_vec = self.bias.to_vec1::<f32>()?;
        let lr = self.config.bias_lr as f32;
        
        for i in 0..n_experts {
            let count = self.ema_counts[i];
            let violation = (target - count) / (target + 1e-6);
            let adjustment = lr * violation.tanh();
            bias_vec[i] += adjustment;
            
            // Clamp to prevent extreme biases
            bias_vec[i] = bias_vec[i].clamp(-2.0, 2.0);
        }
        
        self.bias = Tensor::from_vec(bias_vec, (n_experts,), device)?;
        self.step += 1;
        
        Ok(())
    }
    
    /// Get load balancing statistics
    pub fn get_stats(&self) -> (f32, f32, f32) {
        let counts = &self.ema_counts;
        let mean = counts.iter().sum::<f32>() / counts.len() as f32;
        let max = counts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = counts.iter().cloned().fold(f32::INFINITY, f32::min);
        
        // Imbalance ratio: max/min
        let imbalance = if min > 0.0 { max / min } else { f32::INFINITY };
        
        (mean, imbalance, self.step as f32)
    }
}

// ============================================================================
// Expert Capacity Metrics
// ============================================================================

/// Metrics for tracking expert capacity and token dropping
#[derive(Clone, Debug, Default)]
pub struct CapacityMetrics {
    /// Total tokens processed
    pub total_tokens: usize,
    /// Tokens dropped due to capacity overflow
    pub dropped_tokens: usize,
    /// Per-expert overflow counts
    pub expert_overflow: Vec<usize>,
    /// Per-expert utilization (tokens / capacity)
    pub expert_utilization: Vec<f32>,
}

impl CapacityMetrics {
    pub fn new(n_experts: usize) -> Self {
        Self {
            total_tokens: 0,
            dropped_tokens: 0,
            expert_overflow: vec![0; n_experts],
            expert_utilization: vec![0.0; n_experts],
        }
    }
    
    /// Record token dispatch results
    pub fn record_dispatch(
        &mut self,
        expert_id: usize,
        tokens_routed: usize,
        capacity: usize,
    ) {
        self.total_tokens += tokens_routed.min(capacity);
        
        if tokens_routed > capacity {
            let overflow = tokens_routed - capacity;
            self.dropped_tokens += overflow;
            self.expert_overflow[expert_id] += overflow;
        }
        
        self.expert_utilization[expert_id] = tokens_routed as f32 / capacity.max(1) as f32;
    }
    
    /// Get drop rate (fraction of tokens dropped)
    pub fn drop_rate(&self) -> f32 {
        if self.total_tokens + self.dropped_tokens == 0 {
            0.0
        } else {
            self.dropped_tokens as f32 / (self.total_tokens + self.dropped_tokens) as f32
        }
    }
    
    /// Get average utilization across experts
    pub fn avg_utilization(&self) -> f32 {
        if self.expert_utilization.is_empty() {
            0.0
        } else {
            self.expert_utilization.iter().sum::<f32>() / self.expert_utilization.len() as f32
        }
    }
    
    /// Get most overloaded expert
    pub fn most_overloaded_expert(&self) -> (usize, usize) {
        self.expert_overflow
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, &count)| (idx, count))
            .unwrap_or((0, 0))
    }
    
    /// Reset metrics
    pub fn reset(&mut self) {
        self.total_tokens = 0;
        self.dropped_tokens = 0;
        self.expert_overflow.fill(0);
        self.expert_utilization.fill(0.0);
    }
}

// --- Helper Functions ---

fn gelu(x: &Tensor) -> Result<Tensor> {
    // Approx GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c1 = (2.0f64 / std::f64::consts::PI).sqrt();
    let c2 = 0.044715;
    let x3 = x.powf(3.0)?;
    let inner = ((x + (x3 * c2)?)? * c1)?;
    let tanh = inner.tanh()?;
    let res = ((x * 0.5)? * (tanh + 1.0)?)?;
    Ok(res)
}

// --- Expert Module ---

pub struct Expert {
    fc1: Linear,
    fc2: Linear,
}

impl Expert {
    pub fn new(d_model: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear_no_bias(d_model, hidden, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear_no_bias(hidden, d_model, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = gelu(&x)?;
        self.fc2.forward(&x)
    }
}

// --- DeepSeek MoE ---

pub struct DeepSeekMoE {
    d_model: usize,
    n_routed: usize,
    n_shared: usize,
    top_k: usize,
    routed_experts: Vec<Expert>,
    shared_experts: Vec<Expert>,
    centroids: Tensor, // Parameter
    bias: Tensor,      // Buffer (we'll treat as tensor for now, manual update)
    bias_lr: f64,
}

impl DeepSeekMoE {
    pub fn new(
        d_model: usize,
        n_routed: usize,
        n_shared: usize,
        top_k: usize,
        routed_hidden: usize,
        shared_hidden: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut routed_experts = Vec::new();
        for i in 0..n_routed {
            routed_experts.push(Expert::new(d_model, routed_hidden, vb.pp(&format!("routed.{}", i)))?);
        }

        let mut shared_experts = Vec::new();
        for i in 0..n_shared {
            shared_experts.push(Expert::new(d_model, shared_hidden, vb.pp(&format!("shared.{}", i)))?);
        }

        // Centroids: (n_routed, d_model)
        let centroids = vb.get((n_routed, d_model), "centroids")?;
        
        // Bias: (n_routed) - initialized to zeros
        let bias = Tensor::zeros((n_routed,), DType::F32, vb.device())?;

        Ok(Self {
            d_model,
            n_routed,
            n_shared,
            top_k,
            routed_experts,
            shared_experts,
            centroids,
            bias,
            bias_lr: 0.01,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?; // (N, D)
        let _n_tokens = b * s;

        // 1. Shared Path
        let mut shared_out = Tensor::zeros_like(&x_flat)?;
        for exp in &self.shared_experts {
            shared_out = (shared_out + exp.forward(&x_flat)?)?;
        }

        // 2. Router
        // logits: (N, n_routed) = x_flat @ centroids.T + bias
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;

        let logits = logits.contiguous()?;
        // Top-K
        // topk_vals: (N, k), topk_idx: (N, k)
        // Workaround for missing top_k: arg_sort and gather
        let topk_idx = logits.arg_sort_last_dim(true)?.narrow(1, 0, self.top_k)?.contiguous()?;
        let topk_vals = logits.gather(&topk_idx, 1)?;

        // Softmax over top-k
        let gate = ops::softmax(&topk_vals, 1)?; // (N, k)

        // 3. Dispatch
        let mut routed_out = Tensor::zeros_like(&x_flat)?;
        
        // We iterate over experts and select tokens routed to them
        // Note: This is a naive implementation. Optimized kernels (like in vLLM/DeepSpeed) do this better.
        for i in 0..self.n_routed {
            // mask: (N, k) boolean where index == i
            let _mask = topk_idx.eq(i as u32)?;
            
            // We need to find which tokens (row indices) selected this expert
            // and which of the k slots (col indices) it was.
            // Candle doesn't have `nonzero` easily accessible for general tensors in the same way as PyTorch for this specific masking pattern efficiently without some work.
            // Alternative approach: iterate over tokens? Too slow.
            // Let's use a simpler approach for demonstration:
            // For each expert, gather inputs, process, scatter add.
            
            // Actually, let's try to replicate the PyTorch logic using available ops.
            // mask is (N, k).
            // We want indices where mask is 1.
            
            // Since Candle's indexing is a bit limited, we might iterate over the batch if it's small, 
            // but for a proper layer we should try to be vectorized.
            
            // Let's stick to a slightly less efficient but correct method for now:
            // For each expert `i`:
            // 1. Find indices in `topk_idx` that equal `i`.
            // 2. If none, continue.
            // 3. Gather `x_flat` for those rows.
            // 4. Run expert.
            // 5. Scale by `gate`.
            // 6. Add to `routed_out`.
            
            // To do this efficiently in Candle without `nonzero()` returning a list of indices we can use for `index_select` easily:
            // We can convert mask to float, sum it to check if empty.
            
            // Optimization: Since we are running on CPU/Metal for demo, we can pull to Vec, process, push back? 
            // No, that defeats the purpose of using tensors.
            
            // Let's assume for this demo we can iterate over tokens if N is small, OR
            // we implement the "sparse matrix multiplication" view if possible.
            
            // Let's use the mask approach combined with `where_cond`.
            // This is computationally expensive (runs all experts for all tokens effectively if we mask outputs), 
            // but functionally correct for a demo if we can't do sparse gather/scatter easily.
            // WAIT, the PyTorch code uses `index_select` and `index_add_`.
            // Candle has `index_select` and `index_add`.
            
            // We need the indices.
            // Let's flatten the mask and find indices.
            // Since `nonzero` is not readily available to return a Tensor of indices in Candle (it returns Vec<Vec<usize>> usually or similar depending on version, or might not be exposed),
            // we will simulate it.
            
            // Actually, for the sake of this implementation in Rust/Candle which might lack some sparse ops:
            // We will iterate through the batch (N) in a loop if N is small.
            // But N = B * S = 4 * 64 = 256. That's 256 iterations. A bit much but maybe okay for "demo".
            
            // Better approach:
            // Use `topk_idx` to create a one-hot-like mask for each expert?
            // (N, k) -> (N, k, n_routed) -> sum over k -> (N, n_routed)
            // This tells us weight for each expert per token.
            // But we only want to run the expert on tokens that have non-zero weight.
            
            // Let's try to implement the "Masked" approach:
            // For expert i:
            // 1. Construct a mask (N) = sum(topk_idx == i, dim=1) > 0.
            // 2. If we can't easily gather, we can multiply input by mask? No, expert is non-linear.
            
            // Okay, we will use a simplified routing for the Rust implementation that might not be fully sparse-optimized
            // but demonstrates the logic. We will run ALL experts on ALL tokens, but mask the INPUTS?
            // No, that's wrong (GELU(0) != 0).
            
            // We MUST run only on selected tokens.
            // Let's use a helper to get indices.
            // Since we are on CPU/Metal, we can convert topk_idx to Vec, find indices, and use `index_select`.
            
            let topk_idx_vec = topk_idx.flatten_all()?.to_vec1::<u32>()?;
            let mut indices = Vec::new();
            let mut gate_indices = Vec::new(); // (row, col) in topk
            
            for (flat_idx, &exp_id) in topk_idx_vec.iter().enumerate() {
                if exp_id as usize == i {
                    let row = flat_idx / self.top_k;
                    let col = flat_idx % self.top_k;
                    indices.push(row as u32);
                    gate_indices.push((row, col));
                }
            }
            
            if indices.is_empty() {
                continue;
            }
            
            let indices_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), x.device())?;
            let exp_in = x_flat.index_select(&indices_tensor, 0)?;
            
            let out = self.routed_experts[i].forward(&exp_in)?;
            
            // Get gates
            // We need to gather specific values from `gate` (N, k).
            // gate_indices has (row, col).
            // We can flatten `gate` and select.
            let gate_flat = gate.flatten_all()?;
            let gate_select_indices: Vec<u32> = gate_indices.iter().map(|(r, c)| (r * self.top_k + c) as u32).collect();
            let gate_select_indices_len = gate_select_indices.len();
            let gate_select_indices_tensor = Tensor::from_vec(gate_select_indices, (gate_select_indices_len,), x.device())?;
            let w = gate_flat.index_select(&gate_select_indices_tensor, 0)?.reshape((indices.len(), 1))?;
            
            let weighted_out = out.broadcast_mul(&w)?;
            
            routed_out = routed_out.index_add(&indices_tensor, &weighted_out, 0)?;
        }

        let routed_out = routed_out.reshape((b, s, d))?;
        let shared_out = shared_out.reshape((b, s, d))?;
        
        Ok((x + shared_out + routed_out)?)
    }
    
    // Manual bias update function (simplified)
    pub fn update_bias(&mut self, x: &Tensor) -> Result<()> {
        // x: (B, S, D)
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        
        // logits = x @ centroids.T + bias
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;
        let logits = logits.contiguous()?;
        
        // topk
        let topk_idx = logits.arg_sort_last_dim(true)?.narrow(1, 0, self.top_k)?.contiguous()?; // (N, k)
        
        // Count selections
        let topk_idx_vec = topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let mut counts = vec![0f32; self.n_routed];
        for &idx in &topk_idx_vec {
            if (idx as usize) < self.n_routed {
                counts[idx as usize] += 1.0;
            }
        }
        
        let avg = counts.iter().sum::<f32>() / (self.n_routed as f32).max(1.0);
        
        // Violation = (avg - count) / (avg + 1e-6)
        // Update bias += lr * tanh(violation)
        
        let mut bias_vec = self.bias.to_vec1::<f32>()?;
        for i in 0..self.n_routed {
            let count = counts[i];
            let violation = (avg - count) / (avg + 1e-6);
            bias_vec[i] += (self.bias_lr as f32) * violation.tanh();
        }
        
        self.bias = Tensor::from_vec(bias_vec, (self.n_routed,), x.device())?;
        
        Ok(())
    }
}

// --- Standard MoE ---

pub struct StandardMoE {
    n_routed: usize,
    top_k: usize,
    experts: Vec<Expert>,
    router: Linear,
}

impl StandardMoE {
    pub fn new(
        d_model: usize,
        n_routed: usize,
        top_k: usize,
        hidden_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut experts = Vec::new();
        for i in 0..n_routed {
            experts.push(Expert::new(d_model, hidden_dim, vb.pp(&format!("experts.{}", i)))?);
        }
        
        let router = candle_nn::linear_no_bias(d_model, n_routed, vb.pp("router"))?;
        
        Ok(Self {
            n_routed,
            top_k,
            experts,
            router,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, f32)> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        let n_tokens = b * s;

        let logits = self.router.forward(&x_flat)?;
        let probs = ops::softmax(&logits, 1)?.contiguous()?;
        
        let topk_idx = probs.arg_sort_last_dim(true)?.narrow(1, 0, self.top_k)?.contiguous()?;
        let topk_vals = probs.gather(&topk_idx, 1)?;
        
        // Normalize gates
        let topk_sum = topk_vals.sum_keepdim(1)?;
        let gates = topk_vals.broadcast_div(&topk_sum)?; // (N, k)
        
        // Aux Loss Calculation
        // f_i = fraction of tokens dispatched to expert i
        // p_i = fraction of router probability allocated to expert i
        
        // p_i: mean of probs over batch
        let p_i = probs.mean(0)?; // (n_routed)
        
        // f_i: count selections / N
        let topk_idx_vec = topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let mut counts = vec![0f32; self.n_routed];
        for &idx in &topk_idx_vec {
            if (idx as usize) < self.n_routed {
                counts[idx as usize] += 1.0;
            }
        }
        let f_i_vec: Vec<f32> = counts.iter().map(|&c| c / (n_tokens as f32)).collect();
        let f_i = Tensor::from_vec(f_i_vec, (self.n_routed,), x.device())?;
        
        let aux_loss = (p_i.mul(&f_i)?.sum_all()?.to_scalar::<f32>()?) * (self.n_routed as f32) * 0.01;

        // Dispatch (Same logic as DeepSeekMoE)
        let mut final_out = Tensor::zeros_like(&x_flat)?;
        
        for i in 0..self.n_routed {
            let mut indices = Vec::new();
            let mut gate_indices = Vec::new();
            
            for (flat_idx, &exp_id) in topk_idx_vec.iter().enumerate() {
                if exp_id as usize == i {
                    let row = flat_idx / self.top_k;
                    let col = flat_idx % self.top_k;
                    indices.push(row as u32);
                    gate_indices.push((row, col));
                }
            }
            
            if indices.is_empty() {
                continue;
            }
            
            let indices_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), x.device())?;
            let exp_in = x_flat.index_select(&indices_tensor, 0)?;
            
            let out = self.experts[i].forward(&exp_in)?;
            
            let gate_flat = gates.flatten_all()?;
            let gate_select_indices: Vec<u32> = gate_indices.iter().map(|(r, c)| (r * self.top_k + c) as u32).collect();
            let gate_select_indices_len = gate_select_indices.len();
            let gate_select_indices_tensor = Tensor::from_vec(gate_select_indices, (gate_select_indices_len,), x.device())?;
            let w = gate_flat.index_select(&gate_select_indices_tensor, 0)?.reshape((indices.len(), 1))?;
            
            let weighted_out = out.broadcast_mul(&w)?;
            
            final_out = final_out.index_add(&indices_tensor, &weighted_out, 0)?;
        }
        
        let final_out = final_out.reshape((b, s, d))?;
        
        Ok((final_out, aux_loss))
    }
}

// --- Expert Parallel DeepSeek MoE ---

/// DeepSeekMoE with Expert Parallelism support.
/// 
/// This variant distributes experts across EP ranks and uses
/// all-to-all communication to route tokens to the correct experts.
pub struct ExpertParallelMoE {
    d_model: usize,
    n_routed: usize,
    n_shared: usize,
    top_k: usize,
    /// Local routed experts (only experts assigned to this rank)
    local_experts: Vec<Expert>,
    /// Shared experts (replicated across all ranks)
    shared_experts: Vec<Expert>,
    centroids: Tensor,
    bias: Tensor,
    bias_lr: f64,
    /// EP dispatcher for token routing
    ep_dispatcher: ExpertParallelDispatch,
    /// EP configuration
    _ep_config: ExpertParallelConfig,
}

impl ExpertParallelMoE {
    /// Create a new EP-enabled MoE layer.
    ///
    /// Args:
    ///   d_model: Model dimension
    ///   n_routed: Total number of routed experts across all ranks
    ///   n_shared: Number of shared experts (replicated on each rank)
    ///   top_k: Number of experts to route each token to
    ///   routed_hidden: Hidden dim for routed experts
    ///   shared_hidden: Hidden dim for shared experts
    ///   ep_dispatcher: Expert parallel dispatcher (None for single-rank mode)
    ///   vb: Variable builder
    pub fn new(
        d_model: usize,
        n_routed: usize,
        n_shared: usize,
        top_k: usize,
        routed_hidden: usize,
        shared_hidden: usize,
        ep_dispatcher: Option<ExpertParallelDispatch>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ep_dispatcher = ep_dispatcher.unwrap_or_else(|| ExpertParallelDispatch::new(n_routed));
        let ep_config = ep_dispatcher.config().clone();
        let local_expert_ids = ep_config.local_expert_ids();
        
        // Only initialize local experts
        let mut local_experts = Vec::new();
        for &i in &local_expert_ids {
            local_experts.push(Expert::new(
                d_model,
                routed_hidden,
                vb.pp(&format!("routed.{}", i))
            )?);
        }

        // Shared experts are replicated
        let mut shared_experts = Vec::new();
        for i in 0..n_shared {
            shared_experts.push(Expert::new(
                d_model,
                shared_hidden,
                vb.pp(&format!("shared.{}", i))
            )?);
        }

        // Centroids: (n_routed, d_model) - full routing table
        let centroids = vb.get((n_routed, d_model), "centroids")?;
        
        // Bias: (n_routed)
        let bias = Tensor::zeros((n_routed,), DType::F32, vb.device())?;

        Ok(Self {
            d_model,
            n_routed,
            n_shared,
            top_k,
            local_experts,
            shared_experts,
            centroids,
            bias,
            bias_lr: 0.01,
            ep_dispatcher,
            _ep_config: ep_config,
        })
    }
    
    /// Forward pass with expert parallelism.
    ///
    /// 1. Compute routing for all tokens
    /// 2. Dispatch tokens to experts using all-to-all
    /// 3. Process tokens with local experts
    /// 4. Combine results using all-to-all
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        let _n_tokens = b * s;

        // 1. Shared Path (replicated computation)
        let mut shared_out = Tensor::zeros_like(&x_flat)?;
        for exp in &self.shared_experts {
            shared_out = (shared_out + exp.forward(&x_flat)?)?;
        }

        // 2. Router - compute routing decisions
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;
        let logits = logits.contiguous()?;
        
        // Top-K routing
        let topk_idx = logits.arg_sort_last_dim(true)?
            .narrow(1, 0, self.top_k)?
            .contiguous()?;
        let topk_vals = logits.gather(&topk_idx, 1)?;
        let gate = ops::softmax(&topk_vals, 1)?;

        // 3. Flatten routing for dispatch
        // For EP, we need to send each token to its target expert's rank
        // We'll process each top-k slot separately
        
        let ep_config = self.ep_dispatcher.config();
        let num_local_experts = ep_config.num_local_experts;
        
        let mut routed_out = Tensor::zeros_like(&x_flat)?;
        
        // For each top-k slot
        for k in 0..self.top_k {
            // Get expert indices for this slot
            let expert_indices = topk_idx.narrow(1, k, 1)?.squeeze(1)?;
            let gate_weights = gate.narrow(1, k, 1)?; // (N, 1)
            
            // Dispatch tokens to appropriate EP rank
            let (dispatched, dispatch_info) = self.ep_dispatcher.dispatch(&x_flat, &expert_indices)?;
            
            // Get expert indices for dispatched tokens (need to convert to local expert id)
            let dispatched_expert_indices = if ep_config.ep_size > 1 {
                // Remap global expert ID to local expert ID
                let dispatched_indices_vec = expert_indices.to_vec1::<u32>()?;
                let local_indices: Vec<u32> = dispatched_indices_vec
                    .iter()
                    .map(|&exp_id| (exp_id as usize % num_local_experts) as u32)
                    .collect();
                Tensor::from_vec(local_indices, expert_indices.shape(), x.device())?
            } else {
                expert_indices.clone()
            };
            
            // Process with local experts
            let local_out = self.process_local_experts(&dispatched, &dispatched_expert_indices)?;
            
            // Combine results back
            let combined = self.ep_dispatcher.combine(&local_out, &dispatch_info)?;
            
            // Weight by gate values
            let weighted = combined.broadcast_mul(&gate_weights)?;
            routed_out = (routed_out + weighted)?;
        }

        let routed_out = routed_out.reshape((b, s, d))?;
        let shared_out = shared_out.reshape((b, s, d))?;
        
        // Residual connection
        Ok((x + shared_out + routed_out)?)
    }
    
    /// Process tokens with local experts.
    fn process_local_experts(&self, x: &Tensor, local_expert_indices: &Tensor) -> Result<Tensor> {
        let _shape = x.dims2()?; // Validate 2D shape
        let indices_vec = local_expert_indices.to_vec1::<u32>()?;
        
        let mut output = Tensor::zeros_like(x)?;
        
        for (local_idx, expert) in self.local_experts.iter().enumerate() {
            // Find tokens routed to this local expert
            let mut token_indices = Vec::new();
            for (i, &exp_id) in indices_vec.iter().enumerate() {
                if exp_id as usize == local_idx {
                    token_indices.push(i as u32);
                }
            }
            
            if token_indices.is_empty() {
                continue;
            }
            
            let indices_tensor = Tensor::from_vec(
                token_indices.clone(),
                (token_indices.len(),),
                x.device()
            )?;
            let exp_in = x.index_select(&indices_tensor, 0)?;
            let exp_out = expert.forward(&exp_in)?;
            
            output = output.index_add(&indices_tensor, &exp_out, 0)?;
        }
        
        Ok(output)
    }
    
    /// Update bias for load balancing.
    pub fn update_bias(&mut self, x: &Tensor) -> Result<()> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        
        let logits = x_flat.matmul(&self.centroids.transpose(0, 1)?)?;
        let logits = logits.broadcast_add(&self.bias)?;
        let logits = logits.contiguous()?;
        
        let topk_idx = logits.arg_sort_last_dim(true)?
            .narrow(1, 0, self.top_k)?
            .contiguous()?;
        
        let topk_idx_vec = topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let mut counts = vec![0f32; self.n_routed];
        for &idx in &topk_idx_vec {
            if (idx as usize) < self.n_routed {
                counts[idx as usize] += 1.0;
            }
        }
        
        let avg = counts.iter().sum::<f32>() / (self.n_routed as f32).max(1.0);
        
        let mut bias_vec = self.bias.to_vec1::<f32>()?;
        for i in 0..self.n_routed {
            let count = counts[i];
            let violation = (avg - count) / (avg + 1e-6);
            bias_vec[i] += (self.bias_lr as f32) * violation.tanh();
        }
        
        self.bias = Tensor::from_vec(bias_vec, (self.n_routed,), x.device())?;
        
        Ok(())
    }
}

#[cfg(test)]
mod ep_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    
    #[test]
    fn test_expert_parallel_moe_single_rank() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let d_model = 32;
        let n_routed = 4;
        let n_shared = 1;
        let top_k = 2;
        let routed_hidden = 64;
        let shared_hidden = 64;
        
        let moe = ExpertParallelMoE::new(
            d_model,
            n_routed,
            n_shared,
            top_k,
            routed_hidden,
            shared_hidden,
            None, // Single-rank mode
            vb,
        )?;
        
        let batch_size = 2;
        let seq_len = 8;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, d_model), &device)?;
        
        let output = moe.forward(&x)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, d_model]);
        
        Ok(())
    }
}

// ============================================================================
// DeepSeek-V3 MoE (256 Experts, 8 Active, Hierarchical Routing)
// ============================================================================

/// DeepSeek-V3 style MoE layer with:
/// - 256 routed experts with 8 active per token
/// - Hierarchical routing (group selection → expert selection)
/// - Auxiliary-loss-free load balancing via bias adjustment
/// - Expert capacity for efficient batching
pub struct DeepSeekMoEV3 {
    config: DeepSeekMoEV3Config,
    d_model: usize,
    
    /// Routed experts
    routed_experts: Vec<Expert>,
    /// Shared experts (always active)
    shared_experts: Vec<Expert>,
    
    /// Group centroids for first-stage routing: (n_groups, d_model)
    group_centroids: Tensor,
    /// Expert centroids within groups: (n_routed, d_model)
    expert_centroids: Tensor,
    
    /// Load balancing state
    load_balance: LoadBalancingState,
    
    /// Capacity metrics for tracking dropped tokens
    capacity_metrics: CapacityMetrics,
    
    /// Training mode flag
    training: bool,
}

impl DeepSeekMoEV3 {
    /// Create a new DeepSeek-V3 style MoE layer
    pub fn new(config: DeepSeekMoEV3Config, vb: VarBuilder) -> Result<Self> {
        let d_model = config.d_model;
        let n_routed = config.n_routed_experts;
        let n_shared = config.n_shared_experts;
        let n_groups = config.n_expert_groups;
        
        // Create routed experts
        let mut routed_experts = Vec::with_capacity(n_routed);
        for i in 0..n_routed {
            routed_experts.push(Expert::new(
                d_model,
                config.routed_expert_hidden,
                vb.pp(&format!("routed.{}", i))
            )?);
        }
        
        // Create shared experts
        let mut shared_experts = Vec::with_capacity(n_shared);
        for i in 0..n_shared {
            shared_experts.push(Expert::new(
                d_model,
                config.shared_expert_hidden,
                vb.pp(&format!("shared.{}", i))
            )?);
        }
        
        // Group centroids for hierarchical routing
        let group_centroids = vb.get((n_groups, d_model), "group_centroids")?;
        
        // Expert centroids
        let expert_centroids = vb.get((n_routed, d_model), "expert_centroids")?;
        
        // Load balancing state
        let load_balance = LoadBalancingState::new(&config, vb.device())?;
        
        // Capacity metrics
        let capacity_metrics = CapacityMetrics::new(n_routed);
        
        Ok(Self {
            config,
            d_model,
            routed_experts,
            shared_experts,
            group_centroids,
            expert_centroids,
            load_balance,
            capacity_metrics,
            training: true,
        })
    }
    
    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    
    /// Forward pass with hierarchical routing
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, s, d) = x.dims3()?;
        let x_flat = x.reshape((b * s, d))?;
        let _n_tokens = b * s;  // May be used for metrics/debugging
        
        // 1. Shared expert path (always active)
        let mut shared_out = Tensor::zeros_like(&x_flat)?;
        for exp in &self.shared_experts {
            shared_out = (shared_out + exp.forward(&x_flat)?)?;
        }
        
        // 2. Hierarchical routing
        let (expert_indices, gates, expert_counts) = self.hierarchical_route(&x_flat)?;
        
        // 3. Dispatch and compute routed experts
        let routed_out = self.dispatch_and_compute(&x_flat, &expert_indices, &gates)?;
        
        // 4. Update load balancing (training only)
        if self.training && self.config.aux_loss_free {
            self.load_balance.update(&expert_counts, x.device())?;
        }
        
        // 5. Combine outputs
        let shared_out = shared_out.reshape((b, s, d))?;
        let routed_out = routed_out.reshape((b, s, d))?;
        
        Ok((x + shared_out + routed_out)?)
    }
    
    /// Hierarchical routing: Group selection → Expert selection within groups
    fn hierarchical_route(&self, x: &Tensor) -> Result<(Tensor, Tensor, Vec<f32>)> {
        let _n_tokens = x.dim(0)?;
        let top_k = self.config.top_k;
        let _n_groups = self.config.n_expert_groups;
        let top_k_groups = self.config.top_k_groups;
        let experts_per_group = self.config.experts_per_group();
        let _experts_per_selected_group = self.config.experts_per_selected_group();
        
        // Stage 1: Group selection
        // Compute group affinities: (N, n_groups)
        let group_logits = x.matmul(&self.group_centroids.transpose(0, 1)?)?;
        
        // Select top-k groups
        let group_topk_idx = group_logits
            .arg_sort_last_dim(true)?
            .narrow(1, 0, top_k_groups)?
            .contiguous()?; // (N, top_k_groups)
        
        // Stage 2: Expert selection within selected groups
        // Compute expert affinities: (N, n_routed)
        let mut expert_logits = x.matmul(&self.expert_centroids.transpose(0, 1)?)?;
        
        // Add load balancing bias
        expert_logits = expert_logits.broadcast_add(self.load_balance.get_bias())?;
        
        // Mask experts not in selected groups
        let masked_logits = self.mask_experts_by_group(
            &expert_logits,
            &group_topk_idx,
            experts_per_group
        )?;
        
        // Select top experts from unmasked
        let expert_topk_idx = masked_logits
            .arg_sort_last_dim(true)?
            .narrow(1, 0, top_k)?
            .contiguous()?; // (N, top_k)
        
        let expert_topk_vals = expert_logits.gather(&expert_topk_idx, 1)?;
        
        // Compute gates via softmax over selected experts
        let gates = ops::softmax(&expert_topk_vals, 1)?; // (N, top_k)
        
        // Count expert selections for load balancing
        let expert_counts = self.count_expert_selections(&expert_topk_idx)?;
        
        Ok((expert_topk_idx, gates, expert_counts))
    }
    
    /// Mask experts not in selected groups
    fn mask_experts_by_group(
        &self,
        expert_logits: &Tensor,
        group_topk_idx: &Tensor,
        experts_per_group: usize,
    ) -> Result<Tensor> {
        let n_tokens = expert_logits.dim(0)?;
        let n_routed = self.config.n_routed_experts;
        
        // Create mask: (N, n_routed)
        let group_idx_vec = group_topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let top_k_groups = self.config.top_k_groups;
        
        let mut mask_data = vec![f32::NEG_INFINITY; n_tokens * n_routed];
        
        for token_idx in 0..n_tokens {
            for k in 0..top_k_groups {
                let group_id = group_idx_vec[token_idx * top_k_groups + k] as usize;
                let expert_start = group_id * experts_per_group;
                let expert_end = expert_start + experts_per_group;
                
                for expert_id in expert_start..expert_end {
                    if expert_id < n_routed {
                        mask_data[token_idx * n_routed + expert_id] = 0.0;
                    }
                }
            }
        }
        
        let mask = Tensor::from_vec(mask_data, (n_tokens, n_routed), expert_logits.device())?;
        
        expert_logits + mask
    }
    
    /// Count expert selections for load balancing updates
    fn count_expert_selections(&self, expert_topk_idx: &Tensor) -> Result<Vec<f32>> {
        let idx_vec = expert_topk_idx.flatten_all()?.to_vec1::<u32>()?;
        let n_routed = self.config.n_routed_experts;
        
        let mut counts = vec![0.0f32; n_routed];
        for &idx in &idx_vec {
            if (idx as usize) < n_routed {
                counts[idx as usize] += 1.0;
            }
        }
        
        Ok(counts)
    }
    
    /// Dispatch tokens to experts and compute outputs
    fn dispatch_and_compute(
        &mut self,
        x: &Tensor,
        expert_indices: &Tensor,
        gates: &Tensor,
    ) -> Result<Tensor> {
        let n_tokens = x.dim(0)?;
        let d = self.d_model;
        let top_k = self.config.top_k;
        let n_routed = self.config.n_routed_experts;
        
        // Flatten indices and gates for processing
        let idx_vec = expert_indices.flatten_all()?.to_vec1::<u32>()?;
        let gate_flat = gates.flatten_all()?;
        
        let mut routed_out = Tensor::zeros((n_tokens, d), DType::F32, x.device())?;
        
        // Reset capacity metrics for this forward pass
        self.capacity_metrics.reset();
        
        // Process each expert
        for expert_id in 0..n_routed {
            // Find tokens routed to this expert
            let mut token_indices = Vec::new();
            let mut gate_indices = Vec::new();
            
            for (flat_idx, &exp_id) in idx_vec.iter().enumerate() {
                if exp_id as usize == expert_id {
                    let token_idx = flat_idx / top_k;
                    token_indices.push(token_idx as u32);
                    gate_indices.push(flat_idx as u32);
                }
            }
            
            if token_indices.is_empty() {
                continue;
            }
            
            // Apply capacity constraint
            let capacity = ((n_tokens as f32 / n_routed as f32) * top_k as f32 * self.config.capacity_factor) as usize;
            let capacity = capacity.max(self.config.min_tokens_per_expert);
            
            // Record capacity metrics before truncation
            let tokens_routed = token_indices.len();
            self.capacity_metrics.record_dispatch(expert_id, tokens_routed, capacity);
            
            if token_indices.len() > capacity {
                token_indices.truncate(capacity);
                gate_indices.truncate(capacity);
            }
            
            // Gather inputs
            let indices_tensor = Tensor::from_vec(token_indices.clone(), (token_indices.len(),), x.device())?;
            let exp_in = x.index_select(&indices_tensor, 0)?;
            
            // Process through expert
            let exp_out = self.routed_experts[expert_id].forward(&exp_in)?;
            
            // Gather gates
            let gate_indices_tensor = Tensor::from_vec(gate_indices.clone(), (gate_indices.len(),), x.device())?;
            let token_gates = gate_flat.index_select(&gate_indices_tensor, 0)?.reshape((token_indices.len(), 1))?;
            
            // Weight by gates
            let weighted_out = exp_out.broadcast_mul(&token_gates)?;
            
            // Scatter add to output
            routed_out = routed_out.index_add(&indices_tensor, &weighted_out, 0)?;
        }
        
        Ok(routed_out)
    }
    
    /// Get load balancing statistics
    pub fn get_load_balance_stats(&self) -> (f32, f32, f32) {
        self.load_balance.get_stats()
    }
    
    /// Get capacity metrics
    pub fn get_capacity_metrics(&self) -> &CapacityMetrics {
        &self.capacity_metrics
    }
    
    /// Reset capacity metrics
    pub fn reset_capacity_metrics(&mut self) {
        self.capacity_metrics.reset();
    }
    
    /// Get configuration
    pub fn config(&self) -> &DeepSeekMoEV3Config {
        &self.config
    }
}

#[cfg(test)]
mod v3_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    
    #[test]
    fn test_deepseek_moe_v3_small() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let mut config = DeepSeekMoEV3Config::small_16_2();
        config.d_model = 32;
        config.routed_expert_hidden = 64;
        config.shared_expert_hidden = 64;
        
        let mut moe = DeepSeekMoEV3::new(config, vb)?;
        
        let batch_size = 2;
        let seq_len = 8;
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, 32), &device)?;
        
        let output = moe.forward(&x)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, 32]);
        
        Ok(())
    }
    
    #[test]
    fn test_load_balancing_state() -> Result<()> {
        let device = Device::Cpu;
        let config = DeepSeekMoEV3Config::small_16_2();
        
        let mut state = LoadBalancingState::new(&config, &device)?;
        
        // Simulate uneven expert selection (16 experts total)
        // Total = 100, target per expert = 100/16 = 6.25
        let mut counts = vec![6.25f32; 16]; // Start with balanced counts
        counts[0] = 20.0; // Expert 0 overused (20 > 6.25)
        counts[1] = 0.5;  // Expert 1 underused (0.5 < 6.25)
        
        state.update(&counts, &device)?;
        
        let bias = state.get_bias();
        let bias_vec = bias.to_vec1::<f32>()?;
        
        // Expert 0 should have negative bias (discourage) - count > target
        // Expert 1 should have positive bias (encourage) - count < target
        assert!(bias_vec[0] < 0.0, "Overused expert should have negative bias, got {}", bias_vec[0]);
        assert!(bias_vec[1] > 0.0, "Underused expert should have positive bias, got {}", bias_vec[1]);
        
        Ok(())
    }
    
    #[test]
    fn test_hierarchical_routing() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let mut config = DeepSeekMoEV3Config::small_16_2();
        config.d_model = 32;
        config.routed_expert_hidden = 64;
        config.shared_expert_hidden = 64;
        
        let moe = DeepSeekMoEV3::new(config.clone(), vb)?;
        
        let x = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        
        let (indices, gates, counts) = moe.hierarchical_route(&x)?;
        
        // Check shapes
        assert_eq!(indices.dims(), &[4, config.top_k]);
        assert_eq!(gates.dims(), &[4, config.top_k]);
        assert_eq!(counts.len(), config.n_routed_experts);
        
        // Gates should sum to 1
        let gate_sums = gates.sum(1)?;
        let gate_sums_vec = gate_sums.to_vec1::<f32>()?;
        for sum in gate_sums_vec {
            assert!((sum - 1.0).abs() < 1e-5, "Gates should sum to 1, got {}", sum);
        }
        
        Ok(())
    }
}
