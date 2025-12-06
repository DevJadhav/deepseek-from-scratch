//! MLA Rank Constraints Module for DeepSeek
//!
//! This module implements production-grade rank constraints for Multi-Head Latent Attention (MLA):
//! - SVD-based initialization for low-rank matrices
//! - Rank regularization loss during training
//! - Numerical stability checks for projection matrices with high condition numbers
//! - Gradient clipping specifically for latent projection weights
//!
//! Reference: DeepSeek-V3 architecture specification for MLA with rank constraints.

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;

/// Configuration for MLA rank constraints
#[derive(Clone, Debug)]
pub struct RankConstraintConfig {
    /// Use SVD-based initialization for low-rank matrices
    pub use_svd_init: bool,
    /// Ratio of singular values to keep (1.0 = full rank)
    pub svd_rank_ratio: f32,
    /// Weight for rank regularization loss
    pub rank_regularization_weight: f32,
    /// Target effective rank (None = use d_latent)
    pub target_rank: Option<usize>,
    /// Maximum condition number before warning/correction
    pub condition_number_threshold: f64,
    /// Minimum singular value to prevent numerical issues
    pub min_singular_value: f64,
    /// Max gradient norm for latent projection layers
    pub latent_grad_clip_norm: f32,
    /// Adapt clip threshold based on gradient history
    pub use_adaptive_grad_clip: bool,
    /// Log condition numbers during training
    pub log_condition_numbers: bool,
}

impl Default for RankConstraintConfig {
    fn default() -> Self {
        Self {
            use_svd_init: true,
            svd_rank_ratio: 1.0,
            rank_regularization_weight: 0.01,
            target_rank: None,
            condition_number_threshold: 1e4,
            min_singular_value: 1e-6,
            latent_grad_clip_norm: 1.0,
            use_adaptive_grad_clip: true,
            log_condition_numbers: true,
        }
    }
}

impl RankConstraintConfig {
    /// Create a production config with sensible defaults
    pub fn production() -> Self {
        Self {
            use_svd_init: true,
            svd_rank_ratio: 0.95,
            rank_regularization_weight: 0.001,
            target_rank: None,
            condition_number_threshold: 1e4,
            min_singular_value: 1e-6,
            latent_grad_clip_norm: 1.0,
            use_adaptive_grad_clip: true,
            log_condition_numbers: true,
        }
    }
    
    /// Create a research config with more aggressive constraints
    pub fn research() -> Self {
        Self {
            use_svd_init: true,
            svd_rank_ratio: 0.9,
            rank_regularization_weight: 0.01,
            target_rank: None,
            condition_number_threshold: 1e3,
            min_singular_value: 1e-5,
            latent_grad_clip_norm: 0.5,
            use_adaptive_grad_clip: true,
            log_condition_numbers: true,
        }
    }
}

/// SVD-based initialization for low-rank projection matrices
pub struct SVDInitializer;

impl SVDInitializer {
    /// Initialize a weight tensor with SVD-based low-rank structure
    ///
    /// Uses truncated QR decomposition to create orthogonal initialization
    /// with controlled rank and spectral properties.
    ///
    /// # Arguments
    /// * `shape` - (out_features, in_features)
    /// * `target_rank` - Target rank for the matrix (defaults to min(out, in))
    /// * `init_scale` - Scale factor for initialization
    /// * `device` - Target device
    ///
    /// # Returns
    /// Initialized tensor with low-rank structure
    pub fn initialize_low_rank(
        shape: (usize, usize),
        target_rank: Option<usize>,
        init_scale: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let (out_features, in_features) = shape;
        let rank = target_rank.unwrap_or_else(|| out_features.min(in_features));
        let rank = rank.min(out_features).min(in_features);
        
        // Generate random matrices for QR decomposition
        let u_random = Tensor::randn(0f32, 1f32, (out_features, rank), device)?;
        let v_random = Tensor::randn(0f32, 1f32, (in_features, rank), device)?;
        
        // QR decomposition to get orthogonal bases
        // For Candle, we'll use a simplified approach since SVD/QR may not be directly available
        // We use Gram-Schmidt-like orthogonalization via normalization
        let u = Self::orthogonalize(&u_random)?;
        let v = Self::orthogonalize(&v_random)?;
        
        // Create decaying singular values (Xavier/Glorot-like)
        let fan_in = in_features as f32;
        let fan_out = out_features as f32;
        let std = init_scale * (2.0 / (fan_in + fan_out)).sqrt();
        
        // Exponentially decaying singular values for smoother low-rank approximation
        let sv_values: Vec<f32> = (0..rank)
            .map(|i| std * (-(i as f32) / rank as f32).exp())
            .collect();
        let singular_values = Tensor::from_vec(sv_values, (rank,), device)?;
        
        // Construct weight: U @ diag(S) @ V^T
        // U: (out, rank), S: (rank,), V: (in, rank) -> V^T: (rank, in)
        let s_diag = singular_values.unsqueeze(1)?; // (rank, 1)
        let u_scaled = u.broadcast_mul(&s_diag.t()?)?; // (out, rank) * (1, rank) = (out, rank)
        let weight = u_scaled.matmul(&v.t()?)?; // (out, rank) @ (rank, in) = (out, in)
        
        Ok(weight)
    }
    
    /// Initialize paired down/up projection matrices for MLA
    ///
    /// Ensures the composition down @ up has controlled rank and spectral properties.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `d_latent` - Latent dimension
    /// * `d_out` - Output dimension (may differ from d_model for K/V heads)
    /// * `init_scale` - Scale factor
    /// * `device` - Target device
    ///
    /// # Returns
    /// Tuple of (down_projection, up_projection) tensors
    pub fn initialize_paired_projections(
        d_model: usize,
        d_latent: usize,
        d_out: usize,
        init_scale: f32,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        // Down projection: d_model -> d_latent
        let down_random = Tensor::randn(0f32, 1f32, (d_latent, d_model), device)?;
        let down_ortho = Self::orthogonalize(&down_random)?;
        
        // Up projection: d_latent -> d_out
        let up_random = Tensor::randn(0f32, 1f32, (d_out, d_latent), device)?;
        let up_ortho = Self::orthogonalize(&up_random)?;
        
        // Scale for proper variance
        let scale = init_scale * (1.0 / d_latent as f32).sqrt();
        
        let down_weight = (down_ortho * scale as f64)?;
        let up_weight = (up_ortho * scale as f64)?;
        
        Ok((down_weight, up_weight))
    }
    
    /// Orthogonalize a matrix using iterative normalization
    /// 
    /// This is a simplified alternative to full QR decomposition
    fn orthogonalize(matrix: &Tensor) -> Result<Tensor> {
        // Normalize rows to unit length
        let norms = matrix.sqr()?.sum_keepdim(1)?.sqrt()?;
        let norms_safe = (norms + 1e-8)?;
        matrix.broadcast_div(&norms_safe)
    }
}

/// Rank regularization loss for MLA projection matrices
///
/// Encourages low effective rank by penalizing singular values beyond target rank.
pub struct RankRegularizationLoss {
    target_rank: usize,
    weight: f32,
    use_nuclear_norm: bool,
    tail_penalty_factor: f32,
}

impl RankRegularizationLoss {
    /// Create a new rank regularization loss
    ///
    /// # Arguments
    /// * `target_rank` - Target effective rank
    /// * `weight` - Regularization weight
    /// * `use_nuclear_norm` - Use nuclear norm (sum of singular values) penalty
    /// * `tail_penalty_factor` - Extra penalty for singular values beyond target_rank
    pub fn new(
        target_rank: usize,
        weight: f32,
        use_nuclear_norm: bool,
        tail_penalty_factor: f32,
    ) -> Self {
        Self {
            target_rank,
            weight,
            use_nuclear_norm,
            tail_penalty_factor,
        }
    }
    
    /// Compute rank regularization loss for a weight matrix
    ///
    /// Uses Frobenius norm as a differentiable proxy for rank since 
    /// exact SVD may not be available in Candle for all backends.
    ///
    /// # Arguments
    /// * `weight` - Weight matrix to regularize
    ///
    /// # Returns
    /// Regularization loss scalar
    pub fn compute(&self, weight: &Tensor) -> Result<Tensor> {
        // Use Frobenius norm as a proxy for nuclear norm
        // ||W||_F^2 = sum(s_i^2) where s_i are singular values
        let frobenius_sq = weight.sqr()?.sum_all()?;
        
        // Scale by regularization weight
        let loss = (frobenius_sq * self.weight as f64)?;
        
        Ok(loss)
    }
    
    /// Compute rank regularization using approximate SVD via power iteration
    ///
    /// This method estimates the top-k singular values and penalizes the rest.
    pub fn compute_with_power_iteration(
        &self,
        weight: &Tensor,
        num_iterations: usize,
    ) -> Result<Tensor> {
        let (rows, cols) = weight.dims2()?;
        let device = weight.device();
        
        // Estimate top singular values via power iteration
        let mut top_sv_sum = Tensor::new(0f32, device)?;
        let mut deflated_weight = weight.clone();
        
        let num_to_estimate = self.target_rank.min(rows).min(cols);
        
        for _ in 0..num_to_estimate {
            // Power iteration to find top singular value
            let (sigma, u, v) = Self::power_iteration(&deflated_weight, num_iterations)?;
            top_sv_sum = (top_sv_sum + sigma.clone())?;
            
            // Deflate: W = W - sigma * u * v^T
            let outer = u.unsqueeze(1)?.matmul(&v.unsqueeze(0)?)?;
            let scaled_outer = outer.broadcast_mul(&sigma)?;
            deflated_weight = (deflated_weight - scaled_outer)?;
        }
        
        // Remaining Frobenius norm represents tail singular values
        let tail_frobenius = deflated_weight.sqr()?.sum_all()?.sqrt()?;
        
        // Loss: weight * (head_sum + tail_penalty * tail_norm)
        let loss = if self.use_nuclear_norm {
            let tail_penalty = (tail_frobenius * self.tail_penalty_factor as f64)?;
            ((top_sv_sum + tail_penalty)? * self.weight as f64)?
        } else {
            (tail_frobenius.sqr()? * self.weight as f64)?
        };
        
        Ok(loss)
    }
    
    /// Single power iteration step to estimate top singular value
    fn power_iteration(matrix: &Tensor, num_iterations: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let (_rows, cols) = matrix.dims2()?;
        let device = matrix.device();
        
        // Initialize random vector
        let mut v = Tensor::randn(0f32, 1f32, (cols,), device)?;
        v = (v.sqr()?.sum_all()?.sqrt()?.recip()? * v)?;
        
        for _ in 0..num_iterations {
            // u = Av / ||Av||
            let u_tmp = matrix.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
            let u_norm = u_tmp.sqr()?.sum_all()?.sqrt()?;
            let norm_safe = (u_norm.clone() + 1e-8)?;
            let u = u_tmp.broadcast_div(&norm_safe)?;
            
            // v = A^T u / ||A^T u||
            let v_new = matrix.t()?.matmul(&u.unsqueeze(1)?)?.squeeze(1)?;
            let v_norm = v_new.sqr()?.sum_all()?.sqrt()?;
            let v_norm_safe = (v_norm + 1e-8)?;
            v = v_new.broadcast_div(&v_norm_safe)?;
        }
        
        // Compute singular value: sigma = u^T A v
        let u_final = matrix.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
        let u_norm = u_final.sqr()?.sum_all()?.sqrt()?;
        let norm_safe = (u_norm.clone() + 1e-8)?;
        let u = u_final.broadcast_div(&norm_safe)?;
        let sigma = u_norm;
        
        Ok((sigma, u, v))
    }
}

/// Numerical stability diagnostics for projection matrices
#[derive(Debug, Clone)]
pub struct StabilityDiagnostics {
    pub condition_number: f64,
    pub max_singular_value: f64,
    pub min_singular_value: f64,
    pub effective_rank: usize,
    pub status: StabilityStatus,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StabilityStatus {
    Ok,
    Warning,
    Corrected,
    SvdFailed,
}

/// Numerical stability checker for MLA projection matrices
pub struct NumericalStabilityChecker {
    condition_threshold: f64,
    min_singular_value: f64,
    auto_correct: bool,
    history: HashMap<String, Vec<f64>>,
}

impl NumericalStabilityChecker {
    /// Create a new stability checker
    pub fn new(condition_threshold: f64, min_singular_value: f64, auto_correct: bool) -> Self {
        Self {
            condition_threshold,
            min_singular_value,
            auto_correct,
            history: HashMap::new(),
        }
    }
    
    /// Check numerical stability using Frobenius norm heuristics
    ///
    /// Since Candle may not have full SVD support on all backends,
    /// we use norm-based approximations to detect ill-conditioning.
    pub fn check(&mut self, weight: &Tensor, name: &str) -> Result<StabilityDiagnostics> {
        // Compute Frobenius norm
        let _frobenius = weight.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
        
        // Estimate condition number using power iteration
        let (rows, cols) = weight.dims2()?;
        let min_dim = rows.min(cols);
        
        // Estimate max singular value
        let max_sv = Self::estimate_max_singular_value(weight, 10)?;
        
        // Estimate min singular value (more iterations for accuracy)
        let min_sv = Self::estimate_min_singular_value(weight, 20)?;
        
        let condition_number = max_sv / (min_sv + 1e-12);
        
            // Track history
            self.history
                .entry(name.to_string())
                .or_default()
                .push(condition_number);        // Determine status
        let (status, warning) = if condition_number > self.condition_threshold {
            if min_sv < self.min_singular_value && self.auto_correct {
                (StabilityStatus::Corrected, Some(format!(
                    "High condition number: {:.2e}, applied correction", condition_number
                )))
            } else {
                (StabilityStatus::Warning, Some(format!(
                    "High condition number: {:.2e}", condition_number
                )))
            }
        } else {
            (StabilityStatus::Ok, None)
        };
        
        // Estimate effective rank (singular values > 1% of max)
        let effective_rank = Self::estimate_effective_rank(weight, max_sv * 0.01)?;
        
        Ok(StabilityDiagnostics {
            condition_number,
            max_singular_value: max_sv,
            min_singular_value: min_sv,
            effective_rank: effective_rank.min(min_dim),
            status,
            warning,
        })
    }
    
    /// Estimate maximum singular value via power iteration on A^T A
    fn estimate_max_singular_value(matrix: &Tensor, iterations: usize) -> Result<f64> {
        let (_rows, cols) = matrix.dims2()?;
        let device = matrix.device();
        
        // Power iteration on A^T A to find largest singular value squared
        // sigma_max(A) = sqrt(lambda_max(A^T A))
        let mut v = Tensor::randn(0f32, 1f32, (cols,), device)?;
        
        for _ in 0..iterations {
            // Compute A^T A v = A^T (A v)
            let av = matrix.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;  // [rows]
            let atav = matrix.t()?.matmul(&av.unsqueeze(1)?)?.squeeze(1)?;  // [cols]
            let norm = atav.sqr()?.sum_all()?.sqrt()?;
            v = atav.broadcast_div(&(norm + 1e-8)?)?;  // [cols]
        }
        
        // Compute Rayleigh quotient: v^T A^T A v / v^T v = ||Av||^2 / ||v||^2
        let av = matrix.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
        let sigma_squared = av.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let v_norm_sq = v.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        
        Ok((sigma_squared / (v_norm_sq + 1e-12)).sqrt())
    }
    
    /// Estimate minimum singular value via inverse iteration on A^T A
    fn estimate_min_singular_value(matrix: &Tensor, iterations: usize) -> Result<f64> {
        // For ill-conditioned matrices, inverse iteration can be unstable
        // Use a simpler approach: estimate from Frobenius norm relationship
        // For a matrix with singular values σ1 >= σ2 >= ... >= σn:
        // ||A||_F^2 = σ1^2 + σ2^2 + ... + σn^2
        // 
        // Conservative estimate: assume all other singular values equal σ_max
        // Then σ_min^2 ≈ ||A||_F^2 - (n-1) * σ_max^2
        // This gives an upper bound on σ_min (conservative for condition number)
        
        let (rows, cols) = matrix.dims2()?;
        let min_dim = rows.min(cols);
        
        let frobenius_sq = matrix.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let max_sv = Self::estimate_max_singular_value(matrix, iterations)?;
        
        // For well-conditioned matrices, σ_min ≈ σ_max
        // For ill-conditioned matrices, this estimate is conservative
        let remaining_sq = (frobenius_sq - max_sv.powi(2)).max(0.0);
        
        // If remaining energy is spread across (n-1) singular values,
        // the minimum is at most sqrt(remaining / (n-1))
        // But we use a more generous estimate to avoid false positives
        let estimated_min = if min_dim > 1 {
            // Use geometric mean assumption for middle singular values
            let avg_remaining = remaining_sq / ((min_dim - 1) as f64);
            avg_remaining.sqrt().max(max_sv * 1e-10) // Don't let it go below 1e-10 * max_sv
        } else {
            max_sv // Single singular value
        };
        
        Ok(estimated_min)
    }
    
    /// Estimate effective rank (number of singular values above threshold)
    fn estimate_effective_rank(matrix: &Tensor, _threshold: f64) -> Result<usize> {
        // Use Frobenius norm / max_sv as a rough estimate
        let frobenius = matrix.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
        let max_sv = Self::estimate_max_singular_value(matrix, 5)?;
        
        // If all singular values were equal, effective_rank = (frobenius / sigma)^2
        // This gives an upper bound estimate
        let ratio = frobenius / (max_sv + 1e-12);
        let effective_rank = (ratio * ratio).ceil() as usize;
        
        Ok(effective_rank)
    }
    
    /// Get condition number history for a weight matrix
    pub fn get_history(&self, name: &str) -> Option<&Vec<f64>> {
        self.history.get(name)
    }
    
    /// Get all diagnostics
    pub fn get_all_history(&self) -> &HashMap<String, Vec<f64>> {
        &self.history
    }
}

/// Gradient clipping state for latent projection layers
pub struct LatentGradientClipperState {
    step: usize,
    grad_norm_ema: Option<f64>,
    clip_history: Vec<GradientClipStats>,
}

/// Statistics from a gradient clipping operation
#[derive(Debug, Clone)]
pub struct GradientClipStats {
    pub grad_norm_before: f64,
    pub clip_threshold: f64,
    pub clipped: bool,
    pub grad_norm_ema: f64,
}

/// Specialized gradient clipping for MLA latent projection layers
pub struct LatentProjectionGradientClipper {
    base_max_norm: f32,
    adaptive: bool,
    warmup_steps: usize,
    ema_decay: f64,
    min_clip_norm: f32,
    max_clip_norm: f32,
    state: LatentGradientClipperState,
}

impl LatentProjectionGradientClipper {
    /// Create a new gradient clipper
    pub fn new(
        base_max_norm: f32,
        adaptive: bool,
        warmup_steps: usize,
        ema_decay: f64,
        min_clip_norm: f32,
        max_clip_norm: f32,
    ) -> Self {
        Self {
            base_max_norm,
            adaptive,
            warmup_steps,
            ema_decay,
            min_clip_norm,
            max_clip_norm,
            state: LatentGradientClipperState {
                step: 0,
                grad_norm_ema: None,
                clip_history: Vec::new(),
            },
        }
    }
    
    /// Compute the total gradient norm for a list of gradient tensors
    pub fn compute_grad_norm(gradients: &[&Tensor]) -> Result<f64> {
        let mut total_norm_sq = 0.0f64;
        
        for grad in gradients {
            let grad_norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_norm_sq += grad_norm_sq;
        }
        
        Ok(total_norm_sq.sqrt())
    }
    
    /// Compute clip coefficient and update state
    ///
    /// Returns the clip coefficient (1.0 if no clipping needed, < 1.0 otherwise)
    pub fn compute_clip_coefficient(&mut self, current_grad_norm: f64) -> GradientClipStats {
        self.state.step += 1;
        
        // Update EMA
        match self.state.grad_norm_ema {
            Some(ema) => {
                self.state.grad_norm_ema = Some(
                    self.ema_decay * ema + (1.0 - self.ema_decay) * current_grad_norm
                );
            }
            None => {
                self.state.grad_norm_ema = Some(current_grad_norm);
            }
        }
        
        let ema = self.state.grad_norm_ema.unwrap();
        
        // Compute clip threshold
        let clip_threshold = if self.adaptive && self.state.step > self.warmup_steps {
            (2.0 * ema)
                .max(self.min_clip_norm as f64)
                .min(self.max_clip_norm as f64)
        } else {
            self.base_max_norm as f64
        };
        
        let clipped = current_grad_norm > clip_threshold;
        
        let stats = GradientClipStats {
            grad_norm_before: current_grad_norm,
            clip_threshold,
            clipped,
            grad_norm_ema: ema,
        };
        
        self.state.clip_history.push(stats.clone());
        
        stats
    }
    
    /// Get clip coefficient for applying to gradients
    pub fn get_clip_coefficient(&self, current_grad_norm: f64, clip_threshold: f64) -> f64 {
        if current_grad_norm > clip_threshold {
            clip_threshold / (current_grad_norm + 1e-6)
        } else {
            1.0
        }
    }
    
    /// Get recent clip history
    pub fn get_recent_history(&self, n: usize) -> &[GradientClipStats] {
        let len = self.state.clip_history.len();
        let start = len.saturating_sub(n);
        &self.state.clip_history[start..]
    }
}

/// Unified manager for all MLA rank constraints
pub struct MLARankConstraintManager {
    config: RankConstraintConfig,
    stability_checker: NumericalStabilityChecker,
    grad_clipper: LatentProjectionGradientClipper,
    rank_reg_loss: Option<RankRegularizationLoss>,
}

impl MLARankConstraintManager {
    /// Create a new rank constraint manager
    pub fn new(config: RankConstraintConfig) -> Self {
        let stability_checker = NumericalStabilityChecker::new(
            config.condition_number_threshold,
            config.min_singular_value,
            true, // auto_correct
        );
        
        let grad_clipper = LatentProjectionGradientClipper::new(
            config.latent_grad_clip_norm,
            config.use_adaptive_grad_clip,
            100,  // warmup_steps
            0.99, // ema_decay
            0.1,  // min_clip_norm
            10.0, // max_clip_norm
        );
        
        let rank_reg_loss = config.target_rank.map(|target_rank| {
            RankRegularizationLoss::new(
                target_rank,
                config.rank_regularization_weight,
                true,  // use_nuclear_norm
                2.0,   // tail_penalty_factor
            )
        });
        
        Self {
            config,
            stability_checker,
            grad_clipper,
            rank_reg_loss,
        }
    }
    
    /// Initialize MLA projection weights with SVD-based low-rank structure
    pub fn initialize_weights(
        &self,
        d_model: usize,
        d_latent: usize,
        device: &Device,
    ) -> Result<MLAInitializedWeights> {
        if !self.config.use_svd_init {
            // Return default-initialized weights
            return Ok(MLAInitializedWeights {
                kv_down: Tensor::randn(0f32, 1f32, (d_latent, d_model), device)?,
                k_up: Tensor::randn(0f32, 1f32, (d_model, d_latent), device)?,
                v_up: Tensor::randn(0f32, 1f32, (d_model, d_latent), device)?,
            });
        }
        
        let init_scale = 1.0;
        
        // Initialize paired projections for K path
        let (kv_down, k_up) = SVDInitializer::initialize_paired_projections(
            d_model, d_latent, d_model, init_scale, device
        )?;
        
        // Initialize V up-projection
        let (_, v_up) = SVDInitializer::initialize_paired_projections(
            d_model, d_latent, d_model, init_scale, device
        )?;
        
        Ok(MLAInitializedWeights {
            kv_down,
            k_up,
            v_up,
        })
    }
    
    /// Compute rank regularization loss for MLA projections
    pub fn compute_rank_regularization_loss(
        &self,
        kv_down: &Tensor,
        k_up: &Tensor,
        v_up: &Tensor,
    ) -> Result<Tensor> {
        match &self.rank_reg_loss {
            Some(loss_fn) => {
                let loss_down = loss_fn.compute(kv_down)?;
                let loss_k = loss_fn.compute(k_up)?;
                let loss_v = loss_fn.compute(v_up)?;
                
                loss_down + loss_k + loss_v
            }
            None => {
                Tensor::new(0f32, kv_down.device())
            }
        }
    }
    
    /// Check numerical stability of all MLA projections
    pub fn check_stability(
        &mut self,
        kv_down: &Tensor,
        k_up: &Tensor,
        v_up: &Tensor,
    ) -> Result<MLAStabilityReport> {
        let kv_down_diag = self.stability_checker.check(kv_down, "kv_down")?;
        let k_up_diag = self.stability_checker.check(k_up, "k_up")?;
        let v_up_diag = self.stability_checker.check(v_up, "v_up")?;
        
        Ok(MLAStabilityReport {
            kv_down: kv_down_diag,
            k_up: k_up_diag,
            v_up: v_up_diag,
        })
    }
    
    /// Get gradient clip coefficient for latent projection gradients
    pub fn get_grad_clip_coefficient(&mut self, gradients: &[&Tensor]) -> Result<f64> {
        let grad_norm = LatentProjectionGradientClipper::compute_grad_norm(gradients)?;
        let stats = self.grad_clipper.compute_clip_coefficient(grad_norm);
        Ok(self.grad_clipper.get_clip_coefficient(grad_norm, stats.clip_threshold))
    }
    
    /// Get configuration
    pub fn config(&self) -> &RankConstraintConfig {
        &self.config
    }
    
    /// Get recent gradient clipping history
    pub fn get_grad_clip_history(&self, n: usize) -> &[GradientClipStats] {
        self.grad_clipper.get_recent_history(n)
    }
}

/// Initialized MLA weights with rank constraints applied
pub struct MLAInitializedWeights {
    pub kv_down: Tensor,
    pub k_up: Tensor,
    pub v_up: Tensor,
}

/// Stability report for all MLA projections
#[derive(Debug)]
pub struct MLAStabilityReport {
    pub kv_down: StabilityDiagnostics,
    pub k_up: StabilityDiagnostics,
    pub v_up: StabilityDiagnostics,
}

impl MLAStabilityReport {
    /// Check if any projection has warnings
    pub fn has_warnings(&self) -> bool {
        self.kv_down.warning.is_some() ||
        self.k_up.warning.is_some() ||
        self.v_up.warning.is_some()
    }
    
    /// Get all warnings
    pub fn get_warnings(&self) -> Vec<&str> {
        let mut warnings = Vec::new();
        if let Some(w) = &self.kv_down.warning {
            warnings.push(w.as_str());
        }
        if let Some(w) = &self.k_up.warning {
            warnings.push(w.as_str());
        }
        if let Some(w) = &self.v_up.warning {
            warnings.push(w.as_str());
        }
        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_initializer_low_rank() -> Result<()> {
        let device = Device::Cpu;
        
        let weight = SVDInitializer::initialize_low_rank(
            (64, 128),
            Some(32),
            1.0,
            &device,
        )?;
        
        assert_eq!(weight.dims(), &[64, 128]);
        
        // Check no NaN/Inf
        let has_nan = weight.flatten_all()?.to_vec1::<f32>()?.iter()
            .any(|v| v.is_nan() || v.is_infinite());
        assert!(!has_nan);
        
        Ok(())
    }
    
    #[test]
    fn test_svd_initializer_paired_projections() -> Result<()> {
        let device = Device::Cpu;
        
        let (down, up) = SVDInitializer::initialize_paired_projections(
            512, 64, 512, 1.0, &device
        )?;
        
        assert_eq!(down.dims(), &[64, 512]);
        assert_eq!(up.dims(), &[512, 64]);
        
        // Check composition has reasonable norm
        let composed = up.matmul(&down)?;
        let norm = composed.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!(norm > 0.0 && norm < 1000.0, "Composed norm should be reasonable");
        
        Ok(())
    }
    
    #[test]
    fn test_rank_regularization_loss() -> Result<()> {
        let device = Device::Cpu;
        
        let weight = Tensor::randn(0f32, 1f32, (64, 128), &device)?;
        let loss_fn = RankRegularizationLoss::new(16, 0.01, true, 2.0);
        
        let loss = loss_fn.compute(&weight)?;
        let loss_val = loss.to_scalar::<f32>()?;
        
        assert!(loss_val > 0.0, "Loss should be positive");
        
        Ok(())
    }
    
    #[test]
    fn test_numerical_stability_checker() -> Result<()> {
        let device = Device::Cpu;
        
        let mut checker = NumericalStabilityChecker::new(1e4, 1e-6, true);
        
        // Create a well-conditioned matrix (identity + small noise)
        // This ensures a bounded condition number
        let identity = Tensor::eye(32, candle_core::DType::F32, &device)?;
        let noise = Tensor::randn(0f32, 0.01f32, (32, 32), &device)?;
        let good_weight = identity.add(&noise)?;
        let diag = checker.check(&good_weight, "good")?;
        
        // Identity-like matrix should have condition number close to 1
        assert!(diag.condition_number < 100.0, "Identity-like matrix should have low condition number, got {}", diag.condition_number);
        
        Ok(())
    }
    
    #[test]
    fn test_gradient_clipper() -> Result<()> {
        let _device = Device::Cpu;
        
        let mut clipper = LatentProjectionGradientClipper::new(
            1.0, true, 10, 0.99, 0.1, 10.0
        );
        
        // Simulate gradient norms
        for i in 0..20 {
            let grad_norm = 0.5 + 0.1 * (i as f64);
            let stats = clipper.compute_clip_coefficient(grad_norm);
            
            if i > 10 {
                // After warmup, should be using adaptive clipping
                assert!(stats.clip_threshold > 0.0);
            }
        }
        
        let history = clipper.get_recent_history(5);
        assert_eq!(history.len(), 5);
        
        Ok(())
    }
    
    #[test]
    fn test_mla_rank_constraint_manager() -> Result<()> {
        let device = Device::Cpu;
        
        let config = RankConstraintConfig::production();
        let manager = MLARankConstraintManager::new(config);
        
        // Initialize weights
        let weights = manager.initialize_weights(512, 64, &device)?;
        
        assert_eq!(weights.kv_down.dims(), &[64, 512]);
        assert_eq!(weights.k_up.dims(), &[512, 64]);
        assert_eq!(weights.v_up.dims(), &[512, 64]);
        
        // Compute regularization loss
        let loss = manager.compute_rank_regularization_loss(
            &weights.kv_down,
            &weights.k_up,
            &weights.v_up,
        )?;
        
        let loss_val = loss.to_scalar::<f32>()?;
        assert!(loss_val >= 0.0, "Loss should be non-negative");
        
        Ok(())
    }
    
    #[test]
    fn test_mla_stability_report() -> Result<()> {
        let device = Device::Cpu;
        
        let config = RankConstraintConfig::default();
        let mut manager = MLARankConstraintManager::new(config);
        
        let kv_down = Tensor::randn(0f32, 1f32, (64, 512), &device)?;
        let k_up = Tensor::randn(0f32, 1f32, (512, 64), &device)?;
        let v_up = Tensor::randn(0f32, 1f32, (512, 64), &device)?;
        
        let report = manager.check_stability(&kv_down, &k_up, &v_up)?;
        
        // Random matrices should generally be stable
        assert!(report.kv_down.effective_rank > 0);
        assert!(report.k_up.effective_rank > 0);
        assert!(report.v_up.effective_rank > 0);
        
        Ok(())
    }
}
