# DeepSeek Training Pipeline

## Overview

This document covers the complete pre-training pipeline used by DeepSeek for training their base language models at scale. The training pipeline encompasses:

1. **Data Pipeline**: Efficient data loading, tokenization, and batching
2. **Scaling Laws**: Compute-optimal training formulas
3. **Distributed Training**: Multi-GPU/node parallelism strategies
4. **Curriculum Learning**: Progressive difficulty scheduling
5. **Monitoring & Fault Tolerance**: Robust training infrastructure

**Key Papers:**
- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)

---

## When to Use This

### ✅ Use When:
- **Training models from scratch** - Full pre-training runs
- **Large-scale training** - Multi-GPU/multi-node setups
- **Compute optimization** - Need optimal batch size/LR selection
- **Production training** - Require fault tolerance and monitoring

### ❌ Don't Use When:
- **Fine-tuning only** - Use SFT pipeline instead (Chapter 12)
- **Small experiments** - Simpler training loops suffice
- **Single GPU** - Overhead not justified

---

## Scaling Laws

### DeepSeek Scaling Law

DeepSeek derived their own scaling laws from extensive experiments:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

Where:
- $L$ = Final loss
- $N$ = Number of parameters
- $D$ = Number of training tokens
- $A, B$ = Scaling coefficients
- $\alpha, \beta$ = Scaling exponents
- $L_\infty$ = Irreducible loss

**DeepSeek's Findings:**
- $\alpha \approx 0.34$ (parameter scaling)
- $\beta \approx 0.28$ (data scaling)

### Compute-Optimal Training

Given compute budget $C$ (in FLOPs):

$$C \approx 6 \cdot N \cdot D$$

The optimal allocation follows:

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

This suggests **scaling parameters and data equally** as compute grows.

### Practical Formula

```python
def compute_optimal_config(compute_flops: float) -> dict:
    """
    Compute optimal model size and data amount.
    
    Args:
        compute_flops: Total compute budget in FLOPs
    
    Returns:
        Dictionary with optimal N (params) and D (tokens)
    """
    # Based on Chinchilla/DeepSeek scaling
    # C ≈ 6ND, and N ≈ D for optimal
    optimal_tokens = (compute_flops / 6) ** 0.5
    optimal_params = optimal_tokens
    
    return {
        "optimal_params": int(optimal_params),
        "optimal_tokens": int(optimal_tokens),
        "tokens_per_param": optimal_tokens / optimal_params,
    }
```

---

## Data Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌───────────┐    ┌──────────┐    ┌───────┐  │
│  │ Raw Data│───▶│ Tokenizer │───▶│ Chunking │───▶│Batches│  │
│  │ (Text)  │    │   (BPE)   │    │(Seq Len) │    │       │  │
│  └─────────┘    └───────────┘    └──────────┘    └───────┘  │
│       │                                              │      │
│       ▼                                              ▼      │
│  ┌─────────────┐                            ┌─────────────┐ │
│  │Data Mixing  │                            │  Shuffling  │ │
│  │(code/math/  │                            │  & Caching  │ │
│  │ text ratios)│                            │             │ │
│  └─────────────┘                            └─────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Mixing Strategy

DeepSeek uses domain-specific mixing:

| Domain | Ratio | Description |
|--------|-------|-------------|
| Web Text | 60% | General language understanding |
| Code | 20% | Programming and reasoning |
| Math | 10% | Mathematical reasoning |
| Books | 5% | Long-form coherence |
| Scientific | 5% | Technical knowledge |

The mixing formula:

$$P(\text{domain}_i) = \frac{w_i^\tau}{\sum_j w_j^\tau}$$

Where $\tau$ is a temperature parameter controlling mixing sharpness.

### Implementation

```python
@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    tokenizer_path: str
    data_paths: Dict[str, str]  # domain -> path
    mixing_weights: Dict[str, float]  # domain -> weight
    mixing_temperature: float = 1.0
    seq_length: int = 4096
    batch_size: int = 1024
    num_workers: int = 8
    shuffle_buffer_size: int = 10000
    seed: int = 42


class StreamingDataset:
    """
    Memory-efficient streaming dataset for large-scale training.
    Supports data mixing and curriculum learning.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = self._load_tokenizer()
        self.data_iterators = self._create_iterators()
        self.mixing_probs = self._compute_mixing_probs()
    
    def _compute_mixing_probs(self) -> np.ndarray:
        """Compute sampling probabilities for each domain."""
        weights = np.array(list(self.config.mixing_weights.values()))
        # Temperature scaling
        scaled = weights ** self.config.mixing_temperature
        return scaled / scaled.sum()
    
    def __iter__(self):
        while True:
            # Sample domain according to mixing probabilities
            domain_idx = np.random.choice(
                len(self.mixing_probs), 
                p=self.mixing_probs
            )
            
            # Get next sample from that domain
            domain = list(self.config.data_paths.keys())[domain_idx]
            sample = next(self.data_iterators[domain])
            
            # Tokenize and chunk
            tokens = self.tokenizer.encode(sample)
            
            # Yield fixed-length chunks
            for i in range(0, len(tokens) - self.config.seq_length, self.config.seq_length):
                yield tokens[i:i + self.config.seq_length]
```

---

## Distributed Training

### Parallelism Strategies

DeepSeek uses **3D Parallelism** for large-scale training:

1. **Data Parallelism (DP)**: Replicate model, split batches
2. **Tensor Parallelism (TP)**: Split layers across GPUs
3. **Pipeline Parallelism (PP)**: Split model stages

$$\text{World Size} = DP \times TP \times PP$$

### Memory Analysis

For a model with $P$ parameters, optimizer states require:

| Optimizer | Memory per Param | Total for 7B Model |
|-----------|------------------|-------------------|
| SGD | 4 bytes (fp32) | 28 GB |
| Adam | 12 bytes (fp32 + 2 states) | 84 GB |
| AdamW + FP16 | 18 bytes | 126 GB |

With ZeRO optimization:

$$M_{\text{optimizer}} = \frac{M_{\text{full}}}{DP \times \text{ZeRO stage}}$$

### ZeRO Stages

| Stage | What's Sharded | Memory Reduction |
|-------|---------------|------------------|
| ZeRO-1 | Optimizer states | $\frac{1}{DP}$ |
| ZeRO-2 | + Gradients | $\frac{1}{DP}$ |
| ZeRO-3 | + Parameters | $\frac{1}{DP}$ |

### Implementation

```python
@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # Parallelism
    world_size: int = 8
    dp_size: int = 4
    tp_size: int = 2
    pp_size: int = 1
    
    # ZeRO
    zero_stage: int = 2
    
    # Communication
    backend: str = "nccl"
    gradient_checkpointing: bool = True
    
    # Validation
    def __post_init__(self):
        assert self.dp_size * self.tp_size * self.pp_size == self.world_size, \
            f"DP({self.dp_size}) x TP({self.tp_size}) x PP({self.pp_size}) != World({self.world_size})"


class DistributedTrainer:
    """
    Trainer with support for distributed training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        training_config: TrainingConfig,
    ):
        self.config = config
        self.training_config = training_config
        
        # Initialize distributed
        self._init_distributed()
        
        # Wrap model for distributed training
        self.model = self._wrap_model(model)
        
        # Create optimizer with ZeRO
        self.optimizer = self._create_optimizer()
    
    def _init_distributed(self):
        """Initialize distributed process group."""
        import torch.distributed as dist
        
        if not dist.is_initialized():
            dist.init_process_group(backend=self.config.backend)
        
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size()
        
        torch.cuda.set_device(self.local_rank)
    
    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for data/tensor parallelism."""
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Data parallelism with ZeRO
        if self.config.zero_stage > 0:
            from deepspeed import initialize as ds_init
            model, optimizer, _, _ = ds_init(
                model=model,
                config=self._get_deepspeed_config(),
            )
            return model
        else:
            return DDP(model, device_ids=[self.local_rank])
```

---

## Learning Rate Scheduling

### WSD Schedule (Warmup-Stable-Decay)

DeepSeek uses a three-phase learning rate schedule:

```
Learning Rate
    ^
    |     ┌────────────────────────────────────┐
    |    /                                      \
    |   /                                        \
    |  /           Stable Phase                   \
    | /                                            \
    |/                                              \
    └───────────────────────────────────────────────▶ Steps
     Warmup            ~80%                    Decay
```

$$\eta_t = \begin{cases} 
\eta_{\max} \cdot \frac{t}{T_w} & \text{if } t < T_w \text{ (warmup)}\\
\eta_{\max} & \text{if } T_w \leq t < T_s \text{ (stable)}\\
\eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \frac{T - t}{T - T_s} & \text{otherwise (decay)}
\end{cases}$$

### Peak Learning Rate Selection

Based on model size:

| Model Size | Peak LR | Warmup Steps |
|------------|---------|--------------|
| 1.3B | 3e-4 | 2000 |
| 7B | 2e-4 | 2000 |
| 13B | 1.5e-4 | 2000 |
| 67B | 1e-4 | 2000 |

Scaling rule: $\eta \propto N^{-0.05}$

### Implementation

```python
class WSDScheduler:
    """
    Warmup-Stable-Decay learning rate scheduler.
    Used in DeepSeek training.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def get_lr(self) -> float:
        step = self.current_step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * step / self.warmup_steps
        
        step -= self.warmup_steps
        
        if step < self.stable_steps:
            # Stable phase
            return self.peak_lr
        
        step -= self.stable_steps
        
        # Linear decay
        progress = step / self.decay_steps
        return self.min_lr + (self.peak_lr - self.min_lr) * (1 - progress)
    
    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr
```

---

## Curriculum Learning

### Sequence Length Curriculum

Start with shorter sequences, gradually increase:

$$L_t = \min(L_{\max}, L_{\min} + \frac{L_{\max} - L_{\min}}{T_c} \cdot t)$$

Where:
- $L_t$ = Sequence length at step $t$
- $L_{\min}$ = Starting length (e.g., 512)
- $L_{\max}$ = Target length (e.g., 4096)
- $T_c$ = Curriculum steps

### Difficulty Curriculum

For math/code data, sort by difficulty:

```python
class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive training.
    """
    
    def __init__(
        self,
        start_seq_len: int = 512,
        end_seq_len: int = 4096,
        curriculum_steps: int = 10000,
        difficulty_warmup_steps: int = 5000,
    ):
        self.start_seq_len = start_seq_len
        self.end_seq_len = end_seq_len
        self.curriculum_steps = curriculum_steps
        self.difficulty_warmup_steps = difficulty_warmup_steps
    
    def get_seq_length(self, step: int) -> int:
        """Get sequence length for current step."""
        if step >= self.curriculum_steps:
            return self.end_seq_len
        
        progress = step / self.curriculum_steps
        seq_len = self.start_seq_len + progress * (self.end_seq_len - self.start_seq_len)
        
        # Round to nearest power of 2 for efficiency
        return 2 ** int(np.log2(seq_len) + 0.5)
    
    def get_difficulty_weight(self, step: int) -> float:
        """Get weight for difficult examples (0 = easy only, 1 = all)."""
        if step >= self.difficulty_warmup_steps:
            return 1.0
        return step / self.difficulty_warmup_steps
```

---

## Checkpointing & Fault Tolerance

### Checkpoint Strategy

```python
@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    keep_last_n: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True
    async_save: bool = True  # Non-blocking saves


class CheckpointManager:
    """
    Manages model checkpoints with fault tolerance.
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        os.makedirs(config.save_dir, exist_ok=True)
        self._checkpoint_queue = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        step: int,
        metrics: Dict[str, float],
    ):
        """Save checkpoint."""
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        
        if self.config.save_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if self.config.save_scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        path = os.path.join(self.config.save_dir, f"checkpoint_{step}.pt")
        
        if self.config.async_save:
            # Save in background thread
            self._async_save(checkpoint, path)
        else:
            torch.save(checkpoint, path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def load_latest(self) -> Optional[Dict]:
        """Load most recent checkpoint."""
        checkpoints = glob.glob(os.path.join(self.config.save_dir, "checkpoint_*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest = checkpoints[-1]
        
        return torch.load(latest)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = glob.glob(os.path.join(self.config.save_dir, "checkpoint_*.pt"))
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        while len(checkpoints) > self.config.keep_last_n:
            os.remove(checkpoints.pop(0))
```

---

## Monitoring

### Key Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| Loss | Training loss | Decreasing |
| Gradient Norm | Gradient magnitude | < 1.0 (with clipping) |
| Learning Rate | Current LR | Following schedule |
| Throughput | Tokens/second | Maximizing |
| GPU Memory | Memory utilization | < 90% |
| GPU Utilization | Compute utilization | > 90% |

### Implementation

```python
class TrainingMonitor:
    """
    Monitor training progress and detect issues.
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        use_wandb: bool = True,
        project_name: str = "deepseek-pretrain",
    ):
        self.log_interval = log_interval
        self.metrics_buffer = defaultdict(list)
        
        if use_wandb:
            import wandb
            wandb.init(project=project_name)
            self.wandb = wandb
        else:
            self.wandb = None
    
    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics."""
        for k, v in metrics.items():
            self.metrics_buffer[k].append(v)
        
        if step % self.log_interval == 0:
            # Compute averages
            avg_metrics = {
                k: np.mean(v) for k, v in self.metrics_buffer.items()
            }
            
            # Log to wandb
            if self.wandb:
                self.wandb.log(avg_metrics, step=step)
            
            # Print summary
            self._print_summary(step, avg_metrics)
            
            # Clear buffer
            self.metrics_buffer.clear()
    
    def _print_summary(self, step: int, metrics: Dict[str, float]):
        """Print training summary."""
        loss = metrics.get("loss", 0)
        lr = metrics.get("lr", 0)
        throughput = metrics.get("tokens_per_second", 0)
        
        print(f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e} | {throughput:.0f} tok/s")
    
    def check_health(self, metrics: Dict[str, float]) -> List[str]:
        """Check for training issues."""
        warnings = []
        
        if metrics.get("loss", 0) > 10:
            warnings.append("Loss is very high - possible instability")
        
        if metrics.get("grad_norm", 0) > 10:
            warnings.append("Gradient norm is high - consider lower LR")
        
        if np.isnan(metrics.get("loss", 0)):
            warnings.append("NaN loss detected!")
        
        return warnings
```

---

## Complete Training Loop

```python
class PreTrainer:
    """
    Complete pre-training pipeline for DeepSeek-style training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_config: DataConfig,
        training_config: TrainingConfig,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        self.model = model
        self.data_config = data_config
        self.training_config = training_config
        self.distributed_config = distributed_config
        
        # Initialize components
        self.dataset = StreamingDataset(data_config)
        self.optimizer = self._create_optimizer()
        self.scheduler = WSDScheduler(
            optimizer=self.optimizer,
            warmup_steps=training_config.warmup_steps,
            stable_steps=training_config.stable_steps,
            decay_steps=training_config.decay_steps,
            peak_lr=training_config.learning_rate,
            min_lr=training_config.learning_rate * training_config.min_lr_ratio,
        )
        self.checkpoint_manager = CheckpointManager(
            CheckpointConfig(save_dir=training_config.checkpoint_dir)
        )
        self.monitor = TrainingMonitor()
        self.curriculum = CurriculumScheduler()
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        
        # Resume if checkpoint exists
        start_step = 0
        if resume_from or self.checkpoint_manager.load_latest():
            checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_step = checkpoint["step"]
                print(f"Resumed from step {start_step}")
        
        self.model.train()
        data_iter = iter(self.dataset)
        
        for step in range(start_step, self.training_config.max_steps):
            # Get current curriculum settings
            seq_len = self.curriculum.get_seq_length(step)
            
            # Accumulate gradients
            total_loss = 0
            for _ in range(self.training_config.gradient_accumulation_steps):
                # Get batch
                batch = self._get_batch(data_iter, seq_len)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch)
                    loss = loss / self.training_config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                total_loss += loss.item()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Scheduler step
            lr = self.scheduler.step()
            
            # Logging
            metrics = {
                "loss": total_loss,
                "lr": lr,
                "grad_norm": grad_norm.item(),
                "seq_len": seq_len,
            }
            self.monitor.log(step, metrics)
            
            # Checkpointing
            if step % self.training_config.save_every == 0:
                self.checkpoint_manager.save(
                    self.model, self.optimizer, self.scheduler, step, metrics
                )
        
        print("Training complete!")
```

---

## Rust Implementation

```rust
use candle_core::{Device, DType, Result, Tensor};
use std::collections::HashMap;

/// Scaling law calculator
pub struct ScalingLaws {
    alpha: f64,  // Parameter scaling exponent
    beta: f64,   // Data scaling exponent
    a: f64,      // Parameter coefficient
    b: f64,      // Data coefficient
}

impl ScalingLaws {
    pub fn deepseek() -> Self {
        Self {
            alpha: 0.34,
            beta: 0.28,
            a: 406.4,
            b: 410.7,
        }
    }
    
    /// Predict loss given parameters and data
    pub fn predict_loss(&self, params: f64, tokens: f64) -> f64 {
        self.a / params.powf(self.alpha) + self.b / tokens.powf(self.beta)
    }
    
    /// Compute optimal config for compute budget
    pub fn optimal_config(&self, compute_flops: f64) -> (usize, usize) {
        // C ≈ 6ND, N ≈ D for Chinchilla-optimal
        let optimal = (compute_flops / 6.0).sqrt();
        (optimal as usize, optimal as usize)
    }
}

/// WSD Learning Rate Scheduler
pub struct WSDScheduler {
    warmup_steps: usize,
    stable_steps: usize,
    decay_steps: usize,
    peak_lr: f64,
    min_lr: f64,
}

impl WSDScheduler {
    pub fn new(
        warmup_steps: usize,
        stable_steps: usize,
        decay_steps: usize,
        peak_lr: f64,
        min_lr: f64,
    ) -> Self {
        Self {
            warmup_steps,
            stable_steps,
            decay_steps,
            peak_lr,
            min_lr,
        }
    }
    
    pub fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Warmup
            self.peak_lr * (step as f64) / (self.warmup_steps as f64)
        } else if step < self.warmup_steps + self.stable_steps {
            // Stable
            self.peak_lr
        } else {
            // Decay
            let decay_step = step - self.warmup_steps - self.stable_steps;
            let progress = (decay_step as f64) / (self.decay_steps as f64);
            self.min_lr + (self.peak_lr - self.min_lr) * (1.0 - progress)
        }
    }
}

/// Data mixing configuration
pub struct DataMixingConfig {
    pub weights: HashMap<String, f64>,
    pub temperature: f64,
}

impl DataMixingConfig {
    pub fn deepseek_default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("web".to_string(), 0.60);
        weights.insert("code".to_string(), 0.20);
        weights.insert("math".to_string(), 0.10);
        weights.insert("books".to_string(), 0.05);
        weights.insert("scientific".to_string(), 0.05);
        
        Self {
            weights,
            temperature: 1.0,
        }
    }
    
    pub fn get_sampling_probs(&self) -> Vec<(String, f64)> {
        let total: f64 = self.weights.values()
            .map(|w| w.powf(self.temperature))
            .sum();
        
        self.weights.iter()
            .map(|(k, v)| (k.clone(), v.powf(self.temperature) / total))
            .collect()
    }
}

/// Training configuration
#[derive(Clone)]
pub struct PipelineConfig {
    // Model
    pub model_size: usize,
    
    // Data
    pub seq_length: usize,
    pub batch_size: usize,
    
    // Training
    pub max_steps: usize,
    pub warmup_steps: usize,
    pub stable_ratio: f64,  // Portion of training in stable phase
    pub learning_rate: f64,
    pub min_lr_ratio: f64,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub gradient_accumulation_steps: usize,
    
    // Checkpointing
    pub save_every: usize,
    pub checkpoint_dir: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model_size: 7_000_000_000,  // 7B
            seq_length: 4096,
            batch_size: 1024,
            max_steps: 100_000,
            warmup_steps: 2000,
            stable_ratio: 0.8,
            learning_rate: 2e-4,
            min_lr_ratio: 0.1,
            weight_decay: 0.1,
            max_grad_norm: 1.0,
            gradient_accumulation_steps: 4,
            save_every: 1000,
            checkpoint_dir: "./checkpoints".to_string(),
        }
    }
}

impl PipelineConfig {
    /// Get recommended LR based on model size
    pub fn recommended_lr(model_size: usize) -> f64 {
        // η ∝ N^(-0.05)
        let base_lr = 3e-4;  // For 1B model
        let base_size = 1_000_000_000.0;
        base_lr * (base_size / model_size as f64).powf(0.05)
    }
    
    /// Create WSD scheduler from config
    pub fn create_scheduler(&self) -> WSDScheduler {
        let stable_steps = (self.max_steps as f64 * self.stable_ratio) as usize;
        let decay_steps = self.max_steps - self.warmup_steps - stable_steps;
        
        WSDScheduler::new(
            self.warmup_steps,
            stable_steps,
            decay_steps,
            self.learning_rate,
            self.learning_rate * self.min_lr_ratio,
        )
    }
}

/// Curriculum learning scheduler
pub struct CurriculumScheduler {
    start_seq_len: usize,
    end_seq_len: usize,
    curriculum_steps: usize,
}

impl CurriculumScheduler {
    pub fn new(start_seq_len: usize, end_seq_len: usize, curriculum_steps: usize) -> Self {
        Self {
            start_seq_len,
            end_seq_len,
            curriculum_steps,
        }
    }
    
    pub fn get_seq_length(&self, step: usize) -> usize {
        if step >= self.curriculum_steps {
            return self.end_seq_len;
        }
        
        let progress = step as f64 / self.curriculum_steps as f64;
        let seq_len = self.start_seq_len as f64 + 
            progress * (self.end_seq_len - self.start_seq_len) as f64;
        
        // Round to power of 2
        let log2 = (seq_len.log2() + 0.5) as u32;
        2_usize.pow(log2)
    }
}

/// Demo function
pub fn demo_pipeline() -> Result<()> {
    println!("\n=== Training Pipeline Demo ===\n");
    
    // Scaling laws
    let scaling = ScalingLaws::deepseek();
    println!("Scaling Laws:");
    println!("  Predicted loss (7B, 2T tokens): {:.4}", 
        scaling.predict_loss(7e9, 2e12));
    
    let (opt_n, opt_d) = scaling.optimal_config(1e23);
    println!("  Optimal config for 1e23 FLOPs: {}B params, {}B tokens",
        opt_n / 1_000_000_000, opt_d / 1_000_000_000);
    
    // WSD Scheduler
    let config = PipelineConfig::default();
    let scheduler = config.create_scheduler();
    
    println!("\nWSD Schedule:");
    for step in [0, 1000, 2000, 50000, 90000, 100000] {
        println!("  Step {:6}: LR = {:.2e}", step, scheduler.get_lr(step));
    }
    
    // Curriculum
    let curriculum = CurriculumScheduler::new(512, 4096, 10000);
    println!("\nSequence Length Curriculum:");
    for step in [0, 2500, 5000, 7500, 10000] {
        println!("  Step {:5}: seq_len = {}", step, curriculum.get_seq_length(step));
    }
    
    // Data mixing
    let mixing = DataMixingConfig::deepseek_default();
    println!("\nData Mixing Probabilities:");
    for (domain, prob) in mixing.get_sampling_probs() {
        println!("  {}: {:.1}%", domain, prob * 100.0);
    }
    
    println!("\n=== Demo Complete ===");
    Ok(())
}
```

---

## Hyperparameter Recommendations

### DeepSeek Defaults

| Model Size | LR | Batch Size | Warmup | Total Steps |
|------------|-----|------------|--------|-------------|
| 1.3B | 3e-4 | 4M tokens | 2000 | 50K |
| 7B | 2e-4 | 4M tokens | 2000 | 100K |
| 67B | 1e-4 | 4M tokens | 2000 | 150K |

### Batch Size Scaling

$$B_{\text{optimal}} \propto \sqrt{C}$$

For larger compute budgets, scale batch size with square root of compute.

---

## References

1. [DeepSeek LLM Paper](https://arxiv.org/abs/2401.02954) - Scaling laws and training details
2. [Chinchilla Paper](https://arxiv.org/abs/2203.15556) - Compute-optimal training
3. [ZeRO Paper](https://arxiv.org/abs/1910.02054) - Memory-efficient distributed training
4. [Megatron-LM](https://arxiv.org/abs/1909.08053) - Tensor parallelism
