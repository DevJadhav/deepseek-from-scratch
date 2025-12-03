# Training Infrastructure

## Overview

This document covers the training infrastructure components used in DeepSeek implementations:

1. **Optimizers**: AdamW with decoupled weight decay
2. **Learning Rate Schedulers**: Warmup, Cosine Annealing, WSD
3. **Gradient Accumulation**: Memory-efficient training
4. **Mixed Precision**: FP16/BF16/FP8 training
5. **Checkpointing**: Model saving and resumption
6. **Distributed Training**: Multi-GPU strategies

**Key Papers:**
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2017)
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) (Loshchilov & Hutter, 2016)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (Micikevicius et al., 2017)
- [Large Batch Optimization for Deep Learning](https://arxiv.org/abs/1904.00962) (You et al., 2019)

---

## AdamW Optimizer

### Mathematical Foundation

AdamW is Adam with decoupled weight decay. The update rules are:

**Step 1: Compute gradients**
$$g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$$

**Step 2: Update biased first moment estimate**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Step 3: Update biased second moment estimate**
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Step 4: Bias correction**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Step 5: Update parameters (with decoupled weight decay)**
$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

Key difference from L2 regularization:
- **L2**: $\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2}||\theta||^2$
- **Weight Decay**: Direct subtraction $\theta_t = \theta_{t-1}(1 - \eta\lambda) - \eta \cdot \text{update}$

### Implementation

```python
class AdamW:
    """
    AdamW optimizer with decoupled weight decay.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State initialization
        self.state = {}
        for p in self.params:
            self.state[id(p)] = {
                'm': torch.zeros_like(p.data),
                'v': torch.zeros_like(p.data),
                't': 0,
            }
    
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            grad = p.grad.data
            state = self.state[id(p)]
            
            state['t'] += 1
            t = state['t']
            
            # Update biased moments
            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grad
            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * grad ** 2
            
            # Bias correction
            m_hat = state['m'] / (1 - self.beta1 ** t)
            v_hat = state['v'] / (1 - self.beta2 ** t)
            
            # Adam update
            update = m_hat / (torch.sqrt(v_hat) + self.eps)
            
            # Decoupled weight decay (key difference from L2)
            p.data = p.data - self.lr * (update + self.weight_decay * p.data)
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

### Hyperparameters

| Parameter | Typical Range | DeepSeek Default |
|-----------|---------------|------------------|
| `lr` | 1e-5 to 1e-3 | 1e-4 |
| `beta1` | 0.9 - 0.95 | 0.9 |
| `beta2` | 0.95 - 0.999 | 0.95 |
| `eps` | 1e-8 | 1e-8 |
| `weight_decay` | 0.01 - 0.1 | 0.1 |

---

## Learning Rate Schedulers

### Warmup + Cosine Annealing

Most common schedule for LLM training:

$$\eta_t = \begin{cases} 
\eta_{\max} \cdot \frac{t}{T_w} & \text{if } t < T_w \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t - T_w}{T - T_w}\pi)) & \text{otherwise}
\end{cases}$$

Where:
- $T_w$: Warmup steps
- $T$: Total training steps
- $\eta_{\max}$: Peak learning rate
- $\eta_{\min}$: Minimum learning rate (typically $0.1 \cdot \eta_{\max}$)

```python
class CosineAnnealingWithWarmup:
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.base_lr * (self.min_lr_ratio + 
                                   (1 - self.min_lr_ratio) * cosine_decay)
```

### WSD (Warmup-Stable-Decay)

Used in some DeepSeek models:

```
Learning Rate
    ^
    |     ┌────────────────────┐
    |    /                      \
    |   /                        \
    |  /                          \
    | /                            \
    |/                              \
    └─────────────────────────────────▶ Steps
     warmup    stable phase    decay
```

$$\eta_t = \begin{cases} 
\eta_{\max} \cdot \frac{t}{T_w} & \text{if } t < T_w \text{ (warmup)}\\
\eta_{\max} & \text{if } T_w \leq t < T_s \text{ (stable)}\\
\eta_{\min} + (\eta_{\max} - \eta_{\min})\frac{T - t}{T - T_s} & \text{otherwise (decay)}
\end{cases}$$

```python
class WSDScheduler:
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.lr
        self.current_step = 0
    
    def get_lr(self) -> float:
        t = self.current_step
        
        if t < self.warmup_steps:
            # Warmup phase
            return self.base_lr * (t / self.warmup_steps)
        
        t -= self.warmup_steps
        
        if t < self.stable_steps:
            # Stable phase
            return self.base_lr
        
        t -= self.stable_steps
        
        # Decay phase
        progress = t / self.decay_steps
        return self.base_lr * (self.min_lr_ratio + 
                               (1 - self.min_lr_ratio) * (1 - progress))
```

---

## Gradient Accumulation

### Purpose

Train with effective batch sizes larger than GPU memory allows:

$$\text{Effective Batch} = \text{Micro Batch} \times \text{Accumulation Steps}$$

### Mathematical Equivalence

Accumulating gradients over $K$ steps:

$$\bar{g} = \frac{1}{K}\sum_{k=1}^{K} g^{(k)} = \nabla_\theta \frac{1}{K}\sum_{k=1}^{K} \mathcal{L}^{(k)}$$

This is equivalent to computing gradient over a batch of size $K \cdot B$.

### Implementation

```python
class GradientAccumulator:
    """
    Gradient accumulation for memory-efficient training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int,
        max_grad_norm: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        self.accumulated_loss = 0.0
    
    def backward(self, loss: torch.Tensor):
        """Accumulate gradients."""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_loss += loss.item()
        self.current_step += 1
    
    def step(self) -> Optional[float]:
        """Perform optimizer step if accumulation complete."""
        if self.current_step >= self.accumulation_steps:
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Return average loss and reset
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0
            self.current_step = 0
            
            return avg_loss
        
        return None
    
    def should_step(self) -> bool:
        return self.current_step >= self.accumulation_steps


# Usage
accumulator = GradientAccumulator(model, optimizer, accumulation_steps=4)

for batch in dataloader:
    loss = model(batch)
    accumulator.backward(loss)
    
    avg_loss = accumulator.step()
    if avg_loss is not None:
        # Logging, scheduling, etc.
        print(f"Loss: {avg_loss:.4f}")
```

---

## Mixed Precision Training

### FP16 vs BF16

| Format | Exponent | Mantissa | Range | Precision |
|--------|----------|----------|-------|-----------|
| FP32 | 8 | 23 | ±3.4e38 | 7 digits |
| FP16 | 5 | 10 | ±65504 | 3 digits |
| BF16 | 8 | 7 | ±3.4e38 | 2 digits |

**BF16 preferred** for LLMs: Same range as FP32, sufficient precision.

### Loss Scaling (FP16)

FP16 has limited range, gradients can underflow. Solution: scale loss up.

$$\mathcal{L}_{\text{scaled}} = s \cdot \mathcal{L}$$
$$g_{\text{scaled}} = s \cdot g$$
$$g = g_{\text{scaled}} / s$$

### Implementation

```python
class MixedPrecisionTrainer:
    """
    Mixed precision training with automatic loss scaling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dtype: torch.dtype = torch.bfloat16,
        use_amp: bool = True,
        initial_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dtype = dtype
        self.use_amp = use_amp
        
        # Loss scaler for FP16
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=(dtype == torch.float16)
        )
    
    def forward(self, batch) -> torch.Tensor:
        """Forward pass with autocasting."""
        with torch.cuda.amp.autocast(dtype=self.dtype, enabled=self.use_amp):
            return self.model(batch)
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with scaling."""
        self.scaler.scale(loss).backward()
    
    def step(self):
        """Optimizer step with unscaling."""
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping on unscaled gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
    
    def state_dict(self) -> dict:
        return {
            'scaler': self.scaler.state_dict(),
        }
    
    def load_state_dict(self, state_dict: dict):
        self.scaler.load_state_dict(state_dict['scaler'])
```

### Apple Silicon (MPS)

For Apple Silicon, use FP32 or simulate mixed precision:

```python
def get_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == 'cuda':
        return torch.bfloat16
    elif device.type == 'mps':
        # MPS has limited mixed precision support
        return torch.float32
    else:
        return torch.float32
```

---

## Checkpointing

### Full Checkpoint

Save everything needed to resume training:

```python
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    config: dict,
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
        'config': config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
) -> dict:
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'step': checkpoint['step'],
        'loss': checkpoint['loss'],
        'config': checkpoint['config'],
    }
```

### Sharded Checkpointing

For large models, shard checkpoints across files:

```python
def save_sharded_checkpoint(
    path: str,
    model: nn.Module,
    shard_size_gb: float = 2.0,
):
    """Save model in sharded format."""
    import os
    
    os.makedirs(path, exist_ok=True)
    
    state_dict = model.state_dict()
    
    shards = []
    current_shard = {}
    current_size = 0
    shard_idx = 0
    
    max_shard_size = shard_size_gb * 1024 ** 3  # Convert to bytes
    
    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_size + tensor_size > max_shard_size and current_shard:
            # Save current shard
            shard_path = os.path.join(path, f'model-{shard_idx:05d}.pt')
            torch.save(current_shard, shard_path)
            shards.append({
                'path': f'model-{shard_idx:05d}.pt',
                'keys': list(current_shard.keys())
            })
            
            current_shard = {}
            current_size = 0
            shard_idx += 1
        
        current_shard[key] = tensor
        current_size += tensor_size
    
    # Save last shard
    if current_shard:
        shard_path = os.path.join(path, f'model-{shard_idx:05d}.pt')
        torch.save(current_shard, shard_path)
        shards.append({
            'path': f'model-{shard_idx:05d}.pt',
            'keys': list(current_shard.keys())
        })
    
    # Save index
    index_path = os.path.join(path, 'model.json')
    with open(index_path, 'w') as f:
        json.dump({'shards': shards}, f)
```

---

## Gradient Clipping

### Why Clip Gradients?

Prevent exploding gradients in deep networks:

$$||g|| = \sqrt{\sum_i g_i^2}$$

If $||g|| > \text{max\_norm}$:
$$g_{\text{clipped}} = g \cdot \frac{\text{max\_norm}}{||g||}$$

### Implementation

```python
def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Clip gradient norm of parameters.
    
    Returns:
        Total norm before clipping
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.data, norm_type) for p in parameters
            ]),
            norm_type
        )
    
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm
```

---

## Distributed Training

### Data Parallel (DP/DDP)

Each GPU has full model copy, different data:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank: int, world_size: int):
    dist.init_process_group(
        backend='nccl',  # or 'gloo' for CPU
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# Wrap model
model = Model().to(rank)
model = DDP(model, device_ids=[rank])

# Training loop
for batch in distributed_dataloader:
    loss = model(batch)
    loss.backward()  # DDP handles gradient sync
    optimizer.step()
```

### FSDP (Fully Sharded Data Parallel)

Shard model parameters across GPUs:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

# Mixed precision config
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Wrap model
model = FSDP(
    model,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

### Pipeline Parallel

Split layers across GPUs:

```
GPU 0: Layers 0-11
GPU 1: Layers 12-23
GPU 2: Layers 24-35
GPU 3: Layers 36-47

Micro-batches flow through pipeline:
  GPU0: [B0] [B1] [B2] [B3]
  GPU1:      [B0] [B1] [B2] [B3]
  GPU2:           [B0] [B1] [B2] [B3]
  GPU3:                [B0] [B1] [B2] [B3]
```

---

## Training Loop Template

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: TrainingConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        self.accumulator = GradientAccumulator(
            model, optimizer, config.accumulation_steps
        )
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train(self, train_loader, val_loader):
        for epoch in range(self.config.num_epochs):
            self.train_epoch(train_loader, epoch)
            
            val_loss = self.validate(val_loader)
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best.pt')
    
    def train_epoch(self, loader, epoch: int):
        self.model.train()
        
        for batch in tqdm(loader, desc=f'Epoch {epoch}'):
            loss = self.train_step(batch)
            
            if loss is not None:
                self.global_step += 1
                self.scheduler.step()
                
                if self.global_step % self.config.log_interval == 0:
                    self.log(loss)
                
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')
    
    def train_step(self, batch) -> Optional[float]:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = self.model(batch)
        
        self.accumulator.backward(loss)
        return self.accumulator.step()
    
    def validate(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                loss = self.model(batch)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
```

---

## References

- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [FSDP Documentation](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
