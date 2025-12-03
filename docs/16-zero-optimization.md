# ZeRO Optimization in DeepSeek-V3.2

## Overview

**ZeRO (Zero Redundancy Optimizer)** is a memory optimization technique that eliminates redundant storage of model states across data parallel ranks. DeepSeek-V3.2 uses ZeRO-2 to train the 671B parameter model efficiently.

**Key Papers:**
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2019)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840) (Ren et al., 2021)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) (Rajbhandari et al., 2021)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) (Zhao et al., 2023)

## Memory Analysis

### Without ZeRO (Standard Data Parallelism)

For a model with $\Psi$ parameters using mixed precision training:

| Component | Memory per GPU |
|-----------|---------------|
| FP16 Parameters | $2\Psi$ bytes |
| FP16 Gradients | $2\Psi$ bytes |
| FP32 Parameters (Adam) | $4\Psi$ bytes |
| FP32 Momentum (Adam) | $4\Psi$ bytes |
| FP32 Variance (Adam) | $4\Psi$ bytes |
| **Total** | $16\Psi$ bytes |

For DeepSeek-V3.2 (671B params):
- Memory per GPU: $16 \times 671B = 10.7$ TB ❌ Impossible!

### With ZeRO

ZeRO partitions these states across $N_d$ data parallel ranks:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ZERO MEMORY HIERARCHY                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ZeRO-1: Optimizer State Partitioning                           │   │
│  │  ├── Each rank stores 1/N of optimizer states                   │   │
│  │  ├── Memory: 4Ψ + 4Ψ/N (params+grads + optimizer/N)            │   │
│  │  └── Communication: Gather optimizer states before update       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ZeRO-2: + Gradient Partitioning                                │   │
│  │  ├── Each rank stores 1/N of gradients                          │   │
│  │  ├── Memory: 2Ψ + 2Ψ/N + 12Ψ/N                                 │   │
│  │  └── Communication: Reduce-scatter gradients                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ZeRO-3: + Parameter Partitioning                               │   │
│  │  ├── Each rank stores 1/N of parameters                         │   │
│  │  ├── Memory: 16Ψ/N (everything partitioned)                     │   │
│  │  └── Communication: All-gather params for forward/backward      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## ZeRO Stage Comparison

### Memory per GPU (for 671B model, DP=64)

| Stage | Memory Formula | DeepSeek-V3.2 |
|-------|---------------|---------------|
| Baseline | $16\Psi$ | 10.7 TB |
| ZeRO-1 | $4\Psi + 12\Psi/N_d$ | 2.8 TB |
| ZeRO-2 | $2\Psi + 14\Psi/N_d$ | 1.5 TB |
| ZeRO-3 | $16\Psi/N_d$ | 167 GB ✅ |

DeepSeek-V3.2 uses **ZeRO-2** with DP=64, achieving ~1.5 TB/GPU for model states, leaving memory for activations and KV cache.

## Implementation

### ZeRO-1: Optimizer State Partitioning

```python
class ZeRO1Optimizer:
    """ZeRO Stage 1: Partition optimizer states across DP ranks."""
    
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        optimizer_class: Type[torch.optim.Optimizer],
        dp_world_size: int,
        dp_rank: int,
        **optimizer_kwargs
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        
        # Partition parameters: each rank owns 1/N
        self.owned_params = []
        self.param_to_owner = {}
        
        for i, param in enumerate(params):
            owner_rank = i % dp_world_size
            self.param_to_owner[id(param)] = owner_rank
            if owner_rank == dp_rank:
                self.owned_params.append(param)
        
        # Create optimizer only for owned parameters
        self.optimizer = optimizer_class(
            self.owned_params, **optimizer_kwargs
        )
    
    def step(self):
        """Perform optimization step with all-gather."""
        # Each rank updates its owned parameters
        self.optimizer.step()
        
        # Broadcast updated parameters to all ranks
        for param, owner in self.param_to_owner.items():
            dist.broadcast(param.data, src=owner)
```

### ZeRO-2: Gradient Partitioning

```python
class ZeRO2Optimizer(ZeRO1Optimizer):
    """ZeRO Stage 2: Partition gradients + optimizer states."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_gradient_hooks()
    
    def _register_gradient_hooks(self):
        """Register hooks for gradient reduce-scatter."""
        for param in self.all_params:
            param.register_hook(self._gradient_hook)
    
    def _gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Reduce-scatter gradients to owning rank."""
        # Determine gradient partition boundaries
        chunk_size = grad.numel() // self.dp_world_size
        
        # Reduce-scatter: each rank gets its partition's reduced gradient
        output = torch.empty(chunk_size, device=grad.device)
        dist.reduce_scatter(output, list(grad.chunk(self.dp_world_size)))
        
        # Only owning rank keeps the gradient
        if self.param_to_owner[id(grad)] == self.dp_rank:
            return output
        return None  # Free non-owned gradients
    
    def step(self):
        """Update only owned parameters, then all-gather."""
        # Owned gradients already reduced
        self.optimizer.step()
        
        # All-gather updated parameters
        for param in self.all_params:
            chunk_size = param.numel() // self.dp_world_size
            gathered = [torch.empty(chunk_size, device=param.device) 
                       for _ in range(self.dp_world_size)]
            dist.all_gather(gathered, param.data.view(-1)[:chunk_size])
            param.data = torch.cat(gathered).view(param.shape)
```

### ZeRO-3: Parameter Partitioning

```python
class ZeRO3Optimizer(ZeRO2Optimizer):
    """ZeRO Stage 3: Partition parameters + gradients + optimizer states."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._partition_parameters()
    
    def _partition_parameters(self):
        """Partition parameters across ranks."""
        for param in self.all_params:
            chunk_size = param.numel() // self.dp_world_size
            start = self.dp_rank * chunk_size
            end = start + chunk_size
            
            # Only store owned partition
            param.data = param.data.view(-1)[start:end].clone()
            param._full_shape = param.shape
            param._partition_start = start
    
    def gather_params(self, params: List[torch.nn.Parameter]):
        """All-gather parameters for forward/backward pass."""
        for param in params:
            gathered = [torch.empty_like(param.data) 
                       for _ in range(self.dp_world_size)]
            dist.all_gather(gathered, param.data)
            param.data = torch.cat(gathered).view(param._full_shape)
    
    def release_params(self, params: List[torch.nn.Parameter]):
        """Release gathered parameters to save memory."""
        for param in params:
            start = param._partition_start
            chunk_size = param.numel() // self.dp_world_size
            param.data = param.data.view(-1)[start:start+chunk_size].clone()
```

## Rust Implementation

```rust
// ZeRO-2 implementation in Rust
pub struct ZeRO2Config {
    pub dp_world_size: usize,
    pub dp_rank: usize,
    pub reduce_bucket_size: usize,  // For gradient bucketing
    pub overlap_comm: bool,          // Overlap communication with compute
}

pub struct ZeRO2Optimizer {
    config: ZeRO2Config,
    param_groups: Vec<ParameterGroup>,
    optimizer_states: HashMap<usize, OptimizerState>,
    gradient_buckets: Vec<GradientBucket>,
}

impl ZeRO2Optimizer {
    pub fn new(
        model: &Model,
        config: ZeRO2Config,
        lr: f32,
        betas: (f32, f32),
        weight_decay: f32,
    ) -> Self {
        let mut optimizer = Self {
            config,
            param_groups: Vec::new(),
            optimizer_states: HashMap::new(),
            gradient_buckets: Vec::new(),
        };
        
        // Assign parameters to ranks
        for (i, param) in model.parameters().enumerate() {
            let owner_rank = i % config.dp_world_size;
            
            if owner_rank == config.dp_rank {
                // This rank owns this parameter's optimizer state
                optimizer.optimizer_states.insert(
                    i,
                    OptimizerState::new_adam(param.numel(), lr, betas, weight_decay)
                );
            }
        }
        
        optimizer
    }
    
    pub fn reduce_scatter_gradients(&mut self, gradients: &mut [Tensor]) {
        // Bucket gradients for efficient communication
        for bucket in self.gradient_buckets.iter_mut() {
            bucket.clear();
        }
        
        for (param_id, grad) in gradients.iter().enumerate() {
            let bucket_id = param_id / self.config.reduce_bucket_size;
            self.gradient_buckets[bucket_id].add(grad);
        }
        
        // Reduce-scatter each bucket
        for bucket in &mut self.gradient_buckets {
            let reduced = nccl::reduce_scatter(
                bucket.data(),
                self.config.dp_world_size,
            );
            bucket.set_reduced(reduced);
        }
    }
    
    pub fn step(&mut self) {
        // Update only owned parameters
        for (param_id, state) in self.optimizer_states.iter_mut() {
            let grad = self.get_reduced_gradient(*param_id);
            state.adam_update(grad);
        }
        
        // All-gather updated parameters
        self.all_gather_parameters();
    }
}
```

## Communication Patterns

### ZeRO-2 Communication Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ZeRO-2 COMMUNICATION PATTERN                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Forward Pass:                                                          │
│  ────────────                                                           │
│  [No extra communication - parameters are replicated]                   │
│                                                                         │
│  Backward Pass (overlapped with gradient computation):                  │
│  ─────────────────────────────────────────────────────                  │
│                                                                         │
│  Rank 0         Rank 1         Rank 2         Rank 3                    │
│  ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐                   │
│  │ g0  │───┐    │ g0  │───┐    │ g0  │───┐    │ g0  │───┐               │
│  │ g1  │───┼────│ g1  │───┼────│ g1  │───┼────│ g1  │───┼──► Σg0/N      │
│  │ g2  │───┼────│ g2  │───┼────│ g2  │───┼────│ g2  │───┘  (Rank 0)     │
│  │ g3  │───┘    │ g3  │───┘    │ g3  │───┘    │ g3  │                   │
│  └─────┘        └─────┘        └─────┘        └─────┘                   │
│     │              │              │              │                      │
│     └──────────────┴──────────────┴──────────────┴─► Σg1/N (Rank 1)     │
│     └──────────────┴──────────────┴──────────────┴─► Σg2/N (Rank 2)     │
│     └──────────────┴──────────────┴──────────────┴─► Σg3/N (Rank 3)     │
│                                                                         │
│  Optimizer Step (each rank updates its owned gradients):                │
│  ───────────────────────────────────────────────────────                │
│  Rank 0: Update params for g0                                           │
│  Rank 1: Update params for g1                                           │
│  Rank 2: Update params for g2                                           │
│  Rank 3: Update params for g3                                           │
│                                                                         │
│  All-Gather (collect updated parameters):                               │
│  ────────────────────────────────────────                               │
│  All ranks receive: [p0', p1', p2', p3']                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Communication Volume Comparison

| Operation | Baseline | ZeRO-2 |
|-----------|----------|--------|
| Gradient sync | $2\Psi$ (all-reduce) | $2\Psi$ (reduce-scatter + all-gather) |
| Optimizer state | 0 | 0 |
| **Total** | $2\Psi$ | $2\Psi$ |

ZeRO-2 has the same communication volume as standard DP, but significantly reduces memory!

## Gradient Bucketing

For efficiency, gradients are grouped into buckets before communication:

```python
class GradientBucket:
    """Bucket for gradient accumulation and communication."""
    
    def __init__(self, size: int, dtype: torch.dtype):
        self.buffer = torch.zeros(size, dtype=dtype)
        self.offset = 0
        self.params = []
    
    def add(self, param: torch.nn.Parameter):
        """Add parameter's gradient to bucket."""
        numel = param.numel()
        grad_flat = param.grad.view(-1)
        
        self.buffer[self.offset:self.offset + numel].copy_(grad_flat)
        self.params.append((param, self.offset, numel))
        self.offset += numel
    
    def reduce_scatter(self, dp_group):
        """Perform reduce-scatter on bucket."""
        chunk_size = self.offset // dist.get_world_size(dp_group)
        output = torch.empty(chunk_size, dtype=self.buffer.dtype)
        
        dist.reduce_scatter(
            output,
            list(self.buffer[:self.offset].chunk(dist.get_world_size(dp_group))),
            group=dp_group
        )
        
        return output
```

## Communication Overlap

ZeRO-2 can overlap gradient reduction with backward computation:

```python
class OverlappedZeRO2:
    """ZeRO-2 with communication-computation overlap."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.comm_stream = torch.cuda.Stream()
        
        # Register backward hooks for overlap
        for name, param in model.named_parameters():
            param.register_post_accumulate_grad_hook(
                self._create_reduce_hook(name)
            )
    
    def _create_reduce_hook(self, param_name):
        def hook(param):
            # Launch reduction on separate stream
            with torch.cuda.stream(self.comm_stream):
                self._reduce_scatter_param(param, param_name)
        return hook
    
    def backward_with_overlap(self, loss):
        """Backward pass with overlapped gradient reduction."""
        loss.backward()
        
        # Wait for all reductions to complete
        self.comm_stream.synchronize()
```

## DeepSeek-V3.2 Configuration

```python
# DeepSeek-V3.2 ZeRO-2 configuration
zero2_config = {
    "stage": 2,
    "dp_world_size": 64,
    
    # Gradient bucketing
    "reduce_bucket_size": 500_000_000,  # 500M elements per bucket
    
    # Communication overlap
    "overlap_comm": True,
    "contiguous_gradients": True,
    
    # Memory optimization
    "offload_optimizer": False,  # Keep on GPU for speed
    "offload_param": False,
    
    # For 671B model with DP=64:
    # Memory per GPU: ~1.5 TB model states + activations
}
```

## Memory Savings Summary

For DeepSeek-V3.2 (671B params, DP=64):

| Component | Baseline | ZeRO-2 | Savings |
|-----------|----------|--------|---------|
| Parameters (FP16) | 1.34 TB | 1.34 TB | 0% |
| Gradients (FP16) | 1.34 TB | 21 GB | 98.4% |
| Optimizer (FP32) | 8.05 TB | 126 GB | 98.4% |
| **Total** | 10.7 TB | 1.49 TB | **86%** |

This allows the 671B model to fit on 80GB H100 GPUs with sufficient memory for activations and KV cache.
