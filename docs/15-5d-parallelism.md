# 5D Parallelism in DeepSeek-V3.2

## Overview

DeepSeek-V3.2 employs **5D parallelism** to efficiently train models with 671B parameters across thousands of GPUs. This comprehensive parallelization strategy combines five orthogonal dimensions to achieve near-linear scaling while maintaining high GPU utilization.

**Key Papers:**
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) (Huang et al., 2018)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2019)
- [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120) (Li et al., 2021)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)

## The Five Dimensions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        5D PARALLELISM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  TENSOR PARALLELISM (TP)                                          │  │
│  │  ├── Splits weight matrices across GPUs within a node             │  │
│  │  ├── TP = 8 (one GPU per attention head group)                    │  │
│  │  └── Requires high-bandwidth NVLink connections                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  PIPELINE PARALLELISM (PP)                                        │  │
│  │  ├── Distributes transformer layers across pipeline stages        │  │
│  │  ├── PP = 16 (16 pipeline stages)                                 │  │
│  │  └── Uses DualPipe for bidirectional scheduling                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  DATA PARALLELISM (DP)                                            │  │
│  │  ├── Replicates model, partitions data batches                    │  │
│  │  ├── DP = 64 (64 data parallel replicas)                          │  │
│  │  └── Combined with ZeRO for memory efficiency                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  EXPERT PARALLELISM (EP)                                          │  │
│  │  ├── Distributes MoE experts across nodes                         │  │
│  │  ├── EP = 32 (experts partitioned across 32 groups)               │  │
│  │  └── Hierarchical All-to-All reduces communication                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  SEQUENCE PARALLELISM (SP)                                        │  │
│  │  ├── Partitions sequence dimension for long contexts              │  │
│  │  ├── Ring attention for 128K+ sequences                           │  │
│  │  └── Reduces memory for activation storage                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Total GPU count: TP × PP × DP × EP ≈ 2048 H100 GPUs                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. Tensor Parallelism (TP)

### Concept
Tensor parallelism splits individual weight matrices across GPUs, enabling layers that are too large for a single GPU to fit in distributed memory.

### Implementation in DeepSeek-V3.2

```python
class TensorParallelLinear(nn.Module):
    """Column-parallel linear layer for tensor parallelism."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_world_size: int,
        tp_rank: int,
        gather_output: bool = True
    ):
        super().__init__()
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output
        
        # Each rank handles out_features / tp_world_size columns
        self.local_out_features = out_features // tp_world_size
        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul
        output = F.linear(x, self.weight)
        
        if self.gather_output:
            # All-gather across TP group
            output = all_gather_tp(output, self.tp_world_size)
        
        return output
```

### Communication Pattern
- **Column-parallel**: All-gather on output, no communication on input
- **Row-parallel**: No communication on output, reduce-scatter on input
- **Attention**: Q, K, V projections are column-parallel; output projection is row-parallel

### DeepSeek-V3.2 Configuration
- TP = 8 (within a single node with NVLink)
- Each attention head assigned to one TP rank
- MLA projections partitioned for memory efficiency

## 2. Pipeline Parallelism (PP)

### Concept
Pipeline parallelism assigns different layers to different GPUs, forming a pipeline where micro-batches flow through stages.

### DualPipe: Bidirectional Pipeline

DeepSeek-V3.2 introduces **DualPipe** for near-zero bubble overhead:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DualPipe SCHEDULING                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Time →                                                                  │
│                                                                          │
│  Stage 0: [F0]─────[B0]─────[F4]─────[B4]─────►                          │
│                ╲       ╱         ╲       ╱                               │
│  Stage 1:      [F1]─[B1]─────[F5]─[B5]─────►                             │
│                    ╲   ╱         ╲   ╱                                   │
│  Stage 2:          [F2][B2]───[F6][B6]───►                               │
│                        ╲╱         ╲╱                                     │
│  Stage 3:              [F3/B3]─[F7/B7]───►                               │
│                                                                          │
│  Legend: F = Forward, B = Backward                                       │
│  Key: Forward and Backward can overlap at the middle stages              │
└──────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
// Rust implementation of DualPipe
pub struct DualPipeScheduler {
    num_stages: usize,
    num_micro_batches: usize,
    warmup_batches: usize,
}

impl DualPipeScheduler {
    pub fn schedule(&self) -> Vec<PipelineOp> {
        let mut schedule = Vec::new();
        
        // Warmup phase: fill the pipeline
        for i in 0..self.warmup_batches {
            schedule.push(PipelineOp::Forward(i));
        }
        
        // Steady state: overlapped forward and backward
        for i in self.warmup_batches..self.num_micro_batches {
            // Forward for new micro-batch
            schedule.push(PipelineOp::Forward(i));
            // Backward for completed micro-batch
            schedule.push(PipelineOp::Backward(i - self.warmup_batches));
        }
        
        // Cooldown: drain remaining backwards
        for i in (self.num_micro_batches - self.warmup_batches)..self.num_micro_batches {
            schedule.push(PipelineOp::Backward(i));
        }
        
        schedule
    }
}
```

### Bubble Efficiency
- Traditional 1F1B: ~30% bubble overhead
- DualPipe: ~3-5% bubble overhead
- Key innovation: Bidirectional flow eliminates most idle time

## 3. Data Parallelism (DP)

### Concept
Data parallelism replicates the model across GPUs, with each GPU processing different data batches. Gradients are synchronized via all-reduce.

### Integration with ZeRO

```python
class ZeRODataParallel:
    """ZeRO-powered data parallelism for DeepSeek-V3.2."""
    
    def __init__(
        self,
        model: nn.Module,
        dp_world_size: int,
        dp_rank: int,
        stage: int = 2  # ZeRO-2 for DeepSeek
    ):
        self.model = model
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.stage = stage
        
        # Partition optimizer states (ZeRO-1)
        self._partition_optimizer_states()
        
        # Partition gradients (ZeRO-2)
        if stage >= 2:
            self._partition_gradients()
    
    def _partition_optimizer_states(self):
        """Each rank stores 1/N of optimizer states."""
        for name, param in self.model.named_parameters():
            param_id = hash(name) % self.dp_world_size
            if param_id == self.dp_rank:
                # This rank owns this parameter's optimizer state
                self.owned_params.append(param)
```

### DeepSeek-V3.2 Configuration
- DP = 64 data parallel replicas
- ZeRO-2 for optimizer state and gradient partitioning
- Gradient compression for reduced communication

## 4. Expert Parallelism (EP)

### Concept
Expert parallelism distributes MoE experts across different nodes, with tokens routed to appropriate expert locations.

### Hierarchical All-to-All

DeepSeek-V3.2 uses a two-level communication hierarchy:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL ALL-TO-ALL                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 1: Intra-node (NVLink, 900 GB/s)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  GPU 0       │  │  GPU 1       │  │  GPU 2       │  │  GPU 3       │  │
│  │  Experts     │◄─►│  Experts    │─►│  Experts     │◄─►│  Experts    │  │
│  │  0-31        │  │  32-63       │  │  64-95       │  │  96-127      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                 │                 │                 │          │
│         └─────────────────┼─────────────────┼─────────────────┘          │
│                           │                                              │
│  Level 2: Inter-node (InfiniBand, 400 GB/s)                              │
│                           │                                              │
│         ┌─────────────────┴─────────────────┐                            │
│         ▼                                   ▼                            │
│  ┌──────────────────┐              ┌──────────────────┐                  │
│  │  Node 0          │◄────────────►│  Node 1          │                  │
│  │  Experts 0-127   │              │  Experts 128-255 │                  │
│  └──────────────────┘              └──────────────────┘                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
// Rust hierarchical All-to-All
pub struct HierarchicalAllToAll {
    intra_node_group: ProcessGroup,  // NVLink
    inter_node_group: ProcessGroup,  // InfiniBand
}

impl HierarchicalAllToAll {
    pub fn exchange(&self, tokens: &Tensor, routing: &ExpertRouting) -> Tensor {
        // Step 1: Intra-node shuffle (fast NVLink)
        let local_gathered = self.intra_node_group.all_to_all(
            tokens,
            routing.intra_node_dispatch()
        );
        
        // Step 2: Inter-node exchange (InfiniBand)
        let global_gathered = self.inter_node_group.all_to_all(
            local_gathered,
            routing.inter_node_dispatch()
        );
        
        global_gathered
    }
}
```

### DeepSeek-V3.2 Configuration
- 256 routed experts distributed across 32 expert parallel groups
- 8 experts per GPU for balanced memory
- Hierarchical communication reduces inter-node traffic by 4x

## 5. Sequence Parallelism (SP)

### Concept
Sequence parallelism partitions the sequence dimension across GPUs, enabling training on sequences longer than what fits in single-GPU memory.

### Ring Attention for 128K Context

```python
class RingAttention(nn.Module):
    """Ring attention for extreme sequence lengths."""
    
    def __init__(self, config: RingAttentionConfig):
        super().__init__()
        self.sp_world_size = config.sp_world_size
        self.chunk_size = config.max_seq_len // config.sp_world_size
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Compute attention with sequence partitioned across ranks.
        Uses ring communication for KV exchange.
        """
        sp_rank = dist.get_rank(self.sp_group)
        
        # Local QKV chunk
        local_q = q  # Each rank has its Q chunk
        local_k, local_v = k, v  # Initial KV
        
        output = torch.zeros_like(local_q)
        normalizer = torch.zeros(local_q.shape[:-1], device=q.device)
        
        # Ring rotation: receive KV from prev rank, send to next
        for step in range(self.sp_world_size):
            # Compute attention with current KV chunk
            attn_chunk, norm_chunk = self._chunk_attention(
                local_q, local_k, local_v
            )
            
            # Accumulate (online softmax update)
            output, normalizer = self._online_softmax_update(
                output, normalizer, attn_chunk, norm_chunk
            )
            
            # Ring shift KV to next position
            if step < self.sp_world_size - 1:
                local_k = self._ring_send_recv(local_k)
                local_v = self._ring_send_recv(local_v)
        
        return output / normalizer.unsqueeze(-1)
```

### Memory Benefits
- Standard attention: O(L²) memory for 128K = 16B activations
- Ring attention: O(L²/SP) = 16B/4 = 4B with SP=4
- Enables 128K context without activation checkpointing

## Parallelism Interaction

### GPU Layout for DeepSeek-V3.2

```
Total GPUs: 2048 H100s

Parallelism Degrees:
- TP = 8  (within node)
- PP = 16 (across nodes)
- DP = 64 (across clusters)
- EP = 32 (expert groups)
- SP = 4  (for 128K sequences)

GPU Assignment:
- GPUs 0-7 on Node 0: TP group 0, PP stage 0
- GPUs 8-15 on Node 0: TP group 1, PP stage 0
- ...
- GPUs 0-7 on Node 15: TP group 0, PP stage 15

Communication Volume per Iteration:
- TP (NVLink): ~100 GB/s sustained
- PP (IB): ~10 GB (activation transfers)
- DP (IB): ~50 GB (gradient sync)
- EP (Hierarchical): ~20 GB (expert dispatch)
- SP (Ring): ~5 GB (KV rotation)
```

### Efficiency Analysis

| Parallelism | Communication | Compute Efficiency | Memory Savings |
|------------|---------------|-------------------|----------------|
| TP=8       | NVLink        | 95%               | 8x             |
| PP=16      | InfiniBand    | 97% (DualPipe)    | 16x            |
| DP=64      | InfiniBand    | 98%               | 1x (ZeRO: 64x) |
| EP=32      | Hierarchical  | 96%               | 32x            |
| SP=4       | Ring          | 94%               | 4x             |

## Implementation Tips

### 1. Communication Overlap
```python
# Overlap DP gradient sync with backward pass
with torch.cuda.stream(comm_stream):
    all_reduce_async(gradients)

# Overlap EP dispatch with compute
with torch.cuda.stream(expert_stream):
    dispatch_to_experts_async(tokens, routing)
```

### 2. Memory Management
```python
# Activation checkpointing for PP stages
def checkpoint_stage(stage_fn, *args):
    if training:
        return checkpoint(stage_fn, *args, use_reentrant=False)
    return stage_fn(*args)
```

### 3. Load Balancing
```python
# Dynamic batch sizing based on expert utilization
def adaptive_batch_size(expert_loads: List[float]) -> int:
    max_load = max(expert_loads)
    if max_load > 1.2:  # Overloaded
        return current_batch_size * 0.9
    return current_batch_size
```

---

## Ray Pipeline Integration (PP=3)

The `ray_pipeline` module provides practical 5D parallelism implementation for production training with time-sliced wave execution.

### 3-GPU Pipeline Parallel Setup

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE PARALLEL (PP=3) SETUP                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │    GPU 0     │    │    GPU 1     │    │    GPU 2     │                │
│  │   Stage 0    │ ─► │   Stage 1    │ ─► │   Stage 2    │                │
│  │  Layers 0-N  │    │ Layers N-2N  │    │ Layers 2N-3N │                │
│  └──────────────┘    └──────────────┘    └──────────────┘                │
│         │                   │                   │                        │
│         └───────────────────┴───────────────────┘                        │
│                    Micro-batch Pipeline                                  │
│                                                                          │
│  Configuration:                                                          │
│    - CUDA_VISIBLE_DEVICES=0,1,2                                          │
│    - pipeline_parallel_size=3                                            │
│    - num_workers=3 (one per stage)                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Time-Sliced Wave Scheduling

The pipeline supports alternating between Rust (Candle) and Python (PyTorch) backends across training waves:

```
Time →
        0        5k       10k       15k       20k
        │────────│────────│─────────│─────────│
        │ Wave 1 │ Wave 2 │ Wave 3  │ Wave 4  │
        │ (Rust) │(Python)│ (Rust)  │(Python) │
        │        │        │         │         │
GPU 0  ═╪════════╪════════╪═════════╪═════════╪═►
GPU 1  ═╪════════╪════════╪═════════╪═════════╪═►
GPU 2  ═╪════════╪════════╪═════════╪═════════╪═►
        │        │        │         │         │
     Attention   MoE    RL/Reward  Advanced
     Components         Training   Training
```

### Configuration

```python
from ray_pipeline.config import PipelineConfig, DistributedConfig, TimeSlicedConfig

# Create production 3-GPU config
config = PipelineConfig.production_3gpu_time_sliced()

# Distributed settings (5D Parallelism)
config.distributed = DistributedConfig(
    num_workers=3,
    gpus_per_worker=1,
    pipeline_parallel_size=3,  # PP=3 across 3 GPUs
    data_parallel_size=1,
    tensor_parallel_size=1,
    expert_parallel_size=1,
    sequence_parallel_size=1,
)

# Time-sliced wave config
config.time_sliced = TimeSlicedConfig(
    enabled=True,
    num_waves=4,
    steps_per_wave=5000,
    gpu_ids=[0, 1, 2],
    pipeline_parallel_size=3,
    validation_after_each_wave=True,
)
```

### CLI Usage

```bash
# Full time-sliced execution
python -m ray_pipeline.cli run --time-sliced --gpus 3 --pp-size 3 --max-steps 20000

# Rust-only waves (Waves 1 & 3: Attention + RL)
python -m ray_pipeline.cli run-rust --gpus 3 --pp-size 3 --max-steps 10000

# Python-only waves (Waves 2 & 4: MoE + Advanced)
python -m ray_pipeline.cli run-python --gpus 3 --pp-size 3 --max-steps 10000 \
    --checkpoint-from checkpoints/step_5000
```

### Wave Components by Backend

| Wave | Backend | Components | Why This Backend |
|------|---------|------------|------------------|
| 1 | Rust | MQA, GQA, MLA, DeepSeek Attn | Candle Metal kernel efficiency |
| 2 | Python | Standard MoE, DeepSeek MoE | PyTorch dynamic routing flexibility |
| 3 | Rust | GRPO, R1, DPO, Reward | Low-overhead loss computation |
| 4 | Python | MTP, FP8, Distillation, 5D | Ray Train distributed coordination |

### Checkpoint Consolidation

At each 1k step interval:
1. Save wave checkpoint to `checkpoints/step_{n}/`
2. Record validation loss in `metadata["wave_n_val_loss"]`

At 20k steps:
1. Compare validation loss across all waves
2. Select checkpoint with lowest loss
3. Save as `checkpoints/final/best_model.safetensors`

---

## Summary

5D parallelism enables DeepSeek-V3.2 to efficiently train at scale:

1. **TP** handles large layer sizes with fast NVLink
2. **PP** distributes depth with DualPipe for low bubbles
3. **DP** scales data throughput with ZeRO memory efficiency
4. **EP** distributes 256 experts across the cluster
5. **SP** enables 128K context with ring attention

The careful orchestration of these dimensions achieves 92%+ MFU (Model FLOPS Utilization) during training.

### ray_pipeline Implementation

For practical multi-GPU setups (3-8 GPUs), the `ray_pipeline` provides:
- Time-sliced wave execution alternating Rust/Python backends
- Pipeline parallelism (PP=3) across available GPUs
- Automatic checkpoint handoff between waves
- Best-model selection based on validation loss
