# DeepSeek R1 & V3.2 Implementation Optimization Plan

> **Principal AI Systems Architect Analysis**
> **Generated:** 2025-12-09
> **Codebase Version:** Post-758bc1d (Ray Tune integration)

---

## Executive Summary

This document presents a comprehensive optimization plan for the DeepSeek R1 and V3.2 implementation, synthesized from multi-agent deep research analysis. The codebase demonstrates solid foundational implementations across three backends (Rust, PyTorch, MLX) but requires targeted optimizations for production-grade 5D parallelism and zero-bubble pipeline efficiency.

### Key Findings

1. **DualPipe Foundation Exists** (`rust-src/src/distributed/pipeline.rs`): 1902-line implementation with phase management, but lacks async execution for true zero-bubble
2. **MLX ANE Optimization Strong**: MLA achieves 14-16x KV compression; MoE has intelligent strategy selection (FUSED/BATCHED/HIERARCHICAL)
3. **Critical Gap - Context Parallelism**: Only 4 of 5 parallelism dimensions implemented; context parallelism for 128K+ sequences missing
4. **Cross-Backend Parity Untested**: No automated verification of numerical equivalence between PyTorch/MLX/Rust

### Completeness Matrix

| Backend | Completeness | Critical Gaps |
|---------|--------------|---------------|
| **Rust (Candle)** | 90% | Async DualPipe, context parallelism, custom CUDA/Metal kernels |
| **PyTorch** | 95% | V3.2 sparse attention, context parallelism, ZeRO-Infinity |
| **MLX (Apple)** | 85% | Training mode MoE, DSA dilated patterns, INT8 KV cache |
| **Ray/Modal** | 80% | Heterogeneous clusters, 8-GPU DualPipe verification |

---

## Table of Contents

1. [Rust Backend Optimization](#1-rust-backend-optimization)
2. [PyTorch Backend Optimization](#2-pytorch-backend-optimization)
3. [Python + MLX Backend](#3-python--mlx-backend)
4. [Distributed Infrastructure](#4-distributed-infrastructure)
5. [Testing & Verification](#5-testing--verification)
6. [Appendix: Architecture Diagrams](#appendix-architecture-diagrams)

---

## 1. Rust Backend Optimization

### 1.1 Core Tensor Abstraction

**Current State:** Device selection in `rust-src/src/utils/device.rs` handles CUDA→Metal→CPU fallback, but tensor operations have scattered `#[cfg(feature)]` conditionals.

#### Tasks

- [ ] **1.1.1** Create unified `Tensor<B: Backend>` trait abstraction
  ```rust
  // rust-src/src/tensor/backend.rs
  pub trait Backend: Send + Sync {
      fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
      fn softmax(&self, x: &Tensor, dim: i64) -> Result<Tensor>;
      fn rope(&self, x: &Tensor, freqs: &Tensor) -> Result<Tensor>;
  }

  pub struct CudaBackend { device_id: usize }
  pub struct MetalBackend { device_id: usize }
  pub struct CpuBackend;
  ```
  **File:** `rust-src/src/tensor/backend.rs` (new)
  **Priority:** High | **Complexity:** Medium

- [ ] **1.1.2** Implement `cudarc` bindings for CUDA backend
  - Wrap cuBLAS GEMM for matmul
  - Implement custom CUDA kernels for RoPE, softmax
  - Add memory pool for activation reuse
  **Files:** `rust-src/src/tensor/cuda.rs` (new)
  **Priority:** High | **Complexity:** High

- [ ] **1.1.3** Implement `metal-rs` bindings for Metal backend
  - Metal Performance Shaders for matmul
  - Custom Metal shaders for RoPE (match ANE optimization)
  - Shared memory buffer management
  **Files:** `rust-src/src/tensor/metal.rs` (new)
  **Priority:** High | **Complexity:** High

- [ ] **1.1.4** Refactor existing model code to use `Tensor<B>`
  - Update `rust-src/src/model/mla.rs`
  - Update `rust-src/src/model/moe.rs`
  - Update `rust-src/src/model/sparse_attention.rs`
  **Priority:** Medium | **Complexity:** Medium

### 1.2 DualPipe Scheduler with Tokio

**Current State:** `DualPipeEngine::train_step()` in `pipeline.rs:1680` is synchronous, causing pipeline bubbles during communication.

#### Tasks

- [ ] **1.2.1** Convert `DualPipeEngine` to async with tokio runtime
  ```rust
  // rust-src/src/distributed/pipeline.rs
  impl DualPipeEngine {
      pub async fn train_step_async(
          &mut self,
          micro_batches: Vec<MicroBatch>,
      ) -> Result<(Tensor, PipelineMetrics)> {
          let (compute_tx, compute_rx) = tokio::sync::mpsc::channel(32);
          let (comm_tx, comm_rx) = tokio::sync::mpsc::channel(32);

          // Overlap compute and communication
          tokio::select! {
              _ = self.forward_pass(&micro_batches, &compute_tx) => {},
              _ = self.sync_gradients(&comm_rx) => {},
          }
      }
  }
  ```
  **File:** `rust-src/src/distributed/pipeline.rs`
  **Priority:** Critical | **Complexity:** High

- [ ] **1.2.2** Implement zero-bubble scheduling algorithm
  - Port DeepSeek-V3 paper Algorithm 1 (Section 3.2)
  - Add warmup microbatch prefetching
  - Implement steady-state interleaving
  ```rust
  pub struct ZeroBubbleScheduler {
      warmup_steps: usize,
      steady_interleave_ratio: f32, // 1F1B vs 2F1B adaptive
      backward_prefetch_depth: usize,
  }
  ```
  **File:** `rust-src/src/distributed/zero_bubble.rs` (new)
  **Priority:** Critical | **Complexity:** Very High

- [ ] **1.2.3** Add activation checkpointing to `ActivationBuffer`
  ```rust
  impl ActivationBuffer {
      pub fn checkpoint(&mut self, layer_id: usize) -> CheckpointHandle;
      pub fn recompute(&self, handle: CheckpointHandle) -> Result<Tensor>;
  }
  ```
  **File:** `rust-src/src/distributed/pipeline.rs:200-300`
  **Priority:** High | **Complexity:** Medium

- [ ] **1.2.4** Implement bidirectional stream synchronization
  - Regular stream: stages 0→1→2→3
  - Reverse stream: stages 3→2→1→0
  - Add barrier points for gradient aggregation
  **Priority:** High | **Complexity:** High

- [ ] **1.2.5** Add CUDA stream overlap for compute/communication
  - File: `rust-src/src/distributed/pipeline.rs:1040-1187`
  - Current: Sequential forward → send → receive
  - Target: Pipeline forward[i] with send[i-1] using separate CUDA streams
  **Priority:** High | **Complexity:** High

### 1.3 MoE Expert Kernels

**Current State:** Expert routing exists but lacks fused kernels for top-k gating + expert dispatch.

#### Tasks

- [ ] **1.3.1** Implement fused top-k gating kernel (CUDA)
  ```rust
  // Single kernel for: softmax → top_k → scatter
  #[cuda_kernel]
  pub fn fused_moe_gate(
      hidden: &Tensor,      // [batch, seq, d_model]
      gate_weights: &Tensor, // [d_model, num_experts]
      top_k: usize,
      output_indices: &mut Tensor,  // [batch, seq, top_k]
      output_weights: &mut Tensor,  // [batch, seq, top_k]
  );
  ```
  **File:** `rust-src/src/kernels/moe_gate.cu` (new)
  **Priority:** High | **Complexity:** High

- [ ] **1.3.2** Implement expert parallel dispatch
  - All-to-all communication for expert tokens
  - Load balancing with auxiliary loss
  - Capacity factor overflow handling
  **File:** `rust-src/src/distributed/expert.rs`
  **Priority:** High | **Complexity:** High

- [ ] **1.3.3** Add expert load telemetry
  ```rust
  pub struct ExpertMetrics {
      pub tokens_per_expert: Vec<usize>,
      pub overflow_count: usize,
      pub load_balance_loss: f32,
  }
  ```
  **File:** `rust-src/src/distributed/expert.rs`
  **Priority:** Medium | **Complexity:** Low

- [ ] **1.3.4** Implement Metal compute shader for expert routing
  - File: `rust-src/src/kernels/moe_routing.metal` (new)
  - Use Metal's simdgroup operations for efficient reduction
  - Target Apple M1/M2/M3 with 8-10 GPU cores
  **Priority:** Medium | **Complexity:** High

### 1.4 Modal GPU Verification Harness

#### Tasks

- [ ] **1.4.1** Create Modal stub for Rust CUDA verification
  ```python
  # scripts/verify_rust_cuda.py
  import modal

  app = modal.App("deepseek-rust-verify")

  @app.function(gpu=modal.gpu.A100(count=1, memory=40))
  def verify_rust_kernels():
      """Run Rust CUDA kernels and capture outputs for parity check."""
      import subprocess
      result = subprocess.run([
          "cargo", "test", "--features", "cuda",
          "--", "--test-threads=1"
      ], capture_output=True)
      return result.stdout.decode()
  ```
  **File:** `scripts/verify_rust_cuda.py` (new)
  **Priority:** Medium | **Complexity:** Low

- [ ] **1.4.2** Add PyO3 bindings for Python interop testing
  - Expose `Tensor` to Python
  - Expose `DualPipeEngine` for orchestration
  **Files:** `rust-src/src/pyo3_bindings/`
  **Priority:** Medium | **Complexity:** Medium

- [ ] **1.4.3** Multi-GPU NCCL collective operations verification
  - File: `rust-src/src/distributed/nccl_backend.rs`
  - Test: All-reduce gradient sync across 8 GPUs
  - Command: `cargo run --features cuda -- verify-nccl`
  **Priority:** High | **Complexity:** Medium

---

## 2. PyTorch Backend Optimization

### 2.1 Device-Agnostic V3.2 Model

**Current State:** Model files exist in `src/deepseek/torch/model/` but device handling is inconsistent.

#### Tasks

- [ ] **2.1.1** Implement `DeviceManager` for CUDA→MPS→CPU fallback
  ```python
  # src/deepseek/torch/utils/device.py
  class DeviceManager:
      """Mirrors Rust DeviceSelector for parity."""

      @staticmethod
      def get_device(priority: str = "cuda_first") -> torch.device:
          if priority == "cuda_first":
              if torch.cuda.is_available():
                  return torch.device("cuda")
              elif torch.backends.mps.is_available():
                  return torch.device("mps")
          elif priority == "mps_first":
              if torch.backends.mps.is_available():
                  return torch.device("mps")
              elif torch.cuda.is_available():
                  return torch.device("cuda")
          return torch.device("cpu")
  ```
  **File:** `src/deepseek/torch/utils/device.py` (new)
  **Priority:** High | **Complexity:** Low

- [ ] **2.1.2** Implement V3.2 sparse attention pattern
  ```python
  # src/deepseek/torch/model/sparse_attention.py
  class DeepSeekV32SparseAttention(nn.Module):
      """
      V3.2 uses dynamic sparsity:
      - Local window: 4096 tokens
      - Global tokens: every 512th token attends globally
      - Learned sparsity mask for mid-range
      """
      def __init__(self, config: V32Config):
          self.local_window = config.local_window  # 4096
          self.global_stride = config.global_stride  # 512
          self.sparsity_predictor = nn.Linear(config.d_model, 1)
  ```
  **File:** `src/deepseek/torch/model/sparse_attention.py`
  **Priority:** High | **Complexity:** High

- [ ] **2.1.3** Add automatic mixed precision (AMP) wrapper
  ```python
  class AMPModelWrapper:
      def __init__(self, model, dtype=torch.bfloat16):
          self.model = model
          self.scaler = torch.cuda.amp.GradScaler()

      def forward(self, *args, **kwargs):
          with torch.cuda.amp.autocast(dtype=self.dtype):
              return self.model(*args, **kwargs)
  ```
  **File:** `src/deepseek/torch/utils/amp.py` (new)
  **Priority:** Medium | **Complexity:** Low

- [ ] **2.1.4** Add 256-expert hierarchical routing verification
  - File: `src/deepseek/torch/model/moe.py`
  - Current: Standard MoE with top-k
  - Target: Group selection (top-4 of 8 groups) + expert selection (top-2 per group)
  - Verification: Assert 8 experts active per token
  **Priority:** High | **Complexity:** Medium

### 2.2 5D Parallelism Implementation

**Current State:** Data, tensor, and pipeline parallelism exist. Sequence parallelism partial. Context parallelism missing.

#### Tasks

- [ ] **2.2.1** Implement sequence parallelism for attention
  ```python
  # src/deepseek/torch/distributed/sequence_parallel.py
  class SequenceParallelAttention(nn.Module):
      """
      Splits sequence dimension across devices.
      Each device computes attention for seq_len // world_size tokens.
      """
      def __init__(self, attention: nn.Module, process_group: ProcessGroup):
          self.attention = attention
          self.pg = process_group

      def forward(self, x: Tensor) -> Tensor:
          # Scatter sequence dimension
          x_local = scatter_sequence(x, self.pg)
          # Local attention
          out_local = self.attention(x_local)
          # Gather outputs
          return gather_sequence(out_local, self.pg)
  ```
  **File:** `src/deepseek/torch/distributed/sequence_parallel.py` (new)
  **Priority:** High | **Complexity:** High

- [ ] **2.2.2** Implement context parallelism for 128K+ sequences
  ```python
  # src/deepseek/torch/distributed/context_parallel.py
  class ContextParallelGroup:
      """
      Context parallelism for ultra-long sequences (128K+).

      Architecture:
      - Split sequence into chunks across devices
      - Local attention within chunk
      - Ring all-reduce for cross-chunk KV sharing
      - Global token attention for long-range dependencies
      """
      def __init__(self, world_size: int, chunk_size: int = 32768):
          self.world_size = world_size
          self.chunk_size = chunk_size
          self.kv_comm_stream = torch.cuda.Stream()
  ```
  **File:** `src/deepseek/torch/distributed/context_parallel.py` (new)
  **Priority:** Critical | **Complexity:** Very High

- [ ] **2.2.3** Create 5D parallelism orchestrator
  ```python
  # src/deepseek/torch/distributed/parallelism_5d.py
  @dataclass
  class Parallelism5DConfig:
      data_parallel_size: int = 8
      tensor_parallel_size: int = 4
      pipeline_parallel_size: int = 4
      sequence_parallel_size: int = 2
      context_parallel_size: int = 2

      @property
      def total_devices(self) -> int:
          return (self.data_parallel_size *
                  self.tensor_parallel_size *
                  self.pipeline_parallel_size)  # seq/ctx overlap with tensor
  ```
  **File:** `src/deepseek/torch/distributed/parallelism_5d.py` (new)
  **Priority:** High | **Complexity:** High

- [ ] **2.2.4** Integrate Expert Parallelism with MoE routing
  - File: `src/deepseek/torch/model/expert_parallel.py`
  - Current: All-to-all token routing concept
  - Target: Proper hierarchical communication pattern
  ```
  EP=2: Experts 0-127 on rank 0, Experts 128-255 on rank 1
  All-to-all within DP group, reduce-scatter across EP group
  ```
  **Priority:** High | **Complexity:** High

- [ ] **2.2.5** Implement ZeRO-3 integration with DeepSpeed
  - File: `src/deepseek/cloud/modal/distributed_trainer.py:1252-1276`
  - Current: ZeRO config defined but not fully wired
  - Target: Enable `offload_param.device: "cpu"` for ZeRO-Infinity
  - Verification: Training on model larger than single GPU memory
  **Priority:** High | **Complexity:** Medium

### 2.3 DualPipe with 1F1B Schedule

#### Tasks

- [ ] **2.3.1** Port Rust DualPipe scheduler to PyTorch
  ```python
  # src/deepseek/torch/distributed/dualpipe.py
  class DualPipeScheduler:
      """
      Python implementation matching rust-src/src/distributed/pipeline.rs
      for cross-validation.
      """
      def __init__(self, num_stages: int, num_micro_batches: int):
          self.phase = DualPipePhase.WARMUP
          self.regular_stream = PipelineStream(direction="forward")
          self.reverse_stream = PipelineStream(direction="backward")

      def get_next_action(self) -> DualPipeAction:
          """Returns (stage_id, micro_batch_id, is_forward, stream)"""
          ...
  ```
  **File:** `src/deepseek/torch/distributed/dualpipe.py`
  **Priority:** High | **Complexity:** High

- [ ] **2.3.2** Implement interleaved 1F1B schedule
  ```python
  class Interleaved1F1B:
      """
      Interleaved schedule for reduced memory:
      - Virtual pipeline stages: physical_stages * interleave_factor
      - Each device holds multiple non-contiguous stages
      """
      def __init__(self, num_stages: int, interleave_factor: int = 2):
          self.virtual_stages = num_stages * interleave_factor
  ```
  **File:** `src/deepseek/torch/distributed/schedules.py` (new)
  **Priority:** Medium | **Complexity:** High

- [ ] **2.3.3** Add gradient accumulation with pipeline
  ```python
  class PipelineGradientAccumulator:
      """Accumulates gradients across micro-batches within pipeline."""
      def __init__(self, accumulation_steps: int):
          self.accumulation_steps = accumulation_steps
          self.grad_buffer: Dict[str, Tensor] = {}
  ```
  **File:** `src/deepseek/torch/distributed/dualpipe.py`
  **Priority:** Medium | **Complexity:** Medium

- [ ] **2.3.4** Add activation stashing with memory budget
  - File: `src/deepseek/torch/training/dualpipe.py`
  - Current: Basic activation storage
  - Target: LRU eviction when exceeding `max_activation_memory_gb`
  **Priority:** Medium | **Complexity:** Medium

---

## 3. Python + MLX Backend

### 3.1 MLA Porting with Exact Parity

**Current State:** `src/deepseek/mlx/ane/attention/mla.py` implements ANE-optimized MLA with chunked attention.

#### Tasks

- [ ] **3.1.1** Verify RoPE frequency computation matches PyTorch/Rust
  ```python
  # src/deepseek/mlx/ane/attention/rope.py
  def verify_rope_parity():
      """
      Critical: freqs must match across backends.
      Formula: freqs[i] = 1.0 / (base ** (2*i / d_head))

      Check NTK-aware scaling:
      - base_scaled = base * ((seq_len / base_seq_len) ** (d_head / (d_head - 2)))
      """
      mlx_freqs = compute_rope_freqs_mlx(d_head=64, base=10000.0)
      torch_freqs = compute_rope_freqs_torch(d_head=64, base=10000.0)
      assert mx.allclose(mlx_freqs, mx.array(torch_freqs.numpy()), rtol=1e-5)
  ```
  **File:** `src/deepseek/mlx/ane/attention/rope.py`
  **Priority:** Critical | **Complexity:** Low

- [ ] **3.1.2** Optimize KV cache for ANE memory constraints
  ```python
  # src/deepseek/mlx/ane/kv_cache/optimized.py
  class ANEOptimizedKVCache:
      """
      ANE has 32GB unified memory limit. Optimize for:
      - Latent-only storage (512 dims vs 4096)
      - FP16 throughout
      - Sliding window eviction for sequences > 32K
      """
      def __init__(self, max_memory_gb: float = 24.0):
          self.max_tokens = self._compute_max_tokens(max_memory_gb)
          self.eviction_policy = "sliding_window"
  ```
  **File:** `src/deepseek/mlx/ane/kv_cache/optimized.py` (new)
  **Priority:** High | **Complexity:** Medium

- [ ] **3.1.3** Add INT8 quantized attention path
  ```python
  # Extend ANEMLAConfig
  @dataclass
  class ANEMLAConfig:
      # ... existing fields ...
      use_int8_kv: bool = False  # INT8 KV cache for 2x memory reduction
      use_int8_weights: bool = False  # INT8 weights for ANE efficiency
  ```
  **File:** `src/deepseek/mlx/ane/attention/mla.py`
  **Priority:** Medium | **Complexity:** Medium

- [ ] **3.1.4** Implement extended RoPE scaling for 128K context
  - File: `src/deepseek/mlx/attention.py`
  - Current: Standard RoPE
  - Target: NTK-Aware and YaRN scaling
  ```python
  if scaling_type == "ntk":
      base = base * (scaling_factor ** (d_rope / (d_rope - 2)))
  ```
  **Priority:** High | **Complexity:** Medium

### 3.2 Sparse MoE Gating

**Current State:** `ANEMoE` in `moe.py` supports FUSED/BATCHED/HIERARCHICAL strategies.

#### Tasks

- [ ] **3.2.1** Fix load balance coefficient for training mode
  ```python
  # src/deepseek/mlx/ane/moe/moe.py:45
  # Current: load_balance_alpha=0.001
  # DeepSeek-V3 paper: 0.01 for training

  @dataclass
  class ANEMoEConfig:
      load_balance_alpha: float = 0.01  # Fix: match paper
      mode: Literal["train", "inference"] = "inference"

      def __post_init__(self):
          if self.mode == "inference":
              self.load_balance_alpha = 0.0  # No aux loss at inference
  ```
  **File:** `src/deepseek/mlx/ane/moe/moe.py`
  **Priority:** High | **Complexity:** Low

- [ ] **3.2.2** Implement auxiliary load balancing loss
  ```python
  def compute_load_balance_loss(
      router_probs: mx.array,  # [batch, seq, num_experts]
      expert_mask: mx.array,   # [batch, seq, num_experts] one-hot
      num_experts: int,
  ) -> mx.array:
      """
      L_balance = alpha * num_experts * sum(f_i * P_i)
      where f_i = fraction of tokens to expert i
            P_i = mean router prob for expert i
      """
      f = expert_mask.mean(axis=(0, 1))  # [num_experts]
      P = router_probs.mean(axis=(0, 1))  # [num_experts]
      return num_experts * (f * P).sum()
  ```
  **File:** `src/deepseek/mlx/ane/moe/load_balance.py` (new)
  **Priority:** High | **Complexity:** Medium

- [ ] **3.2.3** Add expert capacity factor tuning
  ```python
  @dataclass
  class ANEMoEConfig:
      # ... existing ...
      capacity_factor: float = 1.25  # Default
      drop_tokens: bool = True  # Drop vs pad overflow

      def get_expert_capacity(self, batch_size: int, seq_len: int) -> int:
          """Tokens per expert = (batch * seq * top_k * capacity_factor) / num_experts"""
          return int(
              (batch_size * seq_len * self.top_k * self.capacity_factor)
              / self.num_routed_experts
          )
  ```
  **File:** `src/deepseek/mlx/ane/moe/moe.py`
  **Priority:** Medium | **Complexity:** Low

- [ ] **3.2.4** Implement hierarchical routing for 256 experts (production)
  ```python
  class HierarchicalRouter(nn.Module):
      """
      Two-level routing for 256 experts:
      1. Route to expert group (16 groups of 16 experts)
      2. Route within group

      This reduces router computation from O(256) to O(16 + 16) = O(32)
      """
      def __init__(self, d_model: int, num_groups: int = 16, experts_per_group: int = 16):
          self.group_router = nn.Linear(d_model, num_groups)
          self.expert_routers = [nn.Linear(d_model, experts_per_group) for _ in range(num_groups)]
  ```
  **File:** `src/deepseek/mlx/ane/moe/hierarchical.py` (new)
  **Priority:** Medium | **Complexity:** High

- [ ] **3.2.5** Implement fused softmax + top-k for MLX
  - File: `src/deepseek/mlx/moe.py`
  - Current: Separate operations
  - Target: Use `mlx.core.custom_function` for fusion
  - Expected speedup: 1.5-2x on M-series chips
  **Priority:** Medium | **Complexity:** Medium

### 3.3 DeepSeek Sparse Attention (DSA) for V3.2

#### Tasks

- [ ] **3.3.1** Implement local window attention for MLX
  - File: `src/deepseek/mlx/sparse_attention.py`
  - Target: 4K token local window with efficient masking
  ```python
  local_mask = mx.triu(mx.ones((seq, seq)), k=-window_size)
  local_mask = mx.tril(local_mask, k=window_size)
  ```
  **Priority:** High | **Complexity:** Medium

- [ ] **3.3.2** Implement dilated global sampling pattern
  - File: `src/deepseek/mlx/sparse_attention.py`
  - Target: Sample every 8th token globally (512-1K global tokens)
  ```python
  global_indices = mx.arange(0, seq_len, dilation_factor)
  global_kv = mx.take(kv, global_indices, axis=1)
  ```
  **Priority:** High | **Complexity:** Medium

- [ ] **3.3.3** Integrate DSA with ANE (Apple Neural Engine)
  - File: `src/deepseek/mlx/ane/sparse_attention_ane.py` (new)
  - Current: ANE supports chunked attention only
  - Target: DSA-aware chunking with ANE acceleration
  **Priority:** Medium | **Complexity:** High

---

## 4. Distributed Infrastructure

### 4.1 UV Environment Configuration

#### Tasks

- [ ] **4.1.1** Create unified `pyproject.toml` with all backends
  ```toml
  # pyproject.toml
  [project]
  name = "deepseek-from-scratch"
  version = "0.1.0"
  requires-python = ">=3.11"

  [project.optional-dependencies]
  torch = ["torch>=2.2.0", "torchvision", "torchaudio"]
  mlx = ["mlx>=0.5.0", "mlx-lm"]
  cuda = ["nvidia-cudnn-cu12", "triton>=2.2.0"]
  distributed = ["ray[default]>=2.9.0", "deepspeed>=0.14.0"]
  modal = ["modal>=0.60.0"]
  dev = ["pytest", "pytest-asyncio", "ruff", "mypy"]
  all = ["deepseek-from-scratch[torch,mlx,distributed,modal,dev]"]

  [tool.uv]
  preview = true
  ```
  **File:** `pyproject.toml`
  **Priority:** High | **Complexity:** Low

- [ ] **4.1.2** Add `uv.lock` sync verification to CI
  ```yaml
  # .github/workflows/ci.yml
  - name: Verify uv.lock is up to date
    run: |
      uv lock --check
      uv sync --all-extras
  ```
  **File:** `.github/workflows/ci.yml`
  **Priority:** Medium | **Complexity:** Low

- [ ] **4.1.3** Create backend-specific virtual environments
  ```bash
  # scripts/setup_envs.sh
  #!/bin/bash
  uv venv .venv-torch --python 3.11
  uv venv .venv-mlx --python 3.11

  source .venv-torch/bin/activate && uv sync --extra torch --extra distributed
  source .venv-mlx/bin/activate && uv sync --extra mlx
  ```
  **File:** `scripts/setup_envs.sh` (new)
  **Priority:** Low | **Complexity:** Low

### 4.2 Ray Pipeline Orchestration

**Current State:** `src/deepseek/cloud/modal/ray_cluster.py` exists but lacks heterogeneous support.

#### Tasks

- [ ] **4.2.1** Implement heterogeneous placement groups
  ```python
  # src/deepseek/cloud/modal/ray_cluster.py
  from ray.util.placement_group import placement_group, PlacementGroup

  def create_hybrid_placement_group(
      mac_studio_count: int = 1,
      a100_count: int = 8,
  ) -> PlacementGroup:
      """
      Create placement group for Mac Studio + A100 cluster.

      Mac Studio: MLX inference, data preprocessing
      A100: Training forward/backward
      """
      bundles = [
          {"CPU": 10, "MAC_STUDIO": 1},  # Mac Studio node
          *[{"GPU": 1, "CPU": 4} for _ in range(a100_count)],  # A100 nodes
      ]
      return placement_group(bundles, strategy="STRICT_SPREAD")
  ```
  **File:** `src/deepseek/cloud/modal/ray_cluster.py`
  **Priority:** High | **Complexity:** Medium

- [ ] **4.2.2** Add context parallel group to Ray
  ```python
  # src/deepseek/cloud/modal/ray_cluster.py
  @ray.remote(num_gpus=1)
  class ContextParallelWorker:
      """Worker for context parallelism in Ray."""

      def __init__(self, rank: int, world_size: int, chunk_size: int):
          self.rank = rank
          self.world_size = world_size
          self.chunk_size = chunk_size
          self.kv_buffer = KVBuffer(max_size=chunk_size * 2)

      def process_chunk(self, hidden: np.ndarray, position_offset: int):
          """Process local chunk and return KV for sharing."""
          ...

      def receive_remote_kv(self, remote_kv: np.ndarray, source_rank: int):
          """Receive KV from adjacent context parallel rank."""
          ...
  ```
  **File:** `src/deepseek/cloud/modal/ray_cluster.py`
  **Priority:** High | **Complexity:** High

- [ ] **4.2.3** Create Ray-based DualPipe orchestrator
  ```python
  # src/deepseek/cloud/modal/dualpipe_ray.py
  @ray.remote
  class DualPipeStage:
      """Ray actor for single pipeline stage."""

      def __init__(self, stage_id: int, model_chunk: nn.Module):
          self.stage_id = stage_id
          self.model = model_chunk
          self.activation_buffer = {}

      async def forward(self, micro_batch_id: int, hidden: np.ndarray):
          """Forward pass, store activation for backward."""
          ...

      async def backward(self, micro_batch_id: int, grad: np.ndarray):
          """Backward pass using stored activation."""
          ...

  class DualPipeOrchestrator:
      """Coordinates DualPipe stages across Ray actors."""

      def __init__(self, stages: List[DualPipeStage], scheduler: DualPipeScheduler):
          self.stages = stages
          self.scheduler = scheduler

      async def train_step(self, micro_batches: List[np.ndarray]):
          """Execute full DualPipe train step."""
          ...
  ```
  **File:** `src/deepseek/cloud/modal/dualpipe_ray.py` (new)
  **Priority:** High | **Complexity:** High

- [ ] **4.2.4** Implement checkpoint sharding across workers
  - File: `src/deepseek/torch/training/distributed_checkpoint.py`
  - Current: Full model checkpoint per node
  - Target: Sharded checkpointing with Ray object store
  ```python
  ray.put(model.state_dict_shard(rank))
  ```
  **Priority:** Medium | **Complexity:** Medium

### 4.3 Modal GPU Configuration

#### Tasks

- [ ] **4.3.1** Update Modal app for 8-GPU DualPipe verification
  ```python
  # src/deepseek/cloud/modal/app.py
  import modal

  app = modal.App("deepseek-dualpipe-verify")

  gpu_image = modal.Image.debian_slim().pip_install(
      "torch>=2.2.0",
      "ray[default]>=2.9.0",
      "deepseek-from-scratch[distributed]",
  )

  @app.function(
      gpu=modal.gpu.A100(count=8, memory=80),  # 8x 80GB A100
      image=gpu_image,
      timeout=3600,
  )
  def verify_dualpipe_8gpu():
      """Run 8-GPU DualPipe verification."""
      import ray
      ray.init()

      # Create 8 pipeline stages
      stages = [DualPipeStage.remote(i, load_model_chunk(i)) for i in range(8)]
      orchestrator = DualPipeOrchestrator(stages, DualPipeScheduler(8, 32))

      # Run verification
      metrics = orchestrator.train_step(dummy_micro_batches(32))
      assert metrics.bubble_ratio < 0.05, f"Bubble ratio too high: {metrics.bubble_ratio}"
  ```
  **File:** `src/deepseek/cloud/modal/app.py`
  **Priority:** High | **Complexity:** Medium

- [ ] **4.3.2** Add Prometheus metrics export
  ```python
  # src/deepseek/cloud/modal/metrics.py
  from prometheus_client import Gauge, Histogram, start_http_server

  dualpipe_bubble_ratio = Gauge(
      'dualpipe_bubble_ratio',
      'Pipeline bubble ratio (0-1)'
  )
  dualpipe_throughput = Gauge(
      'dualpipe_throughput_tokens_per_sec',
      'Training throughput in tokens/second'
  )
  expert_load = Histogram(
      'expert_load_tokens',
      'Tokens per expert',
      buckets=[10, 50, 100, 200, 500, 1000]
  )
  ```
  **File:** `src/deepseek/cloud/modal/metrics.py` (new)
  **Priority:** Medium | **Complexity:** Low

- [ ] **4.3.3** Create Modal secrets for wandb/telemetry
  ```python
  # scripts/setup_modal_secrets.py
  import modal

  modal.Secret.from_name(
      "deepseek-training-secrets",
      {
          "WANDB_API_KEY": "...",
          "HF_TOKEN": "...",
          "PROMETHEUS_PUSHGATEWAY": "...",
      }
  )
  ```
  **File:** `scripts/setup_modal_secrets.py` (new)
  **Priority:** Low | **Complexity:** Low

---

## 5. Testing & Verification

### 5.1 Cross-Backend Logit Comparison

#### Tasks

- [ ] **5.1.1** Create parity test harness
  ```python
  # tests/parity/test_cross_backend.py
  import pytest
  import torch
  import mlx.core as mx
  import numpy as np

  from deepseek.torch.model import DeepSeekV3 as TorchModel
  from deepseek.mlx.model import DeepSeekV3 as MLXModel
  # Rust via PyO3
  from deepseek_rust import DeepSeekV3 as RustModel

  @pytest.fixture
  def shared_weights():
      """Load identical weights for all backends."""
      return load_checkpoint("checkpoints/v3-base.safetensors")

  @pytest.fixture
  def test_input():
      """Fixed input for reproducibility."""
      return {
          "input_ids": [1, 2, 3, 4, 5],  # Token IDs
          "seed": 42,
      }

  def test_logit_parity(shared_weights, test_input):
      """All backends must produce identical logits (within tolerance)."""
      torch_model = TorchModel.from_pretrained(shared_weights)
      mlx_model = MLXModel.from_pretrained(shared_weights)
      rust_model = RustModel.from_pretrained(shared_weights)

      # Run inference
      torch_logits = torch_model(test_input["input_ids"]).detach().numpy()
      mlx_logits = np.array(mlx_model(test_input["input_ids"]))
      rust_logits = rust_model(test_input["input_ids"])

      # Verify parity
      rtol = 1e-4
      np.testing.assert_allclose(torch_logits, mlx_logits, rtol=rtol)
      np.testing.assert_allclose(torch_logits, rust_logits, rtol=rtol)
  ```
  **File:** `tests/parity/test_cross_backend.py` (new)
  **Priority:** Critical | **Complexity:** Medium

- [ ] **5.1.2** Add layer-by-layer parity verification
  ```python
  # tests/parity/test_layer_parity.py
  @pytest.mark.parametrize("layer_type", [
      "embedding",
      "rope",
      "attention",
      "mla",
      "moe_gate",
      "moe_expert",
      "ffn",
      "layernorm",
  ])
  def test_layer_parity(layer_type, shared_weights, test_input):
      """Test individual layer parity."""
      torch_layer = get_torch_layer(layer_type, shared_weights)
      mlx_layer = get_mlx_layer(layer_type, shared_weights)

      torch_out = torch_layer(test_input).detach().numpy()
      mlx_out = np.array(mlx_layer(test_input))

      np.testing.assert_allclose(torch_out, mlx_out, rtol=1e-4)
  ```
  **File:** `tests/parity/test_layer_parity.py` (new)
  **Priority:** High | **Complexity:** Medium

- [ ] **5.1.3** Create numerical stability test suite
  ```python
  # tests/parity/test_numerical_stability.py
  def test_softmax_stability():
      """Test softmax doesn't overflow with large inputs."""
      large_input = torch.randn(1, 1024, 32000) * 100  # Large vocab
      result = F.softmax(large_input, dim=-1)
      assert not torch.isnan(result).any()
      assert not torch.isinf(result).any()

  def test_rope_long_sequence():
      """Test RoPE doesn't degrade at 128K positions."""
      positions = torch.arange(131072)
      freqs = compute_rope_freqs(d_head=64, max_seq_len=131072)
      rope_embeds = apply_rope(freqs, positions)
      assert not torch.isnan(rope_embeds).any()
  ```
  **File:** `tests/parity/test_numerical_stability.py` (new)
  **Priority:** High | **Complexity:** Medium

- [ ] **5.1.4** Add MoE routing parity test
  - Verify expert selection is identical across backends
  - Test with 256 experts, hierarchical routing
  **File:** `tests/parity/test_moe_routing.py` (new)
  **Priority:** High | **Complexity:** Medium

- [ ] **5.1.5** Add DualPipe schedule parity test
  - Verify Rust and PyTorch schedulers produce identical action sequences
  **File:** `tests/parity/test_dualpipe_schedule.py` (new)
  **Priority:** Medium | **Complexity:** Low

### 5.2 DualPipe Bubble Verification

#### Tasks

- [ ] **5.2.1** Create bubble ratio benchmark
  ```python
  # tests/distributed/test_dualpipe_bubble.py
  import pytest
  from deepseek.torch.distributed.dualpipe import DualPipeScheduler, DualPipeMetrics

  @pytest.mark.parametrize("num_stages,num_micro_batches", [
      (4, 16),
      (8, 32),
      (8, 64),
      (16, 128),
  ])
  def test_bubble_ratio(num_stages, num_micro_batches):
      """Verify bubble ratio < 5% for standard configs."""
      scheduler = DualPipeScheduler(num_stages, num_micro_batches)
      metrics = scheduler.simulate()

      # DeepSeek-V3 claims < 3% bubble ratio
      assert metrics.bubble_ratio < 0.05, (
          f"Bubble ratio {metrics.bubble_ratio:.2%} exceeds threshold for "
          f"{num_stages} stages, {num_micro_batches} micro-batches"
      )
  ```
  **File:** `tests/distributed/test_dualpipe_bubble.py` (new)
  **Priority:** High | **Complexity:** Medium

- [ ] **5.2.2** Add tracing spans for bubble measurement
  ```python
  # src/deepseek/torch/distributed/dualpipe.py
  from opentelemetry import trace

  tracer = trace.get_tracer("deepseek.dualpipe")

  class DualPipeEngine:
      def train_step(self, micro_batches):
          with tracer.start_as_current_span("dualpipe_train_step") as span:
              span.set_attribute("num_micro_batches", len(micro_batches))

              for action in self.scheduler:
                  with tracer.start_as_current_span(f"stage_{action.stage_id}"):
                      if action.is_forward:
                          self._forward(action)
                      else:
                          self._backward(action)
  ```
  **File:** `src/deepseek/torch/distributed/dualpipe.py`
  **Priority:** Medium | **Complexity:** Low

- [ ] **5.2.3** Create Modal verification job
  ```python
  # tests/integration/test_modal_dualpipe.py
  import modal
  import pytest

  @pytest.mark.modal
  def test_dualpipe_on_modal():
      """Run DualPipe verification on Modal 8xA100."""
      from deepseek.cloud.modal.app import verify_dualpipe_8gpu

      result = verify_dualpipe_8gpu.remote()
      assert result["bubble_ratio"] < 0.05
      assert result["throughput_tokens_per_sec"] > 10000
  ```
  **File:** `tests/integration/test_modal_dualpipe.py` (new)
  **Priority:** High | **Complexity:** Low

### 5.3 Convergence Pilot Runs

#### Tasks

- [ ] **5.3.1** Create convergence test config
  ```yaml
  # configs/convergence_pilot.yaml
  model:
    architecture: deepseek_v3
    hidden_size: 4096
    num_layers: 32
    num_experts: 32
    top_k: 4

  training:
    batch_size: 512
    max_steps: 1000
    learning_rate: 1e-4
    warmup_steps: 100

  verification:
    log_every_n_steps: 10
    convergence_threshold: 0.01  # Loss decrease per 100 steps

  backends:
    - torch_cuda
    - torch_mps
    - mlx
    - rust_cuda
  ```
  **File:** `configs/convergence_pilot.yaml` (new)
  **Priority:** Medium | **Complexity:** Low

- [ ] **5.3.2** Implement convergence monitoring
  ```python
  # src/deepseek/training/convergence.py
  from dataclasses import dataclass
  from typing import List
  import numpy as np

  @dataclass
  class ConvergenceMetrics:
      losses: List[float]
      learning_rates: List[float]
      gradient_norms: List[float]

      def is_converging(self, window: int = 100, threshold: float = 0.01) -> bool:
          """Check if loss is decreasing."""
          if len(self.losses) < window * 2:
              return True  # Not enough data
          recent = np.mean(self.losses[-window:])
          older = np.mean(self.losses[-2*window:-window])
          return (older - recent) / older > threshold

      def detect_divergence(self) -> bool:
          """Detect if training is diverging."""
          return any(np.isnan(self.losses)) or any(np.isinf(self.losses))
  ```
  **File:** `src/deepseek/training/convergence.py` (new)
  **Priority:** Medium | **Complexity:** Medium

- [ ] **5.3.3** Create multi-backend convergence comparison
  ```python
  # tests/convergence/test_multi_backend.py
  import pytest
  from deepseek.training.trainer import Trainer
  from deepseek.training.convergence import ConvergenceMetrics

  @pytest.mark.slow
  @pytest.mark.parametrize("backend", ["torch_cuda", "torch_mps", "mlx"])
  def test_convergence_1k_steps(backend, convergence_config):
      """Verify all backends converge similarly on 1K steps."""
      trainer = Trainer(backend=backend, config=convergence_config)
      metrics = trainer.train(max_steps=1000)

      assert metrics.is_converging(), f"{backend} not converging"
      assert not metrics.detect_divergence(), f"{backend} diverging"
      assert metrics.losses[-1] < metrics.losses[0] * 0.5, (
          f"{backend} loss not decreasing enough"
      )
  ```
  **File:** `tests/convergence/test_multi_backend.py` (new)
  **Priority:** Medium | **Complexity:** Medium

- [ ] **5.3.4** Run convergence pilot on TinyShakespeare
  ```python
  # scripts/convergence_pilot.py
  def run_convergence_pilot(backend: str = "pytorch"):
      """
      Train on TinyShakespeare for 1000 steps.
      Assert loss decreases from ~4.0 to <2.5.
      """
      data = load_tiny_shakespeare()
      model = build_model(hidden_size=256, num_layers=4)

      initial_loss = evaluate(model, data)
      train(model, data, steps=1000)
      final_loss = evaluate(model, data)

      assert initial_loss > 3.5, f"Initial loss too low: {initial_loss}"
      assert final_loss < 2.5, f"Model didn't converge: {final_loss}"
  ```
  **File:** `scripts/convergence_pilot.py` (new)
  **Priority:** Medium | **Complexity:** Low

### 5.4 Modal GPU Verification Matrix

| Test | Command | Success Criteria |
|------|---------|------------------|
| PyTorch Single GPU | `uv run modal run ray_cluster.py::run_pytorch --scale initial --max-steps 50` | `status: verified`, loss < 10.0 |
| PyTorch Multi-GPU | `uv run modal run ray_cluster.py::run_pytorch --scale initial --max-steps 100` | 8 GPUs used, throughput > 10K tok/s |
| Rust CUDA Build | `uv run modal run ray_cluster.py::run_rust --max-steps 50` | `rust_built: true`, `cuda_available: true` |
| Rust DualPipe Tests | (within run_rust) | `dualpipe_tests.passed: true` |
| 5D Parallelism | `uv run modal run ray_cluster.py::run_full_pipeline --scale initial` | All ranks mapping verified |

---

## Appendix: Architecture Diagrams

### A.1 5D Parallelism Topology

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           5D Parallelism Topology                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Data Parallel (DP=8)                                                        │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬────┐
│  │ Replica │ Replica │ Replica │ Replica │ Replica │ Replica │ Replica │Rep │
│  │    0    │    1    │    2    │    3    │    4    │    5    │    6    │  7 │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┘
│       │         │         │         │         │         │         │          │
│       ▼         ▼         ▼         ▼         ▼         ▼         ▼          │
│  Tensor Parallel (TP=4) within each replica                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐                                         │   │
│  │  │GPU0│ │GPU1│ │GPU2│ │GPU3│  (Attention heads split)                │   │
│  │  └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘                                         │   │
│  │     └──────┴──────┴──────┘                                            │   │
│  │              All-Reduce                                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Pipeline Parallel (PP=4) - DualPipe                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  Regular Stream:  Stage0 ──► Stage1 ──► Stage2 ──► Stage3             │ │
│  │                      │         │         │         │                   │ │
│  │                      ▼         ▼         ▼         ▼                   │ │
│  │  Reverse Stream:  Stage0 ◄── Stage1 ◄── Stage2 ◄── Stage3             │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Sequence Parallel (SP=2) - Attention                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Sequence: [0:32K]              Sequence: [32K:64K]                    │ │
│  │  ┌──────────────────┐          ┌──────────────────┐                   │ │
│  │  │  Local Attention │  ◄────►  │  Local Attention │  (Ring All-Reduce)│ │
│  │  └──────────────────┘          └──────────────────┘                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Context Parallel (CP=2) - 128K+ Sequences                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Context Chunk: [0:64K]        Context Chunk: [64K:128K]               │ │
│  │  ┌──────────────────┐          ┌──────────────────┐                   │ │
│  │  │  KV Cache Chunk  │  ◄────►  │  KV Cache Chunk  │  (KV Sharing)     │ │
│  │  └──────────────────┘          └──────────────────┘                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### A.2 Zero-Bubble DualPipe Schedule

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Zero-Bubble DualPipe Schedule (8 stages)                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Time ─────────────────────────────────────────────────────────────────────► │
│                                                                              │
│  Stage 0: │F0│F1│F2│F3│  │B7│B6│B5│B4│B3│B2│B1│B0│                          │
│  Stage 1:    │F0│F1│F2│F3│  │B7│B6│B5│B4│B3│B2│B1│B0│                       │
│  Stage 2:       │F0│F1│F2│F3│  │B7│B6│B5│B4│B3│B2│B1│B0│                    │
│  Stage 3:          │F0│F1│F2│F3│  │B7│B6│B5│B4│B3│B2│B1│B0│                 │
│                                 ↕ Overlap Communication                      │
│  Stage 4:          │F7│F6│F5│F4│  │B0│B1│B2│B3│B4│B5│B6│B7│                 │
│  Stage 5:       │F7│F6│F5│F4│  │B0│B1│B2│B3│B4│B5│B6│B7│                    │
│  Stage 6:    │F7│F6│F5│F4│  │B0│B1│B2│B3│B4│B5│B6│B7│                       │
│  Stage 7: │F7│F6│F5│F4│  │B0│B1│B2│B3│B4│B5│B6│B7│                          │
│                                                                              │
│  Legend:                                                                     │
│  F# = Forward pass for micro-batch #                                         │
│  B# = Backward pass for micro-batch #                                        │
│     = Bubble (minimized via overlapping)                                     │
│                                                                              │
│  Key Optimization: Bidirectional streams eliminate warmup/cooldown bubbles   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### A.3 MLA KV Compression Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Latent Attention (MLA) Flow                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: x [batch, seq, 4096]                                                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                          Query Path                                     ││
│  │  x ──► Q_content_proj ──► [batch, seq, heads, d_head] ────────┐        ││
│  │    │                                                           │        ││
│  │    └► Q_rope_proj ──► RoPE ──► [batch, seq, heads, d_rope] ──┐│        ││
│  │                                                               ││        ││
│  │                                        ┌──────────────────────┘│        ││
│  │                                        │                       │        ││
│  │                                        ▼                       ▼        ││
│  │                              Q = concat(Q_content, Q_rope)              ││
│  │                              [batch, heads, seq, d_head+d_rope]         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      KV Compression Path (14-16x savings)               ││
│  │                                                                         ││
│  │  x ──► KV_down_proj ──► C_kv [batch, seq, 512] ◄── Latent (cached)     ││
│  │                              │                                          ││
│  │              ┌───────────────┼───────────────┐                          ││
│  │              │               │               │                          ││
│  │              ▼               ▼               ▼                          ││
│  │         K_content_up    K_rope_up        V_up                           ││
│  │              │               │               │                          ││
│  │              ▼               ▼               ▼                          ││
│  │    [batch,heads,seq,d_head] RoPE   [batch,heads,seq,d_head]            ││
│  │              │               │               │                          ││
│  │              └───────┬───────┘               │                          ││
│  │                      ▼                       │                          ││
│  │            K = concat(K_content, K_rope)     │                          ││
│  │                      │                       │                          ││
│  └──────────────────────│───────────────────────│──────────────────────────┘│
│                         │                       │                           │
│                         ▼                       ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        Chunked Attention (ANE)                          ││
│  │                                                                         ││
│  │   Attention(Q, K, V) computed in 128-token chunks                       ││
│  │   ┌────┐ ┌────┐ ┌────┐ ┌────┐                                          ││
│  │   │Ch 0│ │Ch 1│ │Ch 2│ │... │  Parallel on ANE                         ││
│  │   └────┘ └────┘ └────┘ └────┘                                          ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Memory Comparison (per token):                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Standard KV Cache:  2 × heads × d_head = 2 × 32 × 128 = 8192 bytes    │ │
│  │  MLA Latent Cache:   d_latent = 512 bytes                              │ │
│  │  Reduction:          8192 / 512 = 16x                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### A.4 MoE Expert Routing

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MoE Expert Routing (256 Experts)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: x [batch, seq, 4096]                                                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Hierarchical Router (2-level)                        ││
│  │                                                                         ││
│  │  Level 1: Group Selection (16 groups)                                   ││
│  │  x ──► Group_Router ──► softmax ──► top_2 ──► [group_0, group_7]       ││
│  │                                                                         ││
│  │  Level 2: Expert Selection (16 per group)                               ││
│  │  x ──► Expert_Router[group_0] ──► top_2 ──► [expert_3, expert_11]      ││
│  │  x ──► Expert_Router[group_7] ──► top_2 ──► [expert_2, expert_15]      ││
│  │                                                                         ││
│  │  Final: top_k=4 experts selected per token                              ││
│  │         Experts: [3, 11, 114, 127] with routing weights                 ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Expert Dispatch & Execution                          ││
│  │                                                                         ││
│  │  Token Routing Matrix:                                                  ││
│  │  ┌────────────────────────────────────────────────────────────────────┐││
│  │  │ Token │ Expert 3 │ Expert 11 │ Expert 114 │ Expert 127 │ ...      │││
│  │  ├────────────────────────────────────────────────────────────────────┤││
│  │  │   0   │   0.35   │   0.28    │    0.22    │    0.15    │          │││
│  │  │   1   │   0.00   │   0.41    │    0.31    │    0.28    │          │││
│  │  │  ...  │   ...    │   ...     │    ...     │    ...     │          │││
│  │  └────────────────────────────────────────────────────────────────────┘││
│  │                                                                         ││
│  │  Expert Execution (parallel):                                           ││
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                           ││
│  │  │Expert 3│ │Expert11│ │Exp 114 │ │Exp 127 │                           ││
│  │  │  FFN   │ │  FFN   │ │  FFN   │ │  FFN   │                           ││
│  │  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘                           ││
│  │      │          │          │          │                                 ││
│  │      └──────────┴──────────┴──────────┘                                 ││
│  │                      │                                                  ││
│  │                      ▼                                                  ││
│  │            Weighted Sum (routing weights)                               ││
│  │                      │                                                  ││
│  │                      ▼                                                  ││
│  │               Output [batch, seq, 4096]                                 ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Shared Experts (always active)                       ││
│  │                                                                         ││
│  │  x ──► Shared_Expert_0 ──► FFN ──┬──► + ──► Final Output               ││
│  │    └─► Shared_Expert_1 ──► FFN ──┘    ▲                                 ││
│  │                                       │                                 ││
│  │                          Routed Expert Output                           ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Load Balancing Loss:                                                        │
│  L_balance = α × N_experts × Σ(f_i × P_i)                                   │
│  where f_i = fraction of tokens to expert i                                 │
│        P_i = mean routing probability for expert i                          │
│        α = 0.01 (training) / 0.0 (inference)                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### A.5 Hybrid Ray Cluster Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Hybrid Ray Cluster Topology                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────┐        ┌──────────────────────────────────────┐│
│  │     Mac Studio         │        │      Modal GPU Cluster (8xA100)      ││
│  │     (M2 Ultra)         │        │                                      ││
│  │                        │        │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   ││
│  │  ┌──────────────────┐  │        │   │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│   ││
│  │  │   Ray Client     │  │        │   │Stg 0│ │Stg 1│ │Stg 2│ │Stg 3│   ││
│  │  │   (Orchestrator) │  │        │   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   ││
│  │  └────────┬─────────┘  │        │      │       │       │       │      ││
│  │           │            │◄──────►│      │  DualPipe Streams     │      ││
│  │  ┌────────▼─────────┐  │  Ray   │      │       │       │       │      ││
│  │  │   MLX Backend    │  │        │   ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐   ││
│  │  │   (Inference)    │  │        │   │GPU 4│ │GPU 5│ │GPU 6│ │GPU 7│   ││
│  │  └──────────────────┘  │        │   │Stg 4│ │Stg 5│ │Stg 6│ │Stg 7│   ││
│  │                        │        │   └─────┘ └─────┘ └─────┘ └─────┘   ││
│  │  ┌──────────────────┐  │        │                                      ││
│  │  │ Data Preprocessing│  │        │   NCCL All-Reduce Ring              ││
│  │  │ (CPU workers)    │  │        │   ───────────────────►               ││
│  │  └──────────────────┘  │        │                                      ││
│  └────────────────────────┘        └──────────────────────────────────────┘│
│                                                                             │
│  Data Flow:                                                                 │
│  1. Mac Studio: Data loading, tokenization, batching                       │
│  2. Modal: Forward/backward passes on GPU cluster                          │
│  3. Mac Studio: Checkpoint aggregation, MLX inference validation           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority Summary

### Critical Path (Must Complete First)

| Task ID | Description | Effort | Dependencies |
|---------|-------------|--------|--------------|
| 1.2.1 | Async DualPipe with tokio | High | None |
| 1.2.2 | Zero-bubble scheduling | Very High | 1.2.1 |
| 2.2.2 | Context parallelism | Very High | 2.2.1 |
| 5.1.1 | Cross-backend parity tests | Medium | All backends stable |
| 3.1.1 | RoPE parity verification | Low | None |

### High Priority

| Task ID | Description | Effort | Dependencies |
|---------|-------------|--------|--------------|
| 1.1.1-1.1.4 | Unified tensor abstraction | High | None |
| 2.1.2 | V3.2 sparse attention | High | None |
| 2.2.1 | Sequence parallelism | High | None |
| 4.2.1 | Heterogeneous Ray placement | Medium | None |
| 4.3.1 | Modal 8-GPU config | Medium | None |

### Medium Priority

| Task ID | Description | Effort | Dependencies |
|---------|-------------|--------|--------------|
| 1.3.1-1.3.3 | MoE expert kernels | High | 1.1.x |
| 3.2.1-3.2.4 | MLX MoE optimization | Medium | None |
| 5.2.1-5.2.3 | DualPipe bubble verification | Medium | 1.2.x, 2.3.x |
| 5.3.1-5.3.3 | Convergence pilot | Medium | 5.1.1 |

### Lower Priority

| Task ID | Description | Effort | Dependencies |
|---------|-------------|--------|--------------|
| 4.1.1-4.1.3 | UV environment setup | Low | None |
| 4.3.2-4.3.3 | Metrics/secrets setup | Low | None |
| 3.1.3 | INT8 quantized attention | Medium | 3.1.1 |

---

## Appendix B: Current Codebase Metrics

```
Total Files: 150+
Lines of Rust: ~12,000
Lines of Python: ~25,000
Test Coverage: ~70% (1,221 tests passing)
Documentation: 22+ architecture docs
```

### Key File Locations

| Component | Location |
|-----------|----------|
| Rust DualPipe | `rust-src/src/distributed/pipeline.rs` |
| Rust Device Selection | `rust-src/src/utils/device.rs` |
| Rust MoE | `rust-src/src/model/moe.rs` |
| Rust MLA | `rust-src/src/model/mla.rs` |
| PyTorch MLA | `src/deepseek/torch/model/mla.py` |
| PyTorch MoE | `src/deepseek/torch/model/moe.py` |
| MLX MLA (ANE) | `src/deepseek/mlx/ane/attention/mla.py` |
| MLX MoE (ANE) | `src/deepseek/mlx/ane/moe/moe.py` |
| Modal Trainer | `src/deepseek/cloud/modal/distributed_trainer.py` |
| Ray Cluster | `src/deepseek/cloud/modal/ray_cluster.py` |

---

## Appendix C: Cost Estimates

### Modal A100-80GB Training Costs

| Configuration | GPUs | Cost/Hour | 24hr Run |
|---------------|------|-----------|----------|
| Initial (8 GPU) | 8 | $22.40 | $537.60 |
| Scaled (64 GPU) | 64 | $179.20 | $4,300.80 |
| Full V3.2 (256 GPU) | 256 | $716.80 | $17,203.20 |

### Recommended Verification Budget

- **Phase 1**: $50 (initial 8-GPU verification, ~2 hours)
- **Phase 2**: $200 (scaled testing with checkpoints)
- **Phase 3**: $1000 (full convergence runs)

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12 | Principal AI Systems Architect | Initial plan |
| 2.0 | 2025-12-09 | Principal AI Systems Architect | Comprehensive update with multi-agent analysis |

---

*Generated via multi-agent deep research simulation with cross-validation between:*
- *Agent 1: Systems Engineer (Rust/Hardware)*
- *Agent 2: Algorithm Architect (Model/Math)*
- *Agent 3: Infrastructure Orchestrator (Ray/Distributed)*
