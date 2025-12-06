# DeepSeek from Scratch in Rust: Architecture & Implementation Guide

This repository contains a Rust implementation of the DeepSeek-V3 and DeepSeek-R1 architectures, following the "DeepSeek from Scratch" educational series.

## Table of Contents
1. [Chapter 1: Multi-Query & Grouped-Query Attention](#chapter-1-multi-query--grouped-query-attention)
2. [Chapter 2: Multi-Head Latent Attention (MLA)](#chapter-2-multi-head-latent-attention-mla)
3. [Chapter 3: Mixture-of-Experts (MoE)](#chapter-3-mixture-of-experts-moe)
4. [Chapter 4: Multi-Token Prediction & FP8](#chapter-4-multi-token-prediction--fp8)
5. [Chapter 5: DeepSeek-R1 (Reasoning)](#chapter-5-deepseek-r1-reasoning)
6. [Chapter 6: GRPO (Group Relative Policy Optimization)](#chapter-6-grpo-group-relative-policy-optimization)

---

## Chapter 1: Multi-Query & Grouped-Query Attention

### Theory
Standard Multi-Head Attention (MHA) has a large KV cache memory footprint because it stores separate Key and Value matrices for each head.
- **MQA (Multi-Query Attention)**: Uses a single Key and Value head shared across all Query heads. Drastically reduces memory but can degrade performance.
- **GQA (Grouped-Query Attention)**: A middle ground. Groups Query heads and shares KV heads within each group. Offers a better trade-off between memory and performance.

### Implementation (`src/attention.rs`)
- `MultiQueryAttention`: Implements MQA.
- `GroupedQueryAttention`: Implements GQA.
- **Key Detail**: We manually handle causal masking and ensure tensor contiguity for efficient matrix multiplications.

---

## Chapter 2: Multi-Head Latent Attention (MLA)

### Theory
MLA is DeepSeek's innovation to compress the KV cache further without sacrificing performance as much as MQA/GQA.
- **Concept**: Projects Keys and Values into a low-rank latent vector ($c_{KV}$) and then up-projects them for attention. This allows the model to store the compressed latent vector instead of the full KV matrices during inference.
- **RoPE**: Rotary Positional Embeddings are applied to a subset of the query/key dimensions ("pe" part) while the rest ("unpe" part) carry content information.

### Implementation (`src/mla.rs`)
- `MultiHeadLatentAttention`: The core MLA logic with down-projection and up-projection.
- `DeepSeekAttention`: Fuses MLA with RoPE. We split the query/key into `pe` (positional) and `unpe` (content) parts, applying RoPE only to the `pe` part.

---

## Chapter 3: Mixture-of-Experts (MoE)

### Theory
MoE scales model capacity (parameters) without increasing inference cost (FLOPs) linearly.
- **DeepSeekMoE**: Introduces two key innovations:
    1.  **Shared Experts**: A few experts are *always* active for every token. This captures common knowledge.
    2.  **Fine-Grained Routed Experts**: Many small experts are selectively activated by a router.
- **Load Balancing**: A bias term is added to the router logits to prevent expert collapse (where only a few experts get all the tokens).

### Implementation (`src/moe.rs`)
- `DeepSeekMoE`: Implements the Shared + Routed expert architecture.
- `StandardMoE`: A baseline MoE for comparison.
- **Benchmark**: `src/moe_benchmark.rs` compares the two. DeepSeekMoE is slightly heavier per token due to shared experts but offers better training stability and performance.

---

## Chapter 4: Multi-Token Prediction & FP8

### Theory
- **MTP (Multi-Token Prediction)**: Instead of predicting just the next token $t+1$, the model predicts $t+1, t+2, ..., t+k$ sequentially. This densifies training signals and can be used for speculative decoding inference.
- **FP8 Quantization**: Using 8-bit floating point (E4M3) for weights and activations to double throughput and halve memory. DeepSeek uses **Fine-Grained (Tile-based)** quantization to handle outliers.

### Implementation
- `src/mtp.rs`: `MTPModel` wraps a base model and adds sequential MTP modules.
- `src/quantization.rs`: `FP8Linear` simulates:
    - **128x128 Tile-based Weight Quantization**.
    - **Online Activation Quantization**.
    - **Mixed Precision**: FP8 storage -> FP32 accumulation.

---

## Chapter 5: DeepSeek-R1 (Reasoning)

### Theory
DeepSeek-R1 is a "reasoning" model trained to generate a "Chain of Thought" (CoT) before the final answer.
- **Mechanism**: The model outputs a `<think>` token, generates its internal reasoning trace, outputs `</think>`, and then provides the final answer.
- **Training**: Initially supervised on a small high-quality CoT dataset (DeepSeek-R1-Zero), then refined with Reinforcement Learning (GRPO).

### Implementation (`src/r1.rs`)
- `ReasoningModel`: A simulation wrapper that structures the output into `<think>` blocks.

---

## Chapter 6: GRPO (Group Relative Policy Optimization)

### Theory
Standard RL (like PPO) requires a "Critic" (Value Function) model, which is expensive to train and host.
- **GRPO**: Removes the Critic. Instead, it samples a *group* of outputs for the same prompt and uses the group's mean reward as the baseline.
- **Loss**: $Loss = -\frac{1}{G} \sum [Advantage \cdot \ln P(output)] + \beta \cdot KL(P || Ref)$
- **Advantage**: Normalized reward within the group: $A_i = \frac{r_i - \mu}{\sigma}$

### Implementation (`src/grpo.rs`)
- `GRPOTrainer`: Implements the GRPO loss calculation, including advantage computation and KL divergence penalty.
- `GroupSampler`: Logic to sample multiple outputs for a single input.

---

## Running the Code
See [README.md](./README.md) for instructions on how to run the demos and benchmarks.

---

## Chapter 7: Zero-Copy PyO3 Python Interop

### Theory
For production ML pipelines, efficient data transfer between Python (NumPy/PyTorch) and Rust is critical. Traditional approaches copy tensor data across the language boundary, which is expensive for large tensors.

**Zero-Copy Interop** eliminates this overhead using:
1. **PyO3 Buffer Protocol**: Direct memory access to NumPy arrays without copying
2. **Arrow IPC**: Efficient serialization format for tensor transfer between processes
3. **Shared Memory Arena**: mmap-based memory regions for Ray actor communication

### Implementation (`src/pyo3_bindings/`)

#### Module Structure
```
rust-src/src/pyo3_bindings/
├── mod.rs              # Module registration and convenience functions
├── tensor_view.rs      # CandleTensorView - zero-copy NumPy ↔ Candle conversion
├── arrow_interop.rs    # ArrowTensorInterop - Arrow IPC serialization
└── shared_memory.rs    # SharedMemoryArena - mmap-based shared memory
```

#### Key Components

**CandleTensorView** (`tensor_view.rs`):
- `from_numpy_f32/f64/i64/u32/u8`: Create Candle tensor from NumPy array
- `to_numpy_f32/f64/i64`: Convert Candle tensor back to NumPy
- `matmul`, `add`, `mul`, `transpose`: Tensor operations in Rust
- Supported DTypes: F32, F64, F16, BF16, I64, U32, U8 (Note: I32 is NOT supported by Candle)

**ArrowTensorInterop** (`arrow_interop.rs`):
- `serialize_tensor`: Convert tensor to Arrow IPC bytes
- `deserialize_tensor`: Reconstruct tensor from Arrow IPC bytes
- `serialize_batch/deserialize_batch`: Batch tensor serialization with names
- `peek_metadata`: Get tensor shape/dtype without full deserialization

**SharedMemoryArena** (`shared_memory.rs`):
- `allocate_named/allocate`: Store tensors in mmap-backed shared memory
- `get/read`: Retrieve tensors by name or handle
- `free/reset`: Memory management
- `SharedTensorHandle`: Serializable handle for IPC between processes

### Build & Installation

#### Prerequisites
- Rust 1.70+ with `cargo`
- Python 3.10+ with `uv` package manager
- maturin (`uv pip install maturin`)

#### Build Commands
```bash
# Navigate to project root
cd /path/to/DeepSeek-From-Scratch

# Build and install the Rust Python module (development mode)
uv run maturin develop -m rust-src/Cargo.toml --uv

# Or build a wheel
uv run maturin build -m rust-src/Cargo.toml --features pyo3-bindings
```

#### Configuration Files

**rust-src/Cargo.toml** (key sections):
```toml
[lib]
name = "deepseek_rust"
crate-type = ["cdylib", "rlib"]

[features]
default = []
pyo3-bindings = ["dep:pyo3", "dep:numpy", "dep:ndarray"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py310"], optional = true }
numpy = { version = "0.22", optional = true }
```

**rust-src/pyproject.toml**:
```toml
[project]
name = "deepseek-rust"  # Different from main project to avoid conflicts
version = "0.1.0"

[tool.maturin]
features = ["pyo3-bindings"]
module-name = "deepseek_rust"
```

**rust-src/src/lib.rs** (conditional compilation):
```rust
#[cfg(feature = "pyo3-bindings")]
pub mod pyo3_bindings;

#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn deepseek_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_bindings::register_bindings(m)?;
    Ok(())
}
```

### Running Tests

#### Python Tests
```bash
# Run all zero-copy interop tests
uv run pytest tests/rust_interop/ -v

# Run specific test class
uv run pytest tests/rust_interop/test_zero_copy.py::TestCandleTensorView -v
```

#### Rust Tests
```bash
# Run Rust library tests (without PyO3 feature to avoid linker issues)
cd rust-src && cargo test --lib

# Run with verbose output
cargo test --lib -- --nocapture
```

### Python Usage Example
```python
import deepseek_rust
import numpy as np

# Create tensor from NumPy (zero-copy when contiguous)
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
tensor = deepseek_rust.CandleTensorView.from_numpy_f32(arr)

print(tensor.shape())  # [2, 3]
print(tensor.dtype())  # "F32"

# Tensor operations in Rust
tensor_b = deepseek_rust.CandleTensorView.from_numpy_f32(
    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
)
result = tensor.matmul(tensor_b.transpose(0, 1))

# Convert back to NumPy
result_np = result.to_numpy_f32()

# Arrow serialization for IPC
interop = deepseek_rust.ArrowTensorInterop()
serialized = interop.serialize_tensor(tensor)
restored = deepseek_rust.ArrowTensorInterop.deserialize_tensor(serialized)

# Shared memory arena for Ray actors
arena = deepseek_rust.SharedMemoryArena("my_arena", 10 * 1024 * 1024)  # 10MB
handle = arena.allocate_named("weights", tensor)
retrieved = arena.get("weights")
```

### Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'deepseek_rust'`**
- Ensure you ran `uv run maturin develop -m rust-src/Cargo.toml --uv`
- Check that `rust-src/pyproject.toml` has `name = "deepseek-rust"` (different from main project)

**Issue: Linker errors when running `cargo test --lib`**
- PyO3 requires Python libraries at link time for test binaries
- The `pyo3-bindings` feature is optional; tests run without it
- Use `cargo test --lib` (not `cargo test`) to avoid building PyO3 test harness

**Issue: `RuntimeError: NumPy error: Array not contiguous`**
- Rust requires contiguous arrays; use `np.ascontiguousarray()` for sliced views

**Issue: `DType::I32 not found`**
- Candle does not support I32; use I64, U32, or F32 instead

