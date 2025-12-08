# Production Scaling Guide: From Laptop to Datacenter

> **Deploying DeepSeek-From-Scratch at Scale with Three-Backend Strategy**

This guide covers scaling DeepSeek models from local development to production deployment, leveraging the unique strengths of each backend: Rust for inference, PyTorch for training, and MLX for Apple Silicon.

---

## Table of Contents

1. [Scaling Philosophy](#1-scaling-philosophy)
2. [5D Parallelism Deep Dive](#2-5d-parallelism-deep-dive)
3. [Backend Deployment Strategies](#3-backend-deployment-strategies)
4. [Memory Optimization](#4-memory-optimization)
5. [Throughput Optimization](#5-throughput-optimization)
6. [Monitoring and Observability](#6-monitoring-and-observability)
7. [Cost Optimization](#7-cost-optimization)

---

## 1. Scaling Philosophy

### The Three-Stage Journey

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DeepSeek Scaling Path                            │
├─────────────────┬─────────────────────┬─────────────────────────────────┤
│   Stage 1       │      Stage 2        │         Stage 3                 │
│   Development   │      Validation     │         Production              │
├─────────────────┼─────────────────────┼─────────────────────────────────┤
│   MLX Local     │   PyTorch Cloud     │     Rust/PyTorch Multi-GPU      │
│   Apple Silicon │   Single GPU        │     Distributed Cluster         │
├─────────────────┼─────────────────────┼─────────────────────────────────┤
│   • Rapid iter  │   • Full precision  │     • Maximum throughput        │
│   • No cost     │   • Debugging       │     • 5D parallelism            │
│   • Unified mem │   • Profiling       │     • Auto-scaling              │
└─────────────────┴─────────────────────┴─────────────────────────────────┘
```

### Backend Selection Matrix

| Use Case | Recommended Backend | Hardware | Notes |
|----------|---------------------|----------|-------|
| Local dev/debug | MLX | Apple Silicon | Fast iteration, unified memory |
| Training (research) | PyTorch | CUDA GPUs | Flexibility, ecosystem |
| Training (production) | PyTorch + FSDP | Multi-GPU | Distributed, efficient |
| Inference (batch) | Rust/Candle | Any | Best throughput |
| Inference (real-time) | Rust/Candle | CUDA/Metal | Lowest latency |
| Edge deployment | MLX or Rust | Apple/Edge | Optimized for device |

---

## 2. 5D Parallelism Deep Dive

DeepSeek-V3 introduced 5D parallelism for training 671B parameters:

### Parallelism Dimensions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          5D Parallelism                                 │
├───────────┬──────────────────────────────────────────────────────────────
│    DP     │  Data Parallelism: Replicate model, split data              │
│    TP     │  Tensor Parallelism: Split layers horizontally              │
│    PP     │  Pipeline Parallelism: Split layers vertically              │
│    EP     │  Expert Parallelism: Distribute MoE experts                 │
│    SP     │  Sequence Parallelism: Split long sequences                 │
└───────────┴──────────────────────────────────────────────────────────────┘
```

### Configuration by Scale

```python
# config/parallelism.py

@dataclass
class ParallelismConfig:
    """5D parallelism configuration."""
    dp_size: int = 1    # Data parallel replicas
    tp_size: int = 1    # Tensor parallel (within node)
    pp_size: int = 1    # Pipeline parallel stages
    ep_size: int = 1    # Expert parallel groups
    sp_size: int = 1    # Sequence parallel

    def validate(self, world_size: int):
        required = self.dp_size * self.tp_size * self.pp_size * self.ep_size * self.sp_size
        assert required == world_size, f"Config requires {required} GPUs, have {world_size}"


# Recommended configurations by GPU count
CONFIGS = {
    1: ParallelismConfig(dp=1, tp=1, pp=1, ep=1, sp=1),      # Single GPU
    4: ParallelismConfig(dp=2, tp=2, pp=1, ep=1, sp=1),      # Single node, 4 GPU
    8: ParallelismConfig(dp=2, tp=4, pp=1, ep=1, sp=1),      # Single node, 8 GPU
    16: ParallelismConfig(dp=2, tp=4, pp=2, ep=1, sp=1),     # 2 nodes
    64: ParallelismConfig(dp=4, tp=8, pp=2, ep=1, sp=1),     # 8 nodes (7B model)
    256: ParallelismConfig(dp=8, tp=8, pp=4, ep=1, sp=1),    # 32 nodes (70B model)
    2048: ParallelismConfig(dp=32, tp=8, pp=8, ep=1, sp=1),  # 256 nodes (671B)
}
```

### Pipeline Parallelism Implementation

From `src/deepseek/cloud/modal/distributed_trainer.py`:

```python
class PipelineParallelTrainer:
    """
    Pipeline Parallelism with microbatching.

    Architecture (PP=3):
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │    GPU 0     │──▶│    GPU 1     │──▶│    GPU 2     │
    │ Embed+L0-3   │   │   L4-7       │   │ L8-11+Head   │
    └──────────────┘   └──────────────┘   └──────────────┘
           │                                     │
           └────────── Gradient Flow ◀───────────┘
    """

    def __init__(self, model: nn.Module, pp_size: int, num_microbatches: int = 4):
        self.pp_size = pp_size
        self.pp_rank = get_pp_rank()
        self.num_microbatches = num_microbatches

        # Partition model layers
        total_layers = len(model.layers)
        layers_per_stage = total_layers // pp_size
        start_layer = self.pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage

        # Create local stage
        self.stage = PipelineStage(
            layers=model.layers[start_layer:end_layer],
            is_first=(self.pp_rank == 0),
            is_last=(self.pp_rank == pp_size - 1),
            embed=model.embed if self.pp_rank == 0 else None,
            head=model.head if self.pp_rank == pp_size - 1 else None,
        )

    def forward_backward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Execute 1F1B pipeline schedule for memory efficiency.

        Schedule (4 microbatches, 3 stages):
        Time →
        GPU0: F0 F1 F2 F3 B3 B2 B1 B0
        GPU1:    F0 F1 F2 F3 B3 B2 B1 B0
        GPU2:       F0 F1 F2 F3 B3 B2 B1 B0
        """
        microbatches = self._split_batch(batch, self.num_microbatches)

        # Warmup: forward passes to fill pipeline
        forward_outputs = []
        for i in range(min(self.pp_rank + 1, self.num_microbatches)):
            output = self._forward_microbatch(microbatches[i], i)
            forward_outputs.append(output)

        # Steady state: 1F1B
        loss_accum = 0.0
        for i in range(self.pp_rank + 1, self.num_microbatches):
            # Forward
            output = self._forward_microbatch(microbatches[i], i)

            # Backward (for earlier microbatch)
            backward_idx = i - self.pp_rank - 1
            loss = self._backward_microbatch(forward_outputs[backward_idx], backward_idx)
            loss_accum += loss

            forward_outputs.append(output)

        # Cooldown: remaining backwards
        for i in range(self.num_microbatches - self.pp_rank - 1, self.num_microbatches):
            backward_idx = i
            loss = self._backward_microbatch(forward_outputs[backward_idx], backward_idx)
            loss_accum += loss

        return loss_accum / self.num_microbatches
```

### Expert Parallelism for MoE

```python
class ExpertParallelMoE(nn.Module):
    """
    MoE with experts distributed across devices.

    With EP=4 and 256 experts:
    - Device 0: Experts 0-63
    - Device 1: Experts 64-127
    - Device 2: Experts 128-191
    - Device 3: Experts 192-255
    """

    def __init__(self, config: DeepSeekMoEV3Config, ep_group: ProcessGroup):
        super().__init__()
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_group = ep_group

        # Each rank holds subset of experts
        experts_per_rank = config.n_routed_experts // self.ep_size
        self.local_expert_start = self.ep_rank * experts_per_rank
        self.local_expert_end = self.local_expert_start + experts_per_rank

        # Only create local experts
        self.local_experts = nn.ModuleList([
            ExpertV3(config.d_model, config.d_expert)
            for _ in range(experts_per_rank)
        ])

        # Shared experts replicated on all ranks
        self.shared_experts = nn.ModuleList([
            ExpertV3(config.d_model, config.d_expert)
            for _ in range(config.n_shared_experts)
        ])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward with all-to-all communication for expert dispatch.

        1. Each rank determines which tokens go to local experts
        2. All-to-all to send tokens to correct ranks
        3. Local expert computation
        4. All-to-all to return results
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        # Determine routing: which tokens to which EP rank
        expert_rank = expert_indices // (self.config.n_routed_experts // self.ep_size)

        # Count tokens per destination rank
        send_counts = torch.bincount(expert_rank.view(-1), minlength=self.ep_size)

        # All-to-all: exchange tokens
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        # Permute tokens for send
        sorted_indices = torch.argsort(expert_rank.view(-1))
        send_tokens = x_flat[sorted_indices]

        # All-to-all: exchange actual tokens
        recv_tokens = all_to_all_variable(send_tokens, send_counts, recv_counts, self.ep_group)

        # Compute local experts
        local_output = self._compute_local_experts(recv_tokens, expert_indices)

        # All-to-all: return results
        output_tokens = all_to_all_variable(local_output, recv_counts, send_counts, self.ep_group)

        # Unpermute to original order
        output = torch.empty_like(x_flat)
        output[sorted_indices] = output_tokens

        # Add shared experts (computed locally)
        for shared in self.shared_experts:
            output = output + shared(x_flat)

        return output.view(batch, seq_len, d_model)
```

---

## 3. Backend Deployment Strategies

### Rust: High-Throughput Inference Server

```rust
// rust-src/src/server/inference.rs

use axum::{routing::post, Router, Json};
use tokio::sync::Semaphore;

pub struct InferenceServer {
    model: DeepSeekModel,
    batch_queue: BatchQueue,
    max_concurrent: Semaphore,
}

impl InferenceServer {
    pub async fn serve(self, addr: &str) -> Result<()> {
        let app = Router::new()
            .route("/v1/completions", post(Self::completions))
            .route("/v1/chat/completions", post(Self::chat_completions))
            .route("/health", get(Self::health))
            .with_state(Arc::new(self));

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }

    async fn completions(
        State(server): State<Arc<Self>>,
        Json(request): Json<CompletionRequest>,
    ) -> Result<Json<CompletionResponse>> {
        // Acquire semaphore for rate limiting
        let _permit = server.max_concurrent.acquire().await?;

        // Add to batch queue for dynamic batching
        let response = server.batch_queue.submit(request).await?;

        Ok(Json(response))
    }
}

/// Dynamic batching for throughput optimization
pub struct BatchQueue {
    pending: Mutex<Vec<PendingRequest>>,
    batch_timeout_ms: u64,
    max_batch_size: usize,
}

impl BatchQueue {
    pub async fn run_batching_loop(&self, model: &DeepSeekModel) {
        loop {
            // Wait for batch or timeout
            tokio::time::sleep(Duration::from_millis(self.batch_timeout_ms)).await;

            let batch = {
                let mut pending = self.pending.lock().await;
                if pending.is_empty() {
                    continue;
                }
                // Take up to max_batch_size requests
                pending.drain(..pending.len().min(self.max_batch_size)).collect::<Vec<_>>()
            };

            // Process batch
            let results = model.generate_batch(&batch).await?;

            // Send results
            for (req, result) in batch.into_iter().zip(results) {
                req.response_tx.send(result).ok();
            }
        }
    }
}
```

### PyTorch: Training with Modal Cloud

From `src/deepseek/cloud/modal/app.py`:

```python
import modal

app = modal.App("deepseek-training")

# GPU image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "torch", "transformers", "deepspeed", "flash-attn"
]).run_commands([
    "pip install -e /app"
])

@app.function(
    gpu=modal.gpu.A100(count=8, memory=80),
    image=image,
    timeout=86400,  # 24 hours
    volumes={"/data": modal.Volume.from_name("training-data")},
)
def train_distributed(
    config_path: str,
    resume_from: Optional[str] = None,
):
    """Distributed training on Modal with 8×A100."""
    import deepspeed
    from deepseek.torch.training.trainer import DeepSeekTrainer

    # Load config
    config = load_config(config_path)

    # Initialize DeepSpeed
    ds_config = {
        "train_batch_size": config.batch_size * 8,
        "gradient_accumulation_steps": config.grad_accum,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,  # Full sharding
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
        },
        "pipeline": {
            "stages": 4,
            "micro_batch_size": config.micro_batch_size,
        },
    }

    # Create trainer
    trainer = DeepSeekTrainer(
        model_config=config.model,
        training_config=config.training,
        deepspeed_config=ds_config,
    )

    # Train
    trainer.train(
        train_dataset="/data/train",
        eval_dataset="/data/eval",
        resume_from=resume_from,
    )


@app.function(gpu="A100", image=image)
def inference_endpoint(request: dict) -> dict:
    """Serverless inference endpoint."""
    from deepseek.torch.inference import InferenceEngine

    engine = InferenceEngine.load("/checkpoints/latest")
    response = engine.generate(request["prompt"], **request.get("params", {}))

    return {"response": response}
```

### MLX: Apple Silicon Deployment

```python
# src/deepseek/mlx/server.py

from fastapi import FastAPI
from mlx_lm import load, generate

app = FastAPI()

# Load model into unified memory (no copies needed!)
model, tokenizer = load("deepseek-v3-mlx")


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    MLX inference leveraging unified memory.

    On Apple Silicon:
    - No CPU-GPU memory copies
    - Efficient Metal kernels
    - Scales with ANE when available
    """
    tokens = tokenizer.encode(request.prompt)

    output = generate(
        model,
        tokens,
        max_tokens=request.max_tokens,
        temp=request.temperature,
        top_p=request.top_p,
    )

    return {"response": tokenizer.decode(output)}


# Optimized batch inference
class MLXBatchServer:
    def __init__(self, model_path: str, max_batch: int = 8):
        self.model, self.tokenizer = load(model_path)
        self.max_batch = max_batch

    async def process_batch(self, requests: List[str]) -> List[str]:
        """Process batch with unified memory efficiency."""
        # Tokenize all
        all_tokens = [self.tokenizer.encode(r) for r in requests]

        # Pad to same length
        max_len = max(len(t) for t in all_tokens)
        padded = mx.array([
            t + [self.tokenizer.pad_id] * (max_len - len(t))
            for t in all_tokens
        ])

        # Generate (batched, efficient on unified memory)
        outputs = []
        for i in range(len(requests)):
            output = generate(self.model, padded[i], ...)
            outputs.append(self.tokenizer.decode(output))

        return outputs
```

---

## 4. Memory Optimization

### Gradient Checkpointing

From `src/deepseek/torch/model/transformer.py`:

```python
@dataclass
class GradientCheckpointConfig:
    """Configuration for gradient checkpointing."""
    enabled: bool = True
    checkpoint_every_n_layers: int = 1  # Every layer
    checkpoint_moe: bool = True         # Also checkpoint MoE
    use_reentrant: bool = False         # Non-reentrant for torch.compile


class DeepSeekModel(nn.Module):
    def configure_gradient_checkpointing(
        self,
        enabled: bool = True,
        checkpoint_every_n_layers: int = 1,
        checkpoint_moe: bool = True,
    ) -> None:
        """Enable/configure gradient checkpointing at runtime."""
        for layer_idx, layer in enumerate(self.layers):
            should_checkpoint = enabled and (layer_idx % checkpoint_every_n_layers == 0)
            layer.checkpoint_config = GradientCheckpointConfig(
                enabled=should_checkpoint,
                checkpoint_moe=checkpoint_moe,
                use_reentrant=False,
            )


# Memory savings calculation
def estimate_memory_savings(model: DeepSeekModel, batch_size: int, seq_len: int):
    """Estimate memory savings from gradient checkpointing."""
    # Without checkpointing: store all activations
    activations_per_layer = batch_size * seq_len * model.d_model * 4  # FP32
    total_without = activations_per_layer * len(model.layers)

    # With checkpointing: only store checkpointed activations
    # Recompute others during backward
    checkpoint_layers = len(model.layers) // model.checkpoint_config.checkpoint_every_n_layers
    total_with = activations_per_layer * checkpoint_layers

    return {
        'without_checkpoint_gb': total_without / 1e9,
        'with_checkpoint_gb': total_with / 1e9,
        'savings': 1 - (total_with / total_without),
    }
```

### MLA Latent Cache Memory

```python
def estimate_kv_cache_memory(
    model_config: dict,
    batch_size: int,
    max_seq_len: int,
    use_latent_cache: bool = True,
) -> dict:
    """
    Estimate KV cache memory requirements.

    MLA achieves ~14× reduction through latent compression.
    """
    n_layers = model_config['n_layers']
    n_heads = model_config['n_heads']
    head_dim = model_config['d_model'] // n_heads
    d_latent = model_config['d_latent']
    dtype_bytes = 2  # FP16

    if use_latent_cache:
        # Latent cache: [batch, seq, d_latent] per layer
        cache_per_layer = batch_size * max_seq_len * d_latent * dtype_bytes
    else:
        # Standard: [batch, heads, seq, head_dim] × 2 (K, V) per layer
        cache_per_layer = 2 * batch_size * n_heads * max_seq_len * head_dim * dtype_bytes

    total_cache = cache_per_layer * n_layers

    return {
        'total_gb': total_cache / 1e9,
        'per_layer_mb': cache_per_layer / 1e6,
        'compression_ratio': (2 * n_heads * head_dim) / d_latent if use_latent_cache else 1.0,
    }
```

---

## 5. Throughput Optimization

### Continuous Batching

```python
class ContinuousBatchingEngine:
    """
    Continuous batching for maximum GPU utilization.

    Instead of waiting for all sequences to complete,
    immediately add new sequences as old ones finish.
    """

    def __init__(self, model, max_batch_tokens: int = 32768):
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.running_sequences: List[Sequence] = []
        self.waiting_queue: Queue[Sequence] = Queue()

    async def generate_stream(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        """Streaming generation with continuous batching."""
        seq = Sequence(prompt=prompt, max_tokens=max_tokens)
        await self.waiting_queue.put(seq)

        async for token in seq.output_stream:
            yield token

    async def run_engine(self):
        """Main engine loop."""
        while True:
            # Add waiting sequences if we have capacity
            current_tokens = sum(len(s.tokens) for s in self.running_sequences)
            while not self.waiting_queue.empty():
                seq = await self.waiting_queue.get()
                if current_tokens + len(seq.tokens) <= self.max_batch_tokens:
                    self.running_sequences.append(seq)
                    current_tokens += len(seq.tokens)
                else:
                    await self.waiting_queue.put(seq)  # Put back
                    break

            if not self.running_sequences:
                await asyncio.sleep(0.001)
                continue

            # Forward pass for all running sequences
            next_tokens = self.model.forward_batch(self.running_sequences)

            # Update sequences, remove completed
            completed = []
            for seq, token in zip(self.running_sequences, next_tokens):
                seq.append_token(token)
                if seq.is_complete():
                    completed.append(seq)

            for seq in completed:
                self.running_sequences.remove(seq)
                seq.complete()
```

### Speculative Decoding

```python
class SpeculativeDecoder:
    """
    Speculative decoding with draft model for faster generation.

    Uses small draft model to predict N tokens,
    then verifies with main model in single forward pass.
    """

    def __init__(
        self,
        main_model: DeepSeekModel,
        draft_model: DeepSeekModel,
        speculation_length: int = 4,
    ):
        self.main_model = main_model
        self.draft_model = draft_model
        self.speculation_length = speculation_length

    def generate(self, prompt_tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
        generated = prompt_tokens.clone()

        while len(generated) < len(prompt_tokens) + max_tokens:
            # Draft model generates N speculative tokens
            draft_tokens = self._draft_generate(generated, self.speculation_length)

            # Main model verifies all at once
            candidate = torch.cat([generated, draft_tokens], dim=-1)
            main_logits = self.main_model(candidate)

            # Find how many draft tokens to accept
            accepted = 0
            for i in range(self.speculation_length):
                draft_prob = draft_tokens_probs[i]
                main_prob = main_logits[:, -self.speculation_length + i].softmax(-1)

                # Accept if main model agrees
                if main_prob.argmax() == draft_tokens[i]:
                    accepted += 1
                else:
                    # Sample from main model for this position
                    next_token = main_prob.multinomial(1)
                    generated = torch.cat([generated, draft_tokens[:i], next_token], dim=-1)
                    break
            else:
                # All accepted, also sample next
                next_token = main_logits[:, -1].softmax(-1).multinomial(1)
                generated = torch.cat([generated, draft_tokens, next_token], dim=-1)

        return generated
```

---

## 6. Monitoring and Observability

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUESTS_TOTAL = Counter('deepseek_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('deepseek_request_latency_seconds', 'Request latency')
TOKENS_GENERATED = Counter('deepseek_tokens_generated_total', 'Tokens generated')

# Model metrics
GPU_MEMORY_USED = Gauge('deepseek_gpu_memory_bytes', 'GPU memory used', ['device'])
BATCH_SIZE = Histogram('deepseek_batch_size', 'Batch sizes')
TOKENS_PER_SECOND = Gauge('deepseek_tokens_per_second', 'Generation throughput')

# MoE metrics
EXPERT_LOAD = Gauge('deepseek_expert_load', 'Expert utilization', ['expert_id'])
EXPERT_IMBALANCE = Gauge('deepseek_expert_imbalance', 'Load imbalance ratio')


class MetricsMiddleware:
    def __init__(self, model: DeepSeekModel):
        self.model = model
        self._start_background_collection()

    def _start_background_collection(self):
        """Collect GPU and model metrics periodically."""
        async def collect():
            while True:
                # GPU memory
                for i, gpu in enumerate(get_gpus()):
                    GPU_MEMORY_USED.labels(device=f'cuda:{i}').set(gpu.memory_used)

                # Expert load (for MoE layers)
                for layer in self.model.modules():
                    if isinstance(layer, DeepSeekMoEV3):
                        for i, load in enumerate(layer.bias_controller.load_ema):
                            EXPERT_LOAD.labels(expert_id=str(i)).set(load)
                        EXPERT_IMBALANCE.set(layer.bias_controller.load_imbalance())

                await asyncio.sleep(10)

        asyncio.create_task(collect())
```

### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer(__name__)


class TracedInference:
    def __init__(self, model: DeepSeekModel):
        self.model = model

    @tracer.start_as_current_span("generate")
    def generate(self, prompt: str, max_tokens: int) -> str:
        span = trace.get_current_span()
        span.set_attribute("prompt_length", len(prompt))
        span.set_attribute("max_tokens", max_tokens)

        # Tokenize
        with tracer.start_as_current_span("tokenize"):
            tokens = self.tokenizer.encode(prompt)

        # Generate
        with tracer.start_as_current_span("model_forward"):
            output_tokens = []
            for _ in range(max_tokens):
                with tracer.start_as_current_span("forward_step"):
                    logits = self.model(tokens)
                    next_token = logits[:, -1].argmax()
                    output_tokens.append(next_token)
                    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)

        # Decode
        with tracer.start_as_current_span("decode"):
            response = self.tokenizer.decode(output_tokens)

        span.set_attribute("output_length", len(response))
        return response
```

---

## 7. Cost Optimization

### GPU Cost Calculator

```python
@dataclass
class GPUInstance:
    name: str
    hourly_cost: float
    memory_gb: int
    fp16_tflops: float


# Instance pricing (Modal, AWS, GCP averages)
INSTANCES = {
    'a100_40gb': GPUInstance('A100 40GB', 2.50, 40, 312),
    'a100_80gb': GPUInstance('A100 80GB', 3.50, 80, 312),
    'h100_80gb': GPUInstance('H100 80GB', 5.00, 80, 989),
    'l4': GPUInstance('L4', 0.80, 24, 121),
    't4': GPUInstance('T4', 0.50, 16, 65),
}


def calculate_training_cost(
    model_params: int,
    training_tokens: int,
    instance: GPUInstance,
    num_gpus: int,
    efficiency: float = 0.4,  # Typical MFU
) -> dict:
    """
    Estimate training cost.

    Based on: tokens = 6 × params × flops / (efficiency × gpu_flops)
    """
    # FLOPs per token (approximate)
    flops_per_token = 6 * model_params

    # Total FLOPs
    total_flops = flops_per_token * training_tokens

    # Time to train
    effective_flops = num_gpus * instance.fp16_tflops * 1e12 * efficiency
    time_seconds = total_flops / effective_flops
    time_hours = time_seconds / 3600

    # Cost
    total_cost = time_hours * instance.hourly_cost * num_gpus

    return {
        'time_hours': time_hours,
        'time_days': time_hours / 24,
        'total_cost_usd': total_cost,
        'cost_per_billion_tokens': total_cost / (training_tokens / 1e9),
    }


# Example: Training 7B model on 1T tokens
cost = calculate_training_cost(
    model_params=7e9,
    training_tokens=1e12,
    instance=INSTANCES['a100_80gb'],
    num_gpus=64,
)
# Result: ~12 days, ~$67,000
```

### Inference Cost Optimization

```python
def optimize_inference_config(
    model_size: int,
    target_latency_ms: float,
    target_throughput_tps: float,
    budget_hourly: float,
) -> dict:
    """Find optimal instance configuration for inference."""

    configs = []

    for instance_name, instance in INSTANCES.items():
        for num_gpus in [1, 2, 4, 8]:
            # Check if model fits
            required_memory = model_size * 2 / 1e9  # FP16
            if required_memory > instance.memory_gb * num_gpus:
                continue

            # Estimate throughput (simplified)
            tokens_per_second = instance.fp16_tflops * 1e12 / (model_size * 2) * num_gpus * 0.3

            # Estimate latency (for 100 token generation)
            latency_ms = 100 / tokens_per_second * 1000

            # Cost per million tokens
            hourly_cost = instance.hourly_cost * num_gpus
            cost_per_million = hourly_cost / (tokens_per_second * 3600) * 1e6

            # Check constraints
            meets_latency = latency_ms <= target_latency_ms
            meets_throughput = tokens_per_second >= target_throughput_tps
            meets_budget = hourly_cost <= budget_hourly

            if meets_latency and meets_throughput and meets_budget:
                configs.append({
                    'instance': instance_name,
                    'num_gpus': num_gpus,
                    'latency_ms': latency_ms,
                    'throughput_tps': tokens_per_second,
                    'hourly_cost': hourly_cost,
                    'cost_per_million_tokens': cost_per_million,
                })

    # Sort by cost efficiency
    return sorted(configs, key=lambda x: x['cost_per_million_tokens'])
```

---

## Summary

Production scaling for DeepSeek involves:

1. **5D Parallelism**: DP + TP + PP + EP + SP for maximum scale
2. **Backend Selection**: Rust inference, PyTorch training, MLX development
3. **Memory Optimization**: MLA latent cache (14×), gradient checkpointing
4. **Throughput**: Continuous batching, speculative decoding
5. **Observability**: Metrics, tracing, MoE load monitoring
6. **Cost**: Right-size instances, optimize for your latency/throughput targets

---

## Next Steps

- [Build It Yourself Tutorial](./04_step_by_step_creation.md)
- [Architecture Overview](./01_deepseek_architecture_from_scratch.md)
- [MoE Deep Dive](./techniques/mixture_of_experts.md)
