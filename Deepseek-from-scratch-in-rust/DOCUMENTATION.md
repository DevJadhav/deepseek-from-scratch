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
