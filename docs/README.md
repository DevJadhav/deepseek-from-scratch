# DeepSeek Architecture Documentation

This documentation provides in-depth explanations of all architectural components implemented in this repository. Each document includes:

- 📖 **Mathematical foundations** with KaTeX equations
- 🔧 **Implementation examples** in Rust and Python
- ✅ **When to use / When not to use** decision guides
- 📊 **Performance comparisons** and trade-offs

---

## 📚 Table of Contents

### Attention Mechanisms
1. [**Multi-Query Attention (MQA)**](./01-multi-query-attention.md) - Single KV head shared across all query heads
2. [**Grouped-Query Attention (GQA)**](./02-grouped-query-attention.md) - KV heads shared within groups
3. [**Multi-Head Latent Attention (MLA)**](./03-multi-head-latent-attention.md) - Compressed KV via latent projections
4. [**DeepSeek Attention**](./04-deepseek-attention.md) - MLA + Decoupled RoPE

### Mixture of Experts
5. [**Standard MoE**](./05-standard-moe.md) - Classic sparse mixture of experts with load balancing
6. [**DeepSeek MoE**](./06-deepseek-moe.md) - Shared + Routed experts with fine-grained segmentation

### Prediction & Quantization
7. [**Multi-Token Prediction (MTP)**](./07-multi-token-prediction.md) - Predict multiple future tokens for faster training/inference
8. [**FP8 Quantization**](./08-fp8-quantization.md) - 8-bit floating point with tile-based scaling

### Alignment & Training
9. [**GRPO (Group Relative Policy Optimization)**](./09-grpo.md) - DeepSeek's alignment algorithm
10. [**Training Infrastructure**](./10-training-infrastructure.md) - AdamW, schedulers, gradient accumulation, checkpointing

### Advanced Training & Distillation
11. [**Training Pipeline**](./11-training-pipeline.md) - Scaling laws, data mixing, WSD scheduler, distributed training
12. [**Post-Training: SFT & RLHF**](./12-post-training.md) - Supervised fine-tuning, reward models, DPO
13. [**Knowledge Distillation**](./13-knowledge-distillation.md) - Model compression, progressive distillation, SeqKD

### DeepSeek-V3.2 Complete Architecture
14. [**V3.2 Architecture Summary**](./14-v32-architecture.md) - Complete V3.2 implementation guide with DSA, 256-expert MoE, MTP, FP8, and agent training

---

## Quick Reference: When to Use What?


| Component | Use When | Don't Use When |
|-----------|----------|----------------|
| **MQA** | Memory-constrained inference | Training quality is paramount |
| **GQA** | Balance memory vs quality | Extreme memory constraints |
| **MLA** | Long-context, large models | Small models (overhead not worth it) |
| **Standard MoE** | Scaling model capacity | Limited compute budget |
| **DeepSeek MoE** | Production LLMs | Simple tasks |
| **MTP** | Faster training convergence, speculative decoding | Short sequences |
| **FP8** | High-throughput inference, H100/M3+ | Accuracy-critical tasks |
| **GRPO** | LLM alignment, RLHF | Pre-training |
| **WSD Scheduler** | Pre-training large LLMs | Fine-tuning, small models |
| **DPO** | Preference alignment without reward model | Need explicit reward model |
| **Knowledge Distillation** | Model compression, deployment | When teacher not available |

---

## Architecture Comparison

### Attention KV Memory Usage

For sequence length $L$, model dimension $D$, and $H$ heads:

| Method | KV Cache Size | Relative Size |
|--------|---------------|---------------|
| MHA | $2 \times L \times D$ | 1.0x |
| MQA | $2 \times L \times (D/H)$ | $1/H$ |
| GQA | $2 \times L \times (D/H) \times G$ | $G/H$ |
| MLA | $L \times d_{latent} + L \times d_{rope}$ | ~0.1x |

### MoE Compute Scaling

| Method | Active Params | Total Params | Efficiency |
|--------|---------------|--------------|------------|
| Dense | $P$ | $P$ | 100% |
| Standard MoE | $P_{router} + k \times P_{expert}$ | $P_{router} + N \times P_{expert}$ | $k/N$ |
| DeepSeek MoE | $P_{shared} + k \times P_{routed}$ | $P_{shared} + N \times P_{routed}$ | Higher utilization |

---

## Document Structure

Each document follows a consistent structure:

1. **Overview** - What is it and why does it matter?
2. **When to Use** - Decision guide with pros/cons
3. **Mathematical Foundation** - Core equations with explanations
4. **Implementation** - Rust and Python code examples
5. **Performance Analysis** - Memory, compute, accuracy trade-offs
6. **Practical Tips** - Hyperparameters and common issues
7. **References** - Original papers and additional resources

---

## Contributing

When adding new documentation:

1. Follow the numbering scheme (XX-component-name.md)
2. Include both Rust and Python implementations
3. Add mathematical equations using KaTeX
4. Update this README's table of contents
5. Add decision guides (when to use / when not to use)