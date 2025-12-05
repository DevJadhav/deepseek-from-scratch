# Related Work

This document provides a comparison of DeepSeek-V3 with related work in large language models.

## Attention Mechanisms

### Multi-Head Attention (MHA)
- **Paper**: Vaswani et al., "Attention Is All You Need" (2017)
- **Approach**: Each head has independent Q, K, V projections
- **KV Cache**: Full storage of all heads
- **Trade-off**: Maximum expressiveness, highest memory cost

### Multi-Query Attention (MQA)
- **Paper**: Shazeer, "Fast Transformer Decoding" (2019)
- **Approach**: Single shared K, V across all heads
- **KV Cache**: 1/n_heads of MHA
- **Trade-off**: Significant memory savings, some quality loss

### Grouped-Query Attention (GQA)
- **Paper**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- **Approach**: K, V shared within groups of heads
- **KV Cache**: n_groups/n_heads of MHA
- **Trade-off**: Balanced memory-quality trade-off

### Multi-Latent Attention (MLA) - DeepSeek
- **Paper**: DeepSeek-V2 Technical Report (2024)
- **Approach**: Low-rank KV compression with learned up-projections
- **KV Cache**: ~1/14 of MHA
- **Trade-off**: Excellent compression with no quality loss

| Mechanism | KV Cache (per layer) | Quality | Inference Speed |
|-----------|---------------------|---------|-----------------|
| MHA | 100% | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| MQA | 1/n_heads | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GQA | 1/n_groups | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **MLA** | ~7% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Mixture of Experts

### Sparse MoE
- **Paper**: Shazeer et al., "Outrageously Large Neural Networks" (2017)
- **Approach**: Top-k routing with auxiliary load balancing loss
- **Experts**: 8-16 experts typical
- **Issue**: Auxiliary loss contaminates gradients

### GShard
- **Paper**: Lepikhin et al., "GShard" (2020)
- **Approach**: Top-2 routing with capacity factor
- **Scale**: Up to 600B parameters
- **Innovation**: Expert parallelism for distributed training

### Switch Transformer
- **Paper**: Fedus et al., "Switch Transformers" (2021)
- **Approach**: Top-1 routing for simplicity
- **Scale**: 1.6T parameters
- **Innovation**: Simplified routing, better scaling

### Mixtral
- **Paper**: Mistral AI (2024)
- **Approach**: 8 experts, top-2 routing
- **Scale**: 8x7B = 47B total, 13B active
- **Innovation**: Open-source MoE model

### DeepSeekMoE
- **Paper**: DeepSeek-V2/V3 Technical Reports (2024)
- **Approach**: 256 fine-grained experts + shared expert
- **Innovation**: Auxiliary-loss-free load balancing
- **Scale**: 671B total, 37B active

| Model | Total Params | Active Params | Experts | Routing |
|-------|-------------|---------------|---------|---------|
| Switch | 1.6T | 1.6B | 2048 | Top-1 |
| Mixtral | 47B | 13B | 8 | Top-2 |
| DeepSeek-V2 | 236B | 21B | 160 | Top-6 |
| **DeepSeek-V3** | 671B | 37B | 256 | Top-8 |

## Load Balancing Strategies

### Auxiliary Loss Methods
- **Approach**: Add loss term to encourage balanced routing
- **Models**: GShard, Switch, Mixtral
- **Issue**: Loss term affects gradient quality

### Expert Choice Routing
- **Paper**: Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022)
- **Approach**: Experts choose tokens instead of tokens choosing experts
- **Trade-off**: Fixed compute per expert, variable per token

### Auxiliary-Loss-Free (DeepSeek)
- **Approach**: Learnable bias for routing, separate from gating
- **Mechanism**: Post-step bias adjustment based on load
- **Advantage**: Clean gradients, better convergence

## Multi-Token Prediction

### Standard Autoregressive
- **Approach**: Predict one token at a time
- **Limitation**: Sequential generation is slow

### Speculative Decoding
- **Paper**: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2022)
- **Approach**: Draft model generates candidates, main model verifies
- **Speedup**: 2-3x typical

### Multi-Token Prediction (Meta)
- **Paper**: Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (2024)
- **Approach**: Multiple prediction heads during training
- **Benefit**: Better representations, enables speculative decoding

### MTP in DeepSeek-V3
- **Depth**: D=1 (predict 2 tokens total)
- **Training**: Additional heads with λ=0.3 weight
- **Inference**: Self-speculative decoding
- **Speedup**: 1.8x reported

## Distributed Training

### Data Parallelism (DP)
- **Approach**: Replicate model, partition data
- **Frameworks**: PyTorch DDP, Horovod
- **Limitation**: Memory bounded by single GPU

### Model Parallelism (MP)
- **Tensor Parallelism**: Split layers horizontally
- **Pipeline Parallelism**: Split layers vertically
- **Framework**: Megatron-LM

### ZeRO Optimization
- **Paper**: Rajbhandari et al., "ZeRO" (2019)
- **Stages**: ZeRO-1 (optimizer), ZeRO-2 (+gradients), ZeRO-3 (+parameters)
- **Framework**: DeepSpeed

### 5D Parallelism (DeepSeek)
- **Dimensions**: DP, TP, PP, EP (Expert), SP (Sequence)
- **Innovation**: DualPipe for pipeline parallelism
- **Scale**: 2048 H800 GPUs

## Training Efficiency

### Mixed Precision Training
| Precision | Memory | Speed | Stability |
|-----------|--------|-------|-----------|
| FP32 | 1x | 1x | ⭐⭐⭐⭐⭐ |
| BF16 | 0.5x | 1.8x | ⭐⭐⭐⭐ |
| FP16 | 0.5x | 1.9x | ⭐⭐⭐ |
| **FP8** | 0.25x | 2.4x | ⭐⭐⭐⭐ |

### Training Cost Comparison

| Model | Parameters | Training Tokens | GPU Hours | Est. Cost |
|-------|-----------|-----------------|-----------|-----------|
| GPT-4 | ~1.8T | ~13T | ~25M A100 | ~$100M |
| Llama 3.1 405B | 405B | 15T | ~30M H100 | ~$150M |
| **DeepSeek-V3** | 671B | 14.8T | 2.79M H800 | **$5.6M** |

DeepSeek-V3's training cost is remarkably low due to:
1. FP8 training efficiency
2. Auxiliary-loss-free MoE (no wasted computation)
3. Efficient infrastructure (DualPipe)

## Key Differentiators

### DeepSeek-V3 Innovations

1. **Multi-Latent Attention**: 14x KV cache compression without quality loss
2. **Auxiliary-Loss-Free MoE**: Clean gradients for better convergence
3. **256 Fine-Grained Experts**: Better specialization than coarse experts
4. **DualPipe**: ~50% reduction in pipeline bubbles
5. **FP8 Training**: 2.4x throughput improvement
6. **Multi-Token Prediction**: Self-speculative decoding without draft model

### Comparison Summary

| Feature | GPT-4 | Llama 3 | Mixtral | DeepSeek-V3 |
|---------|-------|---------|---------|-------------|
| Architecture | Dense | Dense | MoE | MoE |
| Attention | MHA | GQA | GQA | MLA |
| Experts | N/A | N/A | 8 | 256 |
| Load Balance | N/A | N/A | Aux Loss | Aux-Free |
| Precision | BF16 | BF16 | BF16 | FP8 |
| MTP | No | No | No | Yes |
| Open Weights | No | Yes | Yes | Yes |

## References

1. Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.
2. Shazeer, N. "Fast transformer decoding: One write-head is all you need." arXiv 2019.
3. Ainslie, J., et al. "GQA: Training generalized multi-query transformer models from multi-head checkpoints." EMNLP 2023.
4. Shazeer, N., et al. "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer." ICLR 2017.
5. Fedus, W., et al. "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity." JMLR 2021.
6. Rajbhandari, S., et al. "ZeRO: Memory optimizations toward training trillion parameter models." SC 2020.
7. DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv 2024.
8. DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv 2024.
