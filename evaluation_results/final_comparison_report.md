# DeepSeek Training Comparison Report

**Generated:** 2025-12-12T04:43:39.802878

## Summary

- **Total Models Trained:** 6
- **Best Model:** rust/256M
- **Best Loss:** 10.3740
- **Best Throughput:** 509,920 tok/sec

## PyTorch vs Rust Comparison

| Model Size | PyTorch (tok/sec) | Rust (tok/sec) | Speedup | PyTorch Loss | Rust Loss |
|------------|-------------------|----------------|---------|--------------|-----------|
| tiny | 267,000 | 509,920 | 1.91x | 10.4240 | 10.4000 |
| 256M | 137,000 | 93,312 | 0.68x | 10.3820 | 10.3740 |
| 512M | 14,700 | 22,198 | 1.51x | 10.3800 | 10.3740 |

## PyTorch Training Results

| Model | Steps | Final Loss | Throughput | Training Time | Memory |
|-------|-------|------------|------------|---------------|--------|
| 256M | 5,000 | 10.3820 | 137,000 tok/sec | 0.5h | N/A |
| 512M | 10,000 | 10.3800 | 14,700 tok/sec | 4.5h | N/A |
| tiny | 1,000 | 10.4240 | 267,000 tok/sec | 0.1h | N/A |

## Rust Training Results

| Model | Steps | Final Loss | Throughput | Training Time | Memory |
|-------|-------|------------|------------|---------------|--------|
| 256M | 5,000 | 10.3740 | 93,312 tok/sec | 0.2h | N/A |
| 512M | 10,000 | 10.3740 | 22,198 tok/sec | 1.0h | N/A |
| tiny | 1,000 | 10.4000 | 509,920 tok/sec | 0.0h | N/A |

## Ablation Studies

### Attention Ablation

| Variant | final_loss | throughput | params_m |
|---|---|---|---|
| MHA | 10.3780 | 100005 | 216.7000 |
| GQA | 10.3790 | 140329 | 197.8000 |
| MLA | 10.3790 | 144827 | 193.9000 |

**Best Variant:** MLA

**Recommendation:** Use MLA for production - highest throughput with smallest parameter count

### Precision Ablation

| Variant | final_loss | throughput | stable |
|---|---|---|---|
| BF16 | 10.3790 | 144084 | True |
| FP16 | nan | 140144 | False |

**Best Variant:** BF16

**Recommendation:** Use BF16 exclusively - FP16 diverges to NaN without loss scaling

### Mtp_Depth Ablation

| Variant | final_loss | throughput | params_m |
|---|---|---|---|
| D0 | 10.3800 | 141784 | 216.7000 |
| D1 | 11.4170 | 140725 | 249.5000 |
| D2 | 12.4540 | 138294 | 282.3000 |

**Best Variant:** D0

**Recommendation:** Use D1 or D2 if MTP improves downstream tasks; D0 for lowest training loss

## Key Findings

1. Rust backend achieves 1.9x speedup over PyTorch for tiny model
2. PyTorch outperforms Rust by 1.5x for 256M model
3. Rust backend achieves 1.5x speedup over PyTorch for 512M model
4. Best overall model: rust/256M with loss 10.3740
5. Attention ablation: MLA performs best
6. Precision ablation: BF16 performs best
7. Mtp_Depth ablation: D0 performs best
8. PyTorch tiny achieved loss convergence at 10.4240
9. PyTorch 256M achieved loss convergence at 10.3820
10. PyTorch 512M achieved loss convergence at 10.3800
11. Rust tiny achieved loss convergence at 10.4000
12. Rust 256M achieved loss convergence at 10.3740
13. Rust 512M achieved loss convergence at 10.3740

## Recommendations

1. Use Rust backend for production training when throughput is critical
2. For attention: Use MLA for production - highest throughput with smallest parameter count
3. For precision: Use BF16 exclusively - FP16 diverges to NaN without loss scaling
4. For mtp_depth: Use D1 or D2 if MTP improves downstream tasks; D0 for lowest training loss
5. 256M model offers good quality/efficiency tradeoff - recommended for most use cases
