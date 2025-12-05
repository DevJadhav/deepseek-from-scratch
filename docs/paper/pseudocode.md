# DeepSeek-V3 Pseudocode

This document contains algorithm pseudocode for the key components of DeepSeek-V3.

## Algorithm 1: Multi-Latent Attention (MLA)

```
Algorithm 1: Multi-Latent Attention Forward Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: X ∈ ℝ^(B×T×D), position indices pos ∈ ℤ^T
Output: O ∈ ℝ^(B×T×D)

Parameters:
    W_dkv ∈ ℝ^(D×d_c)       // KV down-projection
    W_uk ∈ ℝ^(d_c×n_h×d_h)  // Key up-projection  
    W_uv ∈ ℝ^(d_c×n_h×d_h)  // Value up-projection
    W_dq ∈ ℝ^(D×d_q)        // Query down-projection
    W_uq ∈ ℝ^(d_q×n_h×d_h)  // Query up-projection
    W_kr ∈ ℝ^(D×d_r)        // RoPE key projection
    W_o ∈ ℝ^(n_h×d_h×D)     // Output projection

1:  // Compress KV to latent space
2:  c_kv ← X @ W_dkv                           // (B, T, d_c)
3:  
4:  // Expand latent to full K, V
5:  K_content ← c_kv @ W_uk                    // (B, T, n_h, d_h)
6:  V ← c_kv @ W_uv                            // (B, T, n_h, d_h)
7:  
8:  // Compress and expand queries
9:  c_q ← X @ W_dq                             // (B, T, d_q)
10: Q_content ← c_q @ W_uq                     // (B, T, n_h, d_h)
11: 
12: // Apply RoPE to dedicated head dimensions
13: k_rope ← X @ W_kr                          // (B, T, d_r)
14: k_rope ← apply_rope(k_rope, pos)           // Apply rotary embeddings
15: q_rope ← Q_content[..., :d_r]              // Extract RoPE dims from Q
16: q_rope ← apply_rope(q_rope, pos)
17: 
18: // Concatenate content and RoPE components
19: K ← concat(K_content, k_rope, dim=-1)      // (B, T, n_h, d_h + d_r)
20: Q ← concat(Q_content, q_rope, dim=-1)      // (B, T, n_h, d_h + d_r)
21: 
22: // Standard attention computation
23: scale ← 1 / sqrt(d_h + d_r)
24: scores ← (Q @ K.transpose(-2, -1)) * scale // (B, n_h, T, T)
25: scores ← apply_causal_mask(scores)
26: attn ← softmax(scores, dim=-1)
27: 
28: // Compute output
29: O ← attn @ V                                // (B, n_h, T, d_h)
30: O ← reshape(O, (B, T, n_h * d_h))
31: O ← O @ W_o                                 // (B, T, D)
32: 
33: return O

// KV Cache: Only store c_kv (d_c dimensions) instead of K, V (n_h × d_h each)
// Compression ratio: 2 × n_h × d_h / d_c ≈ 14×
```

## Algorithm 2: DeepSeekMoE with Auxiliary-Loss-Free Balancing

```
Algorithm 2: DeepSeekMoE Forward Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: X ∈ ℝ^(B×T×D)
Output: Y ∈ ℝ^(B×T×D)

Parameters:
    W_gate ∈ ℝ^(D×N_r)              // Router weights
    bias ∈ ℝ^(N_r)                  // Learnable routing bias (not in loss)
    experts: List[FFN] of size N_r  // Routed experts
    shared_expert: FFN              // Shared expert

Hyperparameters:
    K_r = 8                         // Experts per token
    γ = 0.001                       // Bias update rate

1:  // Flatten batch and sequence
2:  X_flat ← reshape(X, (B*T, D))             // (N, D) where N = B*T
3:  
4:  // Compute routing scores
5:  affinity ← sigmoid(X_flat @ W_gate)       // (N, N_r)
6:  
7:  // Add bias for routing decision (NOT for gating)
8:  routing_scores ← affinity + bias          // (N, N_r)
9:  
10: // Select top-K experts per token
11: _, indices ← topk(routing_scores, K_r)    // (N, K_r)
12: 
13: // Compute gating weights using ORIGINAL affinity
14: selected_affinity ← gather(affinity, indices)  // (N, K_r)
15: gates ← softmax(selected_affinity, dim=-1)     // (N, K_r)
16: 
17: // Initialize output
18: Y_routed ← zeros(N, D)
19: 
20: // Process each expert
21: for e = 0 to N_r - 1:
22:     // Find tokens routed to expert e
23:     mask ← (indices == e).any(dim=-1)      // (N,)
24:     if mask.sum() == 0: continue
25:     
26:     // Get tokens and their gate weights for this expert
27:     X_expert ← X_flat[mask]                // (n_e, D)
28:     gate_idx ← (indices[mask] == e).nonzero()
29:     gate_weights ← gates[mask, gate_idx]   // (n_e,)
30:     
31:     // Expert forward pass
32:     expert_out ← experts[e](X_expert)      // (n_e, D)
33:     
34:     // Accumulate weighted output
35:     Y_routed[mask] += gate_weights.unsqueeze(-1) * expert_out
36: 
37: // Shared expert (always active)
38: Y_shared ← shared_expert(X_flat)           // (N, D)
39: 
40: // Combine outputs
41: Y ← Y_shared + Y_routed                    // (N, D)
42: Y ← reshape(Y, (B, T, D))
43: 
44: return Y


Algorithm 3: Auxiliary-Loss-Free Bias Update (Post-Step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: indices ∈ ℤ^(N×K_r) from forward pass
       bias ∈ ℝ^(N_r) current bias values

1:  // Count tokens per expert
2:  for e = 0 to N_r - 1:
3:      load[e] ← count(indices == e)
4:  
5:  // Compute target load (uniform distribution)
6:  target ← (N * K_r) / N_r
7:  
8:  // Update bias based on load deviation
9:  for e = 0 to N_r - 1:
10:     if load[e] > target:
11:         bias[e] ← bias[e] - γ             // Decrease to reduce load
12:     else:
13:         bias[e] ← bias[e] + γ             // Increase to increase load
14: 
15: return bias

// Key insight: Bias affects routing but not gating weights
// Gradients flow cleanly through affinity → gates → output
```

## Algorithm 3: Expert FFN with SwiGLU Activation

```
Algorithm 4: SwiGLU Expert Forward Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: X ∈ ℝ^(n×D)
Output: Y ∈ ℝ^(n×D)

Parameters:
    W_gate ∈ ℝ^(D×d_ff)     // Gate projection
    W_up ∈ ℝ^(D×d_ff)       // Up projection
    W_down ∈ ℝ^(d_ff×D)     // Down projection

1:  // Parallel projections
2:  gate ← X @ W_gate                          // (n, d_ff)
3:  up ← X @ W_up                              // (n, d_ff)
4:  
5:  // SwiGLU activation
6:  hidden ← SiLU(gate) ⊙ up                   // (n, d_ff)
7:  
8:  // Down projection
9:  Y ← hidden @ W_down                        // (n, D)
10: 
11: return Y

// SiLU(x) = x * sigmoid(x)
// SwiGLU provides smooth gating while maintaining expressivity
```

## Algorithm 4: Multi-Token Prediction

```
Algorithm 5: Multi-Token Prediction Forward Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: hidden_states H ∈ ℝ^(B×T×D) from main model
       input_ids ∈ ℤ^(B×T)
       targets ∈ ℤ^(B×T) ground truth tokens
Output: mtp_loss ∈ ℝ

Parameters:
    embedding: Embedding(V, D)
    mtp_layers: List[TransformerBlock] of size D_mtp
    mtp_heads: List[Linear(D, V)] of size D_mtp

Hyperparameters:
    D_mtp = 1                   // MTP depth
    λ_mtp = 0.3                 // MTP loss weight

1:  mtp_loss ← 0
2:  
3:  for d = 1 to D_mtp:
4:      // Get shifted targets for this depth
5:      shifted_targets ← targets[:, d:]           // (B, T-d)
6:      
7:      // Get hidden states (truncated)
8:      H_trunc ← H[:, :T-d]                       // (B, T-d, D)
9:      
10:     // Get embeddings of previous predictions
11:     if d == 1:
12:         prev_embeds ← embedding(input_ids[:, :T-1])
13:     else:
14:         prev_embeds ← embedding(targets[:, d-1:T-1])
15:     
16:     // Concatenate hidden states with previous embeddings
17:     combined ← concat(H_trunc, prev_embeds, dim=-1)  // (B, T-d, 2D)
18:     combined ← project_down(combined)                 // (B, T-d, D)
19:     
20:     // Apply MTP transformer layers
21:     for layer in mtp_layers[d-1]:
22:         combined ← layer(combined)
23:     
24:     // Compute logits and loss
25:     logits ← mtp_heads[d-1](combined)          // (B, T-d, V)
26:     loss_d ← cross_entropy(logits, shifted_targets)
27:     mtp_loss ← mtp_loss + λ_mtp * loss_d
28: 
29: return mtp_loss


Algorithm 6: MTP Speculative Decoding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: prompt_tokens ∈ ℤ^(T)
Output: generated_tokens ∈ ℤ^(T + max_new_tokens)

1:  tokens ← prompt_tokens
2:  
3:  while len(tokens) < T + max_new_tokens:
4:      // Forward pass generates main + D_mtp speculative tokens
5:      H ← main_model(tokens)
6:      
7:      // Main model prediction
8:      main_logits ← lm_head(H[:, -1])
9:      main_token ← sample(main_logits)
10:     
11:     // Generate D_mtp speculative tokens
12:     speculative_tokens ← [main_token]
13:     H_curr ← H[:, -1]
14:     
15:     for d = 1 to D_mtp:
16:         prev_embed ← embedding(speculative_tokens[-1])
17:         combined ← concat(H_curr, prev_embed)
18:         combined ← mtp_layers[d-1](combined)
19:         spec_logits ← mtp_heads[d-1](combined)
20:         spec_token ← sample(spec_logits)
21:         speculative_tokens.append(spec_token)
22:     
23:     // Verify speculative tokens with full model
24:     candidate_tokens ← concat(tokens, speculative_tokens)
25:     verify_H ← main_model(candidate_tokens)
26:     
27:     // Accept matching tokens
28:     accepted ← 0
29:     for i = 0 to len(speculative_tokens) - 1:
30:         verify_logits ← lm_head(verify_H[:, T + i])
31:         verify_token ← argmax(verify_logits)
32:         if verify_token == speculative_tokens[i]:
33:             accepted ← accepted + 1
34:         else:
35:             break
36:     
37:     // Add accepted tokens
38:     tokens ← concat(tokens, speculative_tokens[:accepted + 1])
39: 
40: return tokens
```

## Algorithm 5: RMSNorm

```
Algorithm 7: RMSNorm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: X ∈ ℝ^(B×T×D)
Output: Y ∈ ℝ^(B×T×D)

Parameters:
    weight ∈ ℝ^(D)          // Learnable scale
    ε = 1e-6                // Numerical stability

1:  // Compute RMS (root mean square)
2:  rms ← sqrt(mean(X², dim=-1, keepdim=True) + ε)  // (B, T, 1)
3:  
4:  // Normalize
5:  X_norm ← X / rms                                 // (B, T, D)
6:  
7:  // Scale
8:  Y ← X_norm * weight                              // (B, T, D)
9:  
10: return Y

// Compared to LayerNorm:
// - No mean subtraction (computationally cheaper)
// - No learnable bias (slightly fewer parameters)
// - Similar empirical performance
```

## Algorithm 6: Rotary Position Embedding (RoPE)

```
Algorithm 8: Rotary Position Embedding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: X ∈ ℝ^(B×T×D), positions ∈ ℤ^(T)
Output: X_rotated ∈ ℝ^(B×T×D)

Hyperparameters:
    θ_base = 10000          // Base frequency
    D_rope = D              // Dimensions to rotate (typically all)

1:  // Compute frequencies for each dimension pair
2:  d ← arange(0, D_rope, 2)                         // [0, 2, 4, ..., D-2]
3:  freqs ← 1 / (θ_base^(d / D_rope))               // (D/2,)
4:  
5:  // Compute position-dependent angles
6:  angles ← outer(positions, freqs)                 // (T, D/2)
7:  
8:  // Compute sin and cos
9:  sin_angles ← sin(angles)                         // (T, D/2)
10: cos_angles ← cos(angles)                         // (T, D/2)
11: 
12: // Split input into pairs
13: X_even ← X[..., 0::2]                            // (B, T, D/2)
14: X_odd ← X[..., 1::2]                             // (B, T, D/2)
15: 
16: // Apply rotation
17: X_even_rot ← X_even * cos_angles - X_odd * sin_angles
18: X_odd_rot ← X_even * sin_angles + X_odd * cos_angles
19: 
20: // Interleave back
21: X_rotated ← stack([X_even_rot, X_odd_rot], dim=-1)
22: X_rotated ← reshape(X_rotated, (B, T, D))
23: 
24: return X_rotated

// RoPE Properties:
// - Relative position information encoded in dot products
// - Decaying attention with distance (natural inductive bias)
// - Extrapolates to longer sequences than seen during training
```

## Algorithm 7: FP8 Mixed Precision Training

```
Algorithm 9: FP8 Forward Pass with Per-Block Scaling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: X ∈ ℝ^(M×K), W ∈ ℝ^(K×N) (in higher precision)
Output: Y ∈ ℝ^(M×N)

Hyperparameters:
    block_size = 128        // Quantization block size
    fp8_max = 448           // E4M3 max value

1:  // Compute scaling factors per block
2:  for i = 0 to M/block_size - 1:
3:      for j = 0 to K/block_size - 1:
4:          X_block ← X[i*block_size:(i+1)*block_size, 
5:                      j*block_size:(j+1)*block_size]
6:          scale_X[i,j] ← fp8_max / max(abs(X_block))
7:  
8:  for i = 0 to K/block_size - 1:
9:      for j = 0 to N/block_size - 1:
10:         W_block ← W[i*block_size:(i+1)*block_size,
11:                     j*block_size:(j+1)*block_size]
12:         scale_W[i,j] ← fp8_max / max(abs(W_block))
13: 
14: // Quantize to FP8
15: X_fp8 ← quantize_fp8(X * scale_X)
16: W_fp8 ← quantize_fp8(W * scale_W)
17: 
18: // FP8 GEMM (hardware accelerated)
19: Y_fp8 ← X_fp8 @ W_fp8                           // FP8 computation
20: 
21: // Dequantize with combined scale
22: Y ← Y_fp8 / (scale_X @ scale_W)                 // Back to higher precision
23: 
24: return Y


Algorithm 10: FP8 Gradient Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: dY ∈ ℝ^(M×N) (gradient w.r.t. output)
       X_fp8, W_fp8 (saved from forward)
       scale_X, scale_W (saved from forward)
Output: dX ∈ ℝ^(M×K), dW ∈ ℝ^(K×N)

1:  // Compute dY scale
2:  scale_dY ← compute_block_scales(dY)
3:  dY_fp8 ← quantize_fp8(dY * scale_dY)
4:  
5:  // Gradient w.r.t. input: dX = dY @ W^T
6:  dX_fp8 ← dY_fp8 @ W_fp8.T                      // FP8 GEMM
7:  dX ← dX_fp8 / (scale_dY @ scale_W.T)
8:  
9:  // Gradient w.r.t. weights: dW = X^T @ dY
10: dW_fp8 ← X_fp8.T @ dY_fp8                      // FP8 GEMM
11: dW ← dW_fp8 / (scale_X.T @ scale_dY)
12: 
13: // Accumulate gradients in higher precision (BF16/FP32)
14: W.grad ← W.grad + dW.to(higher_precision)
15: 
16: return dX, dW

// Key: Forward and most backward in FP8
// Weight updates accumulated in higher precision for stability
```

## Algorithm 8: DualPipe Scheduling

```
Algorithm 11: DualPipe Bidirectional Scheduling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: micro_batches[0:M], model stages[0:P]
Output: processed outputs and gradients

State:
    forward_queue, backward_queue per GPU
    activations[P][M] for backward pass

// Phase 1: Fill pipeline (warmup)
1:  for t = 0 to P - 1:
2:      parallel for gpu = 0 to P - 1:
3:          if t >= gpu:
4:              // GPU can start processing
5:              mb_idx ← t - gpu
6:              activations[gpu][mb_idx] ← forward(stages[gpu], micro_batches[mb_idx])
7:              send_activation_to_next_gpu(activations[gpu][mb_idx])

// Phase 2: Steady state (1F1B with bidirectional)
8:  for t = P to M + P - 1:
9:      parallel for gpu = 0 to P - 1:
10:         // Forward stream (new micro-batch)
11:         if t - gpu < M:
12:             mb_fwd ← t - gpu
13:             recv_from_prev_gpu()
14:             activations[gpu][mb_fwd] ← forward(stages[gpu], ...)
15:             send_to_next_gpu()
16:         
17:         // Backward stream (old micro-batch)
18:         mb_bwd ← t - gpu - (P - 1)
19:         if 0 <= mb_bwd < M:
20:             recv_grad_from_next_gpu()
21:             grad ← backward(stages[gpu], activations[gpu][mb_bwd])
22:             send_grad_to_prev_gpu()
23:         
24:         // Key insight: Forward and backward communications
25:         // happen in opposite directions simultaneously
26:         // maximizing bandwidth utilization

// Phase 3: Drain pipeline (cooldown)
27: for t = M + P - 1 to M + 2*P - 2:
28:     parallel for gpu = 0 to P - 1:
29:         mb_bwd ← t - gpu - (P - 1)
30:         if 0 <= mb_bwd < M:
31:             recv_grad_from_next_gpu()
32:             grad ← backward(stages[gpu], activations[gpu][mb_bwd])
33:             send_grad_to_prev_gpu()

// Synchronize gradients
34: all_reduce_gradients()

// Bubble analysis:
// Standard 1F1B: bubble = (P-1) / (P-1 + M)
// DualPipe:      bubble ≈ (P-1) / (2*(P-1) + M) ≈ half
```
