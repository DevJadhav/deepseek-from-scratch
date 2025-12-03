# Multi-Token Prediction (MTP)

## Overview

Multi-Token Prediction (MTP) is an advanced training and inference technique where the model predicts **multiple future tokens simultaneously** instead of just the next token. This provides:

1. **Richer training signal**: More gradient information per forward pass
2. **Faster inference**: Speculative decoding with self-verification
3. **Better representations**: Forces model to plan ahead

**Key Papers:**
- [Better & Faster Large Language Models via Multi-Token Prediction](https://arxiv.org/abs/2404.19737)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

---

## When to Use MTP

### ✅ Use MTP When:
- **Training large models** - MTP improves sample efficiency
- **Long-form generation** - Planning ahead improves coherence
- **Speculative decoding** - MTP heads enable self-speculation
- **Coding tasks** - Structure prediction benefits from lookahead

### ❌ Don't Use MTP When:
- **Short responses** - Overhead not justified
- **Interactive/streaming** - Need immediate first token
- **Memory constrained** - MTP heads add parameters
- **Simple completion** - Standard autoregressive sufficient

### When to Prefer Over Standard LM:
| Standard LM | MTP |
|-------------|-----|
| Next-token prediction | Multi-token prediction |
| Simple training | Richer gradient signal |
| Token-by-token inference | Speculative decoding |
| Myopic planning | Lookahead planning |

---

## Mathematical Foundation

### Standard Autoregressive LM

Standard language models maximize:

$$\mathcal{L}_{\text{AR}} = \sum_{t=1}^{T} \log P(x_t | x_{<t})$$

Predicting one token at a time: $x_t$ given history $x_{<t}$.

### Multi-Token Prediction Objective

MTP extends this to predict $n$ future tokens:

$$\mathcal{L}_{\text{MTP}} = \sum_{t=1}^{T} \sum_{k=1}^{n} \log P_k(x_{t+k} | x_{\leq t})$$

Where:
- $P_k$: The $k$-th prediction head
- $x_{t+k}$: The token $k$ positions ahead
- $n$: Number of future tokens to predict

### Architecture

```
Hidden States h_t from Backbone
        │
        ├──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
    │Head 1│   │Head 2│   │Head 3│   │Head 4│
    │ +1   │   │ +2   │   │ +3   │   │ +4   │
    └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
       ▼          ▼          ▼          ▼
   Predict    Predict    Predict    Predict
    x_{t+1}    x_{t+2}    x_{t+3}    x_{t+4}
```

---

## MTP Head Architecture

### Basic Structure

Each MTP head transforms shared representations for its specific prediction:

$$h_t^{(k)} = \text{MTPHead}_k(h_t^{\text{backbone}})$$

$$P_k(x_{t+k} | x_{\leq t}) = \text{softmax}(W_k \cdot h_t^{(k)})$$

### Head Options

**Option 1: Independent Heads** (Simpler)
```
Each head: Linear(D → D) + LayerNorm + Linear(D → V)
```

**Option 2: Sequential Heads** (DeepSeek-V3 style)
```
Head k uses output from Head k-1 as additional input
```

### Sequential MTP (DeepSeek Variant)

DeepSeek uses sequential dependency between heads:

$$h_t^{(1)} = \text{Head}_1(h_t)$$
$$h_t^{(k)} = \text{Head}_k(h_t, \text{Embed}(\hat{x}_{t+k-1}))$$

Where $\hat{x}_{t+k-1}$ is the predicted token from head $k-1$.

---

## Training Details

### Loss Function

The total loss combines all prediction heads:

$$\mathcal{L}_{\text{total}} = \sum_{k=1}^{n} \lambda_k \cdot \mathcal{L}_k$$

Where $\lambda_k$ are weighting coefficients.

### Typical Weighting Schemes

| Scheme | $\lambda_1$ | $\lambda_2$ | $\lambda_3$ | $\lambda_4$ |
|--------|-------------|-------------|-------------|-------------|
| Uniform | 1.0 | 1.0 | 1.0 | 1.0 |
| Linear decay | 1.0 | 0.75 | 0.5 | 0.25 |
| Exponential | 1.0 | 0.5 | 0.25 | 0.125 |
| Primary + aux | 1.0 | 0.1 | 0.1 | 0.1 |

**DeepSeek typically uses:** Equal weights with primary head emphasis.

### Gradient Flow

MTP provides denser gradients:

```
Standard LM:    ∂L/∂h_t only from position t+1
MTP (n=4):      ∂L/∂h_t from positions t+1, t+2, t+3, t+4
```

This improves:
- Representation learning
- Long-range dependencies
- Sample efficiency

---

## Implementation

### Rust Implementation

```rust
pub struct MultiTokenPredictor {
    n_heads: usize,
    d_model: usize,
    vocab_size: usize,
    
    // MTP heads (sequential style)
    projection_layers: Vec<Linear>,
    layer_norms: Vec<LayerNorm>,
    output_layers: Vec<Linear>,
    
    // Embedding for sequential dependency
    token_embedding: Embedding,
}

impl MultiTokenPredictor {
    pub fn new(
        d_model: usize,
        vocab_size: usize,
        n_heads: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut projection_layers = Vec::new();
        let mut layer_norms = Vec::new();
        let mut output_layers = Vec::new();
        
        for k in 0..n_heads {
            // First head takes only backbone output
            // Subsequent heads also take previous prediction embedding
            let input_dim = if k == 0 { d_model } else { d_model * 2 };
            
            projection_layers.push(Linear::new(input_dim, d_model, device)?);
            layer_norms.push(LayerNorm::new(d_model, device)?);
            output_layers.push(Linear::new(d_model, vocab_size, device)?);
        }
        
        let token_embedding = Embedding::new(vocab_size, d_model, device)?;
        
        Ok(Self {
            n_heads,
            d_model,
            vocab_size,
            projection_layers,
            layer_norms,
            output_layers,
            token_embedding,
        })
    }
    
    pub fn forward(&self, backbone_hidden: &Tensor) -> Result<Vec<Tensor>> {
        // backbone_hidden: (B, T, D)
        let (b, t, _) = backbone_hidden.dims3()?;
        let mut logits_list = Vec::new();
        let mut prev_hidden = backbone_hidden.clone();
        
        for k in 0..self.n_heads {
            // Prepare input
            let head_input = if k == 0 {
                prev_hidden.clone()
            } else {
                // Get predicted tokens from previous head
                let prev_logits = &logits_list[k - 1];
                let prev_tokens = prev_logits.argmax(D::Minus1)?;  // (B, T)
                let prev_embed = self.token_embedding.forward(&prev_tokens)?;  // (B, T, D)
                
                // Concatenate with backbone hidden
                Tensor::cat(&[&prev_hidden, &prev_embed], 2)?  // (B, T, 2D)
            };
            
            // MTP head forward
            let projected = self.projection_layers[k].forward(&head_input)?;
            let normed = self.layer_norms[k].forward(&projected)?;
            let logits = self.output_layers[k].forward(&normed)?;  // (B, T, V)
            
            logits_list.push(logits);
            prev_hidden = normed;
        }
        
        Ok(logits_list)
    }
    
    pub fn compute_loss(
        &self,
        logits_list: &[Tensor],
        targets: &Tensor,  // (B, T)
        weights: &[f32],
    ) -> Result<Tensor> {
        let mut total_loss = Tensor::zeros((), DType::F32, logits_list[0].device())?;
        
        for (k, (logits, weight)) in logits_list.iter().zip(weights.iter()).enumerate() {
            // Shift targets by k+1 positions
            let shifted_targets = targets.narrow(1, k + 1, targets.dim(1)? - k - 1)?;
            let shifted_logits = logits.narrow(1, 0, logits.dim(1)? - k - 1)?;
            
            let loss = cross_entropy(&shifted_logits, &shifted_targets)?;
            total_loss = (total_loss + loss * (*weight as f64))?;
        }
        
        Ok(total_loss)
    }
}
```

### Python Implementation

```python
class MultiTokenPredictor(nn.Module):
    """
    Multi-Token Prediction heads for predicting multiple future tokens.
    
    Implements sequential MTP where each head conditions on previous predictions.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_heads: int = 4,
        sequential: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.sequential = sequential
        
        # Token embedding for sequential conditioning
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # MTP heads
        self.projection_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        
        for k in range(n_heads):
            input_dim = d_model if k == 0 or not sequential else d_model * 2
            self.projection_layers.append(nn.Linear(input_dim, d_model))
            self.layer_norms.append(nn.LayerNorm(d_model))
            self.output_layers.append(nn.Linear(d_model, vocab_size))
    
    def forward(
        self, 
        backbone_hidden: torch.Tensor,
        targets: Optional[torch.Tensor] = None,  # For teacher forcing
    ) -> List[torch.Tensor]:
        """
        Args:
            backbone_hidden: (B, T, D) hidden states from main model
            targets: (B, T) target tokens for teacher forcing during training
        
        Returns:
            List of logits tensors, one per head: [(B, T, V), ...]
        """
        B, T, D = backbone_hidden.shape
        logits_list = []
        prev_hidden = backbone_hidden
        
        for k in range(self.n_heads):
            # === Prepare Input ===
            if k == 0 or not self.sequential:
                head_input = prev_hidden
            else:
                if targets is not None and self.training:
                    # Teacher forcing: use ground truth
                    # Shift targets by k positions
                    prev_tokens = targets[:, k-1:T+k-1] if k < T else targets[:, -1:].expand(-1, T)
                else:
                    # Autoregressive: use previous predictions
                    prev_tokens = logits_list[k-1].argmax(dim=-1)  # (B, T)
                
                prev_embed = self.token_embedding(prev_tokens)  # (B, T, D)
                head_input = torch.cat([prev_hidden, prev_embed], dim=-1)  # (B, T, 2D)
            
            # === MTP Head ===
            projected = self.projection_layers[k](head_input)
            normed = self.layer_norms[k](projected)
            logits = self.output_layers[k](normed)  # (B, T, V)
            
            logits_list.append(logits)
            prev_hidden = normed
        
        return logits_list
    
    def compute_loss(
        self,
        logits_list: List[torch.Tensor],
        targets: torch.Tensor,
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Compute MTP loss.
        
        Args:
            logits_list: List of (B, T, V) logits from each head
            targets: (B, T) target tokens
            weights: Loss weights for each head
        
        Returns:
            Scalar loss tensor
        """
        if weights is None:
            weights = [1.0] * len(logits_list)
        
        B, T = targets.shape
        total_loss = 0.0
        
        for k, (logits, weight) in enumerate(zip(logits_list, weights)):
            # Head k predicts position t+k+1 from position t
            # So shift targets by k+1
            shift = k + 1
            
            if shift >= T:
                continue
            
            shifted_targets = targets[:, shift:]  # (B, T-shift)
            shifted_logits = logits[:, :T-shift]  # (B, T-shift, V)
            
            # Cross entropy
            loss = F.cross_entropy(
                shifted_logits.reshape(-1, self.vocab_size),
                shifted_targets.reshape(-1),
            )
            
            total_loss += weight * loss
        
        return total_loss


# === Usage Example ===
def train_step_with_mtp(
    model: nn.Module,
    mtp: MultiTokenPredictor,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
):
    # Forward through main model
    backbone_hidden = model(input_ids)  # (B, T, D)
    
    # Multi-token predictions
    mtp_logits = mtp(backbone_hidden, targets=targets)  # List of (B, T, V)
    
    # Compute MTP loss
    mtp_loss = mtp.compute_loss(
        mtp_logits, 
        targets,
        weights=[1.0, 0.5, 0.25, 0.125]  # Decaying weights
    )
    
    return mtp_loss
```

---

## Speculative Decoding with MTP

### The Idea

MTP heads can be used for **speculative decoding**:
1. Generate draft tokens quickly using MTP heads
2. Verify all drafts in parallel with main model
3. Accept matching prefixes, reject mismatches

### Algorithm

```
Algorithm: MTP Speculative Decoding
─────────────────────────────────────
Input: prompt, model, mtp_heads, n_draft
Output: generated tokens

1. hidden = model.backbone(prompt)
2. for step in range(max_tokens):
3.     # Draft with MTP heads
4.     drafts = mtp_heads(hidden[-1])  # n_draft predictions
5.     draft_tokens = [argmax(d) for d in drafts]
6.     
7.     # Verify all drafts in parallel
8.     draft_input = prompt + draft_tokens
9.     verify_logits = model.backbone(draft_input)
10.    
11.    # Check which drafts match
12.    n_accepted = 0
13.    for i, token in enumerate(draft_tokens):
14.        if verify_logits[prompt_len + i].argmax() == token:
15.            n_accepted += 1
16.            accept(token)
17.        else:
18.            accept(verify_logits[prompt_len + i].argmax())
19.            break  # Reject remaining drafts
20.    
21.    # Update hidden for next iteration
22.    hidden = model.backbone(prompt + accepted_tokens)
```

### Speedup Analysis

Let:
- $\alpha$: Acceptance rate (probability draft matches)
- $n$: Number of draft tokens
- $T_d$: Time for draft generation
- $T_v$: Time for verification (parallel)

Expected tokens per step:
$$E[\text{tokens}] = \sum_{k=0}^{n-1} \alpha^k(1-\alpha) \cdot k + \alpha^n \cdot n$$

For $\alpha = 0.7$ and $n = 4$:
$$E[\text{tokens}] \approx 2.4$$

**Speedup** = $\frac{E[\text{tokens}]}{T_d + T_v} \approx 1.5\text{x} - 3\text{x}$

---

## MTP Inference Patterns

### Pattern 1: Independent Generation

```python
def generate_independent(model, mtp, prompt, n_tokens):
    """Each head generates independently (fastest but lowest quality)."""
    hidden = model.backbone(prompt)
    
    all_predictions = []
    for _ in range(n_tokens // mtp.n_heads):
        logits_list = mtp(hidden[:, -1:])
        
        # Take predictions from all heads
        for logits in logits_list:
            token = logits.argmax(dim=-1)
            all_predictions.append(token)
        
        # Update hidden (simplified)
        hidden = model.backbone(torch.cat([prompt] + all_predictions))
    
    return all_predictions
```

### Pattern 2: Verify-and-Accept

```python
def generate_speculative(model, mtp, prompt, n_tokens):
    """Speculative decoding with verification."""
    accepted = []
    hidden = model.backbone(prompt)
    
    while len(accepted) < n_tokens:
        # Draft
        logits_list = mtp(hidden[:, -1:])
        draft = [l.argmax(dim=-1) for l in logits_list]
        
        # Verify
        draft_input = torch.cat([prompt] + accepted + draft)
        verify_hidden = model.backbone(draft_input)
        verify_logits = model.lm_head(verify_hidden)
        
        # Accept prefix
        for i, d in enumerate(draft):
            pos = len(prompt) + len(accepted) + i
            if verify_logits[:, pos].argmax() == d:
                accepted.append(d)
            else:
                accepted.append(verify_logits[:, pos].argmax())
                break
        
        hidden = model.backbone(torch.cat([prompt] + accepted))
    
    return accepted[:n_tokens]
```

### Pattern 3: Parallel Token Generation

```python
def generate_parallel(model, mtp, prompt, n_tokens):
    """Generate multiple positions in parallel (batch-efficient)."""
    B = prompt.size(0)
    
    # Generate n_heads tokens per step
    hidden = model.backbone(prompt)
    logits_list = mtp(hidden)
    
    # All heads predict simultaneously
    predictions = torch.stack([l.argmax(dim=-1) for l in logits_list], dim=-1)
    # predictions: (B, T, n_heads)
    
    return predictions
```

---

## Benefits of MTP Training

### 1. Sample Efficiency

MTP provides more gradient signal per sample:

| Method | Gradients per token |
|--------|---------------------|
| Standard LM | 1 |
| MTP (n=4) | 4 |

### 2. Better Long-Range Planning

Model must predict further ahead → learns better representations:

```
Standard: "The cat sat on the" → "mat"
MTP:      "The cat sat on the" → "mat", "and", "watched", "birds"
```

### 3. Coherence Improvement

Lookahead encourages consistent generation:
- Predict token 4 must be consistent with tokens 1-3
- Implicit planning in representations

---

## Practical Considerations

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `n_heads` | 2-4 | More heads = more compute |
| `head_weights` | [1, 0.5, 0.25, ...] | Decay for distant predictions |
| `sequential` | True | Better quality |
| `teacher_forcing` | 0.9-1.0 | Ratio during training |

### Memory Overhead

MTP adds parameters for each head:
- Projection: $D^2$ or $2D^2$ (sequential)
- LayerNorm: $2D$
- Output: $D \times V$

For $n=4$ heads: ~12-15% parameter increase

### Training Tips

1. **Start with fewer heads**: Add heads gradually
2. **Use teacher forcing**: Essential for stable training
3. **Decay weights**: Distant predictions are harder
4. **Warmup**: MTP benefits from longer warmup

---

## Comparison with Other Techniques

| Technique | Mechanism | Speedup | Quality |
|-----------|-----------|---------|---------|
| Standard AR | Token-by-token | 1x | Baseline |
| MTP Training | Multi-head loss | 1x (train) | Better |
| MTP Inference | Self-speculative | 1.5-2x | Same/Better |
| External Draft | Separate model | 2-3x | Varies |
| Parallel Decoding | Independent | 3-4x | Lower |

---

## References

- [Better & Faster Large Language Models via Multi-Token Prediction](https://arxiv.org/abs/2404.19737)
- [Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
