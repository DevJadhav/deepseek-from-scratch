# Group Relative Policy Optimization (GRPO)

## Overview

GRPO (Group Relative Policy Optimization) is DeepSeek's alignment algorithm that improves upon PPO for language model training. It uses **group-based advantage estimation** instead of a learned value function, making it more stable and memory-efficient.

**Key Paper:** [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)

**Key Innovations:**
1. **No critic network**: Eliminates value function training
2. **Group normalization**: Computes advantages relative to group baseline
3. **Memory efficient**: Single policy, no separate value model
4. **Stable training**: Less variance in advantage estimation

---

## When to Use GRPO

### ✅ Use GRPO When:
- **Fine-tuning LLMs for alignment** - RLHF alternative
- **Mathematical reasoning** - Proven on DeepSeekMath
- **Memory constrained** - No separate value model needed
- **Want stable training** - Less hyperparameter sensitivity

### ❌ Don't Use GRPO When:
- **Pre-training** - Use standard LM loss
- **Short sequences** - Not enough samples for group stats
- **Dense rewards** - GRPO designed for sparse/final rewards
- **Real-time learning** - Batch-based algorithm

### Comparison with Other Methods:
| Method | Critic | Memory | Stability | Complexity |
|--------|--------|--------|-----------|------------|
| PPO | Required | High | Medium | High |
| DPO | None | Low | High | Low |
| **GRPO** | **None** | **Low** | **High** | **Medium** |
| REINFORCE | None | Low | Low | Low |

---

## Mathematical Foundation

### Standard PPO Review

PPO optimizes:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$: Probability ratio
- $\hat{A}_t$: Advantage estimate (from value function)

**Problem**: Value function $V(s)$ is hard to learn and adds memory overhead.

### GRPO's Key Insight

Instead of learning $V(s)$, use **group statistics** as baseline:

$$\hat{A}_i = \frac{R_i - \mu_G}{\sigma_G}$$

Where:
- $R_i$: Reward for sample $i$
- $\mu_G = \frac{1}{|G|}\sum_{j \in G} R_j$: Group mean reward
- $\sigma_G = \sqrt{\frac{1}{|G|}\sum_{j \in G}(R_j - \mu_G)^2}$: Group std

### Group Formation

For each prompt, generate $K$ responses:

$$G = \{(x, y_1, R_1), (x, y_2, R_2), ..., (x, y_K, R_K)\}$$

All responses share the same prompt $x$, enabling fair comparison.

### GRPO Objective

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{G}\left[\frac{1}{K}\sum_{i=1}^{K} \min\left(r_i \hat{A}_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right]$$

With KL penalty:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}} + \beta \cdot D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$$

---

## Algorithm Details

### GRPO Training Loop

```
Algorithm: GRPO Training
────────────────────────────────────────
Input: Policy π_θ, Reference π_ref, Dataset D, Group size K

1. for batch B in D:
2.     for prompt x in B:
3.         # Generate K responses
4.         y_1, ..., y_K ~ π_θ(·|x)
5.         
6.         # Get rewards
7.         R_1, ..., R_K = Reward(x, y_1), ..., Reward(x, y_K)
8.         
9.         # Compute group statistics
10.        μ = mean(R_1, ..., R_K)
11.        σ = std(R_1, ..., R_K)
12.        
13.        # Compute normalized advantages
14.        A_i = (R_i - μ) / (σ + ε)  for i = 1..K
15.        
16.        # Store old log probs
17.        log_π_old_i = log π_θ(y_i|x)  for i = 1..K
18.    
19.    # PPO-style update (multiple epochs)
20.    for epoch in 1..E:
21.        for (x, y_i, A_i, log_π_old_i) in batch:
22.            log_π_new = log π_θ(y_i|x)
23.            ratio = exp(log_π_new - log_π_old_i)
24.            
25.            # Clipped objective
26.            L_clip = min(ratio * A_i, clip(ratio, 1-ε, 1+ε) * A_i)
27.            
28.            # KL penalty
29.            L_kl = β * (log_π_new - log π_ref(y_i|x))
30.            
31.            loss = -L_clip + L_kl
32.            
33.        optimizer.step(loss)
```

### Per-Token vs Per-Sequence

**Per-sequence GRPO** (simpler):
- One reward per response
- One advantage per response

**Per-token GRPO** (finer-grained):
- Apply advantage to each token
- Same advantage for all tokens in a response

$$\mathcal{L} = -\sum_{t=1}^{T} \hat{A} \cdot \log \pi_\theta(y_t | x, y_{<t})$$

---

## Implementation

### Rust Implementation

```rust
pub struct GRPOConfig {
    pub group_size: usize,       // K: responses per prompt
    pub clip_epsilon: f32,       // ε: PPO clip range
    pub kl_coef: f32,            // β: KL penalty coefficient
    pub normalize_advantages: bool,
    pub advantage_eps: f32,      // ε for numerical stability
}

pub struct GRPOTrainer {
    config: GRPOConfig,
    policy: Box<dyn LM>,
    reference: Box<dyn LM>,
    reward_model: Box<dyn RewardModel>,
}

impl GRPOTrainer {
    pub fn compute_grpo_loss(
        &self,
        prompts: &[String],
        responses: &[Vec<String>],  // K responses per prompt
        old_log_probs: &Tensor,     // (batch, K, seq_len)
    ) -> Result<(Tensor, GRPOStats)> {
        let batch_size = prompts.len();
        let k = self.config.group_size;
        
        // === Get Rewards ===
        let rewards = self.get_rewards(prompts, responses)?;  // (batch, K)
        
        // === Compute Group Statistics ===
        let (advantages, stats) = self.compute_advantages(&rewards)?;
        
        // === Compute Policy Loss ===
        let mut total_loss = Tensor::zeros((), DType::F32, &Device::Cpu)?;
        
        for b in 0..batch_size {
            for i in 0..k {
                let response = &responses[b][i];
                
                // Current policy log prob
                let log_prob_new = self.policy.log_prob(
                    &prompts[b], 
                    response
                )?;
                
                // Reference log prob (for KL)
                let log_prob_ref = self.reference.log_prob(
                    &prompts[b],
                    response
                )?;
                
                // Importance sampling ratio
                let log_prob_old = old_log_probs.i((b, i))?;
                let ratio = (log_prob_new.clone() - log_prob_old)?.exp()?;
                
                // Advantage for this response
                let advantage = advantages.i((b, i))?.to_scalar::<f32>()?;
                
                // Clipped objective
                let obj1 = (&ratio * advantage)?;
                let ratio_clipped = ratio.clamp(
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )?;
                let obj2 = (&ratio_clipped * advantage)?;
                
                let policy_loss = obj1.minimum(&obj2)?;
                
                // KL penalty
                let kl = (log_prob_new - log_prob_ref)?;
                let kl_loss = (&kl * self.config.kl_coef)?;
                
                // Combine
                let loss = (policy_loss.neg()? + kl_loss)?;
                total_loss = (total_loss + loss)?;
            }
        }
        
        let avg_loss = (total_loss / ((batch_size * k) as f64))?;
        
        Ok((avg_loss, stats))
    }
    
    fn compute_advantages(&self, rewards: &Tensor) -> Result<(Tensor, GRPOStats)> {
        // rewards: (batch, K)
        
        // Group statistics (per prompt)
        let mean = rewards.mean_keepdim(D::Minus1)?;  // (batch, 1)
        let std = rewards.std_keepdim(D::Minus1)?;   // (batch, 1)
        
        // Normalized advantages
        let advantages = (rewards - &mean)? / (&std + self.config.advantage_eps)?;
        
        let stats = GRPOStats {
            mean_reward: mean.mean_all()?.to_scalar::<f32>()?,
            std_reward: std.mean_all()?.to_scalar::<f32>()?,
            mean_advantage: advantages.mean_all()?.to_scalar::<f32>()?,
        };
        
        Ok((advantages, stats))
    }
    
    fn get_rewards(
        &self,
        prompts: &[String],
        responses: &[Vec<String>],
    ) -> Result<Tensor> {
        let batch_size = prompts.len();
        let k = self.config.group_size;
        
        let mut rewards = vec![0f32; batch_size * k];
        
        for b in 0..batch_size {
            for i in 0..k {
                rewards[b * k + i] = self.reward_model.score(
                    &prompts[b],
                    &responses[b][i],
                )?;
            }
        }
        
        Tensor::from_vec(rewards, (batch_size, k), &Device::Cpu)
    }
}

pub struct GRPOStats {
    pub mean_reward: f32,
    pub std_reward: f32,
    pub mean_advantage: f32,
}
```

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    group_size: int = 8          # K: responses per prompt
    clip_epsilon: float = 0.2    # PPO clip range
    kl_coef: float = 0.1         # KL penalty coefficient
    normalize_advantages: bool = True
    advantage_eps: float = 1e-8
    ppo_epochs: int = 4          # Update epochs per batch


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    
    Key idea: Use group statistics as baseline instead of learned value function.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        config: GRPOConfig,
        optimizer: torch.optim.Optimizer,
    ):
        self.policy = policy
        self.reference = reference.eval()  # Frozen reference
        self.reward_model = reward_model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optimizer
        
        # Freeze reference model
        for param in self.reference.parameters():
            param.requires_grad = False
    
    def train_step(
        self,
        prompts: List[str],
    ) -> Dict[str, float]:
        """
        Single GRPO training step.
        
        Args:
            prompts: List of prompts to generate responses for
        
        Returns:
            Dictionary of training statistics
        """
        batch_size = len(prompts)
        k = self.config.group_size
        
        # === Step 1: Generate K responses per prompt ===
        responses, old_log_probs = self._generate_responses(prompts)
        
        # === Step 2: Get rewards ===
        rewards = self._get_rewards(prompts, responses)  # (batch, K)
        
        # === Step 3: Compute advantages ===
        advantages = self._compute_advantages(rewards)   # (batch, K)
        
        # === Step 4: PPO-style updates ===
        stats = self._ppo_update(
            prompts, responses, old_log_probs, advantages
        )
        
        # Add reward stats
        stats['mean_reward'] = rewards.mean().item()
        stats['std_reward'] = rewards.std().item()
        
        return stats
    
    def _generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """Generate K responses per prompt and store log probs."""
        batch_size = len(prompts)
        k = self.config.group_size
        
        responses = []
        all_log_probs = []
        
        for prompt in prompts:
            prompt_responses = []
            prompt_log_probs = []
            
            for _ in range(k):
                # Generate response
                with torch.no_grad():
                    input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
                    output = self.policy.generate(
                        input_ids,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=1.0,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                response_ids = output.sequences[0][len(input_ids[0]):]
                response = self.tokenizer.decode(response_ids)
                
                # Compute log prob
                log_prob = self._compute_log_prob(prompt, response)
                
                prompt_responses.append(response)
                prompt_log_probs.append(log_prob)
            
            responses.append(prompt_responses)
            all_log_probs.append(torch.stack(prompt_log_probs))
        
        log_probs_tensor = torch.stack(all_log_probs)  # (batch, K)
        
        return responses, log_probs_tensor
    
    def _compute_log_prob(
        self,
        prompt: str,
        response: str,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute log probability of response given prompt."""
        if model is None:
            model = self.policy
        
        full_text = prompt + response
        input_ids = self.tokenizer.encode(full_text, return_tensors='pt')
        prompt_len = len(self.tokenizer.encode(prompt))
        
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)
        
        # Get log probs for response tokens only
        response_logits = logits[0, prompt_len-1:-1]  # (response_len, vocab_size)
        response_ids = input_ids[0, prompt_len:]      # (response_len,)
        
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_ids.unsqueeze(1)).squeeze(1)
        
        return token_log_probs.sum()  # Sum over tokens
    
    def _get_rewards(
        self,
        prompts: List[str],
        responses: List[List[str]],
    ) -> torch.Tensor:
        """Get rewards from reward model."""
        batch_size = len(prompts)
        k = self.config.group_size
        
        rewards = torch.zeros(batch_size, k)
        
        with torch.no_grad():
            for b, prompt in enumerate(prompts):
                for i, response in enumerate(responses[b]):
                    full_text = prompt + response
                    input_ids = self.tokenizer.encode(
                        full_text, return_tensors='pt'
                    )
                    reward = self.reward_model(input_ids)
                    rewards[b, i] = reward.item()
        
        return rewards
    
    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-normalized advantages.
        
        This is the key GRPO innovation: use group statistics
        instead of a learned value function.
        """
        # rewards: (batch, K)
        
        # Compute per-group (per-prompt) statistics
        group_mean = rewards.mean(dim=1, keepdim=True)  # (batch, 1)
        group_std = rewards.std(dim=1, keepdim=True)    # (batch, 1)
        
        # Normalize
        advantages = (rewards - group_mean) / (group_std + self.config.advantage_eps)
        
        return advantages
    
    def _ppo_update(
        self,
        prompts: List[str],
        responses: List[List[str]],
        old_log_probs: torch.Tensor,  # (batch, K)
        advantages: torch.Tensor,      # (batch, K)
    ) -> Dict[str, float]:
        """PPO-style clipped update."""
        batch_size = len(prompts)
        k = self.config.group_size
        
        stats = {
            'policy_loss': 0.0,
            'kl_loss': 0.0,
            'clip_fraction': 0.0,
        }
        
        for epoch in range(self.config.ppo_epochs):
            epoch_policy_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_clip_count = 0
            total_samples = 0
            
            for b in range(batch_size):
                for i in range(k):
                    prompt = prompts[b]
                    response = responses[b][i]
                    old_log_prob = old_log_probs[b, i]
                    advantage = advantages[b, i]
                    
                    # Current log prob
                    self.policy.train()
                    new_log_prob = self._compute_log_prob(prompt, response)
                    
                    # Reference log prob
                    with torch.no_grad():
                        ref_log_prob = self._compute_log_prob(
                            prompt, response, self.reference
                        )
                    
                    # Importance sampling ratio
                    ratio = torch.exp(new_log_prob - old_log_prob.detach())
                    
                    # Clipped objective
                    obj1 = ratio * advantage
                    obj2 = torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    ) * advantage
                    
                    policy_loss = -torch.min(obj1, obj2)
                    
                    # Track clipping
                    clip_fraction = (ratio != obj2 / advantage).float().mean()
                    
                    # KL penalty
                    kl = new_log_prob - ref_log_prob
                    kl_loss = self.config.kl_coef * kl
                    
                    # Total loss
                    loss = policy_loss + kl_loss
                    
                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Track stats
                    epoch_policy_loss += policy_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    epoch_clip_count += clip_fraction.item()
                    total_samples += 1
            
            # Average over epoch
            stats['policy_loss'] += epoch_policy_loss / total_samples
            stats['kl_loss'] += epoch_kl_loss / total_samples
            stats['clip_fraction'] += epoch_clip_count / total_samples
        
        # Average over PPO epochs
        for key in stats:
            stats[key] /= self.config.ppo_epochs
        
        return stats


# === Efficient Batch Implementation ===

class GRPOBatchTrainer(GRPOTrainer):
    """
    More efficient batch-based GRPO implementation.
    Processes all responses in parallel rather than sequentially.
    """
    
    def _ppo_update_batch(
        self,
        prompts: List[str],
        responses: List[List[str]],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """Batched PPO update for efficiency."""
        
        # Flatten for batched processing
        all_prompts = []
        all_responses = []
        flat_old_log_probs = []
        flat_advantages = []
        
        for b, prompt in enumerate(prompts):
            for i, response in enumerate(responses[b]):
                all_prompts.append(prompt)
                all_responses.append(response)
                flat_old_log_probs.append(old_log_probs[b, i])
                flat_advantages.append(advantages[b, i])
        
        flat_old_log_probs = torch.stack(flat_old_log_probs)
        flat_advantages = torch.stack(flat_advantages)
        
        # Batch tokenization
        full_texts = [p + r for p, r in zip(all_prompts, all_responses)]
        encodings = self.tokenizer(
            full_texts, 
            return_tensors='pt', 
            padding=True
        )
        
        # Batched forward pass
        # ... (implementation details)
        
        return {}
```

---

## Advantages Over PPO

### 1. No Critic Network

**PPO**: Needs separate value network $V_\phi(s)$
- Extra parameters to train
- Value estimation is noisy
- Memory overhead

**GRPO**: Uses group statistics
- No extra parameters
- Direct reward comparison
- Memory efficient

### 2. More Stable Advantages

PPO advantage with GAE:
$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

GRPO advantage:
$$\hat{A}_i = \frac{R_i - \mu_G}{\sigma_G}$$

GRPO is simpler and more stable.

### 3. Better for Language Models

| Aspect | PPO | GRPO |
|--------|-----|------|
| State definition | Complex (hidden states) | Simple (prompt) |
| Reward timing | Per-token (approx) | Per-sequence |
| Baseline | Learned (noisy) | Empirical (exact) |

---

## Hyperparameter Guide

### Recommended Settings

| Hyperparameter | Range | DeepSeek Default |
|----------------|-------|------------------|
| `group_size` (K) | 4-16 | 8 |
| `clip_epsilon` | 0.1-0.3 | 0.2 |
| `kl_coef` | 0.01-0.5 | 0.1 |
| `ppo_epochs` | 2-6 | 4 |
| `learning_rate` | 1e-6 to 1e-5 | 5e-6 |
| `batch_size` | 8-64 | 16 |

### Tuning Tips

1. **group_size**: Larger = more stable but slower
2. **clip_epsilon**: Lower = more conservative updates
3. **kl_coef**: Higher = stay closer to reference
4. **ppo_epochs**: More = better sample efficiency, risk overfitting

---

## Training Dynamics

### Reward Curve

Typical GRPO training progression:

```
Reward
   ^
   |          ╱───────────  plateau
   |        ╱
   |      ╱
   |    ╱
   |  ╱
   |╱
   └──────────────────────────▶ Steps
     rapid        gradual
     improvement  improvement
```

### Monitoring Metrics

1. **Mean reward**: Should increase
2. **Advantage std**: Should be ~1 (normalized)
3. **KL divergence**: Should stay bounded
4. **Clip fraction**: Should be 0.1-0.3

---

## Common Issues and Solutions

### Issue 1: Reward Hacking

**Symptom**: High reward but poor quality
**Solution**: Increase KL penalty, use better reward model

### Issue 2: Training Instability

**Symptom**: Loss spikes, NaN
**Solution**: Lower learning rate, increase group_size

### Issue 3: Slow Learning

**Symptom**: Reward barely increases
**Solution**: Decrease KL penalty, check reward model quality

### Issue 4: Mode Collapse

**Symptom**: All responses similar
**Solution**: Increase temperature, use entropy bonus

---

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917)
