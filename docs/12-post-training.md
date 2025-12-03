# Post-Training: SFT and Reinforcement Learning

## Overview

This document covers the complete post-training pipeline used by DeepSeek after pre-training, including:

1. **Supervised Fine-Tuning (SFT)**: Teaching the model to follow instructions
2. **Reward Modeling**: Training a model to predict human preferences
3. **Direct Preference Optimization (DPO)**: Alignment without explicit RL
4. **GRPO Integration**: Combining with DeepSeek's GRPO for advanced alignment

**Key Papers:**
- [DeepSeek LLM](https://arxiv.org/abs/2401.02954) - SFT and DPO pipeline
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) - GRPO algorithm
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO paper

---

## When to Use What

### Decision Guide

| Method | Use When | Pros | Cons |
|--------|----------|------|------|
| **SFT** | Initial instruction following | Simple, stable | Limited preference learning |
| **Reward Model + PPO** | Full RLHF pipeline | Powerful | Complex, unstable |
| **DPO** | Direct preference learning | Simple, stable | Needs paired data |
| **GRPO** | Math/code reasoning | Memory efficient | Needs reward function |

### Typical Pipeline

```
Pre-trained Model
       │
       ▼
┌──────────────┐
│     SFT      │  ← Instruction data (1-2M examples)
└──────────────┘
       │
       ▼
┌──────────────┐
│  DPO/GRPO    │  ← Preference data
└──────────────┘
       │
       ▼
  Aligned Model
```

---

## Supervised Fine-Tuning (SFT)

### Mathematical Foundation

SFT minimizes the negative log-likelihood on instruction-response pairs:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log P_\theta(y_t | x, y_{<t})$$

Where:
- $x$ = instruction/prompt
- $y = (y_1, ..., y_T)$ = response tokens
- $\theta$ = model parameters

### Chat Template Masking

Only compute loss on assistant tokens, not user prompts:

```
<|user|> What is 2+2? <|assistant|> The answer is 4.
[  masked - no loss  ]  [  compute loss here  ]
```

### LoRA for Efficient Fine-Tuning

Instead of fine-tuning all parameters, use Low-Rank Adaptation:

$$W' = W + BA$$

Where:
- $W \in \mathbb{R}^{d \times k}$ = original weights (frozen)
- $B \in \mathbb{R}^{d \times r}$ = low-rank down projection
- $A \in \mathbb{R}^{r \times k}$ = low-rank up projection
- $r \ll \min(d, k)$ = rank (typically 8-64)

Memory savings:
$$\text{Trainable params} = r \times (d + k) \ll d \times k$$

### Implementation

```python
@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    model_name_or_path: str = "deepseek-ai/deepseek-llm-7b-base"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    
    # Regularization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    # NEFTune (noise augmentation)
    use_neftune: bool = True
    neftune_alpha: float = 5.0


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer with LoRA support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SFTConfig,
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Apply LoRA if enabled
        if config.use_lora:
            self.model = self._apply_lora(model)
        else:
            self.model = model
        
        # Chat template
        self.chat_template = DeepSeekChatTemplate(tokenizer)
    
    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA to target modules."""
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        return get_peft_model(model, lora_config)
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SFT loss with optional NEFTune."""
        
        # NEFTune: Add noise to embeddings
        if self.config.use_neftune and self.model.training:
            embeddings = self.model.get_input_embeddings()(input_ids)
            noise = torch.randn_like(embeddings) * self.config.neftune_alpha / (embeddings.size(1) ** 0.5)
            embeddings = embeddings + noise
            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Cross-entropy loss (ignoring masked positions)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        return loss
```

### Chat Template

```python
class DeepSeekChatTemplate:
    """
    DeepSeek chat template for formatting conversations.
    """
    
    SYSTEM_TOKEN = "<|system|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    END_TOKEN = "<|end|>"
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def format_conversation(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Format conversation into model input."""
        formatted = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"{self.SYSTEM_TOKEN}\n{content}\n{self.END_TOKEN}\n"
            elif role == "user":
                formatted += f"{self.USER_TOKEN}\n{content}\n{self.END_TOKEN}\n"
            elif role == "assistant":
                formatted += f"{self.ASSISTANT_TOKEN}\n{content}\n{self.END_TOKEN}\n"
        
        if add_generation_prompt:
            formatted += f"{self.ASSISTANT_TOKEN}\n"
        
        return formatted
    
    def create_labels(
        self,
        input_ids: torch.Tensor,
        assistant_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create labels that mask non-assistant tokens.
        """
        labels = input_ids.clone()
        labels[~assistant_mask] = -100  # Mask non-assistant tokens
        return labels
```

---

## Reward Modeling

### Mathematical Foundation

Train a reward model to predict human preferences using the Bradley-Terry model:

$$P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

Where:
- $y_w$ = preferred (winning) response
- $y_l$ = rejected (losing) response
- $r_\theta$ = reward model
- $\sigma$ = sigmoid function

### Loss Function

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)}[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]$$

### Architecture

```
Input (prompt + response)
        │
        ▼
┌───────────────────┐
│  Transformer      │  (shared with LLM)
│  Backbone         │
└───────────────────┘
        │
        ▼
   [Last Token]
        │
        ▼
┌───────────────────┐
│  Reward Head      │  (linear: hidden_dim → 1)
└───────────────────┘
        │
        ▼
   Scalar Reward
```

### Implementation

```python
class RewardModel(nn.Module):
    """
    Reward model for preference learning.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        self.pad_token_id = pad_token_id
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reward for input sequences.
        
        Returns:
            Tensor of shape (batch_size,) with scalar rewards
        """
        # Get hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Get last non-padding token for each sequence
        batch_size = input_ids.size(0)
        seq_lengths = attention_mask.sum(dim=1) - 1
        
        last_hidden = hidden_states[
            torch.arange(batch_size, device=input_ids.device),
            seq_lengths
        ]
        
        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)
        
        return rewards


class RewardTrainer:
    """
    Trainer for reward model.
    """
    
    def __init__(
        self,
        model: RewardModel,
        config: RewardConfig,
    ):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    
    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        margin: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute preference loss with optional margin.
        
        Args:
            chosen_ids: Token IDs for preferred responses
            chosen_mask: Attention mask for preferred
            rejected_ids: Token IDs for rejected responses
            rejected_mask: Attention mask for rejected
            margin: Optional margin for ranking loss
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary with accuracy and reward stats
        """
        # Get rewards
        chosen_rewards = self.model(chosen_ids, chosen_mask)
        rejected_rewards = self.model(rejected_ids, rejected_mask)
        
        # Bradley-Terry loss with margin
        logits = chosen_rewards - rejected_rewards - margin
        loss = -F.logsigmoid(logits).mean()
        
        # Metrics
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }
        
        return loss, metrics
```

---

## Direct Preference Optimization (DPO)

### Mathematical Foundation

DPO directly optimizes the policy without explicit reward modeling:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

Where:
- $\pi_\theta$ = policy being trained
- $\pi_{\text{ref}}$ = reference policy (frozen)
- $\beta$ = temperature parameter (controls deviation from reference)

### Key Insight

DPO is derived by reparametrizing the RLHF objective:

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

This allows direct optimization without:
1. Training a separate reward model
2. RL training with PPO
3. Value function estimation

### DPO Variants

| Variant | Modification | Use Case |
|---------|--------------|----------|
| **DPO** | Standard | General alignment |
| **IPO** | $\log(1 + e^{-\beta \cdot \Delta})$ | Better calibration |
| **KTO** | Asymmetric loss | When only have good/bad labels |
| **ORPO** | Odds ratio | Reference-free |

### Implementation

```python
@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta: float = 0.1              # KL penalty coefficient
    label_smoothing: float = 0.0   # Label smoothing for stability
    loss_type: str = "sigmoid"     # "sigmoid", "ipo", "kto"
    reference_free: bool = False   # Use log-ratio trick without reference
    
    # Training
    learning_rate: float = 1e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    
    # Regularization
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01


class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        tokenizer,
        config: DPOConfig,
    ):
        self.policy = policy
        self.reference = reference
        self.tokenizer = tokenizer
        self.config = config
        
        # Freeze reference model
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of responses.
        
        Returns:
            Tensor of shape (batch_size,) with sequence log probs
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs of actual tokens
        # labels shape: (batch, seq_len)
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (labels != -100).float()
        token_log_probs = token_log_probs * mask
        
        # Sum over sequence
        seq_log_probs = token_log_probs.sum(dim=-1)
        
        return seq_log_probs
    
    def compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        """
        # Policy log probs
        policy_chosen_logps = self.compute_log_probs(
            self.policy, chosen_ids, chosen_mask, chosen_labels
        )
        policy_rejected_logps = self.compute_log_probs(
            self.policy, rejected_ids, rejected_mask, rejected_labels
        )
        
        # Reference log probs
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.reference, chosen_ids, chosen_mask, chosen_labels
            )
            ref_rejected_logps = self.compute_log_probs(
                self.reference, rejected_ids, rejected_mask, rejected_labels
            )
        
        # Compute log ratios
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # DPO loss
        logits = self.config.beta * (pi_logratios - ref_logratios)
        
        if self.config.loss_type == "sigmoid":
            # Standard DPO
            loss = -F.logsigmoid(logits).mean()
        elif self.config.loss_type == "ipo":
            # Identity Preference Optimization
            loss = (logits - 1 / (2 * self.config.beta)) ** 2
            loss = loss.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Label smoothing
        if self.config.label_smoothing > 0:
            smooth_loss = -F.logsigmoid(-logits).mean()
            loss = (1 - self.config.label_smoothing) * loss + \
                   self.config.label_smoothing * smooth_loss
        
        # Metrics
        chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
        
        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
            "margin": (chosen_rewards - rejected_rewards).mean().item(),
        }
        
        return loss, metrics
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.policy.train()
        
        loss, metrics = self.compute_dpo_loss(
            chosen_ids=batch["chosen_input_ids"],
            chosen_mask=batch["chosen_attention_mask"],
            chosen_labels=batch["chosen_labels"],
            rejected_ids=batch["rejected_input_ids"],
            rejected_mask=batch["rejected_attention_mask"],
            rejected_labels=batch["rejected_labels"],
        )
        
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm,
        )
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        metrics["grad_norm"] = grad_norm.item()
        
        return metrics
```

---

## Rust Implementation

### SFT Trainer

```rust
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{VarBuilder, VarMap, Module, ops};

/// SFT Configuration
#[derive(Clone, Debug)]
pub struct SFTConfig {
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub max_seq_length: usize,
    pub weight_decay: f64,
    pub warmup_ratio: f64,
    pub gradient_accumulation_steps: usize,
    pub max_grad_norm: f64,
}

impl Default for SFTConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-5,
            num_epochs: 3,
            batch_size: 4,
            max_seq_length: 4096,
            weight_decay: 0.01,
            warmup_ratio: 0.03,
            gradient_accumulation_steps: 8,
            max_grad_norm: 1.0,
        }
    }
}

/// Compute SFT loss (cross-entropy on response tokens)
pub fn compute_sft_loss(
    logits: &Tensor,    // (batch, seq, vocab)
    labels: &Tensor,    // (batch, seq) with -100 for masked
) -> Result<Tensor> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    
    // Shift for next-token prediction
    let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
    let shift_labels = labels.narrow(1, 1, seq_len - 1)?;
    
    // Flatten
    let flat_logits = shift_logits.reshape((batch_size * (seq_len - 1), vocab_size))?;
    let flat_labels = shift_labels.reshape((batch_size * (seq_len - 1),))?;
    
    // Create mask for valid labels (not -100)
    // Note: In Candle, we handle this differently than PyTorch
    let mask = flat_labels.ge(&Tensor::zeros((), DType::I64, flat_labels.device())?)?;
    
    // Log softmax
    let log_probs = ops::log_softmax(&flat_logits, D::Minus1)?;
    
    // Gather target log probs
    let flat_labels_u32 = flat_labels.to_dtype(DType::U32)?;
    let target_log_probs = log_probs
        .gather(&flat_labels_u32.unsqueeze(1)?, 1)?
        .squeeze(1)?;
    
    // Apply mask and compute mean
    let masked_log_probs = (target_log_probs * mask.to_dtype(DType::F32)?)?;
    let num_valid = mask.to_dtype(DType::F32)?.sum_all()?;
    
    let loss = (masked_log_probs.sum_all()?.neg()? / num_valid)?;
    
    Ok(loss)
}

/// Chat template for DeepSeek
pub struct ChatTemplate {
    pub system_token: String,
    pub user_token: String,
    pub assistant_token: String,
    pub end_token: String,
}

impl Default for ChatTemplate {
    fn default() -> Self {
        Self {
            system_token: "<|system|>".to_string(),
            user_token: "<|user|>".to_string(),
            assistant_token: "<|assistant|>".to_string(),
            end_token: "<|end|>".to_string(),
        }
    }
}

impl ChatTemplate {
    pub fn format_message(&self, role: &str, content: &str) -> String {
        let token = match role {
            "system" => &self.system_token,
            "user" => &self.user_token,
            "assistant" => &self.assistant_token,
            _ => &self.user_token,
        };
        format!("{}\n{}\n{}\n", token, content, self.end_token)
    }
    
    pub fn format_conversation(&self, messages: &[(String, String)]) -> String {
        messages.iter()
            .map(|(role, content)| self.format_message(role, content))
            .collect()
    }
}
```

### DPO Trainer

```rust
/// DPO Configuration
#[derive(Clone, Debug)]
pub struct DPOConfig {
    pub beta: f64,              // KL penalty coefficient
    pub label_smoothing: f64,   // For stability
    pub learning_rate: f64,
    pub max_grad_norm: f64,
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
            learning_rate: 1e-6,
            max_grad_norm: 1.0,
        }
    }
}

/// Compute sequence log probabilities
pub fn compute_seq_log_probs(
    logits: &Tensor,  // (batch, seq, vocab)
    labels: &Tensor,  // (batch, seq)
) -> Result<Tensor> {
    let (batch_size, seq_len, _vocab_size) = logits.dims3()?;
    
    // Log softmax
    let log_probs = ops::log_softmax(logits, D::Minus1)?;
    
    // Gather target log probs
    let labels_u32 = labels.to_dtype(DType::U32)?;
    let token_log_probs = log_probs
        .gather(&labels_u32.unsqueeze(D::Minus1)?, D::Minus1)?
        .squeeze(D::Minus1)?;
    
    // Create mask (labels != -100)
    let mask = labels.ge(&Tensor::zeros((), DType::I64, labels.device())?)?;
    let masked_log_probs = (token_log_probs * mask.to_dtype(DType::F32)?)?;
    
    // Sum over sequence
    let seq_log_probs = masked_log_probs.sum(D::Minus1)?;
    
    Ok(seq_log_probs)
}

/// Compute DPO loss
pub fn compute_dpo_loss(
    policy_chosen_logps: &Tensor,    // (batch,)
    policy_rejected_logps: &Tensor,  // (batch,)
    ref_chosen_logps: &Tensor,       // (batch,)
    ref_rejected_logps: &Tensor,     // (batch,)
    beta: f64,
    label_smoothing: f64,
) -> Result<(Tensor, DPOMetrics)> {
    // Compute log ratios
    let pi_logratios = (policy_chosen_logps - policy_rejected_logps)?;
    let ref_logratios = (ref_chosen_logps - ref_rejected_logps)?;
    
    // DPO logits
    let logits = ((pi_logratios - ref_logratios)? * beta)?;
    
    // Sigmoid loss: -log(sigmoid(logits))
    let loss = ops::log_softmax(&logits.stack(&[logits.neg()?], 0)?, 0)?
        .i(0)?
        .neg()?
        .mean_all()?;
    
    // Apply label smoothing if needed
    let final_loss = if label_smoothing > 0.0 {
        let smooth_loss = ops::log_softmax(&logits.neg()?.stack(&[logits.clone()], 0)?, 0)?
            .i(0)?
            .neg()?
            .mean_all()?;
        ((loss * (1.0 - label_smoothing))? + (smooth_loss * label_smoothing)?)?
    } else {
        loss
    };
    
    // Compute metrics
    let chosen_rewards = ((policy_chosen_logps - ref_chosen_logps)? * beta)?;
    let rejected_rewards = ((policy_rejected_logps - ref_rejected_logps)? * beta)?;
    
    let accuracy = chosen_rewards.gt(&rejected_rewards)?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?;
    
    let metrics = DPOMetrics {
        loss: final_loss.to_scalar::<f32>()?,
        chosen_reward: chosen_rewards.mean_all()?.to_scalar::<f32>()?,
        rejected_reward: rejected_rewards.mean_all()?.to_scalar::<f32>()?,
        accuracy,
    };
    
    Ok((final_loss, metrics))
}

#[derive(Debug, Clone)]
pub struct DPOMetrics {
    pub loss: f32,
    pub chosen_reward: f32,
    pub rejected_reward: f32,
    pub accuracy: f32,
}
```

---

## Combining with GRPO

DeepSeek uses a two-stage alignment:

### Stage 1: SFT
- Train on instruction-following data
- 1-2M high-quality examples
- Learning rate: 2e-5

### Stage 2: GRPO/DPO
- For general alignment: DPO
- For reasoning (math/code): GRPO

```python
class TwoStageAlignment:
    """
    Two-stage alignment following DeepSeek's approach.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        sft_config: SFTConfig,
        dpo_config: DPOConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sft_config = sft_config
        self.dpo_config = dpo_config
    
    def stage1_sft(
        self,
        sft_data: List[Dict],
        num_epochs: int = 3,
    ):
        """Stage 1: Supervised Fine-Tuning."""
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.sft_config,
        )
        
        # Train
        for epoch in range(num_epochs):
            for batch in sft_data:
                loss = trainer.train_step(batch)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return self.model
    
    def stage2_dpo(
        self,
        preference_data: List[Dict],
        num_epochs: int = 1,
    ):
        """Stage 2: DPO alignment."""
        # Create reference model (copy of current)
        reference = copy.deepcopy(self.model)
        
        trainer = DPOTrainer(
            policy=self.model,
            reference=reference,
            tokenizer=self.tokenizer,
            config=self.dpo_config,
        )
        
        # Train
        for epoch in range(num_epochs):
            for batch in preference_data:
                metrics = trainer.train_step(batch)
                print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}, "
                      f"Acc: {metrics['accuracy']:.2%}")
        
        return self.model
    
    def stage2_grpo(
        self,
        prompts: List[str],
        reward_fn: Callable,
        num_steps: int = 1000,
    ):
        """Stage 2: GRPO alignment (for reasoning tasks)."""
        from grpo import GRPOTrainer, GRPOConfig
        
        trainer = GRPOTrainer(
            policy=self.model,
            reference=copy.deepcopy(self.model),
            reward_fn=reward_fn,
            config=GRPOConfig(),
        )
        
        for step in range(num_steps):
            metrics = trainer.train_step(prompts)
            if step % 10 == 0:
                print(f"Step {step}, Loss: {metrics['loss']:.4f}")
        
        return self.model
```

---

## Hyperparameter Recommendations

### SFT

| Parameter | Range | DeepSeek Default |
|-----------|-------|------------------|
| Learning Rate | 1e-6 to 5e-5 | 2e-5 |
| Epochs | 2-5 | 3 |
| Batch Size | 4-32 | 8-16 |
| LoRA Rank | 8-128 | 64 |
| LoRA Alpha | 16-256 | 128 |

### DPO

| Parameter | Range | DeepSeek Default |
|-----------|-------|------------------|
| Beta | 0.01 to 0.5 | 0.1 |
| Learning Rate | 1e-7 to 5e-6 | 1e-6 |
| Epochs | 1-3 | 1 |
| Label Smoothing | 0.0 to 0.1 | 0.0 |

---

## Evaluation

### SFT Evaluation
- **Perplexity**: On held-out instruction data
- **BLEU/ROUGE**: Response quality
- **Human evaluation**: Instruction following

### DPO/GRPO Evaluation
- **Win rate**: Against reference model
- **MT-Bench**: Multi-turn conversation
- **AlpacaEval**: Instruction following benchmark
- **Task-specific**: MATH, HumanEval, etc.

---

## References

1. [DeepSeek LLM Paper](https://arxiv.org/abs/2401.02954)
2. [DPO Paper](https://arxiv.org/abs/2305.18290)
3. [LoRA Paper](https://arxiv.org/abs/2106.09685)
4. [RLHF Survey](https://arxiv.org/abs/2307.15217)
5. [NEFTune Paper](https://arxiv.org/abs/2310.05914)
