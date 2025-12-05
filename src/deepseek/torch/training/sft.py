"""
Supervised Fine-Tuning (SFT) for DeepSeek

This module implements:
- SFT trainer with LoRA support
- Chat template formatting
- NEFTune noise augmentation
- Efficient data loading for instruction tuning

Based on: DeepSeek LLM paper (https://arxiv.org/abs/2401.02954)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
import os
import json
import copy


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SFTConfig:
    """Configuration for SFT training."""
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
    max_grad_norm: float = 1.0
    
    # NEFTune (noise augmentation)
    use_neftune: bool = True
    neftune_alpha: float = 5.0
    
    # Checkpointing
    save_steps: int = 500
    save_dir: str = "./sft_checkpoints"
    
    # Logging
    log_steps: int = 10


# =============================================================================
# Chat Template
# =============================================================================

class DeepSeekChatTemplate:
    """
    Chat template for DeepSeek models.
    
    Format:
    <|system|>
    {system_message}
    <|end|>
    <|user|>
    {user_message}
    <|end|>
    <|assistant|>
    {assistant_message}
    <|end|>
    """
    
    SYSTEM_TOKEN = "<|system|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    END_TOKEN = "<|end|>"
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def format_message(self, role: str, content: str) -> str:
        """Format a single message."""
        token_map = {
            "system": self.SYSTEM_TOKEN,
            "user": self.USER_TOKEN,
            "assistant": self.ASSISTANT_TOKEN,
        }
        token = token_map.get(role, self.USER_TOKEN)
        return f"{token}\n{content}\n{self.END_TOKEN}\n"
    
    def format_conversation(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Format a full conversation.
        
        Args:
            messages: List of {"role": str, "content": str} dicts
            add_generation_prompt: Whether to add assistant prompt at end
        
        Returns:
            Formatted conversation string
        """
        formatted = ""
        
        for msg in messages:
            formatted += self.format_message(msg["role"], msg["content"])
        
        if add_generation_prompt:
            formatted += f"{self.ASSISTANT_TOKEN}\n"
        
        return formatted
    
    def get_assistant_mask(
        self,
        input_ids: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        """
        Create mask indicating assistant response tokens.
        
        Returns:
            Boolean tensor where True = assistant token (compute loss)
        """
        # Get token IDs for markers
        assistant_token_id = tokenizer.encode(
            self.ASSISTANT_TOKEN, add_special_tokens=False
        )[0]
        end_token_id = tokenizer.encode(
            self.END_TOKEN, add_special_tokens=False
        )[0]
        
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # For each sequence in batch
        for i in range(input_ids.size(0)):
            in_assistant = False
            for j in range(input_ids.size(1)):
                token_id = input_ids[i, j].item()
                
                if token_id == assistant_token_id:
                    in_assistant = True
                    continue
                
                if token_id == end_token_id:
                    in_assistant = False
                    continue
                
                if in_assistant:
                    mask[i, j] = True
        
        return mask
    
    def create_labels(
        self,
        input_ids: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        """
        Create labels that mask non-assistant tokens.
        
        Returns:
            Labels tensor with -100 for masked positions
        """
        assistant_mask = self.get_assistant_mask(input_ids, tokenizer)
        labels = input_ids.clone()
        labels[~assistant_mask] = -100
        return labels


# =============================================================================
# LoRA Implementation
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer.
    
    W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Linear(original_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        result = self.original(x)
        
        # LoRA output
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        
        return result + lora_out * self.scaling
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into original linear layer."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        
        # W' = W + scaling * B @ A
        delta = self.lora_B.weight @ self.lora_A.weight * self.scaling
        merged.weight.data = self.original.weight.data + delta
        
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        
        return merged


def apply_lora(
    model: nn.Module,
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    target_modules: List[str] = None,
) -> nn.Module:
    """
    Apply LoRA to target modules in a model.
    
    Args:
        model: The model to modify
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
        target_modules: List of module name patterns to target
    
    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                parent = model.get_submodule(parent_name) if parent_name else model
                
                lora_linear = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_linear)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {trainable:,} trainable / {total:,} total params "
          f"({100 * trainable / total:.2f}%)")
    
    return model


# =============================================================================
# NEFTune
# =============================================================================

class NEFTuneEmbedding(nn.Module):
    """
    NEFTune: Noise-augmented embedding for better fine-tuning.
    
    Adds uniform noise to embeddings during training:
    e' = e + U(-α/√L, α/√L)
    
    where L is sequence length and α is noise magnitude.
    """
    
    def __init__(self, embedding: nn.Embedding, alpha: float = 5.0):
        super().__init__()
        self.embedding = embedding
        self.alpha = alpha
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        
        if self.training and self.alpha > 0:
            # Add noise scaled by sequence length
            seq_len = embeddings.size(1)
            noise_scale = self.alpha / (seq_len ** 0.5)
            noise = torch.empty_like(embeddings).uniform_(-noise_scale, noise_scale)
            embeddings = embeddings + noise
        
        return embeddings


def apply_neftune(model: nn.Module, alpha: float = 5.0) -> nn.Module:
    """Apply NEFTune to model's embedding layer."""
    # Find embedding layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "embed" in name.lower():
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            parent = model.get_submodule(parent_name) if parent_name else model
            
            neftune_emb = NEFTuneEmbedding(module, alpha=alpha)
            setattr(parent, child_name, neftune_emb)
            print(f"NEFTune applied to {name} with alpha={alpha}")
            break
    
    return model


# =============================================================================
# Dataset
# =============================================================================

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    
    Expects data in format:
    [
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]},
        ...
    ]
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 4096,
        chat_template: Optional[DeepSeekChatTemplate] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = chat_template or DeepSeekChatTemplate(tokenizer)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        messages = item.get("messages", [])
        
        # Format conversation
        text = self.chat_template.format_conversation(messages)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Create labels (mask non-assistant tokens)
        labels = self.chat_template.create_labels(
            input_ids.unsqueeze(0), self.tokenizer
        ).squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =============================================================================
# SFT Trainer
# =============================================================================

class SFTTrainer:
    """
    Supervised Fine-Tuning trainer.
    
    Features:
    - LoRA support
    - NEFTune noise augmentation
    - Gradient accumulation
    - Mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SFTConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.chat_template = DeepSeekChatTemplate(tokenizer)
        
        # Device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        
        # Apply LoRA
        if config.use_lora:
            model = apply_lora(
                model,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
        
        # Apply NEFTune
        if config.use_neftune:
            model = apply_neftune(model, alpha=config.neftune_alpha)
        
        self.model = model.to(device)
        
        # Optimizer (only LoRA parameters if using LoRA)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
        
        # State
        self.global_step = 0
        self.accumulated_loss = 0.0
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        print(f"SFTTrainer initialized on {device}")
        print(f"LoRA: {config.use_lora}, NEFTune: {config.use_neftune}")
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SFT loss (cross-entropy on labeled tokens)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Cross-entropy (ignoring -100)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        return loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        self.model.train()
        
        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass (with mixed precision if available)
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(input_ids, attention_mask, labels)
                loss = loss / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            loss = self.compute_loss(input_ids, attention_mask, labels)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
        
        self.accumulated_loss += loss.item()
        
        return {"loss": loss.item() * self.config.gradient_accumulation_steps}
    
    def optimizer_step(self) -> Dict[str, float]:
        """Perform optimizer step after accumulation."""
        # Gradient clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update state
        self.global_step += 1
        avg_loss = self.accumulated_loss
        self.accumulated_loss = 0.0
        
        return {
            "avg_loss": avg_loss,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "step": self.global_step,
        }
    
    def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
    ):
        """
        Full training loop.
        
        Args:
            train_data: List of training examples
            eval_data: Optional list of evaluation examples
        """
        # Create dataset and dataloader
        dataset = SFTDataset(
            train_data,
            self.tokenizer,
            max_length=self.config.max_seq_length,
            chat_template=self.chat_template,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid tokenizer issues
        )
        
        # Training loop
        accum_step = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                # Train step
                step_metrics = self.train_step(batch)
                accum_step += 1
                
                # Optimizer step
                if accum_step >= self.config.gradient_accumulation_steps:
                    metrics = self.optimizer_step()
                    accum_step = 0
                    
                    # Logging
                    if self.global_step % self.config.log_steps == 0:
                        print(
                            f"  Step {metrics['step']:5d} | "
                            f"Loss: {metrics['avg_loss']:.4f} | "
                            f"Grad: {metrics['grad_norm']:.2f}"
                        )
                    
                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
            
            # Evaluate at end of epoch
            if eval_data:
                eval_loss = self.evaluate(eval_data)
                print(f"  Epoch {epoch + 1} eval loss: {eval_loss:.4f}")
        
        # Final save
        self.save_checkpoint(final=True)
        print("\nTraining complete!")
    
    @torch.no_grad()
    def evaluate(self, eval_data: List[Dict]) -> float:
        """Evaluate on held-out data."""
        self.model.eval()
        
        dataset = SFTDataset(
            eval_data,
            self.tokenizer,
            max_length=self.config.max_seq_length,
            chat_template=self.chat_template,
        )
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            loss = self.compute_loss(input_ids, attention_mask, labels)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.config.save_dir, f"sft_checkpoint_{suffix}.pt")
        
        # Save only LoRA weights if using LoRA
        if self.config.use_lora:
            lora_state = {}
            for name, module in self.model.named_modules():
                if isinstance(module, LoRALinear):
                    lora_state[f"{name}.lora_A"] = module.lora_A.state_dict()
                    lora_state[f"{name}.lora_B"] = module.lora_B.state_dict()
            
            torch.save({
                "lora_state": lora_state,
                "step": self.global_step,
                "config": self.config.__dict__,
            }, path)
        else:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "step": self.global_step,
                "config": self.config.__dict__,
            }, path)
        
        print(f"Checkpoint saved: {path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate response for a prompt."""
        self.model.eval()
        
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted = self.chat_template.format_conversation(
            messages, add_generation_prompt=True
        )
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][input_ids.size(1):],
            skip_special_tokens=True,
        )
        
        return response


# =============================================================================
# Demo
# =============================================================================


