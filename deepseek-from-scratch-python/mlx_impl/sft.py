import mlx.core as mx
from typing import List, Dict, Optional

class DeepSeekChatTemplate:
    """
    Handles chat formatting.
    """
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
        return formatted

class SFTConfig:
    def __init__(self):
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_target_modules = ["q_proj", "v_proj"]
        self.use_neftune = True
        self.neftune_alpha = 5.0

class SFTTrainer:
    """
    Simplified SFT Trainer for MLX.
    """
    def __init__(self, model, config: SFTConfig):
        self.model = model
        self.config = config
        
    def train_step(self, batch):
        # Placeholder for training logic
        pass
        
    def add_neftune_noise(self, embeddings: mx.array) -> mx.array:
        """
        Add NEFTune noise to embeddings.
        noise ~ Uniform(-1, 1) * alpha / sqrt(L*D)
        Actually paper says: alpha / sqrt(L) * dims?
        Usually: scale = alpha / sqrt(seq_len)
        """
        if not self.config.use_neftune:
            return embeddings
            
        seq_len = embeddings.shape[1]
        scale = self.config.neftune_alpha / (seq_len ** 0.5)
        
        noise = mx.random.uniform(-1, 1, embeddings.shape) * scale
        return embeddings + noise
