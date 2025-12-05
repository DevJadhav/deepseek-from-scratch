"""
R1 Reasoning Model - MLX Implementation
"""

import mlx.core as mx
import mlx.nn as nn


class ReasoningModel(nn.Module):
    """
    DeepSeek-R1 style reasoning model.
    Generates chain-of-thought traces before answering.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def generate_with_reasoning(self, prompt: str) -> str:
        """Simulate generation with reasoning trace."""
        reasoning_trace = (
            f"<think>\n"
            f"The user is asking about {prompt}. \n"
            f"1. I need to identify the core question.\n"
            f"2. I should recall relevant information about DeepSeek-R1.\n"
            f"3. I need to formulate a clear and concise answer.\n"
            f"</think>"
        )
        
        final_answer = (
            "\nHere is the answer based on my reasoning:\n"
            "DeepSeek-R1 is a reasoning model that uses Reinforcement Learning to generate "
            "chain-of-thought traces before answering. This improves performance on complex tasks."
        )
        
        return reasoning_trace + final_answer
        
    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass through embedding and LM head."""
        x = self.embed(input_ids)
        return self.lm_head(x)
