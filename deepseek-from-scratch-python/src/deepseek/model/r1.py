import torch
import torch.nn as nn

class ReasoningModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def generate_with_reasoning(self, prompt):
        # Simulate generation trace
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
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        return self.lm_head(x)
