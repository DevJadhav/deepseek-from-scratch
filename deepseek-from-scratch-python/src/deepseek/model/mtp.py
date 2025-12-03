import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModule(nn.Module):
    def __init__(self, d_model, d_embed, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, d_embed)
        self.layer_norm = nn.LayerNorm(d_embed)
        self.head = nn.Linear(d_embed, vocab_size)
        
    def forward(self, x):
        # x: (B, Seq, D_model)
        x = self.proj(x)
        x = self.layer_norm(x)
        return self.head(x)

class MTPModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, k_predictions):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Simplified Transformer Layers
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.main_head = nn.Linear(d_model, vocab_size)
        
        self.k_predictions = k_predictions
        self.mtp_modules = nn.ModuleList([
            MTPModule(d_model, d_model, vocab_size) for _ in range(k_predictions)
        ])
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        
        for layer in self.layers:
            x = F.relu(layer(x)) # Simple non-linearity
            
        main_logits = self.main_head(x)
        
        future_logits = []
        current_state = x
        
        for module in self.mtp_modules:
            # In real MTP, we might concatenate embedding of predicted token?
            # Simplified: Just pass state through module
            logits = module(current_state)
            future_logits.append(logits)
            # Update state? For now keep simple
            
        return main_logits, future_logits
