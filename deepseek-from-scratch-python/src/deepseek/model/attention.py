import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False) # Single KV head
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False) # Single KV head
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape Q: (B, Seq, H, D_head) -> (B, H, Seq, D_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape K, V: (B, Seq, D_head) -> (B, 1, Seq, D_head) (Broadcastable)
        k = k.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # Q: (B, H, Seq, D_head)
        # K: (B, 1, Seq, D_head) -> Broadcasts to (B, H, Seq, D_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # V: (B, 1, Seq, D_head) -> Broadcasts
        output = torch.matmul(attn_weights, v) # (B, H, Seq, D_head)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        self.num_kv_heads = num_groups # One KV head per group
        
        # Check if heads divide evenly by groups
        assert num_heads % num_groups == 0
        self.heads_per_group = num_heads // num_groups
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape Q: (B, Seq, H, D_head) -> (B, H, Seq, D_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape K, V: (B, Seq, G, D_head) -> (B, G, Seq, D_head)
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)
        
        # Repeat K, V for each head in the group
        # (B, G, Seq, D_head) -> (B, G, 1, Seq, D_head) -> (B, G, H_per_G, Seq, D_head) -> (B, H, Seq, D_head)
        k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)
