import torch
import torch.nn as nn
import torch.nn.functional as F

class GRPOTrainer:
    def __init__(self, beta=0.01):
        self.beta = beta
        
    def compute_loss(self, logits, input_ids, rewards, ref_logits):
        # logits: (G, Seq, Vocab)
        # input_ids: (G, Seq)
        # rewards: (G,)
        # ref_logits: (G, Seq, Vocab)
        
        G, Seq, _ = logits.shape
        
        # 1. Advantages
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r # (G,)
        
        # 2. Policy Log Probs
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log probs of actual tokens
        # input_ids: (G, Seq) -> (G, Seq, 1)
        token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1) # (G, Seq)
        seq_log_probs = token_log_probs.sum(dim=1) # (G,)
        
        # 3. KL Divergence
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        # KL(P || Ref) = sum(P * (log P - log Ref))
        # Approx: log P - log Ref (token level)
        # Wait, exact KL is sum P * (log P - log Ref)
        probs = F.softmax(logits, dim=-1)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1) # (G, Seq)
        mean_kl = kl.mean(dim=1) # (G,)
        
        # 4. Loss
        # Loss = - (Adv * seq_log_probs) + beta * mean_kl
        loss = - (advantages * seq_log_probs) + self.beta * mean_kl
        return loss.mean()

class GroupSampler:
    def __init__(self, group_size):
        self.group_size = group_size
        
    def sample(self, prompt):
        return [f"Sampled output {i} for prompt..." for i in range(self.group_size)]
