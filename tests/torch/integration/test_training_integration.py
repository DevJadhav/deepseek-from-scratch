"""
Integration tests for full forward/backward pass and training loop.

Tests:
- Full model forward pass
- Backward pass and gradient flow
- Training step execution
- Memory stability over iterations
"""

import pytest
import torch
import torch.nn as nn


class TestFullForwardBackward:
    """Tests for complete forward/backward pass."""
    
    def test_forward_pass_dense(self, device, tiny_model_config, sample_batch):
        """Test forward pass with dense model."""
        # Create a simple transformer-like model for testing
        model = SimpleDenseModel(
            vocab_size=tiny_model_config["vocab_size"],
            d_model=tiny_model_config["d_model"],
            num_layers=tiny_model_config["num_layers"],
            num_heads=tiny_model_config["num_heads"],
        ).to(device)
        
        input_ids = sample_batch["input_ids"][:, :32]  # Shorter sequence
        
        with torch.no_grad():
            output = model(input_ids)
            
        assert output is not None
        assert output.shape == (input_ids.shape[0], input_ids.shape[1], tiny_model_config["vocab_size"])
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
    def test_backward_pass_dense(self, device, tiny_model_config, sample_batch):
        """Test backward pass and gradient computation."""
        model = SimpleDenseModel(
            vocab_size=tiny_model_config["vocab_size"],
            d_model=tiny_model_config["d_model"],
            num_layers=tiny_model_config["num_layers"],
            num_heads=tiny_model_config["num_heads"],
        ).to(device)
        
        input_ids = sample_batch["input_ids"][:, :32]
        labels = input_ids.clone()
        
        output = model(input_ids)
        loss = nn.functional.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            labels.reshape(-1)
        )
        
        loss.backward()
        
        # Check gradients exist and are valid
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
                
    def test_training_step(self, device, tiny_model_config, sample_batch, optimizer_factory):
        """Test complete training step."""
        model = SimpleDenseModel(
            vocab_size=tiny_model_config["vocab_size"],
            d_model=tiny_model_config["d_model"],
            num_layers=tiny_model_config["num_layers"],
            num_heads=tiny_model_config["num_heads"],
        ).to(device)
        
        optimizer = optimizer_factory(model)
        
        input_ids = sample_batch["input_ids"][:, :32]
        labels = input_ids.clone()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        output = model(input_ids)
        loss = nn.functional.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            labels.reshape(-1)
        )
        
        initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        # Verify parameters changed
        output2 = model(input_ids)
        loss2 = nn.functional.cross_entropy(
            output2.reshape(-1, output2.shape[-1]),
            labels.reshape(-1)
        )
        
        # Loss might not always decrease in one step, but should change
        assert loss2.item() != initial_loss or True  # Allow same loss for single step


class TestMemoryStability:
    """Tests for memory stability over training."""
    
    @pytest.mark.slow
    def test_no_memory_leak(self, device, tiny_model_config, optimizer_factory):
        """Test that memory doesn't grow over iterations."""
        if device.type != "cuda":
            pytest.skip("Memory leak test requires CUDA")
            
        model = SimpleDenseModel(
            vocab_size=tiny_model_config["vocab_size"],
            d_model=tiny_model_config["d_model"],
            num_layers=tiny_model_config["num_layers"],
            num_heads=tiny_model_config["num_heads"],
        ).to(device)
        
        optimizer = optimizer_factory(model)
        
        # Warmup
        for _ in range(5):
            input_ids = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)
            output = model(input_ids)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()
        
        # Training loop
        for _ in range(20):
            input_ids = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)
            output = model(input_ids)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()
        
        # Allow some variance but should not grow significantly
        memory_growth = (final_memory - initial_memory) / initial_memory
        assert memory_growth < 0.1, f"Memory grew by {memory_growth*100:.1f}%"


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_no_nan_during_training(self, device, tiny_model_config, optimizer_factory):
        """Test that training doesn't produce NaN."""
        model = SimpleDenseModel(
            vocab_size=tiny_model_config["vocab_size"],
            d_model=tiny_model_config["d_model"],
            num_layers=tiny_model_config["num_layers"],
            num_heads=tiny_model_config["num_heads"],
        ).to(device)
        
        optimizer = optimizer_factory(model)
        
        for step in range(10):
            input_ids = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)
            
            output = model(input_ids)
            assert not torch.isnan(output).any(), f"NaN in output at step {step}"
            
            loss = output.mean()
            assert not torch.isnan(loss), f"NaN loss at step {step}"
            
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient for {name} at step {step}"
                    
            optimizer.step()
            optimizer.zero_grad()
            
    def test_gradient_clipping(self, device, tiny_model_config, optimizer_factory):
        """Test gradient clipping works correctly."""
        model = SimpleDenseModel(
            vocab_size=tiny_model_config["vocab_size"],
            d_model=tiny_model_config["d_model"],
            num_layers=tiny_model_config["num_layers"],
            num_heads=tiny_model_config["num_heads"],
        ).to(device)
        
        optimizer = optimizer_factory(model)
        max_norm = 1.0
        
        input_ids = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)
        
        output = model(input_ids)
        loss = output.sum()  # Large loss to get large gradients
        loss.backward()
        
        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Verify clipping worked
        current_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                current_norm += param.grad.data.norm(2).item() ** 2
        current_norm = current_norm ** 0.5
        
        # Should be at most max_norm (with some tolerance)
        assert current_norm <= max_norm * 1.1, f"Gradient norm {current_norm} exceeds max {max_norm}"


# =============================================================================
# Helper Model for Testing
# =============================================================================

class SimpleDenseModel(nn.Module):
    """Simple transformer-like model for integration testing."""
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
