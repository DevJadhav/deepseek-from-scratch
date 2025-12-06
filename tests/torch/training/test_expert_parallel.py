"""
Tests for Expert Parallelism (EP) in MoE.

These tests verify the expert parallelism implementation which distributes
experts across multiple ranks and uses all-to-all communication for dispatch.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.deepseek.torch.model.moe import Expert, ExpertParallelMoE


class TestExpertParallelMoEBasic:
    """Basic tests for ExpertParallelMoE (non-distributed)."""
    
    def test_initialization(self):
        """Test ExpertParallelMoE initialization without distributed."""
        model = ExpertParallelMoE(
            num_experts=8,
            d_model=256,
            d_ff=512,
            top_k=2,
        )
        
        # Without distributed, all experts are local
        assert model.ep_world_size == 1
        assert model.ep_rank == 0
        assert model.num_local_experts == 8
        assert model.local_expert_start == 0
        assert len(model.experts) == 8
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=128,
            d_ff=256,
            top_k=2,
        )
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 128)
        
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_forward_finite(self):
        """Test that output contains no NaN or Inf."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=128,
            d_ff=256,
            top_k=2,
        )
        
        x = torch.randn(2, 8, 128)
        output = model(x)
        
        assert torch.all(torch.isfinite(output))
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=128,
            d_ff=256,
            top_k=1,
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 16, 128)
            output = model(x)
            assert output.shape == x.shape


class TestAllToAllDispatch:
    """Tests for the all-to-all dispatch mechanism."""
    
    def test_dispatch_single_rank(self):
        """Test dispatch with single rank (non-distributed)."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=64,
            d_ff=128,
            top_k=1,
        )
        
        x = torch.randn(16, 64)  # 16 tokens
        indices = torch.randint(0, 4, (16,))  # Expert assignments
        
        local_x, local_indices, send_counts, recv_counts, sort_indices = \
            model.all_to_all_dispatch(x, indices)
        
        # With single rank, output should be same as input
        assert local_x.shape == x.shape
        assert torch.all(local_indices == indices)
    
    def test_combine_single_rank(self):
        """Test combine with single rank (non-distributed)."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=64,
            d_ff=128,
            top_k=1,
        )
        
        local_output = torch.randn(16, 64)
        send_counts = torch.tensor([16])
        recv_counts = torch.tensor([16])
        sort_indices = torch.arange(16)
        
        output = model.all_to_all_combine(
            local_output, send_counts, recv_counts, sort_indices, 16
        )
        
        # With single rank, output should be same as input
        torch.testing.assert_close(output, local_output)
    
    def test_dispatch_preserves_tokens(self):
        """Test that dispatch preserves all tokens."""
        model = ExpertParallelMoE(
            num_experts=8,
            d_model=64,
            d_ff=128,
            top_k=1,
        )
        
        x = torch.randn(32, 64)
        indices = torch.randint(0, 8, (32,))
        
        local_x, local_indices, send_counts, recv_counts, sort_indices = \
            model.all_to_all_dispatch(x, indices)
        
        # Should preserve token count
        assert local_x.shape[0] == x.shape[0]


class TestRouting:
    """Tests for routing in ExpertParallelMoE."""
    
    def test_router_output_shape(self):
        """Test router produces correct logit shape."""
        model = ExpertParallelMoE(
            num_experts=8,
            d_model=128,
            d_ff=256,
            top_k=2,
        )
        
        x = torch.randn(4, 16, 128)  # batch=4, seq=16
        x_flat = x.view(-1, 128)  # 64 tokens
        
        logits = model.router(x_flat)
        
        assert logits.shape == (64, 8)  # 64 tokens, 8 experts
    
    def test_top_k_selection(self):
        """Test that top-k correctly selects k experts."""
        model = ExpertParallelMoE(
            num_experts=8,
            d_model=128,
            d_ff=256,
            top_k=3,
        )
        
        x = torch.randn(2, 8, 128)
        x_flat = x.view(-1, 128)
        
        # Get routing
        router_logits = model.router(x_flat)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, 3, dim=-1)
        
        # Should select exactly 3 experts per token
        assert top_k_indices.shape == (16, 3)
        assert top_k_probs.shape == (16, 3)
        
        # Top-k indices should be valid expert indices
        assert torch.all(top_k_indices >= 0)
        assert torch.all(top_k_indices < 8)


class TestGradients:
    """Tests for gradient flow through ExpertParallelMoE."""
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=64,
            d_ff=128,
            top_k=2,
        )
        
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))
    
    def test_parameter_gradients(self):
        """Test that parameters receive gradients."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=64,
            d_ff=128,
            top_k=1,
        )
        
        x = torch.randn(2, 8, 64)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Router should have gradients
        assert model.router.weight.grad is not None
        
        # At least some experts should have gradients
        has_grad = False
        for expert in model.experts:
            # Expert uses nn.Sequential named 'net'
            for param in expert.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    break
        assert has_grad, "No expert received gradients"


class TestEdgeCases:
    """Tests for edge cases in ExpertParallelMoE."""
    
    def test_single_token(self):
        """Test with single token input."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=64,
            d_ff=128,
            top_k=1,
        )
        
        x = torch.randn(1, 1, 64)
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_large_batch(self):
        """Test with large batch."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=64,
            d_ff=128,
            top_k=1,
        )
        
        x = torch.randn(32, 64, 64)
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_num_experts_equals_tokens(self):
        """Test when number of tokens equals number of experts."""
        model = ExpertParallelMoE(
            num_experts=8,
            d_model=64,
            d_ff=128,
            top_k=1,
        )
        
        x = torch.randn(1, 8, 64)  # 8 tokens, 8 experts
        output = model(x)
        
        assert output.shape == x.shape


class TestDevicePlacement:
    """Tests for device placement."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=64,
            d_ff=128,
            top_k=1,
        ).cuda()
        
        x = torch.randn(2, 8, 64).cuda()
        output = model(x)
        
        assert output.is_cuda
        assert output.shape == x.shape
    
    def test_cpu_forward(self):
        """Test forward pass on CPU."""
        model = ExpertParallelMoE(
            num_experts=4,
            d_model=64,
            d_ff=128,
            top_k=1,
        )
        
        x = torch.randn(2, 8, 64)
        output = model(x)
        
        assert not output.is_cuda
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
