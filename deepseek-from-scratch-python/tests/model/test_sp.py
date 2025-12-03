import pytest
import torch
import torch.multiprocessing as mp
import os
from deepseek.model.ring_attention import RingAttention

def run_sp_test(rank, world_size):
    """Worker function for SP test."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    
    # Initialize distributed
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    model = RingAttention(d_model=16, num_heads=2, group=None) # Use default group
    
    x = torch.randn(2, 4, 16) # Batch 2, Seq 4 (local), Dim 16
    out = model(x)
    
    assert out.shape == (2, 4, 16)
    
    torch.distributed.destroy_process_group()

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Distributed tests often flaky on CPU CI without proper setup")
def test_sp_cpu():
    """Test Sequence Parallelism on CPU."""
    world_size = 2
    mp.spawn(run_sp_test,
             args=(world_size,),
             nprocs=world_size,
             join=True)
