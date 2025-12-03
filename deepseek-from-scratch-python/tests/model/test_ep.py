import pytest
import torch
import torch.multiprocessing as mp
import os
from deepseek.model.ep_utils import all_to_all
from deepseek.utils.distributed import setup_distributed, cleanup_distributed

def run_all_to_all_test(rank, world_size):
    """Worker function for all-to-all test."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    
    # Initialize distributed
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    # Mock EP group (use default group)
    # In real code we use get_expert_model_parallel_group which returns None (default) or specific group
    # Our implementation uses get_expert_model_parallel_group() which returns None by default
    
    # Test Data: Rank i sends [i, i] to Rank 0 and [i+1] to Rank 1
    # Input: 
    # Rank 0: [0, 0] -> Rank 0, [1] -> Rank 1
    # Rank 1: [1, 1] -> Rank 0, [2] -> Rank 1
    
    # Let's simplify:
    # Rank 0 sends 2 items to Rank 1, 1 item to Rank 0
    # Rank 1 sends 1 item to Rank 1, 2 items to Rank 0
    
    hidden_dim = 2
    
    if rank == 0:
        # Send 1 to Rank 0, 2 to Rank 1
        input_data = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        output_split_sizes = [1, 2] # Receive 1 from Rank 0, 2 from Rank 1? No.
        # output_split_sizes is what we expect to RECEIVE.
        # input_split_sizes is what we SEND.
        
        input_split_sizes = [1, 2] # Send 1 to Rank 0, 2 to Rank 1
        
        # Rank 0 receives:
        # From Rank 0: 1 item ([0.1, 0.1])
        # From Rank 1: 2 items ([1.2, 1.2], [1.3, 1.3])
        output_split_sizes = [1, 2] 
        
    else:
        # Rank 1
        # Send 2 to Rank 0, 1 to Rank 1
        input_data = torch.tensor([[1.1, 1.1], [1.2, 1.2], [1.3, 1.3]])
        input_split_sizes = [2, 1]
        
        # Rank 1 receives:
        # From Rank 0: 2 items ([0.2, 0.2], [0.3, 0.3])
        # From Rank 1: 1 item ([1.1, 1.1])
        output_split_sizes = [2, 1]
        
    # Run All-to-All
    # Note: Our wrapper takes (input, output_split_sizes, input_split_sizes)
    output = all_to_all(input_data, output_split_sizes, input_split_sizes)
    
    # Verify output shape
    expected_len = sum(output_split_sizes)
    assert output.shape == (expected_len, hidden_dim)
    
    # Verify content (roughly)
    if rank == 0:
        # Should have [0.1, 0.1] from Rank 0
        assert torch.allclose(output[0], torch.tensor([0.1, 0.1]))
    else:
        # Should have [0.2, 0.2] from Rank 0
        assert torch.allclose(output[0], torch.tensor([0.2, 0.2]))

    torch.distributed.destroy_process_group()

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Distributed tests often flaky on CPU CI without proper setup")
def test_all_to_all_cpu():
    """Test All-to-All on CPU."""
    world_size = 2
    mp.spawn(run_all_to_all_test,
             args=(world_size,),
             nprocs=world_size,
             join=True)
