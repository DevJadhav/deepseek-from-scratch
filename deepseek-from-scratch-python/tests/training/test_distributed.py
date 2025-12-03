import pytest
import torch
import torch.multiprocessing as mp
import os
from deepseek.training.training import Trainer, TrainingConfig
from deepseek.utils.distributed import setup_distributed, cleanup_distributed

def run_distributed_test(rank, world_size, use_fsdp=False):
    """Worker function for distributed test."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    # Simple model
    model = torch.nn.Linear(10, 10)
    
    config = TrainingConfig(
        backend="gloo", # Use gloo for CPU testing
        use_fsdp=use_fsdp,
        batch_size=2,
        max_steps=5
    )
    
    trainer = Trainer(model, config, device=torch.device("cpu"))
    
    # Fake data
    input_ids = torch.randn(2, 10)
    labels = torch.randint(0, 10, (2, 10))
    
    # Run one step
    metrics = trainer.train_step(input_ids, labels)
    assert "loss" in metrics
    
    cleanup_distributed()

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Distributed tests often flaky on CPU CI without proper setup")
def test_ddp_cpu():
    """Test DDP on CPU (using gloo)."""
    world_size = 2
    mp.spawn(run_distributed_test,
             args=(world_size, False),
             nprocs=world_size,
             join=True)

@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="FSDP requires CUDA usually or recent PyTorch for CPU")
def test_fsdp_cpu():
    """Test FSDP on CPU (using gloo)."""
    # FSDP on CPU is supported in newer PyTorch versions but might be tricky
    world_size = 2
    mp.spawn(run_distributed_test,
             args=(world_size, True),
             nprocs=world_size,
             join=True)
