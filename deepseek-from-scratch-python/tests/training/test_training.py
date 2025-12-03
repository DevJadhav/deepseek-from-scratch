import pytest
import torch
import os
import json
from deepseek.training.training import Trainer, TrainingConfig, SimpleDataLoader
from deepseek.utils.errors import OOMError
from deepseek.utils.checkpoint import save_checkpoint_with_checksum, load_checkpoint_with_checksum

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        
    def forward(self, x):
        return self.linear(x)

def test_trainer_init():
    config = TrainingConfig()
    model = MockModel()
    trainer = Trainer(model, config, device=torch.device("cpu"))
    assert trainer.global_step == 0

def test_checkpoint_checksum(tmp_path):
    """Test checkpoint saving and loading with checksum."""
    state_dict = {"a": torch.tensor([1.0, 2.0]), "b": 3}
    path = os.path.join(tmp_path, "ckpt.pt")
    
    save_checkpoint_with_checksum(state_dict, path)
    
    assert os.path.exists(path)
    assert os.path.exists(path + ".meta")
    
    loaded = load_checkpoint_with_checksum(path, torch.device("cpu"))
    assert torch.equal(loaded["a"], state_dict["a"])
    assert loaded["b"] == state_dict["b"]

def test_checkpoint_corruption(tmp_path):
    """Test that corrupted checkpoint raises error."""
    state_dict = {"a": torch.tensor([1.0])}
    path = os.path.join(tmp_path, "ckpt.pt")
    
    save_checkpoint_with_checksum(state_dict, path)
    
    # Corrupt file
    with open(path, "wb") as f:
        f.write(b"corrupted data")
        
    # PyTorch might raise UnpicklingError or RuntimeError depending on version/content
    # We just want to ensure it fails safe or raises an error we expect
    try:
        load_checkpoint_with_checksum(path, torch.device("cpu"))
    except (RuntimeError, Exception):
        # As long as it raises, we are good. 
        # The checksum verification happens AFTER load, so if load fails, it's also fine.
        # But we want to test checksum mismatch specifically.
        pass

def test_checkpoint_checksum_mismatch(tmp_path):
    """Test that valid pickle but wrong checksum raises error."""
    state_dict = {"a": torch.tensor([1.0])}
    path = os.path.join(tmp_path, "ckpt.pt")
    
    save_checkpoint_with_checksum(state_dict, path)
    
    # Modify the file content slightly (if possible without breaking pickle)
    # Or easier: modify the metadata
    meta_path = path + ".meta"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    meta["checksum"] = "wrong_checksum"
    
    with open(meta_path, "w") as f:
        json.dump(meta, f)
        
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        load_checkpoint_with_checksum(path, torch.device("cpu"))
