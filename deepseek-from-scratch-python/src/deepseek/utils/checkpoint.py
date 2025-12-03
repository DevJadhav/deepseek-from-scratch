import torch
import hashlib
import os
import json
from typing import Dict, Any, Optional
from deepseek.utils.logging import get_logger

logger = get_logger(__name__)

def compute_checksum(state_dict: Dict[str, Any]) -> str:
    """Compute SHA256 checksum of a state dict."""
    sha256 = hashlib.sha256()
    
    # Sort keys to ensure deterministic order
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if isinstance(val, torch.Tensor):
            # Use a chunk of the tensor data for speed, or full data for accuracy
            # Here we use a str representation of metadata + small chunk
            meta = f"{key}:{val.dtype}:{val.shape}"
            sha256.update(meta.encode('utf-8'))
            # Update with first and last few bytes of data if possible
            if val.numel() > 0:
                # CPU copy for safety
                data_chunk = val.flatten().detach().cpu().numpy()
                if data_chunk.size > 100:
                    sha256.update(data_chunk[:50].tobytes())
                    sha256.update(data_chunk[-50:].tobytes())
                else:
                    sha256.update(data_chunk.tobytes())
        else:
            sha256.update(str(val).encode('utf-8'))
            
    return sha256.hexdigest()

def save_checkpoint_with_checksum(
    state_dict: Dict[str, Any],
    path: str,
    meta: Optional[Dict[str, Any]] = None
):
    """Save checkpoint with a checksum file."""
    # Save main file
    torch.save(state_dict, path)
    
    # Compute checksum
    checksum = compute_checksum(state_dict)
    
    # Save metadata
    meta_path = path + ".meta"
    meta_data = {
        "checksum": checksum,
        "path": os.path.basename(path),
        "extra": meta or {}
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
        
    logger.info("Checkpoint saved with checksum", path=path, checksum=checksum)

def load_checkpoint_with_checksum(path: str, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint and verify checksum if available."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    state_dict = torch.load(path, map_location=device)
    
    meta_path = path + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            
        expected_checksum = meta_data.get("checksum")
        if expected_checksum:
            current_checksum = compute_checksum(state_dict)
            if current_checksum != expected_checksum:
                logger.error("Checkpoint checksum mismatch", 
                             expected=expected_checksum, 
                             actual=current_checksum)
                raise RuntimeError("Checkpoint corruption detected: Checksum mismatch")
            else:
                logger.info("Checkpoint checksum verified", checksum=current_checksum)
    else:
        logger.warning("No checksum metadata found for checkpoint", path=path)
        
    return state_dict
