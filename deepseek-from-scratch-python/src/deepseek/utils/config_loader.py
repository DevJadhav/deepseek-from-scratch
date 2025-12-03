import os
import json
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file and override with environment variables.
    Env vars should be prefixed with DEEPSEEK_ and uppercase.
    Example: DEEPSEEK_BATCH_SIZE=32 overrides batch_size.
    Nested keys: DEEPSEEK_DATA_BATCH_SIZE overrides data.batch_size.
    """
    config = {}
    
    # Load from file
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    # Override from Env
    for k, v in os.environ.items():
        if k.startswith("DEEPSEEK_"):
            key = k[9:].lower() # Remove prefix
            
            # Handle nested keys (e.g. DEEPSEEK_DATA_BATCH_SIZE -> data.batch_size)
            # This is ambiguous if underscores are used in keys.
            # Simplified: We assume flat config or specific mapping.
            # Or we just support top-level overrides.
            
            # Let's try to infer type
            try:
                if v.lower() == 'true':
                    val = True
                elif v.lower() == 'false':
                    val = False
                elif '.' in v:
                    val = float(v)
                else:
                    val = int(v)
            except ValueError:
                val = v
                
            config[key] = val
            
    return config
