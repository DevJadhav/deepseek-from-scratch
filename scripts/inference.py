#!/usr/bin/env python3
"""
Inference Script for Tiny DeepSeek Model
=========================================

Load a trained model and generate text.

Usage:
    python scripts/inference.py --checkpoint ./checkpoints/tiny-mlx/final --prompt "Once upon a time"
    python scripts/inference.py --checkpoint ./checkpoints/tiny-mlx/best --interactive
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deepseek-from-scratch-python" / "mlx_impl"))


def unflatten_params(flat_params):
    """Convert flat dotted keys back to nested dict structure."""
    import mlx.core as mx
    
    def set_nested(d, keys, value):
        """Set a value in a nested dict/list structure."""
        for i, key in enumerate(keys[:-1]):
            next_key = keys[i + 1]
            
            # Determine if we need a list or dict
            if key.isdigit():
                key = int(key)
                # Ensure parent is a list
                if isinstance(d, dict) and key not in d:
                    d[key] = [] if next_key.isdigit() else {}
                elif isinstance(d, list):
                    while len(d) <= key:
                        d.append([] if next_key.isdigit() else {})
                d = d[key]
            else:
                if key not in d:
                    d[key] = [] if next_key.isdigit() else {}
                d = d[key]
        
        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
            while len(d) <= final_key:
                d.append(None)
            d[final_key] = value
        else:
            d[final_key] = value
    
    result = {}
    for key, value in flat_params.items():
        keys = key.split(".")
        set_nested(result, keys, value)
    
    return result


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    import mlx.core as mx
    from tiny_trainer import TinyMTPModel, TinyModelConfig
    
    path = Path(checkpoint_path)
    
    # Load config
    config_file = path / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        config = TinyModelConfig(**config_dict)
    else:
        print("Warning: No config.json found, using default config")
        config = TinyModelConfig()
    
    # Create model
    model = TinyMTPModel(config)
    
    # Load weights
    weights_file = path / "model.safetensors"
    if weights_file.exists():
        flat_weights = mx.load(str(weights_file))
        # Convert flat keys back to nested structure
        weights = unflatten_params(flat_weights)
        model.update(weights)
        print(f"Loaded weights from {weights_file}")
    else:
        print("Warning: No weights found, using random initialization")
    
    return model, config


def load_tokenizer(tokenizer_name: str = "gpt2"):
    """Load tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        sys.exit(1)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text from prompt."""
    import mlx.core as mx
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    return output_text


def interactive_mode(model, tokenizer, args):
    """Interactive generation mode."""
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print(f"Settings: max_tokens={args.max_tokens}, temp={args.temperature}")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating...")
            output = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            print("\n" + "-" * 40)
            print(output)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on trained model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name"
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoints_dir = project_root / "checkpoints"
        if checkpoints_dir.exists():
            for item in checkpoints_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item}")
        sys.exit(1)
    
    print("=" * 60)
    print("Tiny DeepSeek Inference")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model, config = load_model(str(checkpoint_path))
    
    # Count parameters (recursive for nested structure)
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, list):
            for v in params:
                total += count_params(v)
        elif hasattr(params, 'size'):
            total += params.size
        return total
    
    total_params = count_params(model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Run inference
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        
        output = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        print("\n" + "=" * 60)
        print("Generated Text:")
        print("=" * 60)
        print(output)
        print("=" * 60)
    else:
        print("\nError: Specify --prompt or --interactive")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
