#!/usr/bin/env python3
"""
Train Tiny DeepSeek Model
=========================

Main training script that handles:
1. Data loading (TinyStories or custom JSONL)
2. Model creation
3. Training with MTP (Multi-Token Prediction)
4. Checkpointing and logging

Usage:
    # Full training
    python scripts/train_tiny.py --data-dir ./data/stories --output-dir ./checkpoints/tiny-mlx
    
    # Quick test (100 steps)
    python scripts/train_tiny.py --data-dir ./data/stories --output-dir ./checkpoints/test --quick-test
    
    # Resume from checkpoint
    python scripts/train_tiny.py --resume ./checkpoints/tiny-mlx/step_5000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deepseek-from-scratch-python" / "mlx_impl"))


def main():
    parser = argparse.ArgumentParser(description="Train Tiny DeepSeek Model")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="./data/stories",
                        help="Directory containing training data")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Tokenizer name or path")
    
    # Model
    parser.add_argument("--vocab-size", type=int, default=8000,
                        help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num-experts", type=int, default=4,
                        help="Number of MoE experts")
    parser.add_argument("--mtp-k", type=int, default=2,
                        help="Multi-Token Prediction k (0 to disable)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--mtp-weight", type=float, default=0.1,
                        help="Weight for MTP loss")
    
    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="./checkpoints/tiny-mlx",
                        help="Output directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log every N steps")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    # Quick test mode
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test with 100 steps")
    
    args = parser.parse_args()
    
    # Quick test overrides
    if args.quick_test:
        args.max_steps = 100
        args.save_every = 50
        args.eval_every = 25
        args.log_every = 10
        args.warmup_steps = 20
        print("Quick test mode enabled (100 steps)")
    
    # Import MLX modules
    try:
        import mlx.core as mx
        print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    except ImportError:
        print("Error: MLX not installed. Run: pip install mlx")
        sys.exit(1)
    
    from tiny_trainer import (
        TinyMTPModel, 
        TinyModelConfig, 
        TinyMLXTrainer,
        DataLoader,
    )
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        sys.exit(1)
    
    print("=" * 60)
    print("Tiny DeepSeek Training")
    print("=" * 60)
    
    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nTo download TinyStories dataset, run:")
        print("  python scripts/download_tinystories.py")
        print("\nOr create sample data:")
        print("  python scripts/download_tinystories.py --sample-only")
        sys.exit(1)
    
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    
    if not train_dir.exists():
        # Try using data_dir directly
        train_dir = data_dir
        valid_dir = None
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Adjust vocab size to tokenizer
    actual_vocab_size = len(tokenizer)
    if args.vocab_size != actual_vocab_size:
        print(f"Note: Adjusting vocab_size from {args.vocab_size} to {actual_vocab_size}")
        args.vocab_size = actual_vocab_size
    
    # Create model config
    config = TinyModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        num_experts=args.num_experts,
        mtp_k=args.mtp_k,
    )
    
    print("\nModel Configuration:")
    print(f"  vocab_size:    {config.vocab_size}")
    print(f"  d_model:       {config.d_model}")
    print(f"  num_heads:     {config.num_heads}")
    print(f"  num_layers:    {config.num_layers}")
    print(f"  num_experts:   {config.num_experts}")
    print(f"  mtp_k:         {config.mtp_k}")
    print(f"  max_seq_len:   {config.max_seq_len}")
    
    # Create model
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer, model, config = TinyMLXTrainer.load_checkpoint(args.resume)
        trainer.max_steps = args.max_steps  # Update max steps if changed
    else:
        print("\nCreating new model...")
        model = TinyMTPModel(config)
        
        trainer = TinyMLXTrainer(
            model=model,
            config=config,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            gradient_accumulation_steps=args.grad_accum,
            checkpoint_dir=args.output_dir,
            log_every=args.log_every,
            save_every=args.save_every,
            eval_every=args.eval_every,
            mtp_weight=args.mtp_weight,
        )
    
    # Count parameters using the trainer's method
    total_params = trainer._count_params()
    print(f"\nTotal parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    
    # Create data loaders
    print(f"\nLoading training data from: {train_dir}")
    train_loader = DataLoader(
        data_path=str(train_dir),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        shuffle=True,
    )
    print(f"  Training samples: {len(train_loader.samples):,}")
    print(f"  Batches per epoch: {len(train_loader):,}")
    
    valid_loader = None
    if valid_dir and valid_dir.exists():
        print(f"\nLoading validation data from: {valid_dir}")
        valid_loader = DataLoader(
            data_path=str(valid_dir),
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            shuffle=False,
        )
        print(f"  Validation samples: {len(valid_loader.samples):,}")
    
    # Save config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nConfig saved to: {config_path}")
    
    # Start training
    print("\n" + "=" * 60)
    start_time = time.time()
    
    try:
        trainer.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            pad_token_id=tokenizer.pad_token_id,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
    
    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")
    
    # Final info
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print("\nTo run inference:")
    print(f"  python scripts/inference.py --checkpoint {args.output_dir}/final --prompt 'Once upon a time'")
    print("\nTo run in interactive mode:")
    print(f"  python scripts/inference.py --checkpoint {args.output_dir}/final --interactive")


if __name__ == "__main__":
    main()
