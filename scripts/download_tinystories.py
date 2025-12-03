#!/usr/bin/env python3
"""
Download TinyStories Dataset
============================

Downloads the TinyStories dataset from HuggingFace and prepares it
for training a ~10M parameter DeepSeek-like model.

Usage:
    python scripts/download_tinystories.py

The dataset will be saved to:
    - data/stories/train/stories.jsonl
    - data/stories/valid/stories.jsonl
"""

import json
import os
from pathlib import Path


def download_tinystories(output_dir: str = "data/stories", max_train_samples: int = None):
    """
    Download and prepare TinyStories dataset.
    
    Args:
        output_dir: Output directory for the dataset
        max_train_samples: Maximum training samples (None for all)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install with: pip install datasets")
        return False
    
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TinyStories Dataset Downloader")
    print("=" * 60)
    print()
    print("Downloading TinyStories dataset from HuggingFace...")
    print("This may take a few minutes...")
    print()
    
    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Save training data
    train_file = train_dir / "stories.jsonl"
    print(f"Saving training data to {train_file}...")
    
    train_count = 0
    with open(train_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset["train"]):
            if max_train_samples and i >= max_train_samples:
                break
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
            train_count += 1
            if train_count % 100000 == 0:
                print(f"  Processed {train_count:,} training examples...")
    
    print(f"  Total training examples: {train_count:,}")
    
    # Save validation data
    valid_file = valid_dir / "stories.jsonl"
    print(f"\nSaving validation data to {valid_file}...")
    
    valid_count = 0
    with open(valid_file, "w", encoding="utf-8") as f:
        for item in dataset["validation"]:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
            valid_count += 1
    
    print(f"  Total validation examples: {valid_count:,}")
    
    # Print summary
    train_size = train_file.stat().st_size / (1024 * 1024)
    valid_size = valid_file.stat().st_size / (1024 * 1024)
    
    print()
    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print()
    print(f"Training data:   {train_file} ({train_size:.1f} MB)")
    print(f"Validation data: {valid_file} ({valid_size:.1f} MB)")
    print(f"Total examples:  {train_count + valid_count:,}")
    print()
    print("Next steps:")
    print("  1. Run data preparation: python -m ray_pipeline.cli run --model-size tiny --backend mlx --stage data_prep")
    print("  2. Or use the full training script: ./scripts/run_tiny_training.sh")
    print()
    
    return True


def create_sample_data(output_dir: str = "data/stories", num_samples: int = 1000):
    """
    Create sample data for quick testing without downloading full dataset.
    
    Args:
        output_dir: Output directory for the dataset
        num_samples: Number of sample stories to generate
    """
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample TinyStories data for testing...")
    
    # Sample story templates
    templates = [
        "Once upon a time, there was a little {animal} named {name}. {name} loved to play in the {place}.",
        "One day, {name} the {animal} went to the {place}. There, {name} found a special {object}.",
        "In a small {place}, there lived a {adjective} {animal} called {name}. {name} had many friends.",
        "{name} was a {adjective} little {animal} who lived near the {place}. Every day, {name} would explore.",
        "The {adjective} {animal} named {name} discovered something amazing in the {place}.",
    ]
    
    animals = ["cat", "dog", "rabbit", "bird", "mouse", "bear", "fox", "owl", "deer", "squirrel"]
    names = ["Lily", "Max", "Luna", "Oliver", "Bella", "Charlie", "Lucy", "Leo", "Mia", "Jack"]
    places = ["forest", "garden", "meadow", "village", "pond", "hill", "cave", "tree", "field", "stream"]
    objects = ["flower", "stone", "leaf", "feather", "shell", "berry", "acorn", "mushroom", "stick", "pebble"]
    adjectives = ["happy", "curious", "brave", "gentle", "clever", "kind", "playful", "tiny", "fluffy", "bright"]
    
    import random
    random.seed(42)
    
    # Generate training stories
    train_file = train_dir / "stories.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            template = random.choice(templates)
            story = template.format(
                animal=random.choice(animals),
                name=random.choice(names),
                place=random.choice(places),
                object=random.choice(objects),
                adjective=random.choice(adjectives),
            )
            f.write(json.dumps({"text": story}) + "\n")
    
    # Generate validation stories
    valid_file = valid_dir / "stories.jsonl"
    num_valid = max(100, num_samples // 10)
    with open(valid_file, "w", encoding="utf-8") as f:
        for _ in range(num_valid):
            template = random.choice(templates)
            story = template.format(
                animal=random.choice(animals),
                name=random.choice(names),
                place=random.choice(places),
                object=random.choice(objects),
                adjective=random.choice(adjectives),
            )
            f.write(json.dumps({"text": story}) + "\n")
    
    print(f"Created {num_samples} training samples and {num_valid} validation samples")
    print(f"Training data: {train_file}")
    print(f"Validation data: {valid_file}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download TinyStories dataset")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/stories",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Create sample data only (for quick testing)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for sample-only mode"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Maximum training samples to download (None for all)"
    )
    
    args = parser.parse_args()
    
    if args.sample_only:
        create_sample_data(args.output_dir, args.num_samples)
    else:
        download_tinystories(args.output_dir, args.max_train)
