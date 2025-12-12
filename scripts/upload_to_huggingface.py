#!/usr/bin/env python3
"""Upload GGUF model to Hugging Face Hub.

Usage:
    # Upload the 512M GGUF model
    uv run python scripts/upload_to_huggingface.py

    # Upload with custom repo
    uv run python scripts/upload_to_huggingface.py --repo-id YourUsername/model-name

    # Upload specific file
    uv run python scripts/upload_to_huggingface.py --model-path data/exports/deepseek-512M-q8_0.gguf
"""

import argparse
import shutil
from pathlib import Path
from huggingface_hub import login, upload_folder, HfApi


def create_model_card(model_dir: Path, model_name: str = "deepseek-v3.2_512M"):
    """Create a README.md model card for Hugging Face."""
    model_card = f"""---
license: mit
tags:
  - deepseek
  - gguf
  - llm
  - text-generation
  - from-scratch
library_name: gguf
pipeline_tag: text-generation
---

# {model_name}

A 512M parameter DeepSeek-style language model trained from scratch, exported to GGUF format.

## Model Details

- **Architecture**: DeepSeek V3.2 style (MLA attention, MoE, MTP)
- **Parameters**: ~1.34B (512M active)
- **Training**: 9,900 steps on Modal (8x A100-40GB)
- **Final Loss**: 10.385
- **Format**: GGUF (Q8_0 quantization)

### Architecture Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 2048 |
| n_heads | 32 |
| n_layers | 24 |
| vocab_size | 32000 |
| d_ff | 8192 |
| max_seq_len | 1024 |

## Usage

### With llama.cpp

```bash
./main -m deepseek-512M-q8_0.gguf -p "Once upon a time" -n 128
```

### With Ollama

Create a `Modelfile`:

```
FROM ./deepseek-512M-q8_0.gguf

TEMPLATE \"{{{{.Prompt}}}}\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

Then run:

```bash
ollama create deepseek-512m -f Modelfile
ollama run deepseek-512m
```

### With LM Studio

1. Download the GGUF file
2. Import into LM Studio
3. Start chatting!

## Training Details

This model was trained as part of the [DeepSeek-From-Scratch](https://github.com/DevJadhav/deepseek-from-scratch) project, implementing DeepSeek V3 architecture from scratch.

### Training Infrastructure

- **Platform**: Modal Cloud
- **GPUs**: 8x NVIDIA A100-40GB
- **Parallelism**: TP=2, PP=2, DP=2 (5D parallelism)
- **Precision**: BF16

## Limitations

This is a research/educational model trained on synthetic data. It is **not** intended for production use and may generate nonsensical or harmful content.

## License

MIT License - See the repository for details.
"""
    
    readme_path = model_dir / "README.md"
    readme_path.write_text(model_card)
    print(f"Created model card: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Upload GGUF model to Hugging Face Hub")
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/exports/deepseek-512M-q8_0.gguf",
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="DevJadhav/deepseek-v3.2_512M",
        help="Hugging Face repo ID (username/model-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="Skip interactive login (use HF_TOKEN env var)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return 1

    # Create a temporary directory for upload
    upload_dir = Path("data/exports/hf_upload")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Copy the GGUF file
    dest_path = upload_dir / model_path.name
    if not dest_path.exists() or dest_path.stat().st_size != model_path.stat().st_size:
        print(f"📦 Copying {model_path.name} to upload directory...")
        shutil.copy2(model_path, dest_path)

    # Create model card
    create_model_card(upload_dir, args.repo_id.split("/")[-1])

    # Login to Hugging Face
    if not args.no_login:
        print("\n🔐 Logging in to Hugging Face...")
        login()

    # Create repo if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"✅ Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"⚠️  Could not create repo (may already exist): {e}")

    # Upload the folder
    print(f"\n📤 Uploading to {args.repo_id}...")
    upload_folder(
        folder_path=str(upload_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"\n✅ Upload complete!")
    print(f"🔗 View your model: https://huggingface.co/{args.repo_id}")

    return 0


if __name__ == "__main__":
    exit(main())
