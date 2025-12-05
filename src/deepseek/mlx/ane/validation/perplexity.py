"""
ANE Perplexity Evaluation Module

Provides perplexity evaluation on standard benchmarks:
- WikiText-2 perplexity
- C4 perplexity
- Custom dataset support
- Sliding window evaluation for long sequences
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PerplexityConfig:
    """Configuration for perplexity evaluation."""

    # Maximum sequence length for evaluation
    max_seq_len: int = 2048

    # Stride for sliding window (overlap)
    stride: int = 512

    # Batch size for evaluation
    batch_size: int = 1

    # Device for computation
    device: str = "cpu"

    # Use sliding window for long sequences
    use_sliding_window: bool = True

    # Verbose output
    verbose: bool = False

    # Maximum samples to evaluate (None = all)
    max_samples: int | None = None


@dataclass
class PerplexityResult:
    """Result of perplexity evaluation."""

    # Perplexity value
    perplexity: float

    # Total loss
    total_loss: float

    # Number of tokens evaluated
    num_tokens: int

    # Number of sequences evaluated
    num_sequences: int

    # Per-sequence perplexities (if computed)
    sequence_perplexities: list[float] = field(default_factory=list)

    # Dataset name
    dataset: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "perplexity": self.perplexity,
            "total_loss": self.total_loss,
            "num_tokens": self.num_tokens,
            "num_sequences": self.num_sequences,
            "dataset": self.dataset,
        }


class TextDataset:
    """Simple text dataset for perplexity evaluation."""

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 2048,
    ):
        """
        Initialize text dataset.

        Args:
            texts: List of text strings
            tokenizer: Tokenizer with encode method
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get tokenized text."""
        text = self.texts[idx]

        # Tokenize
        if hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        else:
            # Assume tokenizer is callable
            tokens = self.tokenizer(text)

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)

        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]

        return tokens


def load_wikitext2(
    split: str = "test",
    tokenizer=None,
    data_dir: str | None = None,
) -> list[str]:
    """
    Load WikiText-2 dataset.

    Args:
        split: Dataset split ("train", "valid", "test")
        tokenizer: Optional tokenizer (not used for loading)
        data_dir: Optional data directory

    Returns:
        List of text strings
    """
    # Try to load from huggingface datasets
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [item["text"] for item in dataset if item["text"].strip()]
        return texts
    except ImportError:
        pass

    # Fallback: try local file
    if data_dir:
        path = Path(data_dir) / f"wikitext-2-raw/wiki.{split}.raw"
        if path.exists():
            with open(path) as f:
                text = f.read()
            # Split into paragraphs
            texts = [p.strip() for p in text.split("\n\n") if p.strip()]
            return texts

    # Return sample data for testing
    return [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "In a hole in the ground there lived a hobbit. " * 10,
        "It was the best of times, it was the worst of times. " * 10,
    ]


def load_c4(
    split: str = "validation",
    tokenizer=None,
    max_samples: int = 1000,
) -> list[str]:
    """
    Load C4 dataset samples.

    Args:
        split: Dataset split
        tokenizer: Optional tokenizer
        max_samples: Maximum samples to load

    Returns:
        List of text strings
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "allenai/c4",
            "en",
            split=split,
            streaming=True,
        )
        texts = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            if item["text"].strip():
                texts.append(item["text"])
        return texts
    except ImportError:
        pass

    # Return sample data for testing
    return [
        "This is a sample C4 text for testing purposes. " * 20,
        "Machine learning models require large datasets. " * 20,
        "Natural language processing has made great strides. " * 20,
    ]


def compute_perplexity_batch(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    """
    Compute perplexity for a batch.

    Args:
        model: Language model
        input_ids: Input token IDs [batch, seq_len]
        target_ids: Target token IDs [batch, seq_len]
        attention_mask: Optional attention mask

    Returns:
        Tuple of (total_loss, num_tokens)
    """
    with torch.no_grad():
        # Get model outputs
        outputs = model(input_ids)

        # Handle different output types
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            raise ValueError(f"Unsupported model output type: {type(outputs)}")

        # Compute cross-entropy loss
        # logits: [batch, seq_len, vocab_size]
        # targets: [batch, seq_len]
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for loss computation
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)

        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = loss.view(batch_size, seq_len)

        # Apply mask if provided
        if attention_mask is not None:
            loss = loss * attention_mask
            num_tokens = attention_mask.sum().item()
        else:
            num_tokens = batch_size * seq_len

        total_loss = loss.sum().item()

        return total_loss, int(num_tokens)


def compute_perplexity_sliding(
    model: nn.Module,
    tokens: torch.Tensor,
    config: PerplexityConfig,
) -> tuple[float, int]:
    """
    Compute perplexity using sliding window.

    Args:
        model: Language model
        tokens: Token IDs [seq_len]
        config: Perplexity configuration

    Returns:
        Tuple of (total_loss, num_tokens)
    """
    seq_len = tokens.shape[0]
    total_loss = 0.0
    total_tokens = 0

    device = torch.device(config.device)
    model = model.to(device)

    prev_end = 0

    for start in range(0, seq_len, config.stride):
        end = min(start + config.max_seq_len, seq_len)

        # Get input and target
        input_ids = tokens[start:end].unsqueeze(0).to(device)
        target_ids = tokens[start + 1 : end + 1] if end < seq_len else tokens[start + 1 :]
        target_ids = target_ids.unsqueeze(0).to(device)

        # Adjust for different lengths
        min_len = min(input_ids.shape[1], target_ids.shape[1])
        input_ids = input_ids[:, :min_len]
        target_ids = target_ids[:, :min_len]

        if min_len == 0:
            break

        # Only count new tokens (avoid double-counting in overlapping regions)
        if start > 0:
            new_start = max(prev_end - start, 0)
        else:
            new_start = 0

        loss, _ = compute_perplexity_batch(model, input_ids, target_ids)

        # Count only new tokens
        new_tokens = min_len - new_start
        if new_tokens > 0:
            # Approximate: use average loss per token
            avg_loss = loss / min_len
            total_loss += avg_loss * new_tokens
            total_tokens += new_tokens

        prev_end = end

        if end >= seq_len:
            break

    return total_loss, total_tokens


class PerplexityEvaluator:
    """
    Perplexity evaluator for language models.

    Example:
        evaluator = PerplexityEvaluator(config)

        # Evaluate on WikiText-2
        result = evaluator.evaluate_wikitext2(model, tokenizer)
        print(f"WikiText-2 Perplexity: {result.perplexity:.2f}")

        # Evaluate on C4
        result = evaluator.evaluate_c4(model, tokenizer)
        print(f"C4 Perplexity: {result.perplexity:.2f}")
    """

    def __init__(self, config: PerplexityConfig | None = None):
        """Initialize evaluator with configuration."""
        self.config = config or PerplexityConfig()

    def evaluate_texts(
        self,
        model: nn.Module,
        tokenizer,
        texts: list[str],
        dataset_name: str = "custom",
    ) -> PerplexityResult:
        """
        Evaluate perplexity on a list of texts.

        Args:
            model: Language model
            tokenizer: Tokenizer with encode method
            texts: List of text strings
            dataset_name: Name of the dataset

        Returns:
            PerplexityResult
        """
        model.eval()
        device = torch.device(self.config.device)
        model = model.to(device)

        total_loss = 0.0
        total_tokens = 0
        num_sequences = 0
        sequence_perplexities = []

        max_samples = self.config.max_samples or len(texts)

        for i, text in enumerate(texts[:max_samples]):
            if not text.strip():
                continue

            # Tokenize
            if hasattr(tokenizer, "encode"):
                tokens = tokenizer.encode(text)
            else:
                tokens = tokenizer(text)

            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long)

            if len(tokens) < 2:
                continue

            # Compute perplexity
            if self.config.use_sliding_window and len(tokens) > self.config.max_seq_len:
                loss, num_tokens = compute_perplexity_sliding(
                    model, tokens, self.config
                )
            else:
                tokens = tokens[: self.config.max_seq_len]
                input_ids = tokens[:-1].unsqueeze(0).to(device)
                target_ids = tokens[1:].unsqueeze(0).to(device)
                loss, num_tokens = compute_perplexity_batch(
                    model, input_ids, target_ids
                )

            total_loss += loss
            total_tokens += num_tokens
            num_sequences += 1

            # Per-sequence perplexity
            if num_tokens > 0:
                seq_ppl = math.exp(loss / num_tokens)
                sequence_perplexities.append(seq_ppl)

            if self.config.verbose and (i + 1) % 100 == 0:
                current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
                print(f"Processed {i + 1}/{max_samples} sequences, current PPL: {current_ppl:.2f}")

        # Compute final perplexity
        if total_tokens > 0:
            perplexity = math.exp(total_loss / total_tokens)
        else:
            perplexity = float("inf")

        return PerplexityResult(
            perplexity=perplexity,
            total_loss=total_loss,
            num_tokens=total_tokens,
            num_sequences=num_sequences,
            sequence_perplexities=sequence_perplexities,
            dataset=dataset_name,
        )

    def evaluate_wikitext2(
        self,
        model: nn.Module,
        tokenizer,
        split: str = "test",
    ) -> PerplexityResult:
        """
        Evaluate perplexity on WikiText-2.

        Args:
            model: Language model
            tokenizer: Tokenizer
            split: Dataset split

        Returns:
            PerplexityResult
        """
        texts = load_wikitext2(split=split)
        return self.evaluate_texts(model, tokenizer, texts, "wikitext-2")

    def evaluate_c4(
        self,
        model: nn.Module,
        tokenizer,
        max_samples: int = 1000,
    ) -> PerplexityResult:
        """
        Evaluate perplexity on C4.

        Args:
            model: Language model
            tokenizer: Tokenizer
            max_samples: Maximum samples to evaluate

        Returns:
            PerplexityResult
        """
        texts = load_c4(max_samples=max_samples)
        return self.evaluate_texts(model, tokenizer, texts, "c4")

    def evaluate_tokens(
        self,
        model: nn.Module,
        tokens: torch.Tensor,
        dataset_name: str = "custom",
    ) -> PerplexityResult:
        """
        Evaluate perplexity on pre-tokenized data.

        Args:
            model: Language model
            tokens: Token IDs [seq_len]
            dataset_name: Name of the dataset

        Returns:
            PerplexityResult
        """
        model.eval()

        if self.config.use_sliding_window and len(tokens) > self.config.max_seq_len:
            total_loss, total_tokens = compute_perplexity_sliding(
                model, tokens, self.config
            )
        else:
            tokens = tokens[: self.config.max_seq_len]
            device = torch.device(self.config.device)
            model = model.to(device)

            input_ids = tokens[:-1].unsqueeze(0).to(device)
            target_ids = tokens[1:].unsqueeze(0).to(device)
            total_loss, total_tokens = compute_perplexity_batch(
                model, input_ids, target_ids
            )

        perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

        return PerplexityResult(
            perplexity=perplexity,
            total_loss=total_loss,
            num_tokens=total_tokens,
            num_sequences=1,
            dataset=dataset_name,
        )


class SimpleTokenizer:
    """Simple character-level tokenizer for testing."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join(chr(t) for t in tokens)
