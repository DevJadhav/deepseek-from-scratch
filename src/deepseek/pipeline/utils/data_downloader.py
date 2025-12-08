"""Auto-download utilities for pipeline data.

Provides automatic data downloading when data is missing,
integrated with the pipeline to ensure data availability.

Supported Datasets:
- web: HuggingFaceFW/fineweb-edu (FineWeb-Edu)
- code: mlfoundations-dev/stackoverflow
- math: HuggingFaceTB/finemath
- books: CShorten/ML-ArXiv-Papers
- scientific: jamescalam/ai-arxiv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Dataset registry mapping domains to HuggingFace datasets
DOMAIN_DATASETS: dict[str, dict[str, Any]] = {
    "web": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "text_field": "text",
        "split": "train",
        "streaming": True,
    },
    "code": {
        "hf_path": "flytech/python-codes-25k",
        "subset": None,
        "text_field": "output",
        "split": "train",
        "streaming": True,
        "fallback_fields": ["input", "instruction", "code", "text"],
    },
    "math": {
        "hf_path": "HuggingFaceTB/finemath",
        "subset": "finemath-3plus",
        "text_field": "text",
        "split": "train",
        "streaming": True,
        "fallback_fields": ["content", "problem", "solution"],
    },
    "books": {
        "hf_path": "CShorten/ML-ArXiv-Papers",
        "subset": None,
        "text_field": "abstract",
        "split": "train",
        "streaming": True,
        "fallback_fields": ["text", "content", "title"],
    },
    "scientific": {
        "hf_path": "jamescalam/ai-arxiv",
        "subset": None,
        "text_field": "chunk",
        "split": "train",
        "streaming": True,
        "fallback_fields": ["text", "content", "abstract"],
    },
}


class DataDownloader:
    """Automatic data downloader with multi-domain support.

    Downloads and prepares datasets for each domain:
    - web: FineWeb-Edu (general web text)
    - code: StackOverflow (programming Q&A)
    - math: FineMath (mathematical content)
    - books: ML-ArXiv-Papers (ML paper abstracts)
    - scientific: AI-ArXiv (AI research chunks)
    """

    def __init__(
        self,
        output_dir: str | Path = "./data",
        cache_dir: str | Path | None = None,
    ):
        """Initialize the downloader.

        Args:
            output_dir: Directory to store downloaded data
            cache_dir: Optional cache directory for HuggingFace
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def check_domain_exists(self, domain: str, min_samples: int = 100) -> bool:
        """Check if a domain has sufficient data.

        Args:
            domain: Domain name (web, code, math, books, scientific)
            min_samples: Minimum samples required

        Returns:
            True if domain has enough data
        """
        domain_dir = self.output_dir / domain
        if not domain_dir.exists():
            return False

        # Check for data files
        files = (
            list(domain_dir.glob("*.jsonl"))
            + list(domain_dir.glob("*.json"))
            + list(domain_dir.glob("*.parquet"))
        )
        if not files:
            return False

        # Count samples in first few files
        sample_count = 0
        for f in files[:5]:
            if f.suffix == ".jsonl":
                with open(f) as fp:
                    sample_count += sum(1 for _ in fp)
            elif f.suffix == ".parquet":
                try:
                    import pyarrow.parquet as pq

                    table = pq.read_table(f)
                    sample_count += len(table)
                except ImportError:
                    sample_count += 100  # Assume it has data

        return sample_count >= min_samples

    def check_all_domains_exist(self, min_samples: int = 100) -> dict[str, bool]:
        """Check which domains have data.

        Returns:
            Dictionary mapping domain names to existence status
        """
        return {
            domain: self.check_domain_exists(domain, min_samples)
            for domain in DOMAIN_DATASETS
        }

    def _extract_text(self, sample: dict, config: dict) -> str | None:
        """Extract text from a sample using configured field or fallbacks."""
        # Try primary text field
        text = sample.get(config["text_field"])
        if text:
            return str(text)

        # Try fallback fields
        for field in config.get("fallback_fields", []):
            text = sample.get(field)
            if text:
                return str(text)

        # Try common fields
        for field in ["text", "content", "body", "abstract"]:
            text = sample.get(field)
            if text:
                return str(text)

        return None

    def download_domain(
        self,
        domain: str,
        max_samples: int = 5000,
        shard_size: int = 1000,
    ) -> dict:
        """Download a specific domain dataset.

        Args:
            domain: Domain name (web, code, math, books, scientific)
            max_samples: Maximum samples to download
            shard_size: Samples per shard file

        Returns:
            Dictionary with download statistics
        """
        if domain not in DOMAIN_DATASETS:
            raise ValueError(
                f"Unknown domain: {domain}. Available: {list(DOMAIN_DATASETS.keys())}"
            )

        try:
            from datasets import load_dataset
        except ImportError as exc:
            LOGGER.error("datasets package not installed. Run: uv pip install datasets")
            raise RuntimeError(
                "datasets package required for auto-download. "
                "Install with: uv pip install datasets"
            ) from exc

        config = DOMAIN_DATASETS[domain]
        domain_dir = self.output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "Downloading %s dataset from %s (max_samples=%d)",
            domain,
            config["hf_path"],
            max_samples,
        )

        # Load dataset with streaming
        try:
            load_kwargs: dict[str, Any] = {
                "path": config["hf_path"],
                "split": config["split"],
                "streaming": config.get("streaming", True),
            }
            if config.get("subset"):
                load_kwargs["name"] = config["subset"]

            dataset = load_dataset(**load_kwargs)
        except Exception as e:
            LOGGER.warning(
                "Failed to load %s: %s. Trying without streaming...", domain, e
            )
            try:
                load_kwargs["streaming"] = False
                dataset = load_dataset(**load_kwargs)
            except Exception as e2:
                LOGGER.error("Failed to load %s dataset: %s", domain, e2)
                return {"samples_downloaded": 0, "error": str(e2)}

        stats = {
            "domain": domain,
            "hf_path": config["hf_path"],
            "samples_downloaded": 0,
            "shards_written": 0,
            "bytes_written": 0,
            "skipped_empty": 0,
        }

        current_shard: list[dict] = []
        shard_idx = 0

        try:
            for idx, sample in enumerate(dataset):
                if idx >= max_samples:
                    break

                text = self._extract_text(sample, config)
                if not text or len(text.strip()) < 10:
                    stats["skipped_empty"] += 1
                    continue

                current_shard.append(
                    {
                        "text": text,
                        "id": sample.get("id", str(idx)),
                        "domain": domain,
                    }
                )
                stats["samples_downloaded"] += 1

                # Write shard when full
                if len(current_shard) >= shard_size:
                    shard_path = domain_dir / f"shard_{shard_idx:05d}.jsonl"
                    bytes_written = self._write_shard(shard_path, current_shard)
                    stats["bytes_written"] += bytes_written
                    stats["shards_written"] += 1
                    current_shard = []
                    shard_idx += 1

                    if shard_idx % 5 == 0:
                        LOGGER.info(
                            "[%s] Written %d shards (%d samples)",
                            domain,
                            shard_idx,
                            stats["samples_downloaded"],
                        )

        except Exception as e:
            LOGGER.warning("Error during download of %s: %s", domain, e)
            stats["error"] = str(e)

        # Write remaining samples
        if current_shard:
            shard_path = domain_dir / f"shard_{shard_idx:05d}.jsonl"
            bytes_written = self._write_shard(shard_path, current_shard)
            stats["bytes_written"] += bytes_written
            stats["shards_written"] += 1

        # Write manifest
        manifest_path = domain_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(stats, f, indent=2)

        LOGGER.info(
            "[%s] Download complete: %d samples in %d shards (%.2f MB)",
            domain,
            stats["samples_downloaded"],
            stats["shards_written"],
            stats["bytes_written"] / (1024 * 1024),
        )

        return stats

    def download_all_domains(
        self,
        max_samples_per_domain: int = 5000,
        shard_size: int = 1000,
        skip_existing: bool = True,
        domains: list[str] | None = None,
    ) -> dict[str, dict]:
        """Download all domain datasets.

        Args:
            max_samples_per_domain: Max samples per domain
            shard_size: Samples per shard file
            skip_existing: Skip domains that already have data
            domains: Specific domains to download (None = all)

        Returns:
            Dictionary mapping domain names to download stats
        """
        target_domains = domains or list(DOMAIN_DATASETS.keys())
        results: dict[str, dict] = {}

        for domain in target_domains:
            if skip_existing and self.check_domain_exists(domain):
                LOGGER.info("[%s] Already exists, skipping", domain)
                results[domain] = {"skipped": True, "reason": "exists"}
                continue

            LOGGER.info("=" * 60)
            LOGGER.info("Downloading domain: %s", domain)
            LOGGER.info("=" * 60)

            results[domain] = self.download_domain(
                domain=domain,
                max_samples=max_samples_per_domain,
                shard_size=shard_size,
            )

        # Write overall manifest
        manifest_path = self.output_dir / "domains_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _write_shard(self, path: Path, samples: list) -> int:
        """Write samples to a JSONL shard file."""
        bytes_written = 0
        with open(path, "w") as f:
            for sample in samples:
                line = json.dumps(sample, ensure_ascii=False) + "\n"
                f.write(line)
                bytes_written += len(line.encode("utf-8"))
        return bytes_written

    def ensure_data_available(
        self,
        auto_download: bool = True,
        max_samples_per_domain: int = 5000,
        domains: list[str] | None = None,
    ) -> Path:
        """Ensure training data is available, downloading if needed.

        Args:
            auto_download: Whether to auto-download if data missing
            max_samples_per_domain: Max samples per domain
            domains: Specific domains to ensure (None = all)

        Returns:
            Path to the data directory

        Raises:
            RuntimeError: If data is not available and auto_download is False
        """
        target_domains = domains or list(DOMAIN_DATASETS.keys())
        missing_domains = [
            d for d in target_domains if not self.check_domain_exists(d)
        ]

        if not missing_domains:
            LOGGER.info("All requested domains have data")
            return self.output_dir

        if not auto_download:
            raise RuntimeError(
                f"Missing data for domains: {missing_domains}. "
                "Set auto_download=True or manually download data."
            )

        LOGGER.info("Missing domains: %s. Starting auto-download...", missing_domains)
        self.download_all_domains(
            max_samples_per_domain=max_samples_per_domain,
            domains=missing_domains,
        )

        return self.output_dir


def ensure_training_data(
    data_dir: str | Path = "./data",
    auto_download: bool = True,
    max_samples_per_domain: int = 5000,
    domains: list[str] | None = None,
) -> Path:
    """Convenience function to ensure training data is available.

    Args:
        data_dir: Directory to store/find data
        auto_download: Whether to auto-download if missing
        max_samples_per_domain: Max samples per domain
        domains: Specific domains to download

    Returns:
        Path to the data directory
    """
    downloader = DataDownloader(output_dir=data_dir)
    return downloader.ensure_data_available(
        auto_download=auto_download,
        max_samples_per_domain=max_samples_per_domain,
        domains=domains,
    )


def download_all_datasets(
    output_dir: str | Path = "./data",
    max_samples_per_domain: int = 5000,
    skip_existing: bool = True,
) -> dict[str, dict]:
    """Download all domain datasets.

    Args:
        output_dir: Directory to save data
        max_samples_per_domain: Max samples per domain
        skip_existing: Skip domains that already exist

    Returns:
        Download statistics for each domain
    """
    downloader = DataDownloader(output_dir=output_dir)
    return downloader.download_all_domains(
        max_samples_per_domain=max_samples_per_domain,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    # CLI for manual downloading
    import argparse

    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    parser.add_argument(
        "--max-samples", type=int, default=5000, help="Max samples per domain"
    )
    parser.add_argument("--domains", nargs="+", help="Specific domains to download")
    parser.add_argument(
        "--force", action="store_true", help="Re-download existing domains"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    downloader = DataDownloader(output_dir=args.output_dir)
    results = downloader.download_all_domains(
        max_samples_per_domain=args.max_samples,
        skip_existing=not args.force,
        domains=args.domains,
    )

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for domain, stats in results.items():
        if stats.get("skipped"):
            print(f"  {domain}: SKIPPED ({stats.get('reason', 'unknown')})")
        elif stats.get("error"):
            print(f"  {domain}: ERROR - {stats['error']}")
        else:
            print(
                f"  {domain}: {stats['samples_downloaded']} samples "
                f"in {stats['shards_written']} shards"
            )
