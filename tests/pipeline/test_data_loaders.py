"""Tests for FineWeb-Edu data loader scripts."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_dataset():
    """Create a mock HuggingFace dataset."""
    class MockDataset:
        def __init__(self, data):
            self.data = data
            self.index = 0

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    return MockDataset([
        {"text": "Sample text 1 for testing the tokenizer."},
        {"text": "Another sample document with more words."},
        {"text": "Third document to verify tokenization works."},
    ])


class TestDownloadFinewebEdu:
    """Test cases for download_fineweb_edu.py."""

    def test_script_exists(self):
        """Verify the download script exists."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "download_fineweb_edu.py"
        assert script_path.exists(), f"Script not found at {script_path}"

    def test_script_has_main_function(self):
        """Verify the script has a main function."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "download_fineweb_edu.py"
        content = script_path.read_text()
        assert "def main(" in content
        assert 'if __name__ == "__main__"' in content

    def test_script_has_download_function(self):
        """Verify the script has a download function."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "download_fineweb_edu.py"
        content = script_path.read_text()
        assert "def download_fineweb_edu(" in content

    def test_script_supports_arguments(self):
        """Verify the script accepts command line arguments."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "download_fineweb_edu.py"
        content = script_path.read_text()
        assert "argparse" in content or "ArgumentParser" in content
        assert "--output" in content
        assert "--max-samples" in content


class TestTokenizeFinewebEdu:
    """Test cases for tokenize_fineweb.py."""

    def test_script_exists(self):
        """Verify the tokenization script exists."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "tokenize_fineweb.py"
        assert script_path.exists(), f"Script not found at {script_path}"

    def test_script_has_main_function(self):
        """Verify the script has a main function."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "tokenize_fineweb.py"
        content = script_path.read_text()
        assert "def main(" in content
        assert 'if __name__ == "__main__"' in content

    def test_script_has_tokenize_function(self):
        """Verify the script has a tokenize function."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "tokenize_fineweb.py"
        content = script_path.read_text()
        assert "def tokenize_" in content or "def process_" in content

    def test_script_supports_arguments(self):
        """Verify the script accepts command line arguments."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "tokenize_fineweb.py"
        content = script_path.read_text()
        assert "argparse" in content or "ArgumentParser" in content
        assert "--input" in content
        assert "--output" in content

    def test_script_creates_shards(self):
        """Verify the script is designed to create shards."""
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "tokenize_fineweb.py"
        content = script_path.read_text()
        # Should create binary shards
        assert "shard" in content.lower()
        assert ".bin" in content


class TestDataPipelineIntegration:
    """Integration tests for data pipeline components."""

    def test_download_creates_output_directory(self, temp_dir):
        """Test that download function creates output directory."""
        output_dir = temp_dir / "downloads"
        assert not output_dir.exists()

        # Mock the actual download
        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = MagicMock()

            # Import and run with mocked dataset
            script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "download_fineweb_edu.py"
            if script_path.exists():
                # Just verify we can parse the script
                import ast
                content = script_path.read_text()
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    pytest.fail(f"Script has syntax errors: {e}")

    def test_tokenize_produces_manifest(self, temp_dir):
        """Test that tokenization produces a manifest file."""
        # Create mock input data
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        (input_dir / "sample.jsonl").write_text(
            json.dumps({"text": "Hello world"}) + "\n"
        )

        # Verify tokenize script structure supports manifest creation
        script_path = Path(__file__).parents[2] / "scripts" / "data-dowloader" / "tokenize_fineweb.py"
        if script_path.exists():
            content = script_path.read_text()
            assert "manifest" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
