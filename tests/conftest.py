"""
Shared pytest fixtures for the DeepSeek test suite.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def checkpoints_dir(project_root):
    """Return the checkpoints directory."""
    return project_root / "checkpoints"


@pytest.fixture
def config_dir(project_root):
    """Return the config directory."""
    return project_root / "config"


@pytest.fixture
def tiny_config(config_dir):
    """Load the tiny test configuration."""
    import json
    config_path = config_dir / "tiny_test.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}
