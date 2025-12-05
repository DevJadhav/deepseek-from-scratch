"""
Pytest configuration for ANE tests.

Provides shared fixtures and skip markers for Apple Silicon-specific tests.
"""

import platform

import pytest

# Check for Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
IS_MACOS = platform.system() == "Darwin"


@pytest.fixture
def apple_silicon_required():
    """Skip test if not running on Apple Silicon."""
    if not IS_APPLE_SILICON:
        pytest.skip("Test requires Apple Silicon (M1/M2/M3/M4)")


@pytest.fixture
def macos_required():
    """Skip test if not running on macOS."""
    if not IS_MACOS:
        pytest.skip("Test requires macOS")


# Markers for conditional skipping
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "apple_silicon: mark test as requiring Apple Silicon"
    )
    config.addinivalue_line(
        "markers", "macos: mark test as requiring macOS"
    )


def pytest_collection_modifyitems(config, items):
    """Apply skip markers based on platform."""
    skip_apple_silicon = pytest.mark.skip(reason="Test requires Apple Silicon")
    skip_macos = pytest.mark.skip(reason="Test requires macOS")
    
    for item in items:
        if "apple_silicon" in item.keywords and not IS_APPLE_SILICON:
            item.add_marker(skip_apple_silicon)
        if "macos" in item.keywords and not IS_MACOS:
            item.add_marker(skip_macos)
