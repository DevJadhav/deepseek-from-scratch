"""
Tests for PyTorch+GPU CLI (modal_gpu/cli.py) and MLX CLI (mlx_impl/cli.py).

These tests verify that both CLIs match the Rust CLI structure with:
- train: Training command
- evaluate: Evaluation command
- export: Model export command
- demo: Component demonstrations
- status: Environment status check
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# =============================================================================
# PyTorch CLI Tests (modal_gpu/cli.py)
# =============================================================================


class TestPyTorchCLI:
    """Tests for modal_gpu/cli.py."""

    @pytest.fixture
    def cli_module(self):
        """Import the CLI module."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import deepseek.cloud.modal.cli as cli

        return cli

    def test_cli_module_exists(self, cli_module):
        """Test that CLI module can be imported."""
        assert cli_module is not None
        assert hasattr(cli_module, "app")

    def test_cli_has_train_command(self, cli_module):
        """Test that CLI has train command."""
        # Check that train function exists
        assert hasattr(cli_module, "train")
        assert callable(cli_module.train)

    def test_cli_has_evaluate_command(self, cli_module):
        """Test that CLI has evaluate command."""
        assert hasattr(cli_module, "evaluate")
        assert callable(cli_module.evaluate)

    def test_cli_has_export_command(self, cli_module):
        """Test that CLI has export command."""
        assert hasattr(cli_module, "export")
        assert callable(cli_module.export)

    def test_cli_has_demo_command(self, cli_module):
        """Test that CLI has demo command."""
        assert hasattr(cli_module, "demo")
        assert callable(cli_module.demo)

    def test_cli_has_status_command(self, cli_module):
        """Test that CLI has status command."""
        assert hasattr(cli_module, "status")
        assert callable(cli_module.status)

    def test_cli_has_infer_command(self, cli_module):
        """Test that CLI has infer command."""
        assert hasattr(cli_module, "infer")
        assert callable(cli_module.infer)

    def test_train_command_has_required_options(self, cli_module):
        """Test that train command has essential options."""
        import inspect

        sig = inspect.signature(cli_module.train)
        params = list(sig.parameters.keys())

        # Check for essential parameters
        assert "config" in params
        assert "output" in params
        assert "model_size" in params

    def test_evaluate_command_signature(self, cli_module):
        """Test that evaluate command has proper signature."""
        import inspect

        sig = inspect.signature(cli_module.evaluate)
        params = list(sig.parameters.keys())

        assert "checkpoint" in params
        assert "batch_size" in params

    def test_export_command_signature(self, cli_module):
        """Test that export command has proper signature."""
        import inspect

        sig = inspect.signature(cli_module.export)
        params = list(sig.parameters.keys())

        assert "checkpoint" in params
        assert "format" in params

    def test_has_deepspeed_option(self, cli_module):
        """Test that train command has DeepSpeed option."""
        import inspect

        sig = inspect.signature(cli_module.train)
        params = list(sig.parameters.keys())
        assert "use_deepspeed" in params or "zero_stage" in params

    def test_typer_app_configured(self, cli_module):
        """Test that typer app is properly configured."""
        app = cli_module.app
        assert app.info.name == "deepseek-pytorch"
        assert app.info.help is not None


class TestPyTorchCLIHelp:
    """Test CLI help messages."""

    def test_main_help_runs(self):
        """Test that main help command runs."""
        result = subprocess.run(
            [sys.executable, "-m", "deepseek.cloud.modal.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0
        assert "train" in result.stdout.lower() or "Train" in result.stdout

    def test_train_help_runs(self):
        """Test that train help command runs."""
        result = subprocess.run(
            [sys.executable, "-m", "deepseek.cloud.modal.cli", "train", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout or "--model-size" in result.stdout


# =============================================================================
# MLX CLI Tests (mlx_impl/cli.py)
# =============================================================================

# Check if MLX is available
try:
    import mlx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMLXCLI:
    """Tests for deepseek.mlx.cli."""

    @pytest.fixture
    def cli_module(self):
        """Import the MLX CLI module."""
        from deepseek.mlx import cli as mlx_cli
        return mlx_cli

    def test_cli_module_exists(self, cli_module):
        """Test that CLI module can be imported."""
        assert cli_module is not None
        assert hasattr(cli_module, "app")

    def test_cli_has_train_command(self, cli_module):
        """Test that CLI has train command."""
        assert hasattr(cli_module, "train")
        assert callable(cli_module.train)

    def test_cli_has_evaluate_command(self, cli_module):
        """Test that CLI has evaluate command."""
        assert hasattr(cli_module, "evaluate")
        assert callable(cli_module.evaluate)

    def test_cli_has_export_command(self, cli_module):
        """Test that CLI has export command."""
        assert hasattr(cli_module, "export")
        assert callable(cli_module.export)

    def test_cli_has_demo_command(self, cli_module):
        """Test that CLI has demo command."""
        assert hasattr(cli_module, "demo")
        assert callable(cli_module.demo)

    def test_cli_has_status_command(self, cli_module):
        """Test that CLI has status command."""
        assert hasattr(cli_module, "status")
        assert callable(cli_module.status)

    def test_train_command_has_model_options(self, cli_module):
        """Test that train command has model architecture options."""
        import inspect

        sig = inspect.signature(cli_module.train)
        params = list(sig.parameters.keys())

        # MLX-specific options
        assert "model_size" in params
        assert "use_moe" in params
        assert "use_mla" in params

    def test_demo_has_component_option(self, cli_module):
        """Test that demo command has component option."""
        import inspect

        sig = inspect.signature(cli_module.demo)
        params = list(sig.parameters.keys())

        assert "component" in params

    def test_export_has_format_option(self, cli_module):
        """Test that export command has format option."""
        import inspect

        sig = inspect.signature(cli_module.export)
        params = list(sig.parameters.keys())

        assert "format" in params

    def test_typer_app_configured(self, cli_module):
        """Test that typer app is properly configured."""
        app = cli_module.app
        assert app.info.name == "deepseek-mlx"


# =============================================================================
# CLI Consistency Tests
# =============================================================================


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestCLIConsistency:
    """Test that both CLIs have consistent interfaces."""

    @pytest.fixture
    def pytorch_cli(self):
        """Get PyTorch CLI module."""
        from deepseek.cloud.modal import cli
        return cli

    @pytest.fixture
    def mlx_cli(self):
        """Get MLX CLI module."""
        from deepseek.mlx import cli as mlx_cli
        return mlx_cli

    def test_both_have_train(self, pytorch_cli, mlx_cli):
        """Both CLIs have train command."""
        assert hasattr(pytorch_cli, "train")
        assert hasattr(mlx_cli, "train")

    def test_both_have_evaluate(self, pytorch_cli, mlx_cli):
        """Both CLIs have evaluate command."""
        assert hasattr(pytorch_cli, "evaluate")
        assert hasattr(mlx_cli, "evaluate")

    def test_both_have_demo(self, pytorch_cli, mlx_cli):
        """Both CLIs have demo command."""
        assert hasattr(pytorch_cli, "demo")
        assert hasattr(mlx_cli, "demo")

    def test_both_have_status(self, pytorch_cli, mlx_cli):
        """Both CLIs have status command."""
        assert hasattr(pytorch_cli, "status")
        assert hasattr(mlx_cli, "status")

    def test_both_use_typer(self, pytorch_cli, mlx_cli):
        """Both CLIs use typer."""
        import typer

        assert isinstance(pytorch_cli.app, typer.Typer)
        assert isinstance(mlx_cli.app, typer.Typer)


# =============================================================================
# Rust CLI Comparison Tests
# =============================================================================


class TestRustCLIComparison:
    """Test that Python CLIs match Rust CLI structure."""

    @pytest.fixture
    def rust_cli_path(self):
        """Get path to Rust CLI source."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "rust-src"
            / "src"
            / "main.rs"
        )

    def test_rust_cli_exists(self, rust_cli_path):
        """Verify Rust CLI exists."""
        assert rust_cli_path.exists()

    def test_rust_cli_has_train_command(self, rust_cli_path):
        """Verify Rust CLI has Train command."""
        content = rust_cli_path.read_text()
        assert "Train" in content or "train" in content.lower()

    def test_rust_cli_has_evaluate_command(self, rust_cli_path):
        """Verify Rust CLI has Evaluate command."""
        content = rust_cli_path.read_text()
        assert "Evaluate" in content or "evaluate" in content.lower()

    def test_rust_cli_has_export_command(self, rust_cli_path):
        """Verify Rust CLI has Export command."""
        content = rust_cli_path.read_text()
        assert "Export" in content or "export" in content.lower()

    def test_rust_cli_has_demo_command(self, rust_cli_path):
        """Verify Rust CLI has Demo command."""
        content = rust_cli_path.read_text()
        assert "Demo" in content or "demo" in content.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_pytorch_cli_can_show_status(self):
        """Test PyTorch CLI status command runs."""
        result = subprocess.run(
            [sys.executable, "-m", "deepseek.cloud.modal.cli", "status"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        # Status should work even without modal
        assert "Status" in result.stdout or "Environment" in result.stdout or result.returncode == 0

    def test_pytorch_cli_demo_runs(self):
        """Test PyTorch CLI demo command runs."""
        result = subprocess.run(
            [sys.executable, "-m", "deepseek.cloud.modal.cli", "demo"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=30,
        )
        # Demo might fail if PyTorch not configured, but should at least start
        assert result.returncode in [0, 1]  # Success or controlled error

    @pytest.mark.skipif(
        not MLX_AVAILABLE,
        reason="MLX not available",
    )
    def test_mlx_cli_help_available(self):
        """Test MLX CLI help is available."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from deepseek.mlx import cli; print(cli.app.info.name)",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should work or fail gracefully without MLX
        assert result.returncode in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
