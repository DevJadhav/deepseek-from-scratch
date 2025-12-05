# Contributing to DeepSeek From Scratch

First off, thank you for considering contributing to DeepSeek From Scratch! It's people like you that make this project a great learning resource for the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [dev@example.com](mailto:dev@example.com).

## Getting Started

### Prerequisites

- Python 3.10+
- Rust (stable)
- uv package manager
- Git

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/DevJadhav/deepseek-from-scratch.git
cd DeepSeek-From-Scratch

# Install Python dependencies
uv sync

# Install Rust dependencies
cd Deepseek-from-scratch-in-rust
cargo build --release
```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When creating a bug report, include:**

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (OS, Python version, GPU, etc.)
- Relevant logs or error messages
- Code snippets if applicable

**Template:**
```markdown
**Description:**
A clear description of the bug.

**Steps to Reproduce:**
1. Go to '...'
2. Run command '...'
3. See error

**Expected Behavior:**
What you expected to happen.

**Actual Behavior:**
What actually happened.

**Environment:**
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.12.0]
- GPU: [e.g., NVIDIA A100]
- CUDA: [e.g., 12.1]
```

### Suggesting Features

Feature suggestions are welcome! Please include:

- A clear use case
- Potential implementation approach
- Whether you're willing to implement it

### Code Contributions

#### Areas We'd Love Help With

1. **Performance Optimizations**
   - Flash Attention improvements
   - Custom CUDA kernels
   - Memory efficiency

2. **New Features**
   - Additional attention mechanisms
   - New MoE routing strategies
   - Quantization methods

3. **Testing**
   - Expanded test coverage
   - Benchmark additions
   - Edge case testing

4. **Documentation**
   - Tutorial improvements
   - API documentation
   - Blog posts

5. **Infrastructure**
   - CI/CD improvements
   - Docker optimizations
   - Cloud deployment guides

## Development Setup

### Python Development

```bash
# Create virtual environment with uv
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Run type checking
uv run mypy src/deepseek/

# Format code
uv run black .
uv run ruff format .
```

### Rust Development

```bash
cd rust-src

# Build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Format code
cargo fmt

# Lint
cargo clippy
```

### Pre-commit Hooks

We recommend setting up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Style Guidelines

### Python Style

- Follow PEP 8 and PEP 257
- Use type hints for all function signatures
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names
- Document public functions with docstrings

**Example:**
```python
def compute_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.
    
    Args:
        query: Query tensor of shape (batch, heads, seq_len, head_dim)
        key: Key tensor of shape (batch, heads, seq_len, head_dim)
        value: Value tensor of shape (batch, heads, seq_len, head_dim)
        mask: Optional attention mask
        
    Returns:
        Attention output of shape (batch, heads, seq_len, head_dim)
    """
    ...
```

### Rust Style

- Follow Rust conventions and idioms
- Use `rustfmt` for formatting
- Address all `clippy` warnings
- Document public APIs with doc comments

**Example:**
```rust
/// Computes scaled dot-product attention.
///
/// # Arguments
///
/// * `query` - Query tensor of shape (batch, heads, seq_len, head_dim)
/// * `key` - Key tensor
/// * `value` - Value tensor
///
/// # Returns
///
/// Attention output tensor
pub fn compute_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    ...
}
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(mla): add support for variable-length sequences
fix(moe): resolve expert load balancing race condition
docs(readme): update installation instructions
perf(attention): optimize flash attention memory usage
```

## Pull Request Process

### Before Submitting

1. **Fork and branch**: Create a fork and work on a feature branch
2. **Test**: Ensure all tests pass locally
3. **Lint**: Run linting and fix any issues
4. **Document**: Update documentation if needed
5. **Changelog**: Add entry to CHANGELOG.md

### PR Requirements

- [ ] Clear description of changes
- [ ] Tests for new functionality
- [ ] Documentation updates
- [ ] No linting errors
- [ ] All CI checks pass

### Review Process

1. Submit PR with clear description
2. Address reviewer feedback
3. Ensure CI passes
4. Get approval from maintainer
5. Squash and merge

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing done.

## Checklist
- [ ] Tests pass locally
- [ ] Linting passes
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Discord**: [Coming soon]

### Recognition

Contributors are recognized in:
- The project README
- Release notes
- Annual contributor acknowledgments

### Maintainers

- [@DevJadhav](https://github.com/DevJadhav) - Project Lead

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to DeepSeek From Scratch! 🚀
