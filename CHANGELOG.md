# Changelog

All notable changes to DeepSeek From Scratch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

When adding to this changelog:
1. Add entries under `[Unreleased]`
2. Follow the Keep a Changelog format
3. Group changes by type (Added, Changed, Deprecated, Removed, Fixed, Security)

## [Unreleased]

### Added
- **Modal Multi-GPU Training**: Real distributed training with Ray TorchTrainer on up to 8 A100 GPUs per node
- **Cargo Build Caching**: Persistent Modal volumes for `target/` and `.cargo/registry` to eliminate rebuild times
- **Training Checkpointing**: Automatic checkpoints every N steps for fault tolerance in distributed training
- **Sequential Large-Scale Runs**: Support for >10 GPU configurations via sequential 8-GPU runs respecting Modal concurrency limits
- **Scaled Batch Sizes**: Automatic batch size scaling based on world size for distributed training

### Changed
- **Rust Test Features**: Changed from `--features cuda,pyo3-bindings` to `--features cuda` for Modal tests (pyo3-bindings requires Python runtime)
- **Build Performance**: Removed automatic `cargo clean` - builds now use cached artifacts via Modal volumes
- **GPU Verification**: `run_pytorch_verification` now uses actual multi-GPU distributed training instead of single-GPU mock

### Fixed
- **Rust Tests Exit Code 101**: Fixed by removing pyo3-bindings feature that requires Python FFI at runtime
- **Identical Training Times**: Fixed 8 vs 64 GPU training showing same time by implementing real multi-GPU parallelism
- **Build Time Regression**: Fixed full rebuilds on every run by persisting Cargo cache in Modal volumes

