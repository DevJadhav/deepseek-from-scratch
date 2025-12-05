# DeepSeek-V3 Training - Production Dockerfile
# Multi-stage build for CUDA support with exact Python environment
#
# Build: docker build -t deepseek-v3:latest .
# Run: docker run --gpus all -v $(pwd)/data:/workspace/data -v $(pwd)/checkpoints:/workspace/checkpoints deepseek-v3:latest
#
# For development: docker build --target dev -t deepseek-v3:dev .
# For inference: docker build --target inference -t deepseek-v3:inference .

ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.1
ARG UV_VERSION=0.4

# =============================================================================
# Stage 1: Base CUDA image with system dependencies
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu22.04 AS base

# Build arguments
ARG PYTHON_VERSION
ARG UV_VERSION

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TOKENIZERS_PARALLELISM=false \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    ninja-build \
    libssl-dev \
    libffi-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /workspace

# =============================================================================
# Stage 2: Dependencies (cached layer)
# =============================================================================
FROM base AS dependencies

# Copy only dependency files for caching
COPY pyproject.toml uv.lock ./

# Install base dependencies
RUN uv sync --extra cuda --extra dev --no-install-project

# Install Flash Attention (requires CUDA build)
RUN uv pip install flash-attn --no-build-isolation || echo "Flash Attention install skipped"

# =============================================================================
# Stage 3: Development image (full tooling)
# =============================================================================
FROM dependencies AS dev

# Copy all project files
COPY . .

# Install project in editable mode
RUN uv sync --extra cuda --extra dev

# Set up development environment
ENV WANDB_DIR=/workspace/wandb \
    PYTHONPATH=/workspace:/workspace/deepseek-from-scratch-python/src

# Create output directories
RUN mkdir -p /workspace/checkpoints /workspace/logs /workspace/wandb /workspace/data /workspace/profiler_output

# Install additional dev tools
RUN uv pip install jupyterlab ipywidgets

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Development entrypoint
CMD ["bash"]

# =============================================================================
# Stage 4: Training image (optimized for production training)
# =============================================================================
FROM dependencies AS training

# Copy only necessary files for training
COPY deepseek-from-scratch-python/ ./deepseek-from-scratch-python/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY ray_pipeline/ ./ray_pipeline/
COPY pyproject.toml uv.lock ./

# Install project
RUN uv sync --extra cuda

# Set up training environment
ENV WANDB_DIR=/workspace/wandb \
    PYTHONPATH=/workspace:/workspace/deepseek-from-scratch-python/src \
    NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=0

# Create output directories
RUN mkdir -p /workspace/checkpoints /workspace/logs /workspace/wandb /workspace/data

# Healthcheck for distributed training
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Training entrypoint
ENTRYPOINT ["uv", "run", "python"]
CMD ["-m", "ray_pipeline.cli", "--help"]

# =============================================================================
# Stage 5: Inference image (slim, optimized for serving)
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu22.04 AS inference

ARG PYTHON_VERSION
ARG UV_VERSION

# Minimal environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /workspace

# Copy dependency files and install inference-only deps
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-extras || true

# Copy inference scripts and model code
COPY deepseek-from-scratch-python/src/ ./deepseek-from-scratch-python/src/
COPY deepseek-from-scratch-python/mlx_impl/ ./deepseek-from-scratch-python/mlx_impl/
COPY scripts/inference.py ./scripts/

# Set environment
ENV PYTHONPATH=/workspace:/workspace/deepseek-from-scratch-python/src

# Create checkpoint directory
RUN mkdir -p /workspace/checkpoints

# Inference entrypoint
ENTRYPOINT ["uv", "run", "python", "scripts/inference.py"]
CMD ["--help"]

# =============================================================================
# Default target: training
# =============================================================================
FROM training AS default
