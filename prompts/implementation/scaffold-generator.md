# Project Scaffolding Generator

## Objective
Generate complete POC project structure with boilerplate code for `{{USE_CASE_NAME}}`.

## Purpose
This scaffold creates a proof-of-concept structure for:
- **Learning**: Organized codebase to understand GPU acceleration components
- **Experimentation**: Easy-to-modify structure for testing different approaches
- **Benchmarking**: Built-in tools to measure GPU vs CPU performance
- **Demonstration**: Clear structure to share POC results with stakeholders

**Important:** This generates POC-quality scaffolding. Production readiness requires additional hardening.

## Prerequisites
- Completed `planning/01-use-case-discovery.md`
- Completed `planning/02-architecture-design.md`

## Input Variables
- `{{USE_CASE_SLUG}}`: snake_case name (e.g., "realtime_recommendation")
- `{{MODULE_STRUCTURE}}`: From architecture design
- `{{DEPENDENCIES}}`: From architecture design
- `{{SELECTED_STACK}}`: RAPIDS/TensorRT components

## Project Structure Template

Generate the following structure:

```
{{USE_CASE_SLUG}}/
├── .github/
│   └── workflows/
│       ├── test.yml           # CI testing
│       └── benchmark.yml      # Performance benchmarking
├── config/
│   ├── default.yaml           # Default configuration
│   ├── dev.yaml               # Development overrides
│   └── prod.yaml              # Production settings
├── data/
│   ├── raw/.gitkeep           # Raw data storage
│   ├── processed/.gitkeep     # Processed data
│   └── sample/                # Sample datasets for testing
│       └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_benchmarking.ipynb
├── src/
│   └── {{USE_CASE_SLUG}}/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── preprocessor.py
│       │   └── feature_engineering.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── predictor.py
│       │   └── base.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── gpu_utils.py
│       │   ├── metrics.py
│       │   ├── config.py
│       │   └── logging.py
│       └── api/
│           ├── __init__.py
│           ├── server.py
│           └── schemas.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data/
│   ├── test_models/
│   └── test_integration/
├── benchmarks/
│   ├── benchmark_data_pipeline.py
│   ├── benchmark_training.py
│   ├── benchmark_inference.py
│   └── compare_cpu_gpu.py
├── docker/
│   ├── Dockerfile.rapids      # RAPIDS-based GPU image
│   ├── Dockerfile.cpu         # CPU fallback image
│   ├── docker-compose.yml     # Multi-container orchestration
│   ├── .dockerignore          # Docker ignore patterns
│   └── entrypoint.sh          # Container entrypoint script
├── scripts/
│   ├── setup_env.sh           # Environment setup
│   ├── download_data.sh       # Data acquisition
│   ├── train.py               # Training script
│   └── serve.py               # Inference server
├── docs/
│   ├── README.md              # Main documentation
│   ├── SETUP.md               # Setup instructions
│   ├── USAGE.md               # Usage guide
│   └── BENCHMARKS.md          # Performance results
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── LICENSE
```

## Core File Templates

### 1. pyproject.toml

```toml
[project]
name = "{{USE_CASE_SLUG}}"
version = "0.1.0"
description = "{{USE_CASE_NAME}} - GPU-accelerated with NVIDIA RAPIDS and TensorRT"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

dependencies = [
    {{DEPENDENCIES}}
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-benchmark>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
]
notebooks = [
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100
target-version = ['py310', 'py311']

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --strict-markers"
markers = [
    "gpu: tests requiring GPU",
    "slow: slow running tests",
    "integration: integration tests",
]
```

### 2. config/default.yaml

```yaml
# {{USE_CASE_NAME}} Configuration

use_case: "{{USE_CASE_NAME}}"
version: "0.1.0"

# Data Configuration
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  train_split: 0.8
  validation_split: 0.1
  test_split: 0.1
  batch_size: 1024

# GPU Configuration
gpu:
  enabled: true
  device_id: 0
  memory_fraction: 0.8
  multi_gpu: false
  fallback_to_cpu: true

# Model Configuration
model:
  type: "{{MODEL_TYPE}}"
  checkpoint_dir: "models/checkpoints"
  params:
    # Model-specific parameters from architecture design

# Training Configuration
training:
  epochs: 10
  learning_rate: 0.001
  early_stopping_patience: 3
  save_best_only: true

# Inference Configuration
inference:
  batch_size: 256
  precision: "fp16"
  max_latency_ms: 100
  enable_tensorrt: false

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_dir: "logs"

# Benchmarking Configuration
benchmarking:
  enable_cpu_baseline: true
  warmup_iterations: 10
  benchmark_iterations: 100
  metrics:
    - latency_p50
    - latency_p95
    - latency_p99
    - throughput
    - gpu_memory_used
```

### 3. src/{{USE_CASE_SLUG}}/utils/gpu_utils.py

```python
"""GPU utility functions for device management and fallback handling."""

import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["gpu", "cpu"]


def check_gpu_available() -> bool:
    """
    Check if GPU is available for computation.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        import cudf
        # Try to create a small cuDF DataFrame to verify GPU works
        _ = cudf.DataFrame({"test": [1, 2, 3]})
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"GPU not available: {e}")
        return False


def get_device(preferred: DeviceType = "gpu", fallback: bool = True) -> DeviceType:
    """
    Get the computation device based on availability and preferences.

    Args:
        preferred: Preferred device type ("gpu" or "cpu")
        fallback: Whether to fallback to CPU if GPU unavailable

    Returns:
        DeviceType: The device to use for computation

    Raises:
        RuntimeError: If GPU requested but unavailable and fallback=False
    """
    if preferred == "cpu":
        logger.info("Using CPU as requested")
        return "cpu"

    if check_gpu_available():
        logger.info("Using GPU for computation")
        return "gpu"

    if fallback:
        logger.warning("GPU unavailable, falling back to CPU")
        return "cpu"

    raise RuntimeError("GPU requested but not available, and fallback disabled")


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory usage information.

    Returns:
        dict: Memory information (total, used, free in bytes)
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "total": info.total,
            "used": info.used,
            "free": info.free,
        }
    except Exception as e:
        logger.warning(f"Could not get GPU memory info: {e}")
        return {}


def set_gpu_memory_limit(fraction: float = 0.8) -> None:
    """
    Set GPU memory pool size limit.

    Args:
        fraction: Fraction of total GPU memory to use (0.0-1.0)
    """
    try:
        import rmm
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=None,
            maximum_pool_size_limit=fraction,
        )
        logger.info(f"Set GPU memory limit to {fraction*100}% of total")
    except ImportError:
        logger.warning("rmm not available, cannot set memory limit")
```

### 4. src/{{USE_CASE_SLUG}}/utils/config.py

```python
"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Base configuration model."""

    use_case: str
    version: str
    data: Dict[str, Any]
    gpu: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    inference: Dict[str, Any]
    logging: Dict[str, Any]
    benchmarking: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def load(cls, env: str = "default") -> "Config":
        """
        Load configuration for specified environment.

        Args:
            env: Environment name (default, dev, prod)

        Returns:
            Config: Loaded configuration
        """
        config_dir = Path(__file__).parents[3] / "config"
        config_file = config_dir / f"{env}.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        return cls.from_yaml(config_file)
```

### 5. README.md

```markdown
# {{USE_CASE_NAME}}

GPU-accelerated {{USE_CASE_NAME}} using NVIDIA RAPIDS and TensorRT.

## Overview

{{USE_CASE_DESCRIPTION}}

**Key Features:**
- GPU-accelerated data processing with cuDF
- {{MODEL_TYPE}} model training and inference
- {{SPEEDUP}}x faster than CPU baseline
- Production-ready API endpoint
- Comprehensive benchmarking suite

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.0+
- UV package manager

### Installation

\`\`\`bash
# Clone repository
git clone <repo-url>
cd {{USE_CASE_SLUG}}

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
\`\`\`

### Usage

\`\`\`python
from {{USE_CASE_SLUG}}.models import Predictor
from {{USE_CASE_SLUG}}.utils import Config

# Load configuration
config = Config.load("default")

# Initialize predictor
predictor = Predictor(config)

# Make predictions
results = predictor.predict(input_data)
\`\`\`

## Documentation

- [Setup Guide](docs/SETUP.md)
- [Usage Examples](docs/USAGE.md)
- [Benchmark Results](docs/BENCHMARKS.md)

## Performance

| Metric | CPU Baseline | GPU ({{GPU_MODEL}}) | Speedup |
|--------|-------------|---------------------|---------|
| Training Time | {{CPU_TRAIN}} | {{GPU_TRAIN}} | {{TRAIN_SPEEDUP}}x |
| Inference (p99) | {{CPU_P99}} | {{GPU_P99}} | {{INFER_SPEEDUP}}x |

## License

Apache 2.0
\`\`\`

### 6. docker/Dockerfile.rapids

**For RAPIDS Use Cases** (cuDF, cuML, cuGraph):

```dockerfile
# Dockerfile.rapids - RAPIDS-based GPU image for POC deployment
# Based on official NVIDIA RAPIDS container images

# Select RAPIDS version based on CUDA version
# CUDA 12.x: use rapids:24.10-cuda12.0-py3.11
# CUDA 11.x: use rapids:24.10-cuda11.8-py3.11
ARG RAPIDS_VERSION=24.10
ARG CUDA_VERSION=12.0
ARG PYTHON_VERSION=3.11

FROM nvcr.io/nvidia/rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py${PYTHON_VERSION}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Install UV package manager for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Install project dependencies
# RAPIDS packages (cudf, cuml) are already included in base image
RUN uv pip install --system -e .

# Install additional dependencies based on use case
{{#if NEEDS_XGBOOST}}
RUN uv pip install --system xgboost
{{/if}}
{{#if NEEDS_TENSORRT}}
RUN uv pip install --system tensorrt
{{/if}}

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create directories for data and models
RUN mkdir -p /workspace/data /workspace/models /workspace/logs

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose ports for API server and Jupyter
EXPOSE 8000 8888

# Health check for API server
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
```

### 7. docker/Dockerfile.cpu

**For CPU Fallback/Development**:

```dockerfile
# Dockerfile.cpu - CPU-only image for development and testing
FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Install project with CPU-only dependencies
# Use pandas/sklearn instead of cudf/cuml
RUN uv pip install --system -e .
RUN uv pip install --system pandas scikit-learn

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create directories
RUN mkdir -p /workspace/data /workspace/models /workspace/logs

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV CUDA_VISIBLE_DEVICES=""

EXPOSE 8000 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
```

### 8. docker/docker-compose.yml

```yaml
# docker-compose.yml - Multi-container orchestration for POC

version: '3.8'

services:
  # GPU-accelerated service using RAPIDS
  rapids-gpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile.rapids
      args:
        RAPIDS_VERSION: ${RAPIDS_VERSION:-24.10}
        CUDA_VERSION: ${CUDA_VERSION:-12.0}
        PYTHON_VERSION: ${PYTHON_VERSION:-3.11}
    image: {{USE_CASE_SLUG}}:rapids-gpu
    container_name: {{USE_CASE_SLUG}}-gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - CONFIG_ENV=${CONFIG_ENV:-default}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    volumes:
      - ../data:/workspace/data
      - ../models:/workspace/models
      - ../logs:/workspace/logs
      - ../config:/workspace/config
    ports:
      - "8000:8000"
      - "8888:8888"
    networks:
      - {{USE_CASE_SLUG}}-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # CPU fallback service for comparison/testing
  cpu-baseline:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cpu
    image: {{USE_CASE_SLUG}}:cpu
    container_name: {{USE_CASE_SLUG}}-cpu
    environment:
      - CUDA_VISIBLE_DEVICES=""
      - CONFIG_ENV=${CONFIG_ENV:-default}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    volumes:
      - ../data:/workspace/data
      - ../models:/workspace/models
      - ../logs:/workspace/logs
      - ../config:/workspace/config
    ports:
      - "8001:8000"
      - "8889:8888"
    networks:
      - {{USE_CASE_SLUG}}-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Jupyter notebook server for interactive development (GPU)
  jupyter-gpu:
    build:
      context: ..
      dockerfile: docker/Dockerfile.rapids
    image: {{USE_CASE_SLUG}}:rapids-gpu
    container_name: {{USE_CASE_SLUG}}-jupyter-gpu
    runtime: nvidia
    command: jupyter
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - JUPYTER_ENABLE_LAB=yes
    volumes:
      - ../notebooks:/workspace/notebooks
      - ../data:/workspace/data
      - ../models:/workspace/models
      - ../src:/workspace/src
    ports:
      - "8890:8888"
    networks:
      - {{USE_CASE_SLUG}}-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

networks:
  {{USE_CASE_SLUG}}-network:
    driver: bridge

volumes:
  data:
  models:
  logs:
```

### 9. docker/entrypoint.sh

```bash
#!/bin/bash
# entrypoint.sh - Container entrypoint script

set -e

# Function to wait for GPU availability
wait_for_gpu() {
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo "GPU is available"
    else
        echo "Warning: nvidia-smi not found, running in CPU mode"
    fi
}

# Function to run training
run_training() {
    echo "Starting model training..."
    python scripts/train.py \
        --config "config/${CONFIG_ENV:-default}.yaml" \
        --device gpu \
        "$@"
}

# Function to run inference server
run_serve() {
    echo "Starting inference server..."
    python scripts/serve.py \
        --config "config/${CONFIG_ENV:-default}.yaml" \
        --host 0.0.0.0 \
        --port 8000 \
        "$@"
}

# Function to run benchmarks
run_benchmark() {
    echo "Running benchmarks..."
    python benchmarks/compare_cpu_gpu.py \
        --config "config/${CONFIG_ENV:-default}.yaml" \
        "$@"
}

# Function to run Jupyter notebook server
run_jupyter() {
    echo "Starting Jupyter notebook server..."
    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password='' \
        --notebook-dir=/workspace/notebooks
}

# Function to run custom command
run_custom() {
    echo "Running custom command: $@"
    exec "$@"
}

# Wait for GPU if available
wait_for_gpu

# Route to appropriate function based on command
case "$1" in
    train)
        shift
        run_training "$@"
        ;;
    serve)
        shift
        run_serve "$@"
        ;;
    benchmark)
        shift
        run_benchmark "$@"
        ;;
    jupyter)
        shift
        run_jupyter "$@"
        ;;
    bash|sh)
        exec /bin/bash
        ;;
    *)
        run_custom "$@"
        ;;
esac
```

### 10. docker/.dockerignore

```
# Docker ignore patterns
.git
.github
.venv
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info
.pytest_cache
.coverage
htmlcov/
dist/
build/
*.egg

# Data directories (mount as volumes instead)
data/raw/*
data/processed/*
!data/sample

# Model checkpoints (mount as volumes)
models/checkpoints/*

# Logs
logs/*

# Development files
.vscode
.idea
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Documentation
docs/
*.md
!README.md

# Tests (install separately in dev)
tests/
```

### 11. scripts/docker_build.sh

```bash
#!/bin/bash
# docker_build.sh - Build Docker images for the project

set -e

PROJECT_NAME="{{USE_CASE_SLUG}}"
RAPIDS_VERSION="${RAPIDS_VERSION:-24.10}"
CUDA_VERSION="${CUDA_VERSION:-12.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "Building Docker images for ${PROJECT_NAME}..."

# Build RAPIDS GPU image
echo "Building RAPIDS GPU image..."
docker build \
    -f docker/Dockerfile.rapids \
    -t "${PROJECT_NAME}:rapids-gpu" \
    -t "${PROJECT_NAME}:latest" \
    --build-arg RAPIDS_VERSION="${RAPIDS_VERSION}" \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    .

# Build CPU image
echo "Building CPU image..."
docker build \
    -f docker/Dockerfile.cpu \
    -t "${PROJECT_NAME}:cpu" \
    .

echo "Docker images built successfully!"
echo ""
echo "Available images:"
docker images | grep "${PROJECT_NAME}"
echo ""
echo "To run the GPU service:"
echo "  docker run --gpus all -p 8000:8000 ${PROJECT_NAME}:rapids-gpu"
echo ""
echo "To run with docker-compose:"
echo "  docker-compose -f docker/docker-compose.yml up -d"
```

### 12. scripts/docker_run.sh

```bash
#!/bin/bash
# docker_run.sh - Run Docker containers with common configurations

set -e

PROJECT_NAME="{{USE_CASE_SLUG}}"
COMMAND="${1:-serve}"

case "$COMMAND" in
    gpu|serve-gpu)
        echo "Starting GPU inference server..."
        docker run --gpus all \
            -p 8000:8000 \
            -v "$(pwd)/data:/workspace/data" \
            -v "$(pwd)/models:/workspace/models" \
            -v "$(pwd)/config:/workspace/config" \
            -e CONFIG_ENV=default \
            --name "${PROJECT_NAME}-gpu" \
            --rm \
            "${PROJECT_NAME}:rapids-gpu" \
            serve
        ;;

    cpu|serve-cpu)
        echo "Starting CPU inference server..."
        docker run \
            -p 8001:8000 \
            -v "$(pwd)/data:/workspace/data" \
            -v "$(pwd)/models:/workspace/models" \
            -v "$(pwd)/config:/workspace/config" \
            -e CUDA_VISIBLE_DEVICES="" \
            -e CONFIG_ENV=default \
            --name "${PROJECT_NAME}-cpu" \
            --rm \
            "${PROJECT_NAME}:cpu" \
            serve
        ;;

    train-gpu)
        echo "Starting GPU training..."
        docker run --gpus all \
            -v "$(pwd)/data:/workspace/data" \
            -v "$(pwd)/models:/workspace/models" \
            -v "$(pwd)/config:/workspace/config" \
            -e CONFIG_ENV=default \
            --name "${PROJECT_NAME}-train" \
            --rm \
            "${PROJECT_NAME}:rapids-gpu" \
            train
        ;;

    benchmark)
        echo "Running benchmarks..."
        docker run --gpus all \
            -v "$(pwd)/data:/workspace/data" \
            -v "$(pwd)/models:/workspace/models" \
            -v "$(pwd)/config:/workspace/config" \
            --name "${PROJECT_NAME}-benchmark" \
            --rm \
            "${PROJECT_NAME}:rapids-gpu" \
            benchmark
        ;;

    jupyter)
        echo "Starting Jupyter notebook server..."
        docker run --gpus all \
            -p 8890:8888 \
            -v "$(pwd)/notebooks:/workspace/notebooks" \
            -v "$(pwd)/data:/workspace/data" \
            -v "$(pwd)/models:/workspace/models" \
            -v "$(pwd)/src:/workspace/src" \
            --name "${PROJECT_NAME}-jupyter" \
            --rm \
            "${PROJECT_NAME}:rapids-gpu" \
            jupyter
        ;;

    bash)
        echo "Starting interactive bash shell..."
        docker run --gpus all -it \
            -v "$(pwd)/data:/workspace/data" \
            -v "$(pwd)/models:/workspace/models" \
            -v "$(pwd)/config:/workspace/config" \
            --name "${PROJECT_NAME}-bash" \
            --rm \
            "${PROJECT_NAME}:rapids-gpu" \
            bash
        ;;

    *)
        echo "Usage: $0 {gpu|cpu|train-gpu|benchmark|jupyter|bash}"
        echo ""
        echo "Commands:"
        echo "  gpu|serve-gpu  - Start GPU inference server (port 8000)"
        echo "  cpu|serve-cpu  - Start CPU inference server (port 8001)"
        echo "  train-gpu      - Run GPU training"
        echo "  benchmark      - Run performance benchmarks"
        echo "  jupyter        - Start Jupyter notebook server (port 8890)"
        echo "  bash           - Start interactive bash shell"
        exit 1
        ;;
esac
```

## Docker Usage Guide

### Building Images

```bash
# Build all images
./scripts/docker_build.sh

# Or build with specific RAPIDS version
RAPIDS_VERSION=24.10 CUDA_VERSION=12.0 ./scripts/docker_build.sh

# Using docker-compose
docker-compose -f docker/docker-compose.yml build
```

### Running Containers

**GPU Inference Server:**
```bash
# Using helper script
./scripts/docker_run.sh gpu

# Or directly with docker
docker run --gpus all -p 8000:8000 {{USE_CASE_SLUG}}:rapids-gpu serve

# Or with docker-compose
docker-compose -f docker/docker-compose.yml up rapids-gpu
```

**Training:**
```bash
# Using helper script
./scripts/docker_run.sh train-gpu

# Or with docker-compose
docker-compose -f docker/docker-compose.yml run rapids-gpu train
```

**Jupyter Notebooks:**
```bash
# Using helper script
./scripts/docker_run.sh jupyter

# Or with docker-compose
docker-compose -f docker/docker-compose.yml up jupyter-gpu
# Access at http://localhost:8890
```

**Benchmarking:**
```bash
# Run GPU vs CPU comparison
./scripts/docker_run.sh benchmark

# Or run both services and compare
docker-compose -f docker/docker-compose.yml up rapids-gpu cpu-baseline
```

### Docker Deployment Tips

**For NVIDIA Virtual Workstation:**
- NVIDIA Container Toolkit pre-installed
- Use `--gpus all` or `--runtime=nvidia`
- Monitor with `nvidia-smi` from host

**For GCP GPU Instances:**
- Install NVIDIA Container Toolkit:
  ```bash
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

**For Cloud Deployment:**
- Use Docker registry (GCR, ECR, Docker Hub)
- Tag images with versions: `{{USE_CASE_SLUG}}:v0.1.0-rapids-gpu`
- Use kubernetes manifests in `.github/kubernetes/`

## Scaffolding Checklist

Generate the following files in order:

- [ ] Project root structure
- [ ] pyproject.toml with dependencies
- [ ] config/default.yaml
- [ ] src/{{USE_CASE_SLUG}}/__init__.py
- [ ] src/{{USE_CASE_SLUG}}/utils/gpu_utils.py
- [ ] src/{{USE_CASE_SLUG}}/utils/config.py
- [ ] src/{{USE_CASE_SLUG}}/utils/logging.py
- [ ] src/{{USE_CASE_SLUG}}/data/__init__.py (stub)
- [ ] src/{{USE_CASE_SLUG}}/models/__init__.py (stub)
- [ ] docker/Dockerfile.rapids
- [ ] docker/Dockerfile.cpu
- [ ] docker/docker-compose.yml
- [ ] docker/entrypoint.sh
- [ ] docker/.dockerignore
- [ ] scripts/docker_build.sh
- [ ] scripts/docker_run.sh
- [ ] tests/conftest.py
- [ ] .gitignore
- [ ] README.md
- [ ] LICENSE

## Next Steps

After scaffolding:
- Proceed to `data-pipeline.md` for data processing implementation
- Use `model-implementation.md` for training/inference code
- Reference `validation/benchmark-plan.md` for testing strategy
- Test Docker images locally before deploying to GPU environment
