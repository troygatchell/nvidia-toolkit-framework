# Project Scaffolding Generator

## Objective
Generate complete project structure with boilerplate code for `{{USE_CASE_NAME}}`.

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
│   ├── Dockerfile.gpu         # CUDA-enabled image
│   ├── Dockerfile.cpu         # CPU fallback image
│   └── docker-compose.yml
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
- [ ] tests/conftest.py
- [ ] .gitignore
- [ ] README.md
- [ ] LICENSE

## Next Steps

After scaffolding:
- Proceed to `data-pipeline.md` for data processing implementation
- Use `model-implementation.md` for training/inference code
- Reference `validation/benchmark-plan.md` for testing strategy
