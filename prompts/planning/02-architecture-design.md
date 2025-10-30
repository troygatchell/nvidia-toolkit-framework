# Architecture Design Prompt

## Objective
Design the detailed POC architecture for `{{USE_CASE_NAME}}` based on discovery phase findings.

## Purpose
This architecture is designed for:
- **Learning**: Understanding how GPU acceleration works in practice
- **Experimentation**: Testing different NVIDIA components
- **Benchmarking**: Measuring real performance improvements
- **Proof-of-Concept**: Validating approach before production investment

**Important:** This generates POC-quality code. Production deployment requires additional work on error handling, security, monitoring, and optimization.

## Prerequisites
Complete `01-use-case-discovery.md` first to determine technology stack.

## Input Variables
- `{{USE_CASE_NAME}}`: From discovery phase
- `{{SELECTED_STACK}}`: RAPIDS components + TensorRT (from discovery)
- `{{DATA_PIPELINE}}`: Input → Processing → Output flow
- `{{DEPLOYMENT_TARGET}}`: Local dev / Single GPU / Multi-GPU / Cloud

## Architecture Components

### 1. Data Pipeline Architecture

**Input Layer**:
```python
# Define data sources and formats
DATA_SOURCES = {
    "training": "{{TRAINING_DATA_PATH}}",  # e.g., "s3://bucket/train/"
    "inference": "{{INFERENCE_DATA_SOURCE}}",  # e.g., "kafka://topic" or API
}

DATA_FORMATS = "{{FORMAT}}"  # parquet, csv, json, avro
```

**Processing Layer** (cuDF operations):
```python
# Core GPU-accelerated transformations
PIPELINE_STEPS = [
    "load_data",           # cudf.read_parquet()
    "clean_data",          # GPU-accelerated filtering
    "feature_engineering", # cuDF operations
    "train_test_split",    # cuML or custom
]
```

**Output Layer**:
```python
OUTPUT_TYPE = "{{OUTPUT_TYPE}}"  # predictions, embeddings, clusters, rankings
OUTPUT_FORMAT = "{{OUTPUT_FORMAT}}"  # json, parquet, streaming
```

### 2. Model Architecture

**Training Component**:
```
Framework: [cuML / PyTorch with CUDA / XGBoost GPU / Custom]
Model Type: {{MODEL_TYPE}}
Hyperparameters: {{HYPERPARAMS}}

Training Strategy:
- Single GPU training: [yes/no]
- Multi-GPU with DASK: [yes/no]
- Distributed training: [yes/no]
```

**Inference Component**:
```
Optimization Strategy: [TensorRT / cuML native / Hybrid]
Batch Size: {{BATCH_SIZE}}
Precision: [FP32 / FP16 / INT8]
Expected Latency: {{TARGET_LATENCY}}

Deployment:
- Endpoint type: [REST API / gRPC / Batch]
- Concurrency: {{MAX_CONCURRENT_REQUESTS}}
- Load balancing: [yes/no]
```

### 3. Module Structure

Design the codebase modules:

```
src/{{USE_CASE_SLUG}}/
├── data/
│   ├── loader.py          # cuDF data loading
│   ├── preprocessor.py    # GPU transformations
│   └── feature_eng.py     # Feature engineering
├── models/
│   ├── trainer.py         # Model training (RAPIDS/PyTorch)
│   ├── predictor.py       # Inference engine
│   └── optimizer.py       # TensorRT optimization (if needed)
├── utils/
│   ├── gpu_utils.py       # GPU memory management, device selection
│   ├── metrics.py         # Performance tracking
│   └── config.py          # Configuration management
└── api/
    ├── server.py          # FastAPI/Flask endpoint
    └── batch.py           # Batch processing mode
```

### 4. Configuration Schema

Define `config/{{USE_CASE_SLUG}}.yaml`:

```yaml
use_case: "{{USE_CASE_NAME}}"
domain: "{{DOMAIN}}"

data:
  training_path: "{{TRAINING_DATA_PATH}}"
  validation_split: 0.2
  batch_size: {{BATCH_SIZE}}

gpu:
  device_id: 0
  memory_limit: "16GB"
  enable_multi_gpu: false

model:
  type: "{{MODEL_TYPE}}"
  params:
    # Model-specific parameters
    learning_rate: 0.001

inference:
  batch_size: {{INFERENCE_BATCH_SIZE}}
  precision: "fp16"
  max_latency_ms: {{TARGET_LATENCY}}

benchmarking:
  cpu_baseline: true
  metrics: ["latency", "throughput", "accuracy"]
```

### 5. Dependency Specification

Generate `pyproject.toml` with:

```toml
[project]
name = "{{USE_CASE_SLUG}}"
version = "0.1.0"
description = "{{USE_CASE_NAME}} - GPU-accelerated with NVIDIA RAPIDS"
requires-python = ">=3.10"

dependencies = [
    # Core RAPIDS (adjust cu11/cu12 based on CUDA version)
    "cudf-cu12>=24.10.0",
    "cuml-cu12>=24.10.0",

    # Add based on architecture:
    # "dask-cudf>=24.10.0",  # If multi-GPU
    # "cugraph-cu12>=24.10.0",  # If graph algorithms
    # "tensorrt>=8.6.0",  # If using TensorRT

    # Supporting libraries
    "pandas>=2.0.0",  # For CPU fallback
    "scikit-learn>=1.3.0",
    "fastapi>=0.104.0",  # If API endpoint
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-benchmark>=4.0.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
]
```

### 6. GPU Environment Considerations

**Development Environment** (CPU mode for code development):
- Local CPU fallback mode: yes (recommended for initial development)
- Docker container specs: {{DOCKER_IMAGE}}
- CUDA version: {{CUDA_VERSION}}

**POC Environment** (Choose based on learning goals):

**Option A: NVIDIA Virtual Workstation (Recommended for Learning)**
- Pre-configured CUDA environment
- GPU type: Start with T4 or V100
- Setup time: < 30 minutes
- Cost: $2-5/hour
- Best for: Learning, team training, initial POCs

**Option B: GCP GPU Instance (Cost-Conscious)**
- Instance type: `n1-standard-4` or `n1-standard-8`
- GPU type: T4 (learning), V100 (medium POCs), A100 (performance testing)
- Setup time: 1-2 hours (CUDA installation)
- Cost: $0.35-0.95/hour (T4), $2-4/hour (A100)
- Best for: Longer experiments, batch processing

**Option C: Local GPU (If Available)**
- Use local NVIDIA GPU for development
- Best for: Continuous experimentation, no cloud costs

**Note:** Kubernetes deployment is typically not needed for POC/learning phases.

### 7. Error Handling and Fallbacks

```python
# GPU availability check strategy
FALLBACK_STRATEGY = "{{FALLBACK}}"  # "cpu", "error", "degraded"

# Graceful degradation
if not gpu_available:
    if FALLBACK_STRATEGY == "cpu":
        # Use pandas/sklearn CPU versions
    elif FALLBACK_STRATEGY == "error":
        # Raise exception
    else:
        # Run with reduced functionality
```

## Performance Targets

Define expected performance improvements:

| Metric | CPU Baseline | GPU Target | Speedup |
|--------|-------------|------------|---------|
| Training Time | {{CPU_TRAIN_TIME}} | {{GPU_TRAIN_TIME}} | {{SPEEDUP}}x |
| Inference Latency (p50) | {{CPU_P50}} | {{GPU_P50}} | {{SPEEDUP}}x |
| Inference Latency (p99) | {{CPU_P99}} | {{GPU_P99}} | {{SPEEDUP}}x |
| Throughput | {{CPU_THROUGHPUT}} | {{GPU_THROUGHPUT}} | {{SPEEDUP}}x |

## Output Deliverables

1. **Module structure** with file descriptions
2. **Configuration schema** (YAML)
3. **Dependency list** (pyproject.toml format)
4. **Performance targets** for benchmarking
5. **Environment requirements** (CUDA, Docker, etc.)

## Next Steps

Proceed to:
- `implementation/scaffold-generator.md` - Generate project files
- `implementation/data-pipeline.md` - Implement data processing
- `implementation/model-implementation.md` - Build model training/inference
