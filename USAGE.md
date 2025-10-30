# Framework Usage Guide

Complete guide to using the NVIDIA Toolkit Framework for building GPU-accelerated ML proof-of-concepts and learning CUDA/NVIDIA capabilities.

## Purpose of This Framework

This framework is designed for:
- **Learning**: Understanding GPU acceleration benefits and CUDA capabilities
- **Proof-of-Concepts**: Rapid prototyping to validate GPU acceleration potential
- **Experimentation**: Testing different NVIDIA technologies (RAPIDS, TensorRT)
- **Benchmarking**: Measuring actual speedups for your specific use cases

This is **not** production-ready code. Generated toolkits require additional hardening, security review, error handling, and optimization before production deployment.

## Quick Reference

```bash
# 1. Configure use case
cp config/use-case-template.yaml config/my-use-case.yaml
# Edit config/my-use-case.yaml

# 2. Generate toolkit using prompts
# Work through prompts/planning/*.md
# Then prompts/implementation/*.md
# Finally prompts/validation/*.md

# 3. Test generated toolkit
cd <generated-toolkit>/
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/

# 4. Run benchmarks
python benchmarks/benchmark_data_pipeline.py --data data/sample/
```

## Detailed Workflow

### Phase 1: Use Case Configuration (5-10 minutes)

1. **Copy template**:
```bash
cd nvidia-toolkit-framework
cp config/use-case-template.yaml config/my-use-case.yaml
```

2. **Edit key sections**:

**Required fields**:
- `use_case.name` - Human-readable name
- `use_case.slug` - Python module name (snake_case)
- `use_case.domain` - Industry (MarTech/AdTech/FinTech/etc.)
- `model.type` - Algorithm (KNN/XGBoost/KMeans/etc.)
- `data.format` - Data format (parquet/csv/json)
- `data.features` - List of feature columns

**Optional but recommended**:
- `requirements.latency_requirement` - Performance targets
- `model.params` - Hyperparameters
- `preprocessing.steps` - Data cleaning steps
- `feature_engineering.custom` - Custom features

3. **Validate config** (optional):
```bash
python -c "import yaml; yaml.safe_load(open('config/my-use-case.yaml'))"
```

### Phase 2: Planning (20-30 minutes)

Work through planning prompts with your AI agent or manually:

#### 2.1 Use Case Discovery

**Prompt**: `prompts/planning/01-use-case-discovery.md`

**What it does**:
- Analyzes your requirements
- Recommends RAPIDS components (cuDF, cuML, cuGraph, DASK)
- Decides if TensorRT is needed
- Estimates GPU speedups
- Maps dependencies

**Outputs**:
- Technology stack decision
- Expected performance gains
- Dependency list

**Example session**:
```
You: Read my config at config/my-use-case.yaml and follow prompts/planning/01-use-case-discovery.md

Agent: I've analyzed your use case. Based on:
- 10M training samples
- <50ms latency requirement
- KNN model type

I recommend:
✓ cuDF for data processing (20x speedup expected)
✓ cuML KNN for model (30x training speedup)
✗ TensorRT not needed (KNN inference is fast)
✓ Single GPU sufficient (no DASK needed)

Expected end-to-end speedup: 15-20x
```

#### 2.2 Architecture Design

**Prompt**: `prompts/planning/02-architecture-design.md`

**What it does**:
- Designs module structure
- Creates configuration schema
- Plans data pipeline flow
- Defines performance targets

**Outputs**:
- Module breakdown (data/, models/, utils/, api/)
- Configuration files structure
- Benchmarking targets

### Phase 3: Implementation (1-2 hours)

#### 3.1 Project Scaffolding

**Prompt**: `prompts/implementation/scaffold-generator.md`

**What it does**:
- Generates complete directory structure
- Creates boilerplate files
- Sets up pyproject.toml with dependencies
- Initializes configuration files

**Outputs**:
```
my-use-case/
├── config/
│   └── default.yaml
├── src/my_use_case/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── api/
├── tests/
├── benchmarks/
├── scripts/
├── pyproject.toml
└── README.md
```

**Commands**:
```bash
# After scaffolding is generated
cd my-use-case/
uv venv
source .venv/bin/activate
uv pip install -e .
```

#### 3.2 Data Pipeline Implementation

**Prompt**: `prompts/implementation/data-pipeline.md`

**What it does**:
- Implements DataLoader (cuDF/pandas abstraction)
- Creates Preprocessor (GPU-accelerated cleaning)
- Builds FeatureEngineer (custom features)
- Orchestrates pipeline

**Key files created**:
- `src/<slug>/data/loader.py`
- `src/<slug>/data/preprocessor.py`
- `src/<slug>/data/feature_engineering.py`

**Test**:
```bash
# Test data pipeline
python -c "
from my_use_case.data import DataPipeline
pipeline = DataPipeline(config={}, device='cpu')
df = pipeline.run('data/sample/train.parquet')
print(f'Loaded {len(df)} rows')
"
```

#### 3.3 Model Implementation

**Prompt**: `prompts/implementation/model-implementation.md`

**What it does**:
- Implements model class with GPU/CPU support
- Creates Trainer for training orchestration
- Builds Predictor for inference
- Adds model save/load functionality

**Key files created**:
- `src/<slug>/models/base.py`
- `src/<slug>/models/<model_name>.py`
- `src/<slug>/models/trainer.py`
- `src/<slug>/models/predictor.py`
- `scripts/train.py`
- `scripts/serve.py`

**Test training**:
```bash
# CPU mode (local dev)
python scripts/train.py \
  --config config/default.yaml \
  --data data/sample/train.parquet \
  --device cpu

# GPU mode (on GPU machine)
python scripts/train.py \
  --config config/default.yaml \
  --data data/train.parquet \
  --device gpu
```

### Phase 4: Validation (30-60 minutes)

#### 4.1 Benchmarking

**Prompt**: `prompts/validation/benchmark-plan.md`

**What it does**:
- Creates data pipeline benchmarks
- Implements training benchmarks
- Builds inference latency tests
- Measures GPU vs CPU performance

**Key files created**:
- `benchmarks/benchmark_data_pipeline.py`
- `benchmarks/benchmark_training.py`
- `benchmarks/benchmark_inference.py`

**Run benchmarks**:
```bash
# Data pipeline
python benchmarks/benchmark_data_pipeline.py \
  --data data/sample/train.parquet \
  --sizes 10000 100000 1000000

# Training
python benchmarks/benchmark_training.py \
  --data data/sample/train.parquet

# Inference
python benchmarks/benchmark_inference.py \
  --model models/checkpoints/best_model.pkl \
  --data data/sample/test.parquet \
  --batch-sizes 1 8 32 128
```

**Expected output**:
```
================================================================================
BENCHMARK RESULTS
================================================================================
Device     Operation            Data Size    Time (s)     Throughput
================================================================================
gpu        load_parquet         1,000,000    0.1234      8,106,952
cpu        load_parquet         1,000,000    2.4567      407,036
================================================================================
Speedup: 19.9x
```

#### 4.2 Documentation

Update generated docs:
- `docs/BENCHMARKS.md` - Add actual benchmark results
- `README.md` - Update with specific metrics
- `docs/USAGE.md` - Add use-case-specific examples

### Phase 5: Deployment (30 minutes - 2 hours)

#### Option A: Local Docker

```bash
# Build GPU image
docker build -f docker/Dockerfile.gpu -t my-use-case:gpu .

# Run
docker run --gpus all -p 8000:8000 my-use-case:gpu
```

#### Option B: NVIDIA Virtual Workstation (Recommended for Learning)

NVIDIA Virtual Workstation provides pre-configured CUDA environment ideal for POCs:

```bash
# On NVIDIA Virtual Workstation (CUDA pre-installed)
git clone <your-repo>
cd my-use-case

# Setup environment
uv venv && source .venv/bin/activate
uv pip install -e .

# Verify GPU
nvidia-smi

# Install RAPIDS (CUDA drivers already configured)
uv pip install cudf-cu12 cuml-cu12

# Run training
python scripts/train.py --data data/train.parquet --device gpu

# Start API server for testing
python scripts/serve.py --host 0.0.0.0 --port 8000
```

**Tips for Virtual Workstation:**
- CUDA toolkit and drivers pre-installed - skip manual setup
- Use GPU monitoring: `watch -n 1 nvidia-smi`
- Create snapshots after environment setup for quick restarts
- Remember to shut down when not in use to save costs

#### Option C: GCP GPU Instance (For Cost-Conscious POCs)

```bash
# On GCP N1 + T4 instance
# First, install CUDA drivers (one-time setup)
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# Or use RAPIDS Docker to skip setup
docker run --gpus all -v $(pwd):/workspace -it \
  nvcr.io/nvidia/rapidsai/base:24.10-cuda12.0-py3.11

# Inside container
cd /workspace/my-use-case
uv pip install -e .
python scripts/train.py --data data/train.parquet --device gpu
```

#### Option D: Kubernetes (Advanced POC Deployment)

Note: Kubernetes deployment is typically not needed for POC/learning. Consider this only for multi-user POC demonstrations.

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-use-case-poc
spec:
  replicas: 1  # Start with 1 for POC
  template:
    spec:
      containers:
      - name: inference
        image: my-use-case:gpu
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Common Workflows

### Workflow 1: Recommendation System

```bash
# 1. Configure
cp config/examples/realtime-recommendation.yaml config/my-config.yaml

# 2. Generate (using Claude Code)
claude code
> Read config/my-config.yaml and generate toolkit using prompts

# 3. Test locally (CPU)
cd realtime-recommendation/
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/train.py --data data/sample/train.parquet --device cpu

# 4. Deploy to GPU
# Copy to GCP VM
scp -r realtime-recommendation/ gcp-vm:~/

# SSH to VM
ssh gcp-vm
cd realtime-recommendation
uv pip install -e . cudf-cu12 cuml-cu12
python scripts/train.py --data data/train.parquet --device gpu
python scripts/serve.py
```

### Workflow 2: Customer Segmentation

```bash
# 1. Configure
cat > config/segmentation.yaml <<EOF
use_case:
  name: "Customer Segmentation"
  slug: "customer_segmentation"
  domain: "MarTech"

model:
  type: "KMeans"
  task: "clustering"
  params:
    n_clusters: 5

technology_stack:
  model_framework: "cuML"
EOF

# 2. Generate using prompts
# ... follow prompt workflow

# 3. Run clustering
python scripts/train.py --config config/segmentation.yaml --data data/customers.parquet

# 4. Get cluster assignments
python -c "
from customer_segmentation.models import Predictor
predictor = Predictor.load('models/checkpoints/best_model.pkl')
clusters = predictor.predict(new_customers)
print(clusters)
"
```

### Workflow 3: Churn Prediction

```bash
# 1. Configure
cat > config/churn.yaml <<EOF
use_case:
  name: "Churn Prediction"
  slug: "churn_prediction"

model:
  type: "XGBoost"
  task: "classification"
  params:
    max_depth: 6
    learning_rate: 0.1

technology_stack:
  model_framework: "XGBoost"
EOF

# 2. Generate toolkit
# ... follow prompts

# 3. Train with GPU
python scripts/train.py --data data/customers.parquet --device gpu

# 4. API inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "tenure_months": 24,
    "monthly_spend": 89.99
  }'
```

## Testing Strategy

### Unit Tests

```bash
# Run all tests
pytest tests/

# GPU tests only (requires GPU)
pytest tests/ -m gpu

# CPU tests only
pytest tests/ -m "not gpu"

# Specific module
pytest tests/test_data/

# With coverage
pytest tests/ --cov=src/my_use_case --cov-report=html
```

### Integration Tests

```bash
# End-to-end pipeline test
pytest tests/test_integration/test_pipeline.py

# API test
pytest tests/test_integration/test_api.py
```

### Benchmark Tests

```bash
# Quick benchmark (small data)
python benchmarks/benchmark_data_pipeline.py --data data/sample/ --sizes 1000

# Full benchmark (requires GPU + large dataset)
python benchmarks/benchmark_training.py --data data/train.parquet
```

## Performance Optimization

### Data Pipeline

**Issue**: Slow data loading

**Solutions**:
1. Use Parquet instead of CSV (5-10x faster)
2. Enable GPU memory pool:
   ```python
   import rmm
   rmm.reinitialize(pool_allocator=True)
   ```
3. Use column pruning:
   ```python
   df = cudf.read_parquet("data.parquet", columns=["col1", "col2"])
   ```

### Training

**Issue**: GPU memory errors

**Solutions**:
1. Reduce batch size in config
2. Lower GPU memory fraction:
   ```yaml
   gpu:
     memory_fraction: 0.6
   ```
3. Use gradient checkpointing (PyTorch)

### Inference

**Issue**: High latency

**Solutions**:
1. Increase batch size:
   ```yaml
   inference:
     batch_size: 256  # From 32
   ```
2. Enable TensorRT (for neural networks):
   ```yaml
   technology_stack:
     tensorrt:
       enabled: true
       precision: "fp16"
   ```
3. Use model quantization

## Troubleshooting

### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'cudf'
# Solution:
uv pip install cudf-cu12 cuml-cu12

# Error: CUDA driver version insufficient
# Solution: Update CUDA drivers or use CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### GPU Memory Issues

```python
# Error: cudaErrorMemoryAllocation out of memory
# Solution 1: Clear GPU memory
import cudf
import cuml
cudf.core._compat.HAVE_CUPY = False  # Disable CuPy caching

# Solution 2: Reduce memory usage
config["gpu"]["memory_fraction"] = 0.5
```

### Performance Issues

```bash
# Slow GPU performance
# Check GPU utilization
nvidia-smi -l 1

# If GPU util < 50%, possible bottlenecks:
# 1. Data loading from disk (use faster storage)
# 2. CPU preprocessing (move to GPU)
# 3. Small batch size (increase)
```

## Best Practices

### Development

✅ **DO:**
- Develop on CPU, deploy on GPU
- Use small sample datasets locally
- Test GPU/CPU parity: `assert gpu_result.equals(cpu_result)`
- Version control config files

❌ **DON'T:**
- Commit large data files
- Hard-code paths
- Skip CPU fallback testing
- Forget to benchmark

### Production

✅ **DO:**
- Use Docker containers
- Monitor GPU utilization
- Implement health checks
- Log performance metrics
- Set memory limits

❌ **DON'T:**
- Deploy without benchmarking
- Skip error handling
- Ignore latency spikes
- Over-allocate GPU memory

## Publishing Your Toolkit

### As Standalone Repository

```bash
cd my-generated-toolkit/

# Initialize git
git init
git add .
git commit -m "Initial commit: GPU-accelerated <use case>"

# Create repo on GitHub
gh repo create my-org/gpu-toolkit --public --source=. --remote=origin

# Push
git push -u origin main

# Add topics
gh repo edit --add-topic nvidia,rapids,gpu,machine-learning
```

### Documentation Checklist

- [ ] README with quick start
- [ ] Benchmark results in docs/BENCHMARKS.md
- [ ] Usage examples
- [ ] API documentation
- [ ] Deployment guide
- [ ] Performance tuning tips

### Release Checklist

- [ ] All tests passing
- [ ] Benchmarks documented
- [ ] Dependencies pinned in pyproject.toml
- [ ] Docker images built and tagged
- [ ] CI/CD pipeline configured
- [ ] License file included

## Additional Resources

### RAPIDS Resources
- [RAPIDS Docs](https://docs.rapids.ai/)
- [cuDF API](https://docs.rapids.ai/api/cudf/stable/)
- [cuML API](https://docs.rapids.ai/api/cuml/stable/)
- [RAPIDS Examples](https://github.com/rapidsai/notebooks)

### TensorRT Resources
- [TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TensorRT Samples](https://github.com/NVIDIA/TensorRT)

### Framework Resources
- [Framework README](README.md)
- [Quick Start](docs/QUICKSTART.md)
- [Prompt Guide](docs/PROMPT_GUIDE.md)
- [Examples](config/examples/)

---

Questions? Open a GitHub issue or discussion!
