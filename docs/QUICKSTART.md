# Quick Start Guide

Get started with the NVIDIA Toolkit Framework in 10 minutes to build your first GPU-accelerated proof-of-concept.

## What You'll Learn

This guide will help you:
- Build a GPU-accelerated ML proof-of-concept
- Understand CUDA and NVIDIA RAPIDS capabilities
- Benchmark GPU vs CPU performance for your use case
- Experiment with different NVIDIA technologies

**Important:** This framework generates POC code for learning and experimentation. Additional work is needed for production deployment.

## Prerequisites

- Python 3.10 or later
- UV package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Claude Code or similar AI agent (optional but recommended)
- NVIDIA GPU with CUDA 12.0+ (optional for development - CPU mode available)

## Choosing Your GPU Environment

### For Learning & Initial POCs (Recommended)
**NVIDIA Virtual Workstation** - Pre-configured CUDA environment, minimal setup
- Pros: Ready to use, great for beginners, consistent environment
- Cons: Higher cost ($2-5/hour)
- Best for: Learning, team training, quick POCs

### For Cost-Conscious POCs
**GCP N1 + T4 GPU** - Manual setup required, lower cost
- Pros: Lower cost ($0.35-0.95/hour), flexible
- Cons: Requires CUDA setup, more configuration, GPU availability limitations
- Best for: Longer-running POCs, batch experiments

### For Development/Testing
**Local CPU Mode** - No GPU required
- Pros: Free, good for code development
- Cons: No GPU acceleration (can't validate performance)
- Best for: Initial development, testing logic

## Step-by-Step Guide

### 1. Set Up the Framework

**macOS (Local Development):**
```bash
# Clone the framework repository
git clone https://github.com/your-org/nvidia-toolkit-framework.git
cd nvidia-toolkit-framework

# Install framework dependencies (optional, for template generation)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Ubuntu (Virtual Workstation/GCP):**
```bash
# Clone the framework repository
git clone https://github.com/your-org/nvidia-toolkit-framework.git
cd nvidia-toolkit-framework

# Install framework dependencies (optional, for template generation)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Note:** This framework is designed for macOS (local CPU development) and Ubuntu (GPU virtual workstations). Windows is not supported.

### 2. Create Your Use Case Configuration

Copy the template and customize for your use case:

```bash
cp config/use-case-template.yaml config/my-recommendation-system.yaml
```

Edit `config/my-recommendation-system.yaml`:

```yaml
use_case:
  name: "Product Recommendation System"
  slug: "product_recommendations"
  domain: "MarTech"
  description: "GPU-accelerated product recommendations for e-commerce"

requirements:
  primary_goal: "Generate top-10 recommendations in <50ms"
  data_scale:
    training_samples: 5_000_000
    inference_throughput: "5K requests/sec"
  latency_requirement:
    target_p99_ms: 50

technology_stack:
  rapids:
    cudf: true
    cuml: true
  model_framework: "cuML"
  model_type: "KNN"

data:
  format: "parquet"
  features:
    - "user_id"
    - "product_id"
    - "rating"
    - "timestamp"
  target_variable: "rating"

model:
  type: "KNN"
  params:
    n_neighbors: 10
    metric: "euclidean"
```

### 3. Follow the Prompt Workflow

#### Using with Claude Code or Github Copilot

1. Start Claude Code in your working directory:
```bash
cd /path/to/your/projects
claude code
```

2. Tell Claude:
```
I want to build a GPU-accelerated ML toolkit using the NVIDIA Toolkit Framework.
My use case config is at: nvidia-toolkit-framework/config/my-recommendation-system.yaml

Please follow the prompts in order:
1. First: nvidia-toolkit-framework/prompts/planning/01-use-case-discovery.md
2. Then continue through all prompts to generate the complete toolkit
```

3. Claude will:
   - Read your config
   - Work through each prompt file
   - Generate the complete project structure
   - Implement all components
   - Create benchmarking code

#### Manual Approach (Without AI Agent)

Work through each prompt file manually, using them as implementation checklists:

```bash
# 1. Planning Phase
cat prompts/planning/01-use-case-discovery.md
# Read and make decisions about technology stack

cat prompts/planning/02-architecture-design.md
# Design your system architecture

# 2. Implementation Phase
cat prompts/implementation/scaffold-generator.md
# Generate project structure manually

cat prompts/implementation/data-pipeline.md
# Implement data loading and preprocessing

cat prompts/implementation/model-implementation.md
# Implement model training and inference

# 3. Validation Phase
cat prompts/validation/benchmark-plan.md
# Create benchmarking scripts
```

### 4. Project Generation

After following the prompts, you'll have a new project:

```
product-recommendations/
├── config/
│   └── default.yaml
├── src/
│   └── product_recommendations/
│       ├── data/
│       ├── models/
│       ├── utils/
│       └── api/
├── tests/
├── benchmarks/
├── scripts/
│   ├── train.py
│   └── serve.py
├── pyproject.toml
└── README.md
```

### 5. Install and Test

```bash
cd product-recommendations

# Create environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run on CPU (for local development)
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
python scripts/train.py --data data/sample/train.parquet --device cpu

# Run tests
pytest tests/

# Run benchmarks
python benchmarks/benchmark_data_pipeline.py --data data/sample/train.parquet
```

### 6. Deploy to GPU Environment

#### Option A: NVIDIA Virtual Workstation (Recommended for Learning)

```bash
# CUDA and drivers are pre-installed!
# Just verify GPU is available
nvidia-smi

# Setup your POC
git clone <your-generated-repo>
cd product-recommendations
uv venv && source .venv/bin/activate
uv pip install -e .

# Install RAPIDS (CUDA compatibility already handled)
uv pip install cudf-cu12 cuml-cu12

# Train on GPU and see the speedup!
python scripts/train.py --data data/train.parquet --device gpu

# Monitor GPU utilization in another terminal
watch -n 1 nvidia-smi

# Start API server for testing
python scripts/serve.py
```

**Learning Tips:**
- Compare execution time between `--device cpu` and `--device gpu`
- Watch `nvidia-smi` to see GPU memory usage and utilization
- Run benchmarks to measure actual speedup: `python benchmarks/benchmark_training.py`
- Create a snapshot after setup for quick restarts

#### Option B: GCP N1 + T4 GPU (Cost-Conscious POCs)

```bash
# On GCP GPU instance
# First-time setup: Install CUDA drivers
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# OR use RAPIDS Docker to skip manual setup
docker run --gpus all --rm -it \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/rapidsai/base:24.10-cuda12.0-py3.11

# Inside container or on host with CUDA installed
git clone <your-generated-repo>
cd product-recommendations
uv venv && source .venv/bin/activate
uv pip install -e . cudf-cu12 cuml-cu12

python scripts/train.py --data data/train.parquet --device gpu
```

**Cost-Saving Tips:**
- Use preemptible instances for batch experiments (70% cheaper)
- Stop instance when not in use
- Start with smaller datasets for initial testing

#### Option C: Local GPU (If Available)

```bash
# Ensure CUDA is available
nvidia-smi

# Install RAPIDS matching your CUDA version
uv pip install cudf-cu12 cuml-cu12  # For CUDA 12.x

# Train on GPU
python scripts/train.py --data data/train.parquet --device gpu
```

## Example Use Cases

### Real-time Recommendation System
```yaml
model_type: "KNN"
framework: "cuML"
latency: <50ms
```

### Customer Segmentation
```yaml
model_type: "KMeans"
framework: "cuML"
task: "clustering"
```

### Churn Prediction
```yaml
model_type: "XGBoost"
framework: "XGBoost GPU"
task: "classification"
```

### Time Series Forecasting
```yaml
model_type: "LSTM"
framework: "PyTorch"
tensorrt: true
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Configure Use Case (config/my-use-case.yaml)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Planning Phase                                           │
│    • 01-use-case-discovery.md (Tech stack selection)       │
│    • 02-architecture-design.md (System design)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Implementation Phase                                     │
│    • scaffold-generator.md (Project structure)             │
│    • data-pipeline.md (Data processing)                    │
│    • model-implementation.md (Model code)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Validation Phase                                         │
│    • benchmark-plan.md (Performance testing)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Complete GPU-Accelerated Toolkit                        │
│    • Production-ready code                                 │
│    • API endpoints                                         │
│    • Benchmarks showing 10-50x speedup                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Variable Substitution

Prompt files use `{{VARIABLE}}` syntax. When following prompts:

1. **Replace variables** with values from your config
2. Example:
   - Prompt: `{{USE_CASE_NAME}}`
   - Your config: `name: "Product Recommendation System"`
   - Result: Generate code for "Product Recommendation System"

### GPU/CPU Abstraction

All generated code supports both modes:

```python
# Automatic GPU/CPU selection
from product_recommendations.utils import get_device
device = get_device(preferred="gpu", fallback=True)

# Code works on both
from product_recommendations.data import DataLoader
loader = DataLoader(device=device)  # Uses cuDF on GPU, pandas on CPU
```

### Module Structure

Generated projects follow this pattern:

```
src/<use_case_slug>/
├── data/           # Data pipeline (cuDF-based)
├── models/         # Model training/inference (cuML/XGBoost)
├── utils/          # GPU utilities, config management
└── api/            # FastAPI endpoints
```

## Troubleshooting

### CUDA Not Found

If GPU not available:
- Framework automatically falls back to CPU
- Set `gpu.fallback_to_cpu: true` in config
- Use CPU mode for local development

### Import Errors

```bash
# Ensure RAPIDS is installed with correct CUDA version
uv pip install cudf-cu12 cuml-cu12  # For CUDA 12.x
uv pip install cudf-cu11 cuml-cu11  # For CUDA 11.x
```

### Memory Issues

Adjust GPU memory limit in config:

```yaml
gpu:
  memory_fraction: 0.6  # Use 60% of GPU memory (default 80%)
```

## Next Steps

1. **Explore Examples**: See [EXAMPLES.md](EXAMPLES.md) for detailed use case walkthroughs
2. **Customize Prompts**: Learn to adapt prompts in [PROMPT_GUIDE.md](PROMPT_GUIDE.md)
3. **Deploy to Production**: Set up on GCP/AWS with GPU instances
4. **Benchmark**: Run performance tests to verify speedups

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community support
- **RAPIDS Slack**: RAPIDS-specific questions

## Quick Tips

✅ **DO:**
- Start with CPU mode for local development
- Use the configuration template
- Follow prompts in order
- Test on small datasets first
- Run benchmarks to verify GPU speedups

❌ **DON'T:**
- Skip planning prompts
- Manually edit generated code extensively (update config instead)
- Deploy to GPU without testing CPU fallback
- Forget to benchmark CPU vs GPU performance

---

Ready to build your first GPU-accelerated toolkit? Start with [configuring your use case](../config/use-case-template.yaml)!
