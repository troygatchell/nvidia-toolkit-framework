# NVIDIA Toolkit Framework

A **reusable, prompt-driven framework** for building GPU-accelerated ML proof-of-concepts using NVIDIA RAPIDS and TensorRT.

## Overview

This framework provides a systematic approach to creating domain-specific ML proof-of-concepts optimized for GPU acceleration. Designed for **learning and experimentation**, it helps you rapidly prototype and understand CUDA and NVIDIA capabilities for specific use cases in MarTech, AdTech, FinTech, and beyond.

**Key Features:**
- **Learning-focused**: Build proof-of-concepts to understand GPU acceleration benefits
- **Prompt-driven architecture**: Configure entire projects through structured prompts
- **GPU acceleration**: Built-in RAPIDS cuDF/cuML and TensorRT support for hands-on learning
- **Reusable patterns**: Standardized structure for data pipelines, models, and benchmarks
- **CPU fallback**: Graceful degradation for development without GPU
- **Ready for experimentation**: Includes API endpoints, benchmarking, and deployment configs

**What This Framework Is:**
- A learning tool for understanding NVIDIA GPU acceleration
- A rapid prototyping platform for proof-of-concepts
- A starter kit for exploring CUDA capabilities
- A benchmark tool to measure GPU vs CPU performance gains

**What This Framework Is Not:**
- Production-ready code (requires additional hardening and optimization)
- A replacement for comprehensive ML frameworks
- Suitable for critical production workloads without extensive testing

## Repository Structure

```
nvidia-toolkit-framework/
├── prompts/                    # Core prompt templates
│   ├── planning/
│   │   ├── 01-use-case-discovery.md      # Technology stack selection
│   │   └── 02-architecture-design.md     # System architecture design
│   ├── implementation/
│   │   ├── scaffold-generator.md         # Project structure generation
│   │   ├── data-pipeline.md              # Data processing implementation
│   │   └── model-implementation.md       # Model training/inference
│   └── validation/
│       └── benchmark-plan.md             # Performance benchmarking
├── config/
│   └── use-case-template.yaml            # Configuration template
├── docs/
│   ├── QUICKSTART.md                     # Getting started guide
│   ├── PROMPT_GUIDE.md                   # How to customize prompts
│   └── EXAMPLES.md                       # Example use cases
├── starter_kits/                         # Generated toolkit examples
│   └── .template/                        # Template structure
└── pyproject.toml
```

## Quick Start

### 1. Clone the Framework

```bash
git clone https://github.com/your-org/nvidia-toolkit-framework.git
cd nvidia-toolkit-framework
```

### 2. Customize Your Use Case

Copy and edit the configuration template:

```bash
cp config/use-case-template.yaml config/my-use-case.yaml
```

Edit `config/my-use-case.yaml` with your:
- Use case name and domain
- Data sources and formats
- Model type and hyperparameters
- Performance requirements
- GPU configuration

### 3. Follow the Prompt Workflow

Work through the prompt files in order, using them as guides with your AI agent:

1. **Planning Phase**
   - `prompts/planning/01-use-case-discovery.md` - Analyze requirements and select tech stack
   - `prompts/planning/02-architecture-design.md` - Design system architecture

2. **Implementation Phase**
   - `prompts/implementation/scaffold-generator.md` - Generate project structure
   - `prompts/implementation/data-pipeline.md` - Implement data processing
   - `prompts/implementation/model-implementation.md` - Build model training/inference

3. **Validation Phase**
   - `prompts/validation/benchmark-plan.md` - Benchmark GPU vs CPU performance

### 4. Generate Your Toolkit

The prompts will guide you (or your AI agent) to create a complete toolkit with:

```
your-use-case/
├── config/                    # Configuration files
├── src/your_use_case/        # Source code
│   ├── data/                 # Data pipeline
│   ├── models/               # Model implementations
│   ├── utils/                # GPU utilities, config management
│   └── api/                  # API endpoints
├── tests/                    # Unit and integration tests
├── benchmarks/               # Performance benchmarking
├── scripts/                  # Training and serving scripts
└── docs/                     # Documentation
```

## How It Works

### Prompt-Driven Development

Each prompt file contains:

1. **Input Variables**: Placeholders like `{{USE_CASE_NAME}}`, `{{MODEL_TYPE}}`
2. **Decision Framework**: Structured questions to guide technology choices
3. **Code Templates**: Boilerplate with customization points
4. **Checklists**: Ensure all components are implemented

### Example: Creating a Recommendation System

1. **Configure** (`config/recommendation.yaml`):
```yaml
use_case:
  name: "Real-time Recommendation System"
  slug: "realtime_recommendation"
  domain: "MarTech"

model:
  type: "KNN"
  framework: "cuML"

requirements:
  latency_requirement:
    target_p99_ms: 50
```

2. **Follow Prompts**: Each prompt file guides implementation with your config values

3. **Result**: Complete GPU-accelerated recommendation system with:
   - cuDF data pipeline (10-100x faster than pandas)
   - cuML KNN model training
   - FastAPI inference endpoint
   - CPU fallback for local development
   - Comprehensive benchmarks

## Use Case Examples

The framework supports multiple domains:

### MarTech / AdTech
- Real-time recommendation systems
- Customer segmentation at scale
- Churn prediction
- Attribution modeling
- Lookalike audience generation
- RTB bid optimization

### FinTech
- Fraud detection
- Risk scoring
- Portfolio optimization
- Time series forecasting

### Other Domains
- Predictive maintenance (Manufacturing)
- Patient risk stratification (Healthcare)
- Demand forecasting (Retail)

## Technology Stack

### RAPIDS (Data Processing & Traditional ML)
- **cuDF**: GPU DataFrames (10-100x faster than pandas)
- **cuML**: GPU ML algorithms (scikit-learn compatible)
- **cuGraph**: Graph analytics for recommendation graphs
- **DASK-cuDF**: Multi-GPU and distributed computing

### TensorRT (Deep Learning Inference)
- Model optimization and quantization (FP16, INT8)
- Dynamic batching
- Multi-instance GPU (MIG) support

### Supporting Tools
- **UV**: Fast Python package manager
- **FastAPI**: Modern API framework
- **Pydantic**: Configuration validation
- **Pytest**: Testing framework

## Performance Gains

Expected GPU speedups (vs CPU baseline):

| Operation | Typical Speedup |
|-----------|----------------|
| Data loading (Parquet) | 5-20x |
| Data preprocessing | 10-50x |
| Model training (cuML) | 10-50x |
| Model training (XGBoost) | 5-15x |
| Inference (batch) | 5-20x |

## Requirements

### Supported Platforms
- **macOS**: For local CPU development and testing
- **Ubuntu 20.04/22.04**: For GPU virtual workstations and production POCs
- **Windows**: Not supported

### Development (CPU mode)
- Python 3.10+
- UV package manager
- 8GB+ RAM
- macOS or Ubuntu

### Production (GPU mode)
- Python 3.10+
- Ubuntu 20.04/22.04
- NVIDIA GPU with CUDA 11.8+ or 12.0+
- RAPIDS 24.10+
- 16GB+ GPU memory (recommended)

### Cloud Options

#### NVIDIA Virtual Workstation (Recommended for Learning)
NVIDIA Virtual Workstation provides a managed environment ideal for learning and POC development:

**Pros:**
- Pre-configured CUDA toolkit and drivers
- NVIDIA-optimized drivers and software stack
- Simplified setup for RAPIDS and TensorRT
- Good for individual learning and experimentation
- Professional visualization capabilities (if needed)
- Consistent environment across team members

**Cons:**
- Higher cost compared to basic GPU instances
- May include features not needed for ML workloads
- Limited to specific cloud providers

**Best for:** Learning CUDA, prototyping, individual POCs, educational purposes

#### GCP GPU Instances
Standard compute instances with attached GPUs:

**Pros:**
- Cost-effective for burst workloads
- Flexible instance sizing
- Wide GPU selection (T4, V100, A100, L4)
- Pay-per-use pricing
- Easy to scale up/down
- Integration with GCP services (BigQuery, Cloud Storage)

**Cons:**
- Requires manual CUDA setup and configuration
- Driver compatibility management needed
- More setup overhead for beginners

**Best for:** Production POCs, cost-sensitive projects, integration with GCP ecosystem

**Recommended GCP Instance Types:**
- **Learning/Small POCs**: `n1-standard-4` with 1x T4 GPU (16GB)
- **Medium POCs**: `n1-standard-8` with 1x V100 GPU (16GB or 32GB)
- **Large POCs**: `a2-highgpu-1g` with 1x A100 GPU (40GB or 80GB)

#### Other Cloud Options
- **AWS**: EC2 G4/G5/P3/P4 instances with NVIDIA GPUs
- **Azure**: NC/ND series VMs with NVIDIA GPUs
- **Paperspace**: Gradient platform for ML experimentation

## Publishing Your Toolkit

Once generated, publish your use-case-specific toolkit as a standalone repository:

```bash
# Your generated toolkit is standalone
cd realtime-recommendation/
git init
git add .
git commit -m "Initial commit: GPU-accelerated recommendation system"
git remote add origin https://github.com/your-org/gpu-recommendation-system.git
git push -u origin main
```

## Framework vs Use-Case Repositories

- **This framework repository**: Reusable prompts, templates, and configuration
- **Generated use-case repositories**: Complete, standalone implementations

Users can:
1. Star/fork this framework for new projects
2. Use generated repositories directly for specific use cases

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get started in 10 minutes
- [Prompt Customization Guide](docs/PROMPT_GUIDE.md) - How to adapt prompts
- [Use Case Examples](docs/EXAMPLES.md) - Detailed example workflows
- [Architecture Guide](docs/ARCHITECTURE.md) - Framework design principles

## Contributing

Contributions welcome! Areas for improvement:
- Additional prompt templates for more use cases
- Enhanced code templates
- New RAPIDS/TensorRT patterns
- Documentation improvements

## Virtual Workstation Reviews

Community experiences with different GPU environments for POC development:

### NVIDIA Virtual Workstation
**Rating:** ⭐⭐⭐⭐☆ (4/5 for learning & POCs)

**User Experience:**
- **Setup Time**: Minimal (< 30 minutes) - CUDA and drivers pre-configured
- **Learning Curve**: Low - Great for CUDA beginners
- **Stability**: Excellent - Consistent environment across sessions
- **Cost**: Higher ($2-5/hour depending on GPU)
- **Best Use Cases**: Learning RAPIDS, initial POC development, team training

**Tips:**
- Start with smaller GPU types (T4) for learning
- Use snapshots to save configured environments
- Monitor GPU utilization to avoid paying for idle time

### GCP N1 + T4 GPU
**Rating:** ⭐⭐⭐⭐☆ (4/5 for cost-conscious POCs)

**User Experience:**
- **Setup Time**: 1-2 hours (manual CUDA/driver installation)
- **Learning Curve**: Medium - Requires Linux and CUDA knowledge
- **Stability**: Good - Occasional driver update needs
- **Cost**: Lower ($0.35-0.95/hour for T4)
- **Best Use Cases**: Production POCs, cost-sensitive experiments, batch processing

**Tips:**
- Use RAPIDS Docker containers to skip manual setup
- Leverage preemptible instances for even lower costs (non-critical workloads)
- Set up startup scripts for automated environment configuration

### GCP A2 + A100 GPU
**Rating:** ⭐⭐⭐⭐⭐ (5/5 for performance)

**User Experience:**
- **Setup Time**: 1-2 hours (similar to T4)
- **Learning Curve**: Medium
- **Stability**: Excellent
- **Cost**: High ($2-4/hour for single A100)
- **Best Use Cases**: Large-scale POCs, multi-GPU experiments, performance benchmarking

**Tips:**
- Use for final performance validation only (expensive for learning)
- Consider Multi-Instance GPU (MIG) to split A100 into smaller instances
- Great for demonstrating maximum GPU acceleration potential

### Paperspace Gradient
**Rating:** ⭐⭐⭐☆☆ (3/5 for POCs)

**User Experience:**
- **Setup Time**: Minimal (< 15 minutes)
- **Learning Curve**: Very Low - Jupyter-focused interface
- **Stability**: Good
- **Cost**: Moderate ($0.51-2.30/hour)
- **Best Use Cases**: Notebook-based learning, quick experiments, sharing POCs

**Tips:**
- Excellent for Jupyter notebook-based POCs
- Pre-configured ML environments available
- Not ideal for production-style deployments

**Community Recommendation:** Start with NVIDIA Virtual Workstation for initial learning, then migrate to GCP instances once comfortable with CUDA/RAPIDS setup.

## License

MIT License

## Resources

- [NVIDIA RAPIDS](https://rapids.ai/) - GPU-accelerated data science
- [TensorRT](https://developer.nvidia.com/tensorrt) - High-performance deep learning inference
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [Example Notebooks](https://github.com/rapidsai/notebooks)
- [Docker Images](https://hub.docker.com/r/rapidsai/base)
- [Docker Notebook](rapidsai/notebooks)
## Support

- GitHub Issues: For bugs and feature requests
- Discussions: For questions and community support


---

Built with NVIDIA RAPIDS and TensorRT
