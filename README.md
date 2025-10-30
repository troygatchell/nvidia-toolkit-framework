# NVIDIA Toolkit Framework

A **reusable, prompt-driven framework** for building GPU-accelerated ML toolkits using NVIDIA RAPIDS and TensorRT.

## Overview

This framework provides a systematic approach to creating domain-specific ML toolkits optimized for GPU acceleration. By customizing prompt files and configuration, you can generate complete, production-ready implementations for various use cases in MarTech, AdTech, FinTech, and beyond.

**Key Features:**
- **Prompt-driven architecture**: Configure entire projects through structured prompts
- **GPU acceleration**: Built-in RAPIDS cuDF/cuML and TensorRT support
- **Reusable patterns**: Standardized structure for data pipelines, models, and benchmarks
- **CPU fallback**: Graceful degradation for development without GPU
- **Production-ready**: Includes API endpoints, benchmarking, and deployment configs

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

### Development (CPU mode)
- Python 3.10+
- UV package manager
- 8GB+ RAM

### Production (GPU mode)
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ or 12.0+
- RAPIDS 24.10+
- 16GB+ GPU memory (recommended)

### Cloud Options
- GCP: N1 instances with T4, V100, or A100 GPUs
- AWS: EC2 G4/G5/P3/P4 instances
- Azure: NC/ND series VMs

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

## License

Apache 2.0

## Resources

- [NVIDIA RAPIDS](https://rapids.ai/) - GPU-accelerated data science
- [TensorRT](https://developer.nvidia.com/tensorrt) - High-performance deep learning inference
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [Example Notebooks](https://github.com/rapidsai/notebooks)

## Support

- GitHub Issues: For bugs and feature requests
- Discussions: For questions and community support
- RAPIDS Slack: For RAPIDS-specific questions

---

Built with NVIDIA RAPIDS and TensorRT
