# Use Case Discovery Prompt

## Objective
Analyze the target use case and determine optimal GPU acceleration strategy using NVIDIA RAPIDS and/or TensorRT for **proof-of-concept development and learning**.

## Purpose
This prompt guides you through building a POC to:
- **Learn** about GPU acceleration benefits for your specific use case
- **Experiment** with NVIDIA RAPIDS and TensorRT technologies
- **Validate** potential performance gains before production investment
- **Benchmark** GPU vs CPU performance with real data

**Note:** This will generate POC code for learning and experimentation, not production-ready implementations.

## Input Variables
Replace these variables when customizing for your use case:

- `{{USE_CASE_NAME}}`: Name of the use case (e.g., "Real-time Recommendation System")
- `{{DOMAIN}}`: Industry domain (e.g., "MarTech", "AdTech", "FinTech", "Healthcare")
- `{{PRIMARY_GOAL}}`: Main objective (e.g., "Generate personalized product recommendations in <50ms")
- `{{DATA_SCALE}}`: Expected data volume (e.g., "10M users, 100K products, 1B interactions")
- `{{LATENCY_REQUIREMENT}}`: Performance target (e.g., "p99 <100ms", "batch processing 1M records/sec")
- `{{MODEL_TYPES}}`: ML models needed (e.g., "collaborative filtering, matrix factorization, deep learning")

## Analysis Framework

### 1. Data Characteristics Assessment
Analyze the following aspects of `{{USE_CASE_NAME}}`:

**Data Volume**
- Training data size: `{{DATA_SCALE}}`
- Inference throughput requirements: `{{LATENCY_REQUIREMENT}}`
- Real-time vs batch processing needs

**Data Operations**
- Primary operations: filtering, aggregations, joins, transformations
- Data formats: Parquet, CSV, JSON, streaming
- Feature engineering complexity

**Output Requirements**
- Expected output: predictions, rankings, clusters, scores
- Batch size for inference
- Concurrent request handling

### 2. Technology Stack Selection

Based on the analysis, recommend:

**RAPIDS Components** (for data processing and training):
- [ ] cuDF - GPU DataFrames for ETL and feature engineering
- [ ] cuML - GPU-accelerated ML algorithms (if traditional ML models)
- [ ] cuGraph - Graph algorithms (if network/relationship analysis)
- [ ] DASK-cuDF - Multi-GPU distributed computing (if >1 GPU or large scale)

**TensorRT Components** (for inference optimization):
- [ ] TensorRT for deep learning inference
- [ ] Triton Inference Server for production deployment
- [ ] Model optimization: FP16, INT8 quantization
- [ ] Multi-instance GPU (MIG) support

**Hybrid Approach**:
- RAPIDS for: _________________
- TensorRT for: _________________
- CPU fallback for: _________________

### 3. Architecture Decision Record

**Chosen Architecture**: [RAPIDS-only / TensorRT-only / Hybrid]

**Rationale**:
```
Based on {{USE_CASE_NAME}} requirements:
- Data operations: [describe]
- Model training: [describe approach]
- Inference needs: [describe requirements]
- Scaling strategy: [single GPU / multi-GPU / distributed]
```

**Expected Performance Gains**:
- Training speedup: __x faster than CPU baseline
- Inference latency: __ms (vs __ms CPU)
- Throughput: __ requests/sec/GPU

### 4. Dependency Mapping

**Core Dependencies**:
```toml
[dependencies]
# RAPIDS
cudf-cu12 = ">=24.10.0"  # Adjust CUDA version
cuml-cu12 = ">=24.10.0"
# Add others as needed
```

**Model Dependencies** (if applicable):
- HuggingFace models: `{{MODEL_NAMES}}`
- Pre-trained weights: `{{WEIGHT_SOURCES}}`

## Recommended GPU Environment for POC

**For Learning & Initial POCs:**
- **NVIDIA Virtual Workstation**: Pre-configured CUDA, minimal setup, ideal for beginners
  - Setup time: < 30 minutes
  - Cost: $2-5/hour
  - Best for: Learning, team training, initial POC development

**For Cost-Conscious POCs:**
- **GCP N1 + T4 GPU**: Lower cost, requires CUDA setup
  - Setup time: 1-2 hours
  - Cost: $0.35-0.95/hour
  - Best for: Longer experiments, batch processing

**For Development:**
- **CPU Mode**: Free, no GPU required
  - Best for: Code development, logic testing (no performance validation)

## Output Specification

Generate a structured POC plan with:

1. **Technology Stack**: List of RAPIDS/TensorRT components to experiment with
2. **Data Pipeline**: cuDF transformations, feature engineering steps for POC
3. **Model Architecture**: Training and inference approach for learning
4. **Performance Benchmarks**: Target metrics vs CPU baseline to measure speedup
5. **Deployment Strategy**: Single GPU (typical for POC), multi-GPU if needed for learning
6. **Learning Objectives**: What CUDA/NVIDIA capabilities this POC will demonstrate

## Next Steps

After completing this discovery phase:
- Proceed to `02-architecture-design.md` for detailed system design
- Reference `implementation/scaffold-generator.md` for project structure
- Use `validation/benchmark-plan.md` for performance testing strategy
