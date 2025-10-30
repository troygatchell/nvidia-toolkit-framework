# Prompt Customization Guide

Learn how to adapt the framework prompts for your specific use cases.

## Overview

The NVIDIA Toolkit Framework uses **structured prompt files** to guide the creation of GPU-accelerated ML toolkits. Each prompt file contains:

1. **Variables**: Placeholders like `{{USE_CASE_NAME}}` that you replace with your values
2. **Decision frameworks**: Questions and checklists to guide technology choices
3. **Code templates**: Boilerplate code with customization points
4. **Next steps**: Links to the next prompt in the workflow

## Prompt Workflow

```
Planning → Implementation → Validation
```

### Planning Phase

| Prompt File | Purpose | Outputs |
|-------------|---------|---------|
| `01-use-case-discovery.md` | Analyze requirements and select tech stack | Technology decisions, dependency list |
| `02-architecture-design.md` | Design system architecture | Module structure, configuration schema |

### Implementation Phase

| Prompt File | Purpose | Outputs |
|-------------|---------|---------|
| `scaffold-generator.md` | Generate project structure | Directory tree, boilerplate files |
| `data-pipeline.md` | Implement data processing | DataLoader, Preprocessor, FeatureEngineer |
| `model-implementation.md` | Build model training/inference | Model classes, Trainer, Predictor |

### Validation Phase

| Prompt File | Purpose | Outputs |
|-------------|---------|---------|
| `benchmark-plan.md` | Create performance tests | Benchmark scripts, results documentation |

## Variable Reference

### Core Variables

These appear in most prompt files:

| Variable | Description | Example |
|----------|-------------|---------|
| `{{USE_CASE_NAME}}` | Human-readable name | "Real-time Recommendation System" |
| `{{USE_CASE_SLUG}}` | Snake_case identifier | "realtime_recommendation" |
| `{{DOMAIN}}` | Industry domain | "MarTech", "AdTech", "FinTech" |
| `{{PRIMARY_GOAL}}` | Main objective | "Generate recommendations in <50ms" |

### Data Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{DATA_SCALE}}` | Data volume | "10M users, 100K products" |
| `{{DATA_FORMAT}}` | File format | "parquet", "csv", "json" |
| `{{FEATURES}}` | Feature list | ["user_id", "product_id", "rating"] |
| `{{TARGET_VARIABLE}}` | Prediction target | "rating", "churned", "click" |

### Model Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{MODEL_TYPE}}` | Algorithm name | "KNN", "XGBoost", "LSTM" |
| `{{MODEL_FRAMEWORK}}` | Implementation | "cuML", "XGBoost GPU", "PyTorch" |
| `{{HYPERPARAMETERS}}` | Model parameters | n_neighbors: 10 |

### Performance Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{LATENCY_REQUIREMENT}}` | Target latency | "p99 <100ms" |
| `{{SPEEDUP}}` | Expected GPU speedup | "20x" |
| `{{GPU_MODEL}}` | GPU hardware | "T4", "V100", "A100" |

## Customization Examples

### Example 1: Customer Segmentation

**Original config** (recommendation system):
```yaml
use_case:
  name: "Real-time Recommendation System"
  slug: "realtime_recommendation"

model:
  type: "KNN"
  task: "recommendation"
  params:
    n_neighbors: 10
```

**Customized config** (segmentation):
```yaml
use_case:
  name: "Customer Segmentation System"
  slug: "customer_segmentation"

model:
  type: "KMeans"
  task: "clustering"
  params:
    n_clusters: 5
    max_iter: 300
```

**Prompt adjustments**:

In `prompts/implementation/model-implementation.md`:

**Replace:**
```python
from cuml import {{CUML_MODEL}}
# For KNN: from cuml.neighbors import NearestNeighbors
```

**With:**
```python
from cuml import KMeans  # {{CUML_MODEL}} = KMeans
```

### Example 2: Churn Prediction

**Config:**
```yaml
use_case:
  name: "Customer Churn Prediction"
  slug: "churn_prediction"

model:
  type: "XGBoost"
  task: "classification"
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
```

**Prompt adjustments**:

In `prompts/implementation/model-implementation.md`, use the XGBoost template instead of cuML template.

**Update metrics** in `_compute_metrics`:
```python
# For classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

return {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred)),
    "recall": float(recall_score(y_true, y_pred)),
    "auc": float(roc_auc_score(y_true, y_pred_proba)),
}
```

### Example 3: Time Series Forecasting

**Config:**
```yaml
use_case:
  name: "Demand Forecasting"
  slug: "demand_forecasting"

model:
  type: "LSTM"
  task: "regression"
  framework: "PyTorch"

technology_stack:
  tensorrt:
    enabled: true
    precision: "fp16"
```

**Prompt adjustments**:

1. In `01-use-case-discovery.md`, select:
   - RAPIDS for data preprocessing
   - PyTorch for model training
   - TensorRT for inference optimization

2. In `model-implementation.md`, use PyTorch template and add TensorRT conversion:
```python
def _optimize_with_tensorrt(self):
    """Convert PyTorch model to TensorRT."""
    import torch
    import tensorrt as trt

    # Export to ONNX
    dummy_input = torch.randn(1, self.config["sequence_length"], self.config["input_dim"])
    torch.onnx.export(self.model, dummy_input, "model.onnx")

    # Convert to TensorRT
    # ... TensorRT conversion code
```

## Decision Trees

### Choosing Model Framework

```
Is it traditional ML (trees, linear, clustering)?
├─ Yes → Use cuML
│   ├─ KNN, KMeans, DBSCAN, RandomForest, etc.
│   └─ Benefits: Simple API, scikit-learn compatible
│
└─ No → Is it gradient boosting?
    ├─ Yes → Use XGBoost GPU
    │   └─ Benefits: Best performance for tabular data
    │
    └─ No → Is it deep learning?
        └─ Yes → Use PyTorch + TensorRT
            └─ Benefits: Flexibility, production optimization
```

### Choosing RAPIDS Components

```
Do you need multi-GPU or distributed computing?
├─ Yes → Use DASK-cuDF
│   └─ Scale: 10M+ samples, multiple GPUs
│
└─ No → Use cuDF only
    └─ Scale: Up to 10M samples, single GPU

Do you have graph/network data?
├─ Yes → Add cuGraph
│   └─ Use cases: Social networks, recommendation graphs
│
└─ No → Skip cuGraph
```

### TensorRT Decision

```
Is your model a neural network?
├─ Yes → Consider TensorRT
│   ├─ Latency critical (<10ms)? → Enable TensorRT with FP16/INT8
│   └─ Latency acceptable? → Use native PyTorch/TF
│
└─ No → TensorRT not applicable
    └─ Use cuML or XGBoost native inference
```

## Adding Custom Features

### Adding a New Preprocessing Step

1. **Update config**:
```yaml
preprocessing:
  steps:
    - "my_custom_step"

  my_custom_step:
    param1: value1
    param2: value2
```

2. **Add method in `data-pipeline.md` template**:
```python
def my_custom_step(self, df, param1, param2):
    """
    Custom preprocessing step.

    Args:
        df: Input DataFrame
        param1: Description
        param2: Description

    Returns:
        Processed DataFrame
    """
    # GPU-accelerated implementation
    # Use cuDF operations for best performance
    return df
```

### Adding a New Feature Engineering Function

1. **Update config**:
```yaml
feature_engineering:
  custom:
    - name: "my_feature"
      description: "Custom feature for my use case"
      params:
        window: 7
```

2. **Add method in `data-pipeline.md` template**:
```python
def create_my_feature(self, df, window=7):
    """
    Create my custom feature.

    Args:
        df: Input DataFrame
        window: Window size

    Returns:
        DataFrame with new feature
    """
    # Implement using cuDF operations
    df["my_feature"] = df["column"].rolling(window).mean()
    return df
```

### Adding a New Model Type

1. **Create new template in `model-implementation.md`**:

```python
class MyCustomModel(BaseModel):
    """
    {{MODEL_TYPE}} model with GPU acceleration.

    Uses: {{MODEL_FRAMEWORK}}
    """

    def __init__(self, config: dict, device: str = "gpu"):
        super().__init__(config, device)
        # Initialize model components

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train model."""
        # Implement training logic
        pass

    def predict(self, X):
        """Make predictions."""
        # Implement prediction logic
        pass

    # ... implement other BaseModel methods
```

2. **Update config**:
```yaml
model:
  type: "MyCustomModel"
  framework: "cuML"  # or PyTorch, XGBoost, etc.
  params:
    # Model-specific parameters
```

## Best Practices

### 1. Start with Templates

✅ **DO:**
- Use provided templates as starting points
- Customize gradually
- Test each component before moving to next

❌ **DON'T:**
- Write everything from scratch
- Skip prompt files
- Make large changes without testing

### 2. Follow Variable Naming

✅ **DO:**
- Use `{{UPPERCASE_SNAKE_CASE}}` for variables
- Be consistent across files
- Document custom variables

❌ **DON'T:**
- Mix naming conventions
- Use ambiguous variable names
- Forget to replace placeholders

### 3. Maintain GPU/CPU Abstraction

✅ **DO:**
```python
if device == "gpu":
    try:
        import cudf
        df_lib = cudf
    except ImportError:
        import pandas as pd
        df_lib = pd
else:
    import pandas as pd
    df_lib = pd
```

❌ **DON'T:**
```python
import cudf  # Hard dependency on GPU
df = cudf.DataFrame(data)
```

### 4. Document Decisions

✅ **DO:**
- Keep notes on why you chose specific components
- Document expected vs actual performance
- Update README with customizations

❌ **DON'T:**
- Make undocumented changes
- Skip benchmarking
- Forget to update docs

## Troubleshooting

### Issue: Variable Not Found in Config

**Problem:** Prompt references `{{MY_VARIABLE}}` but it's not in your config.

**Solution:**
1. Add variable to your config file
2. Or replace with a sensible default
3. Or remove the feature if not needed

### Issue: Model Template Doesn't Fit

**Problem:** Your model type doesn't match any provided template.

**Solution:**
1. Find the closest template (cuML, XGBoost, or PyTorch)
2. Use it as a skeleton
3. Modify `train()` and `predict()` methods
4. Keep the BaseModel interface intact

### Issue: Performance Not Meeting Targets

**Problem:** GPU speedup less than expected.

**Solution:**
1. Check data size (small data = small speedup)
2. Profile bottlenecks with benchmarks
3. Ensure GPU memory not exceeded
4. Consider batch size adjustments

## Advanced Customization

### Multi-GPU Support

Add to config:
```yaml
gpu:
  multi_gpu: true
  num_gpus: 4

technology_stack:
  rapids:
    dask_cudf: true
```

Update `data-pipeline.md` to use DASK:
```python
import dask_cudf

# Distributed DataFrame
ddf = dask_cudf.read_parquet("data/*.parquet")
result = ddf.groupby("column").mean().compute()
```

### Custom Metrics

Add to config:
```yaml
model:
  custom_metrics:
    - "ndcg@10"
    - "hit_rate@5"
```

Implement in model:
```python
def _compute_metrics(self, y_true, y_pred):
    """Compute custom metrics."""
    from my_metrics import ndcg_at_k

    return {
        "ndcg@10": ndcg_at_k(y_true, y_pred, k=10),
        # ...
    }
```

## Summary

| Task | File to Modify | Section |
|------|---------------|---------|
| Change model type | `config/*.yaml` | `model.type` |
| Add preprocessing | `config/*.yaml` + `data-pipeline.md` | `preprocessing.steps` |
| Add features | `config/*.yaml` + `data-pipeline.md` | `feature_engineering.custom` |
| Change GPU config | `config/*.yaml` | `gpu.*` |
| Add dependencies | `config/*.yaml` | `dependencies.*` |
| Custom metrics | `model-implementation.md` | `_compute_metrics()` |

## Next Steps

- See [EXAMPLES.md](EXAMPLES.md) for complete customization examples
- Review [QUICKSTART.md](QUICKSTART.md) for basic workflow
- Check [README.md](../README.md) for framework overview

---

Need help? Open an issue or discussion on GitHub!
