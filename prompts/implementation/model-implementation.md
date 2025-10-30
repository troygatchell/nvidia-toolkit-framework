# Model Implementation Prompt

## Objective
Implement GPU-accelerated model training and inference for `{{USE_CASE_NAME}}`.

## Prerequisites
- Completed data pipeline from `data-pipeline.md`
- Preprocessed data available
- Model architecture decided in `planning/02-architecture-design.md`

## Input Variables
- `{{MODEL_TYPE}}`: Model algorithm (e.g., "XGBoost", "Neural Collaborative Filtering", "K-Means")
- `{{MODEL_FRAMEWORK}}`: cuML, XGBoost GPU, PyTorch, TensorFlow
- `{{TRAINING_STRATEGY}}`: single-GPU, multi-GPU, distributed
- `{{INFERENCE_OPTIMIZATION}}`: TensorRT, cuML native, ONNX
- `{{HYPERPARAMETERS}}`: Model-specific parameters

## Implementation Guide

### 1. Base Model Interface (src/{{USE_CASE_SLUG}}/models/base.py)

Define abstract interface for all models:

```python
"""Base model interface for {{USE_CASE_NAME}}."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for GPU-accelerated models.

    Ensures consistent interface across different model types.
    """

    def __init__(self, config: dict, device: str = "gpu"):
        """
        Initialize model.

        Args:
            config: Model configuration dictionary
            device: Computation device ("gpu" or "cpu")
        """
        self.config = config
        self.device = device
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training history/metrics
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        # Compute metrics based on task type
        return self._compute_metrics(y_test, predictions)

    @abstractmethod
    def _compute_metrics(self, y_true, y_pred):
        """Compute task-specific metrics."""
        pass
```

### 2. Model Implementation (src/{{USE_CASE_SLUG}}/models/{{MODEL_NAME}}.py)

Implement specific model using `{{MODEL_FRAMEWORK}}`:

#### Example: cuML-based Model

```python
"""{{MODEL_TYPE}} implementation using cuML."""

import logging
from pathlib import Path
from typing import Optional
import joblib

from .base import BaseModel

logger = logging.getLogger(__name__)


class {{MODEL_CLASS_NAME}}(BaseModel):
    """
    {{MODEL_TYPE}} model with GPU acceleration.

    Uses: {{MODEL_FRAMEWORK}}
    """

    def __init__(self, config: dict, device: str = "gpu"):
        super().__init__(config, device)

        if device == "gpu":
            try:
                from cuml import {{CUML_MODEL}}
                self.model_class = {{CUML_MODEL}}
                logger.info("Using cuML GPU-accelerated {{MODEL_TYPE}}")
            except ImportError:
                logger.warning("cuML not available, falling back to sklearn")
                from sklearn import {{SKLEARN_MODEL}}
                self.model_class = {{SKLEARN_MODEL}}
                self.device = "cpu"
        else:
            from sklearn import {{SKLEARN_MODEL}}
            self.model_class = {{SKLEARN_MODEL}}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train {{MODEL_TYPE}} model.

        Args:
            X_train: Training features (cuDF DataFrame or numpy array)
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            dict: Training metrics
        """
        logger.info(f"Training {{MODEL_TYPE}} on {self.device}")

        # Initialize model with hyperparameters
        self.model = self.model_class(
            **self.config.get("model", {}).get("params", {})
        )

        # Train
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train)
        logger.info(f"Training metrics: {train_metrics}")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info(f"Validation metrics: {val_metrics}")
            return {"train": train_metrics, "val": val_metrics}

        return {"train": train_metrics}

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input features (cuDF DataFrame or numpy array)

        Returns:
            Predictions (cuDF Series or numpy array)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities (for classification).

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support predict_proba")

    def save(self, path: str):
        """Save model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # For cuML models, convert to CPU before saving
        if self.device == "gpu" and hasattr(self.model, "convert_to_sklearn"):
            model_to_save = self.model.convert_to_sklearn()
        else:
            model_to_save = self.model

        joblib.dump(model_to_save, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def _compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics."""
        # Import appropriate metrics based on task
        if self.config.get("task") == "classification":
            if self.device == "gpu":
                try:
                    from cuml.metrics import accuracy_score, roc_auc_score
                except ImportError:
                    from sklearn.metrics import accuracy_score, roc_auc_score
            else:
                from sklearn.metrics import accuracy_score, roc_auc_score

            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                # Add more metrics as needed
            }
        else:  # regression
            if self.device == "gpu":
                try:
                    from cuml.metrics import mean_squared_error, r2_score
                except ImportError:
                    from sklearn.metrics import mean_squared_error, r2_score
            else:
                from sklearn.metrics import mean_squared_error, r2_score

            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
```

#### Example: XGBoost GPU Model

```python
"""XGBoost GPU-accelerated model."""

import logging
from pathlib import Path
import xgboost as xgb

from .base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostGPUModel(BaseModel):
    """XGBoost with GPU acceleration."""

    def __init__(self, config: dict, device: str = "gpu"):
        super().__init__(config, device)
        self.tree_method = "hist" if device == "cpu" else "gpu_hist"

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model."""
        params = self.config.get("model", {}).get("params", {})
        params["tree_method"] = self.tree_method

        if self.device == "gpu":
            params["gpu_id"] = self.config.get("gpu", {}).get("device_id", 0)

        dtrain = xgb.DMatrix(X_train, label=y_train)

        eval_list = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_list.append((dval, "val"))

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.get("training", {}).get("epochs", 100),
            evals=eval_list,
            early_stopping_rounds=self.config.get("training", {}).get(
                "early_stopping_patience", 10
            ),
            verbose_eval=True,
        )

        self.is_trained = True
        return {"train_metrics": "logged"}

    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, path: str):
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        self.model = xgb.Booster()
        self.model.load_model(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    def _compute_metrics(self, y_true, y_pred):
        """Compute metrics."""
        from sklearn.metrics import mean_squared_error, r2_score
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
```

### 3. Trainer (src/{{USE_CASE_SLUG}}/models/trainer.py)

High-level training orchestration:

```python
"""Model training orchestration."""

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for {{USE_CASE_NAME}}.

    Handles: training loop, checkpointing, logging, early stopping
    """

    def __init__(self, model, config: dict):
        """
        Initialize trainer.

        Args:
            model: Model instance (implements BaseModel)
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(config.get("model", {}).get("checkpoint_dir", "models/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Execute training pipeline.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            dict: Training results and metrics
        """
        logger.info("Starting training...")
        start_time = time.time()

        # Train model
        history = self.model.train(X_train, y_train, X_val, y_val)

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s")

        # Save model
        if self.config.get("training", {}).get("save_best_only", True):
            model_path = self.checkpoint_dir / "best_model.pkl"
            self.model.save(model_path)

        return {
            "history": history,
            "training_time": training_time,
            "model_path": str(model_path),
        }

    def evaluate(self, X_test, y_test):
        """
        Evaluate trained model.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model...")
        start_time = time.time()

        metrics = self.model.evaluate(X_test, y_test)

        eval_time = time.time() - start_time
        logger.info(f"Evaluation completed in {eval_time:.2f}s")
        logger.info(f"Metrics: {metrics}")

        return {
            "metrics": metrics,
            "eval_time": eval_time,
        }
```

### 4. Predictor (src/{{USE_CASE_SLUG}}/models/predictor.py)

Inference engine with optimization:

```python
"""Optimized inference engine."""

import logging
import time
from pathlib import Path
from typing import Union, List

logger = logging.getLogger(__name__)


class Predictor:
    """
    High-performance predictor for {{USE_CASE_NAME}}.

    Features:
    - Batch prediction
    - GPU acceleration
    - TensorRT optimization (if configured)
    - Latency tracking
    """

    def __init__(self, model, config: dict):
        """
        Initialize predictor.

        Args:
            model: Trained model instance
            config: Inference configuration
        """
        self.model = model
        self.config = config
        self.batch_size = config.get("inference", {}).get("batch_size", 256)

        # Optional TensorRT optimization
        if config.get("inference", {}).get("enable_tensorrt", False):
            self._optimize_with_tensorrt()

    def predict(self, X, return_proba: bool = False):
        """
        Make predictions with latency tracking.

        Args:
            X: Input features
            return_proba: Return probabilities (classification only)

        Returns:
            Predictions (and optionally metadata)
        """
        start_time = time.time()

        if return_proba and hasattr(self.model, "predict_proba"):
            predictions = self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)

        latency = (time.time() - start_time) * 1000  # ms

        logger.debug(f"Prediction latency: {latency:.2f}ms")

        return {
            "predictions": predictions,
            "latency_ms": latency,
            "num_samples": len(X),
        }

    def predict_batch(self, X_batches: List):
        """
        Batch prediction with throughput tracking.

        Args:
            X_batches: List of input batches

        Returns:
            Combined predictions with metrics
        """
        all_predictions = []
        total_samples = 0
        start_time = time.time()

        for batch in X_batches:
            result = self.predict(batch)
            all_predictions.append(result["predictions"])
            total_samples += result["num_samples"]

        total_time = time.time() - start_time
        throughput = total_samples / total_time

        logger.info(f"Batch prediction: {throughput:.0f} samples/sec")

        return {
            "predictions": all_predictions,
            "total_time": total_time,
            "throughput": throughput,
            "total_samples": total_samples,
        }

    def _optimize_with_tensorrt(self):
        """Apply TensorRT optimization (if applicable)."""
        # TODO: Implement TensorRT conversion
        # This is model-dependent and requires ONNX export
        logger.info("TensorRT optimization requested but not yet implemented")
        pass
```

### 5. Module Exports (src/{{USE_CASE_SLUG}}/models/__init__.py)

```python
"""Model implementations for {{USE_CASE_NAME}}."""

from .base import BaseModel
from .{{MODEL_NAME}} import {{MODEL_CLASS_NAME}}
from .trainer import Trainer
from .predictor import Predictor

__all__ = [
    "BaseModel",
    "{{MODEL_CLASS_NAME}}",
    "Trainer",
    "Predictor",
]
```

## Training Script (scripts/train.py)

```python
"""Training script for {{USE_CASE_NAME}}."""

import argparse
import logging
from pathlib import Path

from {{USE_CASE_SLUG}}.data import DataPipeline
from {{USE_CASE_SLUG}}.models import {{MODEL_CLASS_NAME}}, Trainer
from {{USE_CASE_SLUG}}.utils import Config, get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Check device
    device = get_device(preferred=args.device, fallback=True)

    # Load and process data
    logger.info("Loading data...")
    pipeline = DataPipeline(config.model_dump(), device=device)
    df = pipeline.run(args.data, mode="train")

    # Split features and target
    target_col = config.model["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Initialize model and trainer
    model = {{MODEL_CLASS_NAME}}(config.model_dump(), device=device)
    trainer = Trainer(model, config.model_dump())

    # Train
    results = trainer.train(X, y)
    logger.info(f"Training results: {results}")


if __name__ == "__main__":
    main()
```

## Testing Strategy

Create `tests/test_models/test_{{MODEL_NAME}}.py`:

```python
import pytest
from {{USE_CASE_SLUG}}.models import {{MODEL_CLASS_NAME}}


@pytest.fixture
def sample_data():
    # Create sample training data
    pass


@pytest.mark.gpu
def test_model_train_gpu(sample_data):
    model = {{MODEL_CLASS_NAME}}(config={}, device="gpu")
    # Test training
    pass


def test_model_train_cpu(sample_data):
    model = {{MODEL_CLASS_NAME}}(config={}, device="cpu")
    # Test training
    pass
```

## Checklist

- [ ] Implement BaseModel interface
- [ ] Create {{MODEL_CLASS_NAME}} with {{MODEL_FRAMEWORK}}
- [ ] Build Trainer orchestrator
- [ ] Implement Predictor with latency tracking
- [ ] Create training script
- [ ] Add unit tests
- [ ] Document model architecture and hyperparameters

## Next Steps

- Proceed to `validation/benchmark-plan.md` for performance testing
- Use `validation/accuracy-validation.md` for model quality checks
