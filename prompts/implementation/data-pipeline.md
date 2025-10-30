# Data Pipeline Implementation

## Objective
Implement GPU-accelerated data loading, preprocessing, and feature engineering for `{{USE_CASE_NAME}}`.

## Prerequisites
- Completed scaffolding from `scaffold-generator.md`
- Sample dataset available in `data/sample/`

## Input Variables
- `{{DATA_SOURCE}}`: Data location (file path, S3, database, etc.)
- `{{DATA_FORMAT}}`: Format (parquet, csv, json, etc.)
- `{{FEATURES}}`: List of features to extract/engineer
- `{{TARGET_VARIABLE}}`: Prediction target (if supervised learning)
- `{{PREPROCESSING_STEPS}}`: Required transformations

## Implementation Guide

### 1. Data Loader (src/{{USE_CASE_SLUG}}/data/loader.py)

Implement data loading with GPU/CPU abstraction:

```python
"""Data loading utilities with GPU acceleration."""

import logging
from pathlib import Path
from typing import Union, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    GPU-accelerated data loader with CPU fallback.

    Supports: {{SUPPORTED_FORMATS}}
    """

    def __init__(self, device: str = "gpu"):
        """
        Initialize data loader.

        Args:
            device: Computation device ("gpu" or "cpu")
        """
        self.device = device
        if device == "gpu":
            try:
                import cudf
                self.df_lib = cudf
                logger.info("Using cuDF for GPU-accelerated data loading")
            except ImportError:
                logger.warning("cuDF not available, falling back to pandas")
                self.df_lib = pd
                self.device = "cpu"
        else:
            self.df_lib = pd

    def load_parquet(
        self,
        path: Union[str, Path],
        columns: Optional[list[str]] = None,
        filters: Optional[list] = None,
    ):
        """
        Load Parquet file(s).

        Args:
            path: Path to parquet file or directory
            columns: Columns to load (None for all)
            filters: Row filters (pyarrow-style)

        Returns:
            DataFrame (cuDF or pandas)
        """
        logger.info(f"Loading parquet from {path}")
        return self.df_lib.read_parquet(path, columns=columns, filters=filters)

    def load_csv(
        self,
        path: Union[str, Path],
        **kwargs,
    ):
        """
        Load CSV file.

        Args:
            path: Path to CSV file
            **kwargs: Additional arguments for read_csv

        Returns:
            DataFrame (cuDF or pandas)
        """
        logger.info(f"Loading CSV from {path}")
        return self.df_lib.read_csv(path, **kwargs)

    def load_json(
        self,
        path: Union[str, Path],
        **kwargs,
    ):
        """
        Load JSON file.

        Args:
            path: Path to JSON file
            **kwargs: Additional arguments for read_json

        Returns:
            DataFrame (cuDF or pandas)
        """
        logger.info(f"Loading JSON from {path}")
        return self.df_lib.read_json(path, **kwargs)

    def to_pandas(self, df):
        """Convert DataFrame to pandas (if cuDF)."""
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        return df

    def to_gpu(self, df):
        """Convert pandas DataFrame to cuDF (if available)."""
        if self.device == "gpu" and not hasattr(df, "to_pandas"):
            import cudf
            return cudf.from_pandas(df)
        return df
```

**Customization Points:**
- Add format-specific loaders for `{{DATA_FORMAT}}`
- Implement streaming/chunked loading for large datasets
- Add data validation checks

### 2. Preprocessor (src/{{USE_CASE_SLUG}}/data/preprocessor.py)

Implement GPU-accelerated preprocessing:

```python
"""Data preprocessing with GPU acceleration."""

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    GPU-accelerated data preprocessor.

    Handles: {{PREPROCESSING_STEPS}}
    """

    def __init__(self, device: str = "gpu"):
        self.device = device
        if device == "gpu":
            try:
                import cudf
                self.df_lib = cudf
            except ImportError:
                import pandas as pd
                self.df_lib = pd
                self.device = "cpu"
        else:
            import pandas as pd
            self.df_lib = pd

    def clean_missing_values(
        self,
        df,
        strategy: str = "drop",
        fill_value=None,
        columns: Optional[List[str]] = None,
    ):
        """
        Handle missing values.

        Args:
            df: Input DataFrame
            strategy: "drop", "fill", or "interpolate"
            fill_value: Value for fill strategy
            columns: Columns to process (None for all)

        Returns:
            Cleaned DataFrame
        """
        if columns:
            subset = columns
        else:
            subset = df.columns

        if strategy == "drop":
            return df.dropna(subset=subset)
        elif strategy == "fill":
            return df.fillna(fill_value)
        elif strategy == "interpolate":
            # cuDF supports interpolation
            return df.interpolate()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def filter_outliers(
        self,
        df,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5,
    ):
        """
        Filter outliers using GPU-accelerated operations.

        Args:
            df: Input DataFrame
            column: Column to check
            method: "iqr", "zscore", or "quantile"
            threshold: Threshold for outlier detection

        Returns:
            Filtered DataFrame
        """
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return df[(df[column] >= lower) & (df[column] <= upper)]
        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()
            z_scores = (df[column] - mean) / std
            return df[z_scores.abs() <= threshold]
        else:
            raise ValueError(f"Unknown method: {method}")

    def encode_categorical(
        self,
        df,
        columns: List[str],
        method: str = "label",
    ):
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame
            columns: Categorical columns
            method: "label" or "onehot"

        Returns:
            DataFrame with encoded columns
        """
        if method == "label":
            for col in columns:
                df[f"{col}_encoded"] = df[col].astype("category").cat.codes
        elif method == "onehot":
            if self.device == "gpu":
                # cuDF one-hot encoding
                df = self.df_lib.get_dummies(
                    df, columns=columns, prefix=columns, dtype="float32"
                )
            else:
                df = self.df_lib.get_dummies(
                    df, columns=columns, prefix=columns
                )
        return df

    def normalize_features(
        self,
        df,
        columns: List[str],
        method: str = "minmax",
    ):
        """
        Normalize numeric features.

        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: "minmax", "zscore", or "robust"

        Returns:
            DataFrame with normalized columns
        """
        for col in columns:
            if method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                df[f"{col}_norm"] = (df[col] - mean) / std
        return df
```

**Customization Points:**
- Add use-case-specific cleaning logic
- Implement custom outlier detection for `{{DOMAIN}}`
- Add feature validation

### 3. Feature Engineering (src/{{USE_CASE_SLUG}}/data/feature_engineering.py)

```python
"""Feature engineering with GPU acceleration."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    GPU-accelerated feature engineering.

    Creates features for: {{USE_CASE_NAME}}
    """

    def __init__(self, device: str = "gpu"):
        self.device = device
        if device == "gpu":
            try:
                import cudf
                self.df_lib = cudf
            except ImportError:
                import pandas as pd
                self.df_lib = pd
        else:
            import pandas as pd
            self.df_lib = pd

    def create_time_features(self, df, timestamp_col: str):
        """
        Extract time-based features from timestamp.

        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column name

        Returns:
            DataFrame with time features
        """
        df[timestamp_col] = self.df_lib.to_datetime(df[timestamp_col])
        df["hour"] = df[timestamp_col].dt.hour
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        df["day_of_month"] = df[timestamp_col].dt.day
        df["month"] = df[timestamp_col].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        return df

    def create_aggregation_features(
        self,
        df,
        group_cols: List[str],
        agg_col: str,
        agg_funcs: List[str] = ["mean", "sum", "count"],
    ):
        """
        Create aggregation features (GPU-accelerated).

        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_col: Column to aggregate
            agg_funcs: Aggregation functions

        Returns:
            DataFrame with aggregation features
        """
        agg_df = df.groupby(group_cols)[agg_col].agg(agg_funcs).reset_index()
        # Rename columns
        agg_df.columns = group_cols + [
            f"{agg_col}_{func}" for func in agg_funcs
        ]
        # Merge back
        df = df.merge(agg_df, on=group_cols, how="left")
        return df

    def create_interaction_features(
        self,
        df,
        col1: str,
        col2: str,
        operations: List[str] = ["multiply", "divide"],
    ):
        """
        Create interaction features between two columns.

        Args:
            df: Input DataFrame
            col1: First column
            col2: Second column
            operations: Operations to apply

        Returns:
            DataFrame with interaction features
        """
        if "multiply" in operations:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        if "divide" in operations and (df[col2] != 0).all():
            df[f"{col1}_div_{col2}"] = df[col1] / df[col2]
        if "add" in operations:
            df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
        if "subtract" in operations:
            df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
        return df

    # ADD USE-CASE-SPECIFIC FEATURE ENGINEERING BELOW
    # --------------------------------------------------

    def create_{{FEATURE_NAME}}(self, df, **kwargs):
        """
        Create {{FEATURE_NAME}} feature.

        Specific to {{USE_CASE_NAME}}.

        Args:
            df: Input DataFrame
            **kwargs: Feature-specific parameters

        Returns:
            DataFrame with new feature
        """
        # TODO: Implement use-case-specific feature
        pass
```

### 4. Pipeline Orchestration (src/{{USE_CASE_SLUG}}/data/__init__.py)

```python
"""Data pipeline orchestration."""

from .loader import DataLoader
from .preprocessor import Preprocessor
from .feature_engineering import FeatureEngineer

__all__ = ["DataLoader", "Preprocessor", "FeatureEngineer", "DataPipeline"]


class DataPipeline:
    """
    End-to-end data pipeline for {{USE_CASE_NAME}}.

    Orchestrates: Load → Preprocess → Feature Engineering
    """

    def __init__(self, config: dict, device: str = "gpu"):
        self.config = config
        self.device = device
        self.loader = DataLoader(device=device)
        self.preprocessor = Preprocessor(device=device)
        self.feature_engineer = FeatureEngineer(device=device)

    def run(self, data_path: str, mode: str = "train"):
        """
        Execute full pipeline.

        Args:
            data_path: Path to input data
            mode: "train" or "inference"

        Returns:
            Processed DataFrame
        """
        # 1. Load data
        df = self.loader.load_{{DATA_FORMAT}}(data_path)

        # 2. Preprocess
        df = self.preprocessor.clean_missing_values(df, strategy="drop")
        # Add more preprocessing steps based on config

        # 3. Feature engineering
        if "timestamp_col" in self.config:
            df = self.feature_engineer.create_time_features(
                df, self.config["timestamp_col"]
            )

        # Add use-case-specific feature engineering
        # df = self.feature_engineer.create_{{FEATURE_NAME}}(df)

        return df
```

## Testing Strategy

Create `tests/test_data/test_loader.py`:

```python
import pytest
from {{USE_CASE_SLUG}}.data import DataLoader


@pytest.mark.gpu
def test_loader_gpu():
    loader = DataLoader(device="gpu")
    # Test with sample data
    assert loader.device == "gpu"


def test_loader_cpu():
    loader = DataLoader(device="cpu")
    assert loader.device == "cpu"
```

## Performance Benchmarking

Add to `benchmarks/benchmark_data_pipeline.py`:

```python
import time
from {{USE_CASE_SLUG}}.data import DataPipeline


def benchmark_pipeline():
    """Compare GPU vs CPU data pipeline performance."""
    # GPU version
    pipeline_gpu = DataPipeline(config={}, device="gpu")
    start = time.time()
    df_gpu = pipeline_gpu.run("data/sample/train.parquet")
    gpu_time = time.time() - start

    # CPU version
    pipeline_cpu = DataPipeline(config={}, device="cpu")
    start = time.time()
    df_cpu = pipeline_cpu.run("data/sample/train.parquet")
    cpu_time = time.time() - start

    print(f"GPU: {gpu_time:.2f}s, CPU: {cpu_time:.2f}s, Speedup: {cpu_time/gpu_time:.1f}x")
```

## Checklist

- [ ] Implement DataLoader with {{DATA_FORMAT}} support
- [ ] Add preprocessing steps: {{PREPROCESSING_STEPS}}
- [ ] Create feature engineering functions for {{FEATURES}}
- [ ] Build DataPipeline orchestrator
- [ ] Write unit tests for each component
- [ ] Create benchmarking script
- [ ] Document expected speedups

## Next Steps

After data pipeline implementation:
- Proceed to `model-implementation.md` for training/inference
- Use `validation/benchmark-plan.md` to measure performance
