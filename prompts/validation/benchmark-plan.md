# Benchmark Plan

## Objective
Create comprehensive performance benchmarks comparing GPU-accelerated POC implementation against CPU baseline for `{{USE_CASE_NAME}}`.

## Purpose
Benchmarking is critical for POC validation:
- **Learn**: Understand where GPU acceleration provides the most benefit
- **Measure**: Quantify actual speedups on your data and use case
- **Validate**: Prove ROI of GPU acceleration before production investment
- **Demonstrate**: Show stakeholders concrete performance improvements

**Key POC Question to Answer:** "Is GPU acceleration worth it for this use case?"

## Prerequisites
- Completed model implementation
- Sample datasets for testing
- Both GPU and CPU environments available (or CPU fallback mode)

## Recommended Benchmark Environments

**Option 1: NVIDIA Virtual Workstation** (Learning)
- Pre-configured for benchmarking
- Run CPU baseline in Docker or separate VM
- Best for: Initial POC validation

**Option 2: GCP Dual Setup** (Cost-Effective)
- GPU instance: N1 + T4 for GPU benchmarks
- CPU instance: N1 without GPU for CPU baseline
- Run benchmarks in parallel, compare results

**Option 3: Single Instance with CPU Mode** (Development)
- Use `--device cpu` flag for CPU baseline
- Use `--device gpu` flag for GPU tests
- Convenient but not true apples-to-apples comparison

## Input Variables
- `{{BENCHMARK_METRICS}}`: Metrics to measure (latency, throughput, memory, accuracy)
- `{{DATASET_SIZES}}`: Test data sizes (e.g., [1K, 10K, 100K, 1M])
- `{{BATCH_SIZES}}`: Inference batch sizes to test
- `{{GPU_MODEL}}`: GPU type (T4, V100, A100, etc.)

## Benchmark Dimensions

### 1. Data Processing Benchmarks

**Metrics to Measure:**
- Data loading time (Parquet/CSV reading)
- Preprocessing time (cleaning, transformations)
- Feature engineering time
- End-to-end pipeline time

**Test Cases:**
```python
PIPELINE_BENCHMARKS = {
    "load_parquet": {
        "data_sizes": [10_000, 100_000, 1_000_000, 10_000_000],
        "metrics": ["time", "throughput_rows_per_sec"],
    },
    "preprocessing": {
        "operations": ["filter", "groupby", "join", "normalize"],
        "data_sizes": [100_000, 1_000_000],
        "metrics": ["time", "speedup_vs_cpu"],
    },
    "feature_engineering": {
        "features": {{FEATURE_LIST}},
        "data_sizes": [100_000, 1_000_000],
        "metrics": ["time", "memory_usage"],
    },
}
```

### 2. Training Benchmarks

**Metrics to Measure:**
- Training time per epoch
- Total training time
- GPU memory usage
- Training throughput (samples/sec)
- Convergence speed (epochs to target accuracy)

**Test Cases:**
```python
TRAINING_BENCHMARKS = {
    "single_epoch": {
        "data_sizes": [10_000, 100_000, 1_000_000],
        "metrics": ["time", "samples_per_sec", "gpu_memory_mb"],
    },
    "full_training": {
        "data_sizes": [100_000, 1_000_000],
        "metrics": ["total_time", "epochs_to_convergence", "final_accuracy"],
    },
    "scaling": {
        "num_gpus": [1, 2, 4] if {{MULTI_GPU}} else [1],
        "data_size": 1_000_000,
        "metrics": ["speedup", "efficiency"],
    },
}
```

### 3. Inference Benchmarks

**Metrics to Measure:**
- Latency (p50, p95, p99)
- Throughput (requests/sec)
- Batch inference time
- GPU memory usage
- Cold start time

**Test Cases:**
```python
INFERENCE_BENCHMARKS = {
    "latency": {
        "batch_sizes": [1, 8, 16, 32, 64, 128, 256],
        "num_requests": 1000,
        "metrics": ["p50_ms", "p95_ms", "p99_ms", "max_ms"],
    },
    "throughput": {
        "batch_sizes": [32, 64, 128, 256],
        "duration_sec": 60,
        "metrics": ["requests_per_sec", "samples_per_sec"],
    },
    "concurrent": {
        "concurrency_levels": [1, 5, 10, 20],
        "batch_size": 32,
        "metrics": ["throughput", "avg_latency"],
    },
}
```

## Implementation Template

### benchmarks/benchmark_data_pipeline.py

```python
"""Benchmark data pipeline performance."""

import time
import numpy as np
from typing import List, Dict
import logging

from {{USE_CASE_SLUG}}.data import DataPipeline
from {{USE_CASE_SLUG}}.utils import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipelineBenchmark:
    """Benchmark data pipeline operations."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.config = Config.load("default").model_dump()

    def benchmark_load(self, device: str, data_size: int) -> Dict:
        """Benchmark data loading."""
        pipeline = DataPipeline(self.config, device=device)

        # Warmup
        _ = pipeline.loader.load_parquet(self.data_path)

        # Measure
        times = []
        for _ in range(5):
            start = time.time()
            df = pipeline.loader.load_parquet(self.data_path)
            elapsed = time.time() - start
            times.append(elapsed)
            rows = len(df)

        avg_time = np.mean(times)
        throughput = rows / avg_time

        return {
            "device": device,
            "operation": "load_parquet",
            "data_size": rows,
            "avg_time_sec": avg_time,
            "throughput_rows_per_sec": throughput,
        }

    def benchmark_preprocessing(self, device: str) -> Dict:
        """Benchmark preprocessing operations."""
        pipeline = DataPipeline(self.config, device=device)
        df = pipeline.loader.load_parquet(self.data_path)

        # Warmup
        _ = pipeline.preprocessor.clean_missing_values(df)

        # Measure
        start = time.time()
        df_clean = pipeline.preprocessor.clean_missing_values(df)
        elapsed = time.time() - start

        return {
            "device": device,
            "operation": "preprocessing",
            "data_size": len(df),
            "time_sec": elapsed,
        }

    def compare_cpu_gpu(self, data_sizes: List[int]) -> None:
        """Compare CPU vs GPU performance."""
        results = []

        for size in data_sizes:
            logger.info(f"Benchmarking data size: {size}")

            # GPU
            try:
                gpu_result = self.benchmark_load("gpu", size)
                results.append(gpu_result)
            except Exception as e:
                logger.warning(f"GPU benchmark failed: {e}")

            # CPU
            cpu_result = self.benchmark_load("cpu", size)
            results.append(cpu_result)

            # Calculate speedup
            if "gpu" in [r["device"] for r in results[-2:]]:
                gpu_time = [r for r in results[-2:] if r["device"] == "gpu"][0]["avg_time_sec"]
                cpu_time = [r for r in results[-2:] if r["device"] == "cpu"][0]["avg_time_sec"]
                speedup = cpu_time / gpu_time
                logger.info(f"Speedup: {speedup:.2f}x")

        # Print summary table
        self._print_results(results)

    def _print_results(self, results: List[Dict]) -> None:
        """Print benchmark results as table."""
        print("\n" + "="*80)
        print(f"{'Device':<10} {'Operation':<20} {'Data Size':<12} {'Time (s)':<12} {'Throughput':<15}")
        print("="*80)
        for r in results:
            print(
                f"{r['device']:<10} "
                f"{r['operation']:<20} "
                f"{r['data_size']:<12} "
                f"{r.get('avg_time_sec', r.get('time_sec', 0)):<12.4f} "
                f"{r.get('throughput_rows_per_sec', 0):<15,.0f}"
            )
        print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to benchmark data")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10_000, 100_000, 1_000_000])
    args = parser.parse_args()

    benchmark = DataPipelineBenchmark(args.data)
    benchmark.compare_cpu_gpu(args.sizes)
```

### benchmarks/benchmark_training.py

```python
"""Benchmark model training performance."""

import time
import numpy as np
from typing import Dict
import logging

from {{USE_CASE_SLUG}}.data import DataPipeline
from {{USE_CASE_SLUG}}.models import {{MODEL_CLASS_NAME}}, Trainer
from {{USE_CASE_SLUG}}.utils import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingBenchmark:
    """Benchmark model training."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.config = Config.load("default")

    def benchmark_training(self, device: str) -> Dict:
        """Benchmark full training loop."""
        # Load data
        pipeline = DataPipeline(self.config.model_dump(), device=device)
        df = pipeline.run(self.data_path, mode="train")

        # Prepare features
        target_col = self.config.model["target_column"]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Initialize model
        model = {{MODEL_CLASS_NAME}}(self.config.model_dump(), device=device)
        trainer = Trainer(model, self.config.model_dump())

        # Measure training time
        start = time.time()
        results = trainer.train(X, y)
        training_time = time.time() - start

        return {
            "device": device,
            "training_time_sec": training_time,
            "num_samples": len(X),
            "samples_per_sec": len(X) / training_time,
        }

    def compare_cpu_gpu(self) -> None:
        """Compare training performance."""
        logger.info("Benchmarking GPU training...")
        gpu_result = self.benchmark_training("gpu")

        logger.info("Benchmarking CPU training...")
        cpu_result = self.benchmark_training("cpu")

        speedup = cpu_result["training_time_sec"] / gpu_result["training_time_sec"]

        print("\n" + "="*70)
        print("TRAINING BENCHMARK RESULTS")
        print("="*70)
        print(f"{'Device':<10} {'Time (s)':<15} {'Samples/sec':<15}")
        print("-"*70)
        print(f"GPU        {gpu_result['training_time_sec']:<15.2f} {gpu_result['samples_per_sec']:<15,.0f}")
        print(f"CPU        {cpu_result['training_time_sec']:<15.2f} {cpu_result['samples_per_sec']:<15,.0f}")
        print("-"*70)
        print(f"Speedup: {speedup:.2f}x")
        print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    benchmark = TrainingBenchmark(args.data)
    benchmark.compare_cpu_gpu()
```

### benchmarks/benchmark_inference.py

```python
"""Benchmark inference performance."""

import time
import numpy as np
from typing import List, Dict
import logging

from {{USE_CASE_SLUG}}.models import {{MODEL_CLASS_NAME}}, Predictor
from {{USE_CASE_SLUG}}.utils import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceBenchmark:
    """Benchmark inference latency and throughput."""

    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.config = Config.load("default")

    def benchmark_latency(
        self,
        device: str,
        batch_sizes: List[int],
        num_requests: int = 1000,
    ) -> List[Dict]:
        """Benchmark inference latency across batch sizes."""
        # Load model
        model = {{MODEL_CLASS_NAME}}(self.config.model_dump(), device=device)
        model.load(self.model_path)
        predictor = Predictor(model, self.config.model_dump())

        # Load test data
        # TODO: Load test data based on format

        results = []
        for batch_size in batch_sizes:
            latencies = []

            # Warmup
            for _ in range(10):
                # TODO: Get batch of data
                _ = predictor.predict(batch)

            # Measure
            for _ in range(num_requests):
                start = time.time()
                _ = predictor.predict(batch)
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)

            results.append({
                "device": device,
                "batch_size": batch_size,
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "max_ms": np.max(latencies),
                "avg_ms": np.mean(latencies),
            })

        return results

    def benchmark_throughput(
        self,
        device: str,
        batch_size: int,
        duration_sec: int = 60,
    ) -> Dict:
        """Benchmark inference throughput."""
        model = {{MODEL_CLASS_NAME}}(self.config.model_dump(), device=device)
        model.load(self.model_path)
        predictor = Predictor(model, self.config.model_dump())

        # Warmup
        for _ in range(10):
            _ = predictor.predict(batch)

        # Measure
        start = time.time()
        num_requests = 0
        while time.time() - start < duration_sec:
            _ = predictor.predict(batch)
            num_requests += 1

        elapsed = time.time() - start
        throughput = num_requests / elapsed

        return {
            "device": device,
            "batch_size": batch_size,
            "duration_sec": elapsed,
            "num_requests": num_requests,
            "requests_per_sec": throughput,
            "samples_per_sec": throughput * batch_size,
        }

    def print_latency_results(self, results: List[Dict]) -> None:
        """Print latency benchmark results."""
        print("\n" + "="*80)
        print("INFERENCE LATENCY BENCHMARK")
        print("="*80)
        print(f"{'Device':<10} {'Batch':<8} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12} {'Max (ms)':<12}")
        print("-"*80)
        for r in results:
            print(
                f"{r['device']:<10} "
                f"{r['batch_size']:<8} "
                f"{r['p50_ms']:<12.2f} "
                f"{r['p95_ms']:<12.2f} "
                f"{r['p99_ms']:<12.2f} "
                f"{r['max_ms']:<12.2f}"
            )
        print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    args = parser.parse_args()

    benchmark = InferenceBenchmark(args.model, args.data)

    # Latency benchmark
    logger.info("Running latency benchmarks...")
    gpu_latency = benchmark.benchmark_latency("gpu", args.batch_sizes)
    cpu_latency = benchmark.benchmark_latency("cpu", args.batch_sizes)

    benchmark.print_latency_results(gpu_latency + cpu_latency)
```

## Benchmark Execution Plan

### Phase 1: Data Pipeline Benchmarks
```bash
python benchmarks/benchmark_data_pipeline.py \
    --data data/sample/train.parquet \
    --sizes 10000 100000 1000000
```

### Phase 2: Training Benchmarks
```bash
python benchmarks/benchmark_training.py \
    --data data/sample/train.parquet
```

### Phase 3: Inference Benchmarks
```bash
python benchmarks/benchmark_inference.py \
    --model models/checkpoints/best_model.pkl \
    --data data/sample/test.parquet \
    --batch-sizes 1 8 32 128 256
```

## Results Documentation Template

Create `docs/BENCHMARKS.md`:

```markdown
# Performance Benchmarks - {{USE_CASE_NAME}}

## Test Environment

- **GPU**: {{GPU_MODEL}}
- **CPU**: {{CPU_MODEL}}
- **CUDA Version**: {{CUDA_VERSION}}
- **RAPIDS Version**: {{RAPIDS_VERSION}}
- **Dataset Size**: {{DATASET_SIZE}}

## Results Summary

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Data Loading | {{CPU_LOAD}}s | {{GPU_LOAD}}s | {{LOAD_SPEEDUP}}x |
| Preprocessing | {{CPU_PREP}}s | {{GPU_PREP}}s | {{PREP_SPEEDUP}}x |
| Training | {{CPU_TRAIN}}s | {{GPU_TRAIN}}s | {{TRAIN_SPEEDUP}}x |
| Inference (p99) | {{CPU_P99}}ms | {{GPU_P99}}ms | {{INFER_SPEEDUP}}x |

## Detailed Results

[Insert detailed benchmark tables here]

## Conclusions

GPU acceleration provides {{OVERALL_SPEEDUP}}x end-to-end speedup for {{USE_CASE_NAME}}.
\`\`\`

## Checklist

- [ ] Implement data pipeline benchmarks
- [ ] Implement training benchmarks
- [ ] Implement inference benchmarks (latency & throughput)
- [ ] Run benchmarks on GPU
- [ ] Run benchmarks on CPU
- [ ] Document results in docs/BENCHMARKS.md
- [ ] Create visualization charts (optional)
- [ ] Verify latency meets requirements ({{TARGET_LATENCY}})

## Next Steps

- Use results to optimize bottlenecks
- Update README with benchmark highlights
- Consider TensorRT optimization if inference latency insufficient
