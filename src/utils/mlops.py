"""
MLOps Utilities for Experiment Tracking and Model Management
Provides integration with MLflow and model optimization

Author: Mohamed salem eddah
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn

from .logging import get_logger

logger = get_logger(__name__)


class MLflowExperimentTracker:
    """
    MLflow experiment tracking wrapper.

    Provides convenient methods for logging parameters, metrics, and artifacts.
    """

    def __init__(
        self, experiment_name: str, tracking_uri: Optional[str] = None, run_name: Optional[str] = None
    ):
        """
        Initialize MLflow experiment tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (default: from config)
            run_name: Optional name for the run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id

            logger.info(f"MLflow experiment: {experiment_name} (ID: {self.experiment_id})")

        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            self.experiment_id = None

    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        if run_name is None:
            run_name = self.run_name

        try:
            self.run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
            logger.info(f"Started MLflow run: {self.run.info.run_id}")
        except Exception as e:
            logger.warning(f"Could not start MLflow run: {e}")

    def end_run(self):
        """End the current MLflow run."""
        if self.run:
            try:
                mlflow.end_run()
                logger.info("MLflow run ended")
            except Exception as e:
                logger.warning(f"Error ending MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if self.run:
            try:
                mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Could not log params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.run:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Could not log metrics: {e}")

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """Log PyTorch model."""
        if self.run:
            try:
                mlflow.pytorch.log_model(
                    model, artifact_path=artifact_path, registered_model_name=registered_model_name
                )
                logger.info(f"Model logged to MLflow: {artifact_path}")
            except Exception as e:
                logger.warning(f"Could not log model: {e}")

    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if self.run:
            try:
                mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
            except Exception as e:
                logger.warning(f"Could not log artifact: {e}")

    def log_dict(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact."""
        if self.run:
            try:
                mlflow.log_dict(dictionary, filename)
            except Exception as e:
                logger.warning(f"Could not log dict: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """Set run tags."""
        if self.run:
            try:
                mlflow.set_tags(tags)
            except Exception as e:
                logger.warning(f"Could not set tags: {e}")


class ModelOptimizer:
    """
    Model optimization utilities for production deployment.

    Provides quantization, ONNX export, and other optimizations.
    """

    @staticmethod
    def quantize_dynamic(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Apply dynamic quantization to model.

        Args:
            model: PyTorch model to quantize
            dtype: Target dtype for quantization

        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")

        # Quantize linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=dtype
        )

        # Compare sizes
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        quantized_size = (
            sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024**2
        )

        logger.info(
            f"Model quantized: {original_size:.2f}MB -> {quantized_size:.2f}MB "
            f"({100 * quantized_size / original_size:.1f}% of original)"
        )

        return quantized_model

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: tuple,
        output_path: Path,
        input_names: list[str] = ["node_features", "edge_index"],
        output_names: list[str] = ["output"],
        dynamic_axes: Optional[Dict] = None,
    ):
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model to export
            dummy_input: Example input for tracing
            output_path: Path to save ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
        """
        logger.info(f"Exporting model to ONNX format: {output_path}")

        model.eval()

        # Default dynamic axes for variable batch size
        if dynamic_axes is None:
            dynamic_axes = {
                "node_features": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "output": {0: "num_nodes"},
            }

        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

            logger.info("ONNX export successful")

            # Verify the model
            import onnx

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verified")

        except Exception as e:
            logger.error(f"ONNX export failed: {e}", exc_info=True)
            raise

    @staticmethod
    def export_to_torchscript(
        model: nn.Module, dummy_input: tuple, output_path: Path, method: str = "trace"
    ):
        """
        Export model to TorchScript format.

        Args:
            model: PyTorch model to export
            dummy_input: Example input for tracing
            output_path: Path to save TorchScript model
            method: Export method ('trace' or 'script')
        """
        logger.info(f"Exporting model to TorchScript format: {output_path}")

        model.eval()

        try:
            if method == "trace":
                scripted_model = torch.jit.trace(model, dummy_input)
            elif method == "script":
                scripted_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown export method: {method}")

            scripted_model.save(output_path)
            logger.info("TorchScript export successful")

        except Exception as e:
            logger.error(f"TorchScript export failed: {e}", exc_info=True)
            raise

    @staticmethod
    def benchmark_model(
        model: nn.Module, dummy_input: tuple, num_iterations: int = 100, warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model inference time.

        Args:
            model: Model to benchmark
            dummy_input: Example input
            num_iterations: Number of iterations to run
            warmup: Number of warmup iterations

        Returns:
            Dictionary with benchmark results
        """
        import time

        model.eval()
        device = next(model.parameters()).device

        # Move input to device
        if isinstance(dummy_input, tuple):
            dummy_input = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in dummy_input)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(*dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(*dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        results = {
            "mean_ms": sum(times) / len(times) * 1000,
            "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5
            * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "throughput_samples_per_sec": 1.0 / (sum(times) / len(times)),
        }

        logger.info(f"Benchmark results: {results['mean_ms']:.2f}ms ± {results['std_ms']:.2f}ms")

        return results
