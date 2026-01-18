"""
Configuration Management with Environment Variables and Validation
Provides type-safe configuration using Pydantic with support for .env files

Author: Mohamed salem eddah
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    """Data-related configuration."""

    raw_dir: Path = Field(
        default=Path("data/raw/CICIDS2017"),
        description="Directory containing raw CICIDS2017 data",
    )
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Directory for processed data"
    )
    graph_dir: Path = Field(default=Path("data/graphs"), description="Directory for graph data")
    cache_dir: Path = Field(default=Path("data/cache"), description="Directory for cached data")
    sample_size: Optional[int] = Field(
        default=None, description="Sample size for testing (None for full dataset)"
    )

    @field_validator("raw_dir", "processed_dir", "graph_dir", "cache_dir", mode="before")
    @classmethod
    def expand_path(cls, v):
        """Expand environment variables and relative paths."""
        if isinstance(v, str):
            v = os.path.expandvars(v)
            v = os.path.expanduser(v)
        return Path(v)


class PreprocessingConfig(BaseSettings):
    """Preprocessing configuration."""

    normalize: bool = Field(default=True, description="Apply normalization")
    handle_outliers: bool = Field(default=True, description="Handle outliers")
    engineer_features: bool = Field(default=True, description="Engineer additional features")
    outlier_method: Literal["iqr", "zscore"] = Field(
        default="iqr", description="Outlier detection method"
    )
    outlier_threshold: float = Field(default=3.0, description="Outlier threshold")
    normalization_method: Literal["robust", "standard", "minmax"] = Field(
        default="robust", description="Normalization method"
    )
    use_caching: bool = Field(default=True, description="Use caching for preprocessed data")
    cache_format: Literal["parquet", "pickle"] = Field(
        default="parquet", description="Cache file format"
    )


class GraphConfig(BaseSettings):
    """Graph construction configuration."""

    construction_method: Literal["host_based", "flow_based", "hierarchical"] = Field(
        default="host_based", description="Graph construction method"
    )
    time_window: int = Field(default=300, description="Time window for temporal snapshots (seconds)")
    directed: bool = Field(default=True, description="Create directed graph")
    aggregate_edges: bool = Field(default=True, description="Aggregate parallel edges")
    min_flows: int = Field(default=1, description="Minimum flows to create an edge")
    self_loops: bool = Field(default=False, description="Include self-loops")
    include_edge_features: bool = Field(default=True, description="Include edge features")
    parallel_construction: bool = Field(
        default=True, description="Use parallel processing for graph construction"
    )
    num_workers: int = Field(
        default=4, description="Number of parallel workers for graph construction"
    )


class ModelConfig(BaseSettings):
    """Model architecture configuration."""

    type: Literal["gcn", "gat", "sage", "hybrid", "temporal"] = Field(
        default="gcn", description="Model type"
    )
    hidden_dim: int = Field(default=128, description="Hidden dimension size")
    num_layers: int = Field(default=3, description="Number of GNN layers")
    dropout: float = Field(default=0.5, description="Dropout rate")
    use_batch_norm: bool = Field(default=True, description="Use batch normalization")

    # GAT-specific
    num_heads: int = Field(default=4, description="Number of attention heads (GAT)")

    # GraphSAGE-specific
    aggregator: Literal["mean", "max", "lstm"] = Field(
        default="mean", description="Aggregator type (GraphSAGE)"
    )

    # Temporal GNN
    lstm_layers: int = Field(default=2, description="Number of LSTM layers (Temporal GNN)")
    gnn_type: str = Field(default="gcn", description="Base GNN type for temporal model")


class TrainingConfig(BaseSettings):
    """Training configuration."""

    num_epochs: int = Field(default=200, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    weight_decay: float = Field(default=0.0005, description="Weight decay (L2 regularization)")
    batch_size: int = Field(default=512, description="Batch size for training")
    patience: int = Field(default=30, description="Early stopping patience")
    min_delta: float = Field(default=0.0001, description="Minimum improvement for early stopping")
    scheduler_type: Literal["plateau", "cosine"] = Field(
        default="plateau", description="Learning rate scheduler"
    )
    use_class_weights: bool = Field(default=True, description="Use class weights for imbalance")
    gradient_clip: float = Field(default=1.0, description="Gradient clipping threshold")

    # Performance optimizations
    use_mixed_precision: bool = Field(default=True, description="Use mixed precision training")
    use_graph_sampling: bool = Field(default=True, description="Use graph sampling for large graphs")
    num_neighbors: list[int] = Field(
        default=[15, 10, 5], description="Number of neighbors to sample per layer"
    )


class EvaluationConfig(BaseSettings):
    """Evaluation configuration."""

    train_ratio: float = Field(default=0.6, description="Training set ratio")
    val_ratio: float = Field(default=0.2, description="Validation set ratio")
    test_ratio: float = Field(default=0.2, description="Test set ratio")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    metrics: list[str] = Field(
        default=["accuracy", "precision", "recall", "f1", "roc_auc"],
        description="Metrics to compute",
    )


class OutputConfig(BaseSettings):
    """Output configuration."""

    model_dir: Path = Field(default=Path("models"), description="Model checkpoint directory")
    results_dir: Path = Field(default=Path("results"), description="Results directory")
    visualizations_dir: Path = Field(
        default=Path("results/visualizations"), description="Visualizations directory"
    )
    save_best_only: bool = Field(default=True, description="Save only best model")
    save_interval: int = Field(default=10, description="Save checkpoint every N epochs")

    @field_validator("model_dir", "results_dir", "visualizations_dir", mode="before")
    @classmethod
    def expand_path(cls, v):
        """Expand paths."""
        if isinstance(v, str):
            v = os.path.expandvars(v)
            v = os.path.expanduser(v)
        return Path(v)


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    console_output: bool = Field(default=True, description="Output logs to console")
    use_structured_logging: bool = Field(
        default=True, description="Use structured JSON logging"
    )

    @field_validator("log_dir", mode="before")
    @classmethod
    def expand_path(cls, v):
        """Expand paths."""
        if isinstance(v, str):
            v = os.path.expandvars(v)
        return Path(v)


class ResourceConfig(BaseSettings):
    """Resource configuration."""

    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", description="Device to use for training"
    )
    num_workers: int = Field(default=4, description="Number of data loading workers")
    pin_memory: bool = Field(default=True, description="Pin memory for faster GPU transfer")


class APIConfig(BaseSettings):
    """API configuration for production serving."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of API workers")
    reload: bool = Field(default=False, description="Enable auto-reload (dev only)")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    max_request_size: int = Field(
        default=10 * 1024 * 1024, description="Maximum request size in bytes"
    )


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""

    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")
    enable_mlflow: bool = Field(default=False, description="Enable MLflow tracking")
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="gnn_cyber_experiment", description="MLflow experiment name"
    )


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Global settings
    seed: int = Field(default=42, description="Random seed")
    deterministic: bool = Field(default=True, description="Use deterministic algorithms")
    experiment_name: str = Field(
        default="cicids2017_gnn_experiment", description="Experiment name"
    )

    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.data.processed_dir,
            self.data.graph_dir,
            self.data.cache_dir,
            self.output.model_dir,
            self.output.results_dir,
            self.output.visualizations_dir,
            self.logging.log_dir,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "data": self.data.model_dump(),
            "preprocessing": self.preprocessing.model_dump(),
            "graph": self.graph.model_dump(),
            "model": self.model.model_dump(),
            "training": self.training.model_dump(),
            "evaluation": self.evaluation.model_dump(),
            "output": self.output.model_dump(),
            "logging": self.logging.model_dump(),
            "resources": self.resources.model_dump(),
            "api": self.api.model_dump(),
            "monitoring": self.monitoring.model_dump(),
            "seed": self.seed,
            "deterministic": self.deterministic,
            "experiment_name": self.experiment_name,
        }


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from file and environment variables.

    Args:
        config_path: Optional path to YAML config file (for backward compatibility)

    Returns:
        AppConfig instance
    """
    if config_path and config_path.exists():
        import yaml

        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Merge YAML config with environment variables
        # Environment variables take precedence
        return AppConfig(**yaml_config)

    # Load from environment variables and .env file
    return AppConfig()


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global config instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig):
    """Set global config instance."""
    global _config
    _config = config
