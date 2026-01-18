"""
Pytest Configuration and Fixtures
Provides shared fixtures for testing GNN Cyber Project
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import AppConfig


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    config = AppConfig(
        data={"raw_dir": "tests/fixtures/data", "sample_size": 1000},
        model={"type": "gcn", "hidden_dim": 64, "num_layers": 2},
        training={"num_epochs": 5, "batch_size": 32},
    )
    return config


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataframe():
    """Create sample network flow dataframe for testing."""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "Source IP": [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        "Destination IP": [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        "Source Port": np.random.randint(1024, 65535, n_samples),
        "Destination Port": np.random.choice([80, 443, 22, 3389, 8080], n_samples),
        "Protocol": np.random.choice([6, 17], n_samples),  # TCP, UDP
        "Flow Duration": np.random.exponential(1000, n_samples),
        "Total Fwd Packets": np.random.poisson(10, n_samples),
        "Total Backward Packets": np.random.poisson(8, n_samples),
        "Total Length of Fwd Packets": np.random.poisson(1000, n_samples),
        "Total Length of Bwd Packets": np.random.poisson(800, n_samples),
        "Fwd Packet Length Max": np.random.randint(0, 1500, n_samples),
        "Fwd Packet Length Min": np.random.randint(0, 100, n_samples),
        "Fwd Packet Length Mean": np.random.uniform(0, 500, n_samples),
        "Fwd Packet Length Std": np.random.uniform(0, 200, n_samples),
        "Bwd Packet Length Max": np.random.randint(0, 1500, n_samples),
        "Bwd Packet Length Min": np.random.randint(0, 100, n_samples),
        "Bwd Packet Length Mean": np.random.uniform(0, 500, n_samples),
        "Bwd Packet Length Std": np.random.uniform(0, 200, n_samples),
        "Flow Bytes/s": np.random.exponential(10000, n_samples),
        "Flow Packets/s": np.random.exponential(100, n_samples),
        "Flow IAT Mean": np.random.exponential(100, n_samples),
        "Flow IAT Std": np.random.exponential(50, n_samples),
        "Fwd IAT Mean": np.random.exponential(100, n_samples),
        "Fwd IAT Std": np.random.exponential(50, n_samples),
        "Bwd IAT Mean": np.random.exponential(100, n_samples),
        "Bwd IAT Std": np.random.exponential(50, n_samples),
        "Label": np.random.choice(["BENIGN", "DoS", "PortScan"], n_samples, p=[0.8, 0.1, 0.1]),
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_csv_file(sample_dataframe, temp_dir):
    """Create sample CSV file for testing."""
    csv_path = temp_dir / "sample_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_graph_data():
    """Create sample PyTorch Geometric graph for testing."""
    # Create a small graph
    num_nodes = 50
    num_edges = 150
    num_features = 15
    num_classes = 2

    # Node features
    x = torch.randn(num_nodes, num_features)

    # Edge index (directed graph)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Edge features
    edge_attr = torch.randn(num_edges, 8)

    # Labels (binary classification: benign vs attack)
    y = torch.randint(0, num_classes, (num_nodes,))

    # Create masks for train/val/test splits
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:30] = True
    val_mask[30:40] = True
    test_mask[40:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data


@pytest.fixture
def sample_model():
    """Create sample GNN model for testing."""
    from src.models.gnn_models import GraphConvolutionalNetwork

    model = GraphConvolutionalNetwork(
        input_dim=15, hidden_dim=32, output_dim=2, num_layers=2, dropout=0.5
    )
    return model


@pytest.fixture
def mock_checkpoint(temp_dir, sample_model):
    """Create mock model checkpoint."""
    checkpoint_path = temp_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": sample_model.state_dict(),
            "epoch": 10,
            "val_loss": 0.5,
            "val_f1": 0.85,
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_environment_vars(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "DATA_RAW_DIR": "/tmp/test_data/raw",
        "DATA_PROCESSED_DIR": "/tmp/test_data/processed",
        "LOG_LEVEL": "DEBUG",
        "DEVICE": "cpu",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def gpu_available():
    """Check if GPU is available for GPU tests."""
    return torch.cuda.is_available()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if GPU is not available."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
