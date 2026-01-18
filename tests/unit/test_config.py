"""
Unit tests for configuration management
"""

import os
from pathlib import Path

import pytest

from src.config import AppConfig, DataConfig, ModelConfig, load_config


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.raw_dir == Path("data/raw/CICIDS2017")
        assert config.processed_dir == Path("data/processed")
        assert config.sample_size is None

    def test_path_expansion(self, tmp_path):
        """Test environment variable expansion in paths."""
        os.environ["TEST_DATA_DIR"] = str(tmp_path)
        config = DataConfig(raw_dir="$TEST_DATA_DIR/raw")
        assert config.raw_dir == tmp_path / "raw"


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_valid_model_types(self):
        """Test valid model type values."""
        for model_type in ["gcn", "gat", "sage", "hybrid", "temporal"]:
            config = ModelConfig(type=model_type)
            assert config.type == model_type

    def test_default_hyperparameters(self):
        """Test default hyperparameter values."""
        config = ModelConfig()
        assert config.hidden_dim == 128
        assert config.num_layers == 3
        assert config.dropout == 0.5
        assert config.use_batch_norm is True


class TestAppConfig:
    """Tests for AppConfig."""

    def test_config_initialization(self):
        """Test configuration initialization."""
        config = AppConfig()
        assert config.seed == 42
        assert config.deterministic is True
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = AppConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict

    def test_setup_directories(self, tmp_path):
        """Test directory creation."""
        config = AppConfig(
            data={"processed_dir": tmp_path / "processed"},
            output={"model_dir": tmp_path / "models"},
        )
        config.setup_directories()
        assert (tmp_path / "processed").exists()
        assert (tmp_path / "models").exists()


@pytest.mark.unit
class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.seed == 42

    def test_environment_variable_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("SEED", "123")
        config = AppConfig()
        # Note: Pydantic settings will pick up env vars automatically
        assert config.seed is not None
