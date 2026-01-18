"""
Integration tests for the full pipeline
"""

import pytest
import torch
from pathlib import Path

from src.config import AppConfig


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    def test_config_loading_integration(self, temp_dir):
        """Test configuration loading and validation."""
        config = AppConfig(
            data={"raw_dir": temp_dir / "raw", "processed_dir": temp_dir / "processed"}
        )
        config.setup_directories()

        assert (temp_dir / "processed").exists()
        assert config.data.raw_dir == temp_dir / "raw"

    @pytest.mark.slow
    def test_small_training_run(self, sample_graph_data, temp_dir):
        """Test a small training run end-to-end."""
        from src.models.gnn_models import create_model
        from src.training.trainer import GNNTrainer

        # Create model
        model = create_model(
            "gcn",
            input_dim=sample_graph_data.x.shape[1],
            hidden_dim=16,
            output_dim=2,
            num_layers=2,
        )

        # Setup trainer
        trainer = GNNTrainer(model, device="cpu", learning_rate=0.01, patience=5)
        trainer.setup_training(class_weights=None, scheduler_type="plateau")

        # Train for a few epochs
        history = trainer.train(
            sample_graph_data, num_epochs=3, verbose=False, save_dir=temp_dir
        )

        # Verify training completed
        assert len(history["train_loss"]) == 3
        assert all(isinstance(loss, float) for loss in history["train_loss"])

    def test_model_save_and_load(self, sample_model, temp_dir):
        """Test saving and loading model checkpoints."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        # Save model
        torch.save(
            {"model_state_dict": sample_model.state_dict(), "epoch": 5}, checkpoint_path
        )

        # Load model
        checkpoint = torch.load(checkpoint_path)
        assert "model_state_dict" in checkpoint
        assert checkpoint["epoch"] == 5
