"""
Unit tests for GNN models
"""

import pytest
import torch

from src.models.gnn_models import (
    GraphAttentionNetwork,
    GraphConvolutionalNetwork,
    GraphSAGE,
    HybridGNN,
    create_model,
)


@pytest.mark.unit
class TestGCN:
    """Tests for Graph Convolutional Network."""

    def test_model_creation(self):
        """Test GCN model creation."""
        model = GraphConvolutionalNetwork(
            input_dim=10, hidden_dim=32, output_dim=2, num_layers=2, dropout=0.5
        )
        assert model is not None
        assert hasattr(model, "convs")
        assert len(model.convs) == 2

    def test_forward_pass(self, sample_graph_data):
        """Test forward pass through GCN."""
        model = GraphConvolutionalNetwork(
            input_dim=sample_graph_data.x.shape[1], hidden_dim=32, output_dim=2
        )
        output = model(sample_graph_data.x, sample_graph_data.edge_index)
        assert output.shape == (sample_graph_data.num_nodes, 2)
        assert not torch.isnan(output).any()

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = GraphConvolutionalNetwork(input_dim=10, hidden_dim=32, output_dim=2, num_layers=3)
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        assert num_params < 1e6  # Should be reasonable for small model


@pytest.mark.unit
class TestGAT:
    """Tests for Graph Attention Network."""

    def test_model_creation(self):
        """Test GAT model creation."""
        model = GraphAttentionNetwork(
            input_dim=10, hidden_dim=32, output_dim=2, num_heads=4, num_layers=2
        )
        assert model is not None
        assert hasattr(model, "convs")

    def test_forward_pass(self, sample_graph_data):
        """Test forward pass through GAT."""
        model = GraphAttentionNetwork(
            input_dim=sample_graph_data.x.shape[1], hidden_dim=32, output_dim=2, num_heads=2
        )
        output = model(sample_graph_data.x, sample_graph_data.edge_index)
        assert output.shape == (sample_graph_data.num_nodes, 2)
        assert not torch.isnan(output).any()


@pytest.mark.unit
class TestGraphSAGE:
    """Tests for GraphSAGE."""

    @pytest.mark.parametrize("aggregator", ["mean", "max"])
    def test_different_aggregators(self, sample_graph_data, aggregator):
        """Test GraphSAGE with different aggregators."""
        model = GraphSAGE(
            input_dim=sample_graph_data.x.shape[1],
            hidden_dim=32,
            output_dim=2,
            aggregator=aggregator,
        )
        output = model(sample_graph_data.x, sample_graph_data.edge_index)
        assert output.shape == (sample_graph_data.num_nodes, 2)
        assert not torch.isnan(output).any()


@pytest.mark.unit
class TestHybridGNN:
    """Tests for Hybrid GNN."""

    def test_model_creation(self):
        """Test Hybrid model creation."""
        model = HybridGNN(input_dim=10, hidden_dim=32, output_dim=2)
        assert model is not None
        assert hasattr(model, "gcn")
        assert hasattr(model, "gat")
        assert hasattr(model, "sage")

    def test_forward_pass(self, sample_graph_data):
        """Test forward pass through Hybrid model."""
        model = HybridGNN(input_dim=sample_graph_data.x.shape[1], hidden_dim=32, output_dim=2)
        output = model(sample_graph_data.x, sample_graph_data.edge_index)
        assert output.shape == (sample_graph_data.num_nodes, 2)


@pytest.mark.unit
class TestModelFactory:
    """Tests for model factory function."""

    @pytest.mark.parametrize("model_type", ["gcn", "gat", "sage", "hybrid"])
    def test_create_all_model_types(self, model_type):
        """Test creating all model types through factory."""
        model = create_model(
            model_type=model_type, input_dim=10, hidden_dim=32, output_dim=2, num_layers=2
        )
        assert model is not None

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError):
            create_model(model_type="invalid", input_dim=10, hidden_dim=32, output_dim=2)
