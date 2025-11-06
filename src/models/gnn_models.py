"""
Graph Neural Network Model Architectures for Cyber Threat Prediction
Implements multiple state-of-the-art GNN architectures

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph Convolutional Network (GCN) for node classification.
    
    Based on Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 use_batch_norm: bool = True):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"Initialized GCN: {num_layers} layers, hidden_dim={hidden_dim}")
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector for graph-level tasks
            
        Returns:
            Node embeddings and predictions
        """
        # GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batch_norms:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        if self.batch_norms:
            x = self.batch_norms[-1](x)
        
        embeddings = x
        
        # Classification
        out = self.classifier(x)
        
        return out, embeddings


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT) with multi-head attention.
    
    Based on Veličković et al. (2018): "Graph Attention Networks"
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.5,
                 use_batch_norm: bool = True):
        """
        Initialize GAT model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Number of output classes
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(GraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim * num_heads))
        
        # Output layer (concatenate heads)
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"Initialized GAT: {num_layers} layers, {num_heads} heads, hidden_dim={hidden_dim}")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass with attention mechanism."""
        # GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batch_norms:
                x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        if self.batch_norms:
            x = self.batch_norms[-1](x)
        
        embeddings = x
        
        # Classification
        out = self.classifier(x)
        
        return out, embeddings


class GraphSAGE(nn.Module):
    """
    GraphSAGE with multiple aggregation strategies.
    
    Based on Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs"
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 aggregator: str = 'mean'):
        """
        Initialize GraphSAGE model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Number of output classes
            num_layers: Number of SAGE layers
            dropout: Dropout probability
            aggregator: Aggregation method ('mean', 'max', 'lstm')
        """
        super(GraphSAGE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"Initialized GraphSAGE: {num_layers} layers, aggregator={aggregator}")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        
        embeddings = x
        out = self.classifier(x)
        
        return out, embeddings


class TemporalGNN(nn.Module):
    """
    Temporal GNN combining spatial GNN with LSTM for time-series prediction.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 gnn_layers: int = 2,
                 lstm_layers: int = 2,
                 dropout: float = 0.5,
                 gnn_type: str = 'gcn'):
        """
        Initialize Temporal GNN.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Number of output classes
            gnn_layers: Number of GNN layers
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            gnn_type: Type of GNN ('gcn', 'gat', 'sage')
        """
        super(TemporalGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        
        # Spatial GNN component
        if gnn_type == 'gcn':
            self.gnn = GraphConvolutionalNetwork(
                input_dim, hidden_dim, hidden_dim, gnn_layers, dropout, use_batch_norm=True
            )
        elif gnn_type == 'gat':
            self.gnn = GraphAttentionNetwork(
                input_dim, hidden_dim, hidden_dim, gnn_layers, num_heads=4, dropout=dropout
            )
        elif gnn_type == 'sage':
            self.gnn = GraphSAGE(
                input_dim, hidden_dim, hidden_dim, gnn_layers, dropout
            )
        
        # Temporal LSTM component
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"Initialized TemporalGNN: {gnn_type.upper()} + LSTM")
    
    def forward(self, x_sequence, edge_index_sequence, batch=None):
        """
        Forward pass for temporal graphs.
        
        Args:
            x_sequence: List of node features for each timestep
            edge_index_sequence: List of edge indices for each timestep
            batch: Batch vector
            
        Returns:
            Predictions for the sequence
        """
        # Process each graph snapshot with GNN
        gnn_outputs = []
        for x_t, edge_index_t in zip(x_sequence, edge_index_sequence):
            _, emb = self.gnn(x_t, edge_index_t, batch)
            
            # Aggregate node embeddings (mean pooling)
            if batch is not None:
                emb = global_mean_pool(emb, batch)
            else:
                emb = emb.mean(dim=0, keepdim=True)
            
            gnn_outputs.append(emb)
        
        # Stack temporal embeddings
        temporal_emb = torch.stack(gnn_outputs, dim=1)  # [batch, seq_len, hidden_dim]
        
        # LSTM for temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(temporal_emb)
        
        # Use last hidden state for prediction
        final_emb = h_n[-1]  # [batch, hidden_dim]
        
        # Classification
        out = self.classifier(final_emb)
        
        return out, final_emb


class HybridGNN(nn.Module):
    """
    Hybrid GNN combining multiple GNN architectures for robust predictions.
    Ensemble approach for improved performance.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        """
        Initialize Hybrid GNN with multiple architectures.
        """
        super(HybridGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Multiple GNN branches
        self.gcn = GraphConvolutionalNetwork(
            input_dim, hidden_dim, hidden_dim, num_layers, dropout
        )
        self.gat = GraphAttentionNetwork(
            input_dim, hidden_dim // 2, hidden_dim, num_layers, num_heads=2, dropout=dropout
        )
        self.sage = GraphSAGE(
            input_dim, hidden_dim, hidden_dim, num_layers, dropout
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fusion_bn = BatchNorm1d(hidden_dim)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        logger.info("Initialized HybridGNN with GCN + GAT + GraphSAGE")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass combining multiple GNN outputs."""
        # Get embeddings from each GNN
        _, gcn_emb = self.gcn(x, edge_index, batch)
        _, gat_emb = self.gat(x, edge_index, batch)
        _, sage_emb = self.sage(x, edge_index, batch)
        
        # Concatenate embeddings
        combined_emb = torch.cat([gcn_emb, gat_emb, sage_emb], dim=-1)
        
        # Fusion
        fused = self.fusion(combined_emb)
        fused = self.fusion_bn(fused)
        fused = F.relu(fused)
        
        # Classification
        out = self.classifier(fused)
        
        return out, fused


def create_model(model_type: str,
                input_dim: int,
                hidden_dim: int = 128,
                output_dim: int = 2,
                num_layers: int = 3,
                dropout: float = 0.5,
                **kwargs) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: 'gcn', 'gat', 'sage', 'temporal', or 'hybrid'
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (number of classes)
        num_layers: Number of GNN layers
        dropout: Dropout probability
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized GNN model
    """
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return GraphConvolutionalNetwork(
            input_dim, hidden_dim, output_dim, num_layers, dropout, **kwargs
        )
    elif model_type == 'gat':
        return GraphAttentionNetwork(
            input_dim, hidden_dim, output_dim, num_layers, 
            num_heads=kwargs.get('num_heads', 4), dropout=dropout, **kwargs
        )
    elif model_type == 'sage':
        return GraphSAGE(
            input_dim, hidden_dim, output_dim, num_layers, dropout,
            aggregator=kwargs.get('aggregator', 'mean')
        )
    elif model_type == 'temporal':
        return TemporalGNN(
            input_dim, hidden_dim, output_dim, num_layers,
            lstm_layers=kwargs.get('lstm_layers', 2), dropout=dropout,
            gnn_type=kwargs.get('gnn_type', 'gcn')
        )
    elif model_type == 'hybrid':
        return HybridGNN(
            input_dim, hidden_dim, output_dim, num_layers, dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    input_dim = 15  # From node features
    hidden_dim = 128
    output_dim = 2
    
    print("Testing GNN Models...")
    print("="*60)
    
    # Test each model
    models = {
        'GCN': create_model('gcn', input_dim, hidden_dim, output_dim),
        'GAT': create_model('gat', input_dim, hidden_dim, output_dim),
        'GraphSAGE': create_model('sage', input_dim, hidden_dim, output_dim),
        'Hybrid': create_model('hybrid', input_dim, hidden_dim, output_dim)
    }
    
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {num_params:,} parameters")
    
    print("\n✅ All models initialized successfully!")