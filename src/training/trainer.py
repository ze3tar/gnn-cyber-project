"""
Training and Evaluation Pipeline for GNN-based Cyber Threat Prediction
Implements comprehensive training with early stopping, checkpointing, and evaluation

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNTrainer:
    """
    Comprehensive training pipeline for GNN models with advanced features:
    - Early stopping with patience
    - Model checkpointing
    - Learning rate scheduling
    - Class imbalance handling
    - Comprehensive metrics tracking
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 5e-4,
                 patience: int = 20,
                 min_delta: float = 1e-4):
        """
        Initialize trainer.
        
        Args:
            model: GNN model to train
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization coefficient
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0
        self.best_model_state = None
        
        logger.info(f"Initialized trainer on device: {device}")
    
    def setup_training(self,
                      class_weights: Optional[torch.Tensor] = None,
                      scheduler_type: str = 'plateau'):
        """
        Setup optimizer, loss function, and scheduler.
        
        Args:
            class_weights: Weights for imbalanced classes
            scheduler_type: 'plateau' or 'cosine'
        """
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
        
        logger.info("Training setup complete")
    
    def compute_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        unique, counts = torch.unique(labels, return_counts=True)
        total = len(labels)
        weights = total / (len(unique) * counts.float())
        logger.info(f"Class weights: {weights}")
        return weights
    
    def train_epoch(self, data: Data) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out, _ = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        
        # Compute loss
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask].to(self.device))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute metrics
        pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
        true = data.y[data.train_mask].cpu().numpy()
        
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average='binary')
        
        return loss.item(), acc, f1
    
    @torch.no_grad()
    def evaluate(self, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on given mask.
        
        Args:
            data: PyTorch Geometric Data object
            mask: Boolean mask for evaluation
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        # Forward pass
        out, embeddings = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        
        # Predictions
        pred_probs = torch.softmax(out[mask], dim=1)
        pred = pred_probs.argmax(dim=1).cpu().numpy()
        pred_probs = pred_probs[:, 1].cpu().numpy()  # Probability of positive class
        true = data.y[mask].cpu().numpy()
        
        # Compute loss
        loss = self.criterion(out[mask], data.y[mask].to(self.device))
        
        # Metrics
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy_score(true, pred),
            'precision': precision_score(true, pred, average='binary', zero_division=0),
            'recall': recall_score(true, pred, average='binary', zero_division=0),
            'f1': f1_score(true, pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(true, pred_probs) if len(np.unique(true)) > 1 else 0.0
        }
        
        return metrics
    
    def train(self,
             data: Data,
             num_epochs: int = 200,
             verbose: bool = True,
             save_dir: Optional[str] = None) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            data: PyTorch Geometric Data object with train/val/test masks
            num_epochs: Maximum number of epochs
            verbose: Whether to print progress
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(data)
            
            # Validate
            val_metrics = self.evaluate(data, data.val_mask)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.best_val_f1 = val_metrics['f1']
                self.epochs_no_improve = 0
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                if save_dir:
                    checkpoint_path = save_dir / 'best_model.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_f1': val_metrics['f1']
                    }, checkpoint_path)
            else:
                self.epochs_no_improve += 1
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Early stopping
            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with val_loss={self.best_val_loss:.4f}, val_f1={self.best_val_f1:.4f}")
        
        return self.history
    
    def test(self, data: Data) -> Dict[str, float]:
        """
        Evaluate on test set and generate comprehensive metrics.
        
        Args:
            data: PyTorch Geometric Data object with test_mask
            
        Returns:
            Dictionary of test metrics
        """
        logger.info("Evaluating on test set...")
        
        test_metrics = self.evaluate(data, data.test_mask)
        
        # Get predictions for confusion matrix
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
            true = data.y[data.test_mask].cpu().numpy()
        
        # Confusion matrix
        cm = confusion_matrix(true, pred)
        test_metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(true, pred, output_dict=True)
        test_metrics['classification_report'] = report
        
        # Log results
        logger.info("Test Results:")
        for key, value in test_metrics.items():
            if key not in ['confusion_matrix', 'classification_report']:
                logger.info(f"  {key}: {value:.4f}")
        
        return test_metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Accuracy', alpha=0.8)
        axes[0, 1].plot(self.history['val_acc'], label='Val Accuracy', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1', alpha=0.8)
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['learning_rates'], alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, test_metrics: Dict, save_dir: str):
        """Save all results to files."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_to_save = {k: v for k, v in test_metrics.items() 
                          if k not in ['confusion_matrix']}
        
        with open(save_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot and save figures
        self.plot_training_history(save_dir / 'training_history.png')
        
        if 'confusion_matrix' in test_metrics:
            cm = np.array(test_metrics['confusion_matrix'])
            self.plot_confusion_matrix(cm, save_dir / 'confusion_matrix.png')
        
        logger.info(f"Results saved to {save_dir}")


def prepare_data_splits(data: Data, 
                       train_ratio: float = 0.6,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.2,
                       random_state: int = 42) -> Data:
    """
    Create train/val/test splits for node classification.
    
    Args:
        data: PyTorch Geometric Data object
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed
        
    Returns:
        Data object with train_mask, val_mask, test_mask
    """
    assert train_ratio + val_ratio + test_ratio == 1.0
    
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    
    # Split indices
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=random_state, 
        stratify=data.y.numpy()
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=random_state,
        stratify=data.y[temp_idx].numpy()
    )
    
    # Create masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    logger.info(f"Data splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    return data


if __name__ == "__main__":
    from gnn_models import create_model
    
    # Load data
    data_path = "data/graphs/cicids_pyg_data.pt"
    
    if not Path(data_path).exists():
        logger.error(f"Data not found at {data_path}")
        logger.error("Please run the graph construction script first!")
        exit(1)
    
    logger.info(f"Loading data from {data_path}...")
    data = torch.load(data_path)
    
    # Prepare splits
    data = prepare_data_splits(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(
        'gcn',
        input_dim=data.x.shape[1],
        hidden_dim=128,
        output_dim=2,
        num_layers=3,
        dropout=0.5
    )
    
    # Initialize trainer
    trainer = GNNTrainer(model, device=device, learning_rate=0.001, patience=30)
    
    # Compute class weights
    class_weights = trainer.compute_class_weights(data.y)
    trainer.setup_training(class_weights=class_weights, scheduler_type='plateau')
    
    # Train
    history = trainer.train(data, num_epochs=200, verbose=True, save_dir='models/checkpoints')
    
    # Test
    test_metrics = trainer.test(data)
    
    # Save results
    trainer.save_results(test_metrics, 'results/gnn_evaluation')
    
    logger.info("\n✅ Training and evaluation complete!")