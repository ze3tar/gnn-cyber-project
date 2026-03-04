"""
Optimized Training Pipeline with Performance Enhancements
- Mini-batch training with graph sampling
- Mixed precision training
- Gradient accumulation
- Optimized memory management

Author: Mohamed salem eddah
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from ..utils.logging import get_logger

logger = get_logger(__name__)


class OptimizedGNNTrainer:
    """
    Optimized GNN trainer with advanced performance features:
    - Mini-batch training with neighbor sampling
    - Mixed precision (FP16) training
    - Gradient accumulation for larger effective batch sizes
    - Memory-efficient graph sampling
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 5e-4,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize optimized trainer.

        Args:
            model: GNN model to train
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            use_mixed_precision: Use automatic mixed precision (FP16)
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_mixed_precision = use_mixed_precision and device == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None

        # Training components
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = None
        self.criterion = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
        }

        logger.info(
            f"Initialized OptimizedGNNTrainer on {device} "
            f"(Mixed Precision: {self.use_mixed_precision})"
        )

    def create_mini_batch_loaders(
        self,
        data: Data,
        batch_size: int = 512,
        num_neighbors: List[int] = [15, 10, 5],
        num_workers: int = 4,
    ) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
        """
        Create optimized mini-batch data loaders with neighbor sampling.

        Args:
            data: PyTorch Geometric Data object
            batch_size: Batch size for training
            num_neighbors: Number of neighbors to sample per layer [L1, L2, L3]
            num_workers: Number of parallel data loading workers

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Training loader with neighbor sampling
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Validation loader (can use larger batch since no backprop)
        val_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size * 2,
            input_nodes=data.val_mask,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
        )

        # Test loader
        test_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size * 2,
            input_nodes=data.test_mask,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
        )

        logger.info(
            f"Created mini-batch loaders: batch_size={batch_size}, "
            f"neighbors={num_neighbors}, workers={num_workers}"
        )

        return train_loader, val_loader, test_loader

    def setup_training(
        self, class_weights: Optional[torch.Tensor] = None, scheduler_type: str = "plateau"
    ):
        """Setup loss function and learning rate scheduler."""
        # Setup loss function with class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Setup scheduler
        if scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )

        logger.info(f"Training setup complete: scheduler={scheduler_type}")

    def train_epoch(self, train_loader: NeighborLoader) -> Tuple[float, float, float]:
        """
        Train for one epoch using mini-batch sampling.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy, f1_score)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        self.optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            batch = batch.to(self.device)

            # Mixed precision forward pass
            if self.use_mixed_precision:
                with autocast():
                    out = self.model(batch.x, batch.edge_index)
                    # Only compute loss for nodes in the current batch
                    loss = self.criterion(out[: batch.batch_size], batch.y[: batch.batch_size])
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Update weights after accumulating gradients
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard forward pass
                out = self.model(batch.x, batch.edge_index)
                loss = self.criterion(out[: batch.batch_size], batch.y[: batch.batch_size])
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (i + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

            # Collect predictions
            pred = out[: batch.batch_size].argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y[: batch.batch_size].cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return avg_loss, accuracy, f1

    @torch.no_grad()
    def evaluate(self, loader: NeighborLoader) -> Tuple[float, float, float]:
        """
        Evaluate model on validation/test set.

        Args:
            loader: Data loader for evaluation

        Returns:
            Tuple of (average_loss, accuracy, f1_score)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(self.device)

            if self.use_mixed_precision:
                with autocast():
                    out = self.model(batch.x, batch.edge_index)
                    loss = self.criterion(out[: batch.batch_size], batch.y[: batch.batch_size])
            else:
                out = self.model(batch.x, batch.edge_index)
                loss = self.criterion(out[: batch.batch_size], batch.y[: batch.batch_size])

            total_loss += loss.item()

            pred = out[: batch.batch_size].argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y[: batch.batch_size].cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return avg_loss, accuracy, f1

    def train(
        self,
        data: Data,
        num_epochs: int = 100,
        batch_size: int = 512,
        num_neighbors: List[int] = [15, 10, 5],
        patience: int = 30,
        save_dir: Optional[Path] = None,
        checkpoint_name: str = "best_model.pt",
    ) -> Dict:
        """
        Train model with mini-batch sampling and early stopping.

        Args:
            data: PyTorch Geometric Data object
            num_epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            num_neighbors: Number of neighbors to sample per layer
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
            checkpoint_name: Name of the checkpoint file

        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_mini_batch_loaders(
            data, batch_size=batch_size, num_neighbors=num_neighbors
        )

        best_val_f1 = 0.0
        epochs_no_improve = 0

        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
            )

            # Early stopping and checkpointing
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0

                if save_dir:
                    save_path = Path(save_dir) / checkpoint_name
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_f1": val_f1,
                            "val_loss": val_loss,
                        },
                        save_path,
                    )
                    logger.info(f"Saved best model (F1: {val_f1:.4f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info(f"Training complete! Best Val F1: {best_val_f1:.4f}")
        return self.history
