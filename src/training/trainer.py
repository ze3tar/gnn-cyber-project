"""
Training and Evaluation Pipeline for GNN-based Cyber Threat Prediction
Implements comprehensive training with early stopping, checkpointing,
AMP autocast, TensorBoard logging, temporal training, and SHAP analysis.

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
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

# TensorBoard — optional (Phase 4)
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

# SHAP — optional (Phase 5)
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


class GNNTrainer:
    """
    Comprehensive training pipeline for GNN models with:
    - Early stopping with patience
    - Model checkpointing
    - Learning rate scheduling
    - Class imbalance handling via weighted cross-entropy
    - Mixed-precision training via AMP (Phase 2)
    - TensorBoard logging (Phase 4)
    - Temporal GNN training loop (Phase 3)
    - SHAP feature importance analysis (Phase 5)
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 5e-4,
                 patience: int = 20,
                 min_delta: float = 1e-4,
                 use_amp: bool = False,
                 tensorboard_dir: Optional[str] = None):
        """
        Initialize trainer.

        Args:
            model: GNN model to train
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization coefficient
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            use_amp: Enable automatic mixed precision (Phase 2; CUDA only)
            tensorboard_dir: Directory for TensorBoard logs (Phase 4)
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta

        # AMP (Phase 2)
        self.use_amp = use_amp and device != 'cpu'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled")

        # TensorBoard (Phase 4)
        self.writer = None
        if tensorboard_dir and _TENSORBOARD_AVAILABLE:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(tensorboard_dir) / run_name
            self.writer = SummaryWriter(log_dir=str(log_path))
            logger.info(f"TensorBoard logging to: {log_path}")
        elif tensorboard_dir and not _TENSORBOARD_AVAILABLE:
            logger.warning("tensorboard not installed. Run: pip install tensorboard")

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
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )

        logger.info("Training setup complete")

    def compute_class_weights(self, labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for imbalanced datasets.
        If fewer classes are present in labels than num_classes, the weight
        tensor is padded with 1.0 so CrossEntropyLoss never gets a size mismatch.
        """
        unique, counts = torch.unique(labels, return_counts=True)
        total = len(labels)
        observed_weights = total / (len(unique) * counts.float())

        # Build full-size weight tensor (one entry per expected class)
        weights = torch.ones(num_classes, dtype=torch.float)
        for cls_idx, cls_id in enumerate(unique):
            if cls_id.item() < num_classes:
                weights[cls_id.item()] = observed_weights[cls_idx]

        logger.info(f"Class weights ({len(unique)}/{num_classes} classes present): {weights}")
        return weights

    # ------------------------------------------------------------------
    # Single-epoch train / evaluate
    # ------------------------------------------------------------------

    def train_epoch(self, data: Data) -> Tuple[float, float, float]:
        """
        Train for one epoch with optional AMP.

        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.train()
        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                out, _ = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                loss = self.criterion(out[data.train_mask], data.y[data.train_mask].to(self.device))
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out, _ = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask].to(self.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
        true = data.y[data.train_mask].cpu().numpy()

        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average='binary', zero_division=0)

        return loss.item(), acc, f1

    @torch.no_grad()
    def evaluate(self, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on given node mask."""
        self.model.eval()

        out, embeddings = self.model(data.x.to(self.device), data.edge_index.to(self.device))

        pred_probs = torch.softmax(out[mask], dim=1)
        pred = pred_probs.argmax(dim=1).cpu().numpy()
        pred_probs_np = pred_probs[:, 1].cpu().numpy()
        true = data.y[mask].cpu().numpy()

        loss = self.criterion(out[mask], data.y[mask].to(self.device))

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy_score(true, pred),
            'precision': precision_score(true, pred, average='binary', zero_division=0),
            'recall': recall_score(true, pred, average='binary', zero_division=0),
            'f1': f1_score(true, pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(true, pred_probs_np) if len(np.unique(true)) > 1 else 0.0
        }

        return metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self,
              data: Data,
              num_epochs: int = 200,
              verbose: bool = True,
              save_dir: Optional[str] = None) -> Dict:
        """
        Full training loop with early stopping and TensorBoard logging.

        Args:
            data: PyTorch Geometric Data object with train/val/test masks
            num_epochs: Maximum number of epochs
            verbose: Whether to print progress
            save_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(data)
            val_metrics = self.evaluate(data, data.val_mask)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # TensorBoard logging (Phase 4)
            if self.writer:
                self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_metrics['loss']}, epoch)
                self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_metrics['accuracy']}, epoch)
                self.writer.add_scalars('F1', {'train': train_f1, 'val': val_metrics['f1']}, epoch)
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)

            # LR scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.best_val_f1 = val_metrics['f1']
                self.epochs_no_improve = 0
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                if save_dir:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_f1': val_metrics['f1']
                    }, save_dir / 'best_model.pt')
            else:
                self.epochs_no_improve += 1

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model: val_loss={self.best_val_loss:.4f}, val_f1={self.best_val_f1:.4f}")

        if self.writer:
            self.writer.close()

        return self.history

    # ------------------------------------------------------------------
    # Temporal GNN training (Phase 3)
    # ------------------------------------------------------------------

    def train_temporal(self,
                       snapshots: List[Data],
                       num_epochs: int = 100,
                       sequence_len: int = 5,
                       save_dir: Optional[str] = None) -> Dict:
        """
        Training loop for TemporalGNN operating on graph snapshot sequences.

        The model receives a sliding window of `sequence_len` consecutive
        snapshots and predicts the attack label for the final snapshot.

        Args:
            snapshots: Ordered list of PyG Data objects (one per time window)
            num_epochs: Training epochs
            sequence_len: Number of snapshots per sequence
            save_dir: Checkpoint directory

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting temporal training: {len(snapshots)} snapshots, "
                    f"sequence_len={sequence_len}, {num_epochs} epochs")

        if len(snapshots) < sequence_len + 1:
            raise ValueError(
                f"Need at least {sequence_len + 1} snapshots for temporal training, "
                f"got {len(snapshots)}"
            )

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        history = {'train_loss': [], 'val_loss': []}
        num_sequences = len(snapshots) - sequence_len
        split = int(num_sequences * 0.8)

        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []

            for start in range(split):
                seq = snapshots[start: start + sequence_len]
                x_seq = [s.x.to(self.device) for s in seq]
                ei_seq = [s.edge_index.to(self.device) for s in seq]

                # Target: majority label in the next snapshot
                target_snap = snapshots[start + sequence_len]
                target = torch.tensor(
                    [int(target_snap.y.float().mean().item() > 0.5)],
                    dtype=torch.long
                ).to(self.device)

                self.optimizer.zero_grad()
                out, _ = self.model(x_seq, ei_seq)
                loss = self.criterion(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for start in range(split, num_sequences):
                    seq = snapshots[start: start + sequence_len]
                    x_seq = [s.x.to(self.device) for s in seq]
                    ei_seq = [s.edge_index.to(self.device) for s in seq]

                    target_snap = snapshots[start + sequence_len]
                    target = torch.tensor(
                        [int(target_snap.y.float().mean().item() > 0.5)],
                        dtype=torch.long
                    ).to(self.device)

                    out, _ = self.model(x_seq, ei_seq)
                    loss = self.criterion(out, target)
                    val_losses.append(loss.item())

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_losses)) if val_losses else 0.0
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)

            if self.writer:
                self.writer.add_scalar('Temporal/TrainLoss', avg_train, epoch)
                self.writer.add_scalar('Temporal/ValLoss', avg_val, epoch)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Temporal Epoch {epoch + 1}/{num_epochs} | "
                            f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

        if self.writer:
            self.writer.close()

        return history

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------

    def test(self, data: Data) -> Dict[str, float]:
        """Evaluate on test set and generate comprehensive metrics."""
        logger.info("Evaluating on test set...")

        test_metrics = self.evaluate(data, data.test_mask)

        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
            true = data.y[data.test_mask].cpu().numpy()

        cm = confusion_matrix(true, pred)
        test_metrics['confusion_matrix'] = cm.tolist()

        report = classification_report(true, pred, output_dict=True)
        test_metrics['classification_report'] = report

        logger.info("Test Results:")
        for key, value in test_metrics.items():
            if key not in ('confusion_matrix', 'classification_report'):
                logger.info(f"  {key}: {value:.4f}")

        return test_metrics

    # ------------------------------------------------------------------
    # SHAP feature importance (Phase 5)
    # ------------------------------------------------------------------

    def compute_shap_importance(self,
                                data: Data,
                                num_background: int = 50,
                                num_test: int = 50,
                                save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Compute SHAP feature importance for the GNN node embeddings.

        Uses a background sample of node features to build a DeepExplainer
        and evaluates on a test sample, returning mean |SHAP| per feature.

        Args:
            data: PyG Data object
            num_background: Background samples for SHAP
            num_test: Test samples to explain
            save_path: If set, saves a bar chart of feature importances

        Returns:
            Array of mean absolute SHAP values per feature, or None if SHAP unavailable.
        """
        if not _SHAP_AVAILABLE:
            logger.warning("shap not installed. Run: pip install shap")
            return None

        logger.info("Computing SHAP feature importance...")

        self.model.eval()
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        # Wrapper: pass only node features (treat edge_index as fixed context)
        class _GNNWrapper(nn.Module):
            def __init__(self, model, edge_index):
                super().__init__()
                self._model = model
                self._ei = edge_index

            def forward(self, x_in):
                out, _ = self._model(x_in, self._ei)
                return out

        wrapper = _GNNWrapper(self.model, edge_index)

        background = x[:num_background]
        test_data = x[data.test_mask][:num_test]

        try:
            explainer = shap.DeepExplainer(wrapper, background)
            shap_values = explainer.shap_values(test_data)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return None

        # shap_values is list [class0, class1]; use class-1 (attack)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs_shap = np.abs(sv).mean(axis=0)

        logger.info("Top 10 features by SHAP importance:")
        for rank, (idx, val) in enumerate(
            sorted(enumerate(mean_abs_shap), key=lambda x: x[1], reverse=True)[:10]
        ):
            logger.info(f"  [{rank + 1}] Feature {idx}: {val:.4f}")

        if save_path:
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(mean_abs_shap)), mean_abs_shap, color='steelblue')
            plt.yticks(range(len(mean_abs_shap)),
                       [f"Feature {i}" for i in range(len(mean_abs_shap))])
            plt.xlabel("Mean |SHAP|")
            plt.title("GNN Feature Importance (SHAP)")
            plt.tight_layout()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP importance chart saved to {save_path}")

        return mean_abs_shap

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training/validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', alpha=0.8)
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', alpha=0.8)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['train_f1'], label='Train F1', alpha=0.8)
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', alpha=0.8)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history['learning_rates'], alpha=0.8)
        axes[1, 1].set_title('Learning Rate')
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
        """Plot confusion matrix heatmap."""
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
        """Save metrics, training history, and figures to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        serializable = {k: v for k, v in test_metrics.items()
                        if k not in ('confusion_matrix', 'classification_report')}
        with open(save_dir / 'test_metrics.json', 'w') as f:
            json.dump(serializable, f, indent=2)

        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        self.plot_training_history(str(save_dir / 'training_history.png'))

        if 'confusion_matrix' in test_metrics:
            cm = np.array(test_metrics['confusion_matrix'])
            self.plot_confusion_matrix(cm, str(save_dir / 'confusion_matrix.png'))

        logger.info(f"Results saved to {save_dir}")


# ------------------------------------------------------------------
# Data split utilities
# ------------------------------------------------------------------

def prepare_data_splits(data: Data,
                        train_ratio: float = 0.6,
                        val_ratio: float = 0.2,
                        test_ratio: float = 0.2,
                        random_state: int = 42) -> Data:
    """
    Create stratified train/val/test node splits.

    Args:
        data: PyTorch Geometric Data object
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed

    Returns:
        Data object with train_mask, val_mask, test_mask
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    labels = data.y.numpy()

    # Stratified split — handles class imbalance correctly
    try:
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, random_state=random_state,
            stratify=labels
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size, random_state=random_state,
            stratify=labels[temp_idx]
        )
    except ValueError:
        # Fall back to random split if stratification fails (e.g., single class)
        logger.warning("Stratified split failed; falling back to random split")
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, random_state=random_state
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size, random_state=random_state
        )

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

    data_path = "data/graphs/cicids_pyg_data.pt"

    if not Path(data_path).exists():
        logger.error(f"Data not found at {data_path}")
        logger.error("Please run the graph construction script first!")
        exit(1)

    logger.info(f"Loading data from {data_path}...")
    data = torch.load(data_path)

    data = prepare_data_splits(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(
        'gcn',
        input_dim=data.x.shape[1],
        hidden_dim=128,
        output_dim=2,
        num_layers=3,
        dropout=0.5
    )

    trainer = GNNTrainer(
        model,
        device=device,
        learning_rate=0.001,
        patience=30,
        tensorboard_dir='logs/tensorboard'
    )

    class_weights = trainer.compute_class_weights(data.y)
    trainer.setup_training(class_weights=class_weights, scheduler_type='plateau')

    history = trainer.train(data, num_epochs=200, verbose=True, save_dir='models/checkpoints')

    test_metrics = trainer.test(data)
    trainer.save_results(test_metrics, 'results/gnn_evaluation')

    logger.info("Training and evaluation complete!")
