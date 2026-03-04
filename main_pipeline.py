"""
Main Execution Pipeline for GNN-based Cyber Threat Prediction
Orchestrates the entire workflow from data loading to model evaluation.

Includes:
- Config validation at startup (Phase 2)
- Temporal GNN training (Phase 3)
- Model benchmarking across all architectures (Phase 3)
- Interactive HTML graph export (Phase 4)
- SHAP feature importance (Phase 5)

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks

Usage:
    python main_pipeline.py --config config.yaml
    python main_pipeline.py --mode train --model gcn
    python main_pipeline.py --mode benchmark
    python main_pipeline.py --mode temporal
    python main_pipeline.py --mode evaluate --checkpoint models/best_model.pt
"""

import argparse
import sys
import time
import yaml
import torch
import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from preprocessing.cicids_loader import CICIDS2017Loader
from preprocessing.graph_constructor import AdvancedGraphConstructor
from models.gnn_models import create_model
from training.trainer import GNNTrainer, prepare_data_splits

# Setup logging
Path('logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config validation (Phase 2)
# ------------------------------------------------------------------

_CONFIG_SCHEMA = {
    'data': ['raw_dir', 'processed_dir', 'graph_dir'],
    'preprocessing': ['normalize', 'handle_outliers', 'engineer_features'],
    'graph': ['construction_method', 'time_window', 'directed'],
    'model': ['type', 'hidden_dim', 'num_layers', 'dropout'],
    'training': ['num_epochs', 'learning_rate', 'weight_decay', 'patience'],
    'evaluation': ['train_ratio', 'val_ratio', 'test_ratio'],
    'output': ['model_dir', 'results_dir', 'visualizations_dir'],
}

_VALID_MODEL_TYPES = {'gcn', 'gat', 'sage', 'hybrid', 'temporal'}
_VALID_GRAPH_METHODS = {'host_based', 'flow_based', 'hierarchical'}


def validate_config(config: dict) -> None:
    """
    Validate configuration schema and value ranges at startup.

    Raises:
        ValueError if any required key is missing or has an invalid value.
    """
    for section, keys in _CONFIG_SCHEMA.items():
        if section not in config:
            raise ValueError(f"Config missing required section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Config missing required key: '{section}.{key}'")

    model_type = config['model']['type']
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Invalid model type '{model_type}'. Must be one of {_VALID_MODEL_TYPES}"
        )

    graph_method = config['graph']['construction_method']
    if graph_method not in _VALID_GRAPH_METHODS:
        raise ValueError(
            f"Invalid graph method '{graph_method}'. Must be one of {_VALID_GRAPH_METHODS}"
        )

    ev = config['evaluation']
    total = ev.get('train_ratio', 0) + ev.get('val_ratio', 0) + ev.get('test_ratio', 0)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.4f}")

    lr = config['training'].get('learning_rate', 0)
    if not (0 < lr < 1):
        raise ValueError(f"learning_rate must be in (0, 1), got {lr}")

    logger.info("Configuration validated successfully.")


# ------------------------------------------------------------------
# Pipeline class
# ------------------------------------------------------------------

class Pipeline:
    """
    Main pipeline for GNN-based cyber threat prediction.
    Handles data loading, preprocessing, graph construction, training, and evaluation.
    """

    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path) if config_path else self.get_default_config()
        validate_config(self.config)
        self.setup_directories()
        self.device = self._resolve_device()

        logger.info("=" * 80)
        logger.info("GNN-BASED CYBER THREAT PREDICTION PIPELINE")
        logger.info("=" * 80)

    @staticmethod
    def get_default_config() -> dict:
        return {
            'data': {
                'raw_dir': 'data/raw/CICIDS2017',
                'processed_dir': 'data/processed',
                'graph_dir': 'data/graphs',
                'sample_size': None,
            },
            'preprocessing': {
                'normalize': True,
                'handle_outliers': True,
                'engineer_features': True,
                'deduplicate': True,
                'augment_rare': False,
            },
            'graph': {
                'construction_method': 'host_based',
                'time_window': 300,
                'directed': True,
                'aggregate_edges': True,
                'min_flows': 1,
                'self_loops': False,
            },
            'model': {
                'type': 'gcn',
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.5,
                'num_heads': 4,
                'lstm_layers': 2,
                'gnn_type': 'gcn',
            },
            'training': {
                'num_epochs': 200,
                'learning_rate': 0.001,
                'weight_decay': 5e-4,
                'batch_size': 1,
                'patience': 30,
                'min_delta': 1e-4,
                'scheduler_type': 'plateau',
                'use_class_weights': True,
                'use_amp': False,
            },
            'evaluation': {
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'random_state': 42,
            },
            'output': {
                'model_dir': 'models',
                'results_dir': 'results',
                'visualizations_dir': 'results/visualizations',
            }
        }

    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_directories(self):
        dirs = [
            self.config['data']['processed_dir'],
            self.config['data']['graph_dir'],
            self.config['output']['model_dir'],
            self.config['output']['results_dir'],
            self.config['output']['visualizations_dir'],
            'logs', 'logs/tensorboard',
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _resolve_device(self) -> str:
        device_cfg = self.config.get('resources', {}).get('device', 'auto')
        if device_cfg == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_cfg

    # ------------------------------------------------------------------
    # Phase 1: Data preprocessing
    # ------------------------------------------------------------------

    def run_data_preprocessing(self) -> str:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: DATA PREPROCESSING")
        logger.info("=" * 80)

        loader = CICIDS2017Loader(
            data_dir=self.config['data']['raw_dir'],
            cache_dir='data/cache',
            log_dir='logs'
        )

        pre_cfg = self.config.get('preprocessing', {})
        df = loader.preprocess_pipeline(
            sample_size=self.config['data']['sample_size'],
            normalize=pre_cfg.get('normalize', True),
            handle_outliers=pre_cfg.get('handle_outliers', True),
            engineer_features=pre_cfg.get('engineer_features', True),
            deduplicate=pre_cfg.get('deduplicate', True),
            augment_rare=pre_cfg.get('augment_rare', False),
        )

        output_path = Path(self.config['data']['processed_dir']) / 'cicids2017_processed.csv'
        loader.save_processed_data(str(output_path), save_metadata=True)

        logger.info(f"Data preprocessing complete. Saved to {output_path}")
        return str(output_path)

    # ------------------------------------------------------------------
    # Phase 2: Graph construction
    # ------------------------------------------------------------------

    def run_graph_construction(self, data_path: str) -> str:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: GRAPH CONSTRUCTION")
        logger.info("=" * 80)

        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df):,} records")

        graph_cfg = self.config['graph']
        constructor = AdvancedGraphConstructor(
            construction_method=graph_cfg['construction_method'],
            time_window=graph_cfg['time_window'],
            directed=graph_cfg['directed'],
            self_loops=graph_cfg.get('self_loops', False),
        )

        G = constructor.build_networkx_graph(
            df,
            aggregate_edges=graph_cfg.get('aggregate_edges', True),
            min_flows=graph_cfg.get('min_flows', 1)
        )

        stats = constructor.compute_graph_statistics(G)
        logger.info(f"Graph statistics: {json.dumps(stats, indent=2)}")

        pyg_data = constructor.networkx_to_pyg(G, df, include_edge_features=True)

        graph_path = Path(self.config['data']['graph_dir']) / 'cicids_graph.gpickle'
        pyg_path = Path(self.config['data']['graph_dir']) / 'cicids_pyg_data.pt'

        constructor.save_graph(G, str(graph_path))
        torch.save(pyg_data, pyg_path)

        # Static visualization
        vis_dir = Path(self.config['output']['visualizations_dir'])
        constructor.visualize_graph_sample(G, str(vis_dir / 'network_graph.png'),
                                           max_nodes=100, highlight_attacks=True)

        # Interactive HTML export (Phase 4)
        constructor.export_interactive_html(G, str(vis_dir / 'network_graph_interactive.html'),
                                            max_nodes=200)

        logger.info(f"Graph construction complete. Saved to {pyg_path}")
        return str(pyg_path)

    # ------------------------------------------------------------------
    # Phase 3: Training
    # ------------------------------------------------------------------

    def _build_trainer(self, model: torch.nn.Module) -> GNNTrainer:
        """Create a GNNTrainer with settings from config."""
        train_cfg = self.config['training']
        return GNNTrainer(
            model,
            device=self.device,
            learning_rate=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
            patience=train_cfg['patience'],
            min_delta=train_cfg.get('min_delta', 1e-4),
            use_amp=train_cfg.get('use_amp', False),
            tensorboard_dir='logs/tensorboard',
        )

    def _build_model(self, model_type: str, input_dim: int) -> torch.nn.Module:
        """Instantiate a GNN model from config."""
        model_cfg = self.config['model']
        return create_model(
            model_type,
            input_dim=input_dim,
            hidden_dim=model_cfg['hidden_dim'],
            output_dim=2,
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout'],
            num_heads=model_cfg.get('num_heads', 4),
            lstm_layers=model_cfg.get('lstm_layers', 2),
            gnn_type=model_cfg.get('gnn_type', 'gcn'),
        )

    def run_training(self, graph_path: str) -> dict:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: MODEL TRAINING")
        logger.info("=" * 80)

        data = torch.load(graph_path, weights_only=False)
        logger.info(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")

        ev = self.config['evaluation']
        data = prepare_data_splits(
            data,
            train_ratio=ev['train_ratio'],
            val_ratio=ev['val_ratio'],
            test_ratio=ev['test_ratio'],
            random_state=ev.get('random_state', 42),
        )

        logger.info(f"Using device: {self.device}")
        model = self._build_model(self.config['model']['type'], data.x.shape[1])

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {self.config['model']['type'].upper()} with {num_params:,} parameters")

        trainer = self._build_trainer(model)

        class_weights = None
        if self.config['training']['use_class_weights']:
            class_weights = trainer.compute_class_weights(data.y)

        trainer.setup_training(
            class_weights=class_weights,
            scheduler_type=self.config['training']['scheduler_type']
        )

        checkpoint_dir = Path(self.config['output']['model_dir']) / 'checkpoints'
        checkpoint_name = f"{self.config['model']['type']}_{self.run_id}_best.pt"
        history = trainer.train(
            data,
            num_epochs=self.config['training']['num_epochs'],
            verbose=True,
            save_dir=str(checkpoint_dir)
        )

        test_metrics = trainer.test(data)

        results_dir = (Path(self.config['output']['results_dir']) /
                       f"{self.config['model']['type']}_evaluation")
        trainer.save_results(test_metrics, str(results_dir))

        # SHAP analysis (Phase 5)
        shap_path = str(Path(self.config['output']['visualizations_dir']) / 'shap_importance.png')
        trainer.compute_shap_importance(data, save_path=shap_path)

        logger.info(f"Training complete. Results saved to {results_dir}")
        return {'history': history, 'test_metrics': test_metrics}

    # ------------------------------------------------------------------
    # Temporal training (Phase 3)
    # ------------------------------------------------------------------

    def run_temporal_training(self, data_path: str) -> dict:
        """
        Run TemporalGNN training on time-ordered graph snapshots.

        Args:
            data_path: Path to processed CSV file

        Returns:
            Training history dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEMPORAL GNN TRAINING")
        logger.info("=" * 80)

        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df):,} records for temporal graph construction")

        graph_cfg = self.config['graph']
        constructor = AdvancedGraphConstructor(
            construction_method='host_based',
            time_window=graph_cfg.get('time_window', 300),
            directed=graph_cfg.get('directed', True),
        )

        snapshots = constructor.create_temporal_snapshots(
            df,
            timestamp_col='timestamp',
            max_snapshots=50,
        )

        if not snapshots:
            logger.error("No temporal snapshots created. Ensure the data contains a 'timestamp' column.")
            return {}

        logger.info(f"Created {len(snapshots)} temporal snapshots")
        input_dim = snapshots[0].x.shape[1]
        model = self._build_model('temporal', input_dim)

        trainer = self._build_trainer(model)
        trainer.setup_training(scheduler_type=self.config['training']['scheduler_type'])

        history = trainer.train_temporal(
            snapshots,
            num_epochs=self.config['training']['num_epochs'],
            sequence_len=5,
            save_dir=str(Path(self.config['output']['model_dir']) / 'checkpoints_temporal'),
        )

        results_dir = Path(self.config['output']['results_dir']) / 'temporal_evaluation'
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Temporal training complete. Results saved to {results_dir}")
        return history

    # ------------------------------------------------------------------
    # Benchmark mode (Phase 3)
    # ------------------------------------------------------------------

    def run_benchmark(self, graph_path: str) -> None:
        """
        Run all GNN architectures sequentially and produce a comparison table.

        Benchmarks: GCN, GAT, GraphSAGE, HybridGNN, TemporalGNN (static mode).
        """
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK: ALL GNN ARCHITECTURES")
        logger.info("=" * 80)

        data = torch.load(graph_path, weights_only=False)
        ev = self.config['evaluation']
        data = prepare_data_splits(
            data,
            train_ratio=ev['train_ratio'],
            val_ratio=ev['val_ratio'],
            test_ratio=ev['test_ratio'],
            random_state=ev.get('random_state', 42),
        )

        model_types = ['gcn', 'gat', 'sage', 'hybrid']
        results = []

        for model_type in model_types:
            logger.info(f"\n--- Benchmarking: {model_type.upper()} ---")
            t_start = time.time()

            model = self._build_model(model_type, data.x.shape[1])
            num_params = sum(p.numel() for p in model.parameters())

            trainer = self._build_trainer(model)

            class_weights = None
            if self.config['training']['use_class_weights']:
                class_weights = trainer.compute_class_weights(data.y)

            trainer.setup_training(
                class_weights=class_weights,
                scheduler_type=self.config['training']['scheduler_type']
            )

            # Use reduced epochs for benchmarking
            bench_epochs = min(50, self.config['training']['num_epochs'])
            history = trainer.train(data, num_epochs=bench_epochs, verbose=False)
            test_metrics = trainer.test(data)

            elapsed = time.time() - t_start

            row = {
                'model': model_type.upper(),
                'params': num_params,
                'accuracy': round(test_metrics.get('accuracy', 0), 4),
                'precision': round(test_metrics.get('precision', 0), 4),
                'recall': round(test_metrics.get('recall', 0), 4),
                'f1': round(test_metrics.get('f1', 0), 4),
                'roc_auc': round(test_metrics.get('roc_auc', 0), 4),
                'train_time_s': round(elapsed, 1),
            }
            results.append(row)

            # Save per-model results
            model_results_dir = (Path(self.config['output']['results_dir']) /
                                  f"{model_type}_benchmark")
            trainer.save_results(test_metrics, str(model_results_dir))

        # Print comparison table
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 80)
        header = f"{'Model':<12} {'Params':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'Time(s)':>10}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in results:
            logger.info(
                f"{r['model']:<12} {r['params']:>10,} {r['accuracy']:>10.4f} "
                f"{r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} "
                f"{r['roc_auc']:>10.4f} {r['train_time_s']:>10.1f}"
            )

        # Save table to JSON
        bench_path = Path(self.config['output']['results_dir']) / 'benchmark_results.json'
        with open(bench_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nBenchmark results saved to {bench_path}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self):
        start_time = datetime.now()

        logger.info("\n" + "=" * 80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 80)

        try:
            data_path = self.run_data_preprocessing()
            graph_path = self.run_graph_construction(data_path)
            results = self.run_training(graph_path)

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Total time: {elapsed / 60:.2f} minutes")
            logger.info("Test Results:")
            for key, value in results['test_metrics'].items():
                if key not in ('confusion_matrix', 'classification_report'):
                    logger.info(f"  {key}: {value:.4f}")

            summary = {
                'config': self.config,
                'elapsed_time': elapsed,
                'test_metrics': {
                    k: v for k, v in results['test_metrics'].items()
                    if k not in ('confusion_matrix', 'classification_report')
                }
            }

            summary_path = Path(self.config['output']['results_dir']) / 'pipeline_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Pipeline summary saved to {summary_path}")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Evaluate only
    # ------------------------------------------------------------------

    def run_evaluation_only(self, checkpoint_path: str):
        """Re-evaluate a saved checkpoint without retraining."""
        logger.info("Running evaluation on trained model checkpoint...")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        graph_path = Path(self.config['data']['graph_dir']) / 'cicids_pyg_data.pt'
        data = torch.load(str(graph_path), weights_only=False)
        data = prepare_data_splits(data)

        model = self._build_model(self.config['model']['type'], data.x.shape[1])
        model.load_state_dict(checkpoint['model_state_dict'])

        trainer = GNNTrainer(model, device=self.device)
        trainer.criterion = torch.nn.CrossEntropyLoss()

        test_metrics = trainer.test(data)

        results_dir = Path(self.config['output']['results_dir']) / 'evaluation_only'
        trainer.save_results(test_metrics, str(results_dir))

        logger.info("Evaluation complete!")
        return test_metrics


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GNN-based Cyber Threat Prediction Pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'preprocess', 'graph', 'train',
                                 'evaluate', 'benchmark', 'temporal'],
                        help='Pipeline execution mode')
    parser.add_argument('--model', type=str, default=None,
                        choices=['gcn', 'gat', 'sage', 'hybrid', 'temporal'],
                        help='Override model type from config')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (required for evaluate mode)')

    args = parser.parse_args()

    pipeline = Pipeline(args.config)

    # Override model type if specified on CLI
    if args.model:
        pipeline.config['model']['type'] = args.model

    if args.mode == 'full':
        pipeline.run_full_pipeline()

    elif args.mode == 'preprocess':
        pipeline.run_data_preprocessing()

    elif args.mode == 'graph':
        data_path = Path(pipeline.config['data']['processed_dir']) / 'cicids2017_processed.csv'
        pipeline.run_graph_construction(str(data_path))

    elif args.mode == 'train':
        graph_path = Path(pipeline.config['data']['graph_dir']) / 'cicids_pyg_data.pt'
        pipeline.run_training(str(graph_path))

    elif args.mode == 'evaluate':
        if not args.checkpoint:
            logger.error("--checkpoint is required for evaluate mode")
            sys.exit(1)
        pipeline.run_evaluation_only(args.checkpoint)

    elif args.mode == 'benchmark':
        graph_path = Path(pipeline.config['data']['graph_dir']) / 'cicids_pyg_data.pt'
        if not graph_path.exists():
            logger.error(f"Graph file not found: {graph_path}. Run --mode graph first.")
            sys.exit(1)
        pipeline.run_benchmark(str(graph_path))

    elif args.mode == 'temporal':
        data_path = Path(pipeline.config['data']['processed_dir']) / 'cicids2017_processed.csv'
        if not data_path.exists():
            logger.error(f"Processed data not found: {data_path}. Run --mode preprocess first.")
            sys.exit(1)
        pipeline.run_temporal_training(str(data_path))


if __name__ == "__main__":
    main()
