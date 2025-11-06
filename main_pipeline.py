"""
Main Execution Pipeline for GNN-based Cyber Threat Prediction
Orchestrates the entire workflow from data loading to model evaluation

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks

Usage:
    python main_pipeline.py --config config.yaml
    python main_pipeline.py --mode train --model gcn
    python main_pipeline.py --mode evaluate --checkpoint models/best_model.pt
"""

import argparse
import sys
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

from preprocessing.cicids_loader import CICIDS2017Loader
from preprocessing.graph_constructor import AdvancedGraphConstructor
from models.gnn_models import create_model
from training.trainer import GNNTrainer, prepare_data_splits

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline for GNN-based cyber threat prediction.
    Handles data loading, preprocessing, graph construction, training, and evaluation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self.load_config(config_path) if config_path else self.get_default_config()
        self.setup_directories()
        
        logger.info("="*80)
        logger.info("GNN-BASED CYBER THREAT PREDICTION PIPELINE")
        logger.info("="*80)
        logger.info(f"Configuration loaded: {json.dumps(self.config, indent=2)}")
    
    @staticmethod
    def get_default_config() -> dict:
        """Get default configuration."""
        return {
            'data': {
                'raw_dir': '/root/gnn-cyber-project/data/raw/CICIDS2017',
                'processed_dir': 'data/processed',
                'graph_dir': 'data/graphs',
                'sample_size': None,  # None for full dataset
            },
            'preprocessing': {
                'normalize': True,
                'handle_outliers': True,
                'engineer_features': True,
            },
            'graph': {
                'construction_method': 'host_based',
                'time_window': 300,
                'directed': True,
                'aggregate_edges': True,
                'min_flows': 1,
            },
            'model': {
                'type': 'gcn',  # gcn, gat, sage, hybrid
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.5,
                'num_heads': 4,  # For GAT
            },
            'training': {
                'num_epochs': 200,
                'learning_rate': 0.001,
                'weight_decay': 5e-4,
                'batch_size': 1,  # For full graph
                'patience': 30,
                'scheduler_type': 'plateau',
                'use_class_weights': True,
            },
            'evaluation': {
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
            },
            'output': {
                'model_dir': 'models',
                'results_dir': 'results',
                'visualizations_dir': 'results/visualizations',
            }
        }
    
    @staticmethod
    def load_config(config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config['data']['processed_dir'],
            self.config['data']['graph_dir'],
            self.config['output']['model_dir'],
            self.config['output']['results_dir'],
            self.config['output']['visualizations_dir'],
            'logs'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def run_data_preprocessing(self) -> str:
        """
        Execute data preprocessing pipeline.
        
        Returns:
            Path to processed data file
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: DATA PREPROCESSING")
        logger.info("="*80)
        
        # Initialize loader
        loader = CICIDS2017Loader(
            data_dir=self.config['data']['raw_dir'],
            cache_dir='data/cache',
            log_dir='logs'
        )
        
        # Run preprocessing
        df = loader.preprocess_pipeline(
            sample_size=self.config['data']['sample_size'],
            normalize=self.config['preprocessing']['normalize'],
            handle_outliers=self.config['preprocessing']['handle_outliers'],
            engineer_features=self.config['preprocessing']['engineer_features']
        )
        
        # Save processed data
        output_path = Path(self.config['data']['processed_dir']) / 'cicids2017_processed.csv'
        loader.save_processed_data(output_path, save_metadata=True)
        
        logger.info(f"✅ Data preprocessing complete. Saved to {output_path}")
        
        return str(output_path)
    
    def run_graph_construction(self, data_path: str) -> str:
        """
        Execute graph construction.
        
        Args:
            data_path: Path to processed data file
            
        Returns:
            Path to PyG graph file
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: GRAPH CONSTRUCTION")
        logger.info("="*80)
        
        # Load data
        import pandas as pd
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df):,} records")
        
        # Initialize constructor
        constructor = AdvancedGraphConstructor(
            construction_method=self.config['graph']['construction_method'],
            time_window=self.config['graph']['time_window'],
            directed=self.config['graph']['directed']
        )
        
        # Build graph
        G = constructor.build_networkx_graph(
            df,
            aggregate_edges=self.config['graph']['aggregate_edges'],
            min_flows=self.config['graph']['min_flows']
        )
        
        # Compute statistics
        stats = constructor.compute_graph_statistics(G)
        logger.info(f"Graph statistics: {json.dumps(stats, indent=2)}")
        
        # Convert to PyG
        pyg_data = constructor.networkx_to_pyg(G, df, include_edge_features=True)
        
        # Save
        graph_path = Path(self.config['data']['graph_dir']) / 'cicids_graph.gpickle'
        pyg_path = Path(self.config['data']['graph_dir']) / 'cicids_pyg_data.pt'
        
        constructor.save_graph(G, graph_path)
        torch.save(pyg_data, pyg_path)
        
        # Visualize
        vis_path = Path(self.config['output']['visualizations_dir']) / 'network_graph.png'
        constructor.visualize_graph_sample(G, vis_path, max_nodes=100, highlight_attacks=True)
        
        logger.info(f"✅ Graph construction complete. Saved to {pyg_path}")
        
        return str(pyg_path)
    
    def run_training(self, graph_path: str) -> dict:
        """
        Execute model training.
        
        Args:
            graph_path: Path to PyG graph file
            
        Returns:
            Training history and test metrics
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: MODEL TRAINING")
        logger.info("="*80)
        
        # Load data
        data = torch.load(graph_path)
        logger.info(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")
        
        # Prepare splits
        data = prepare_data_splits(
            data,
            train_ratio=self.config['evaluation']['train_ratio'],
            val_ratio=self.config['evaluation']['val_ratio'],
            test_ratio=self.config['evaluation']['test_ratio']
        )
        
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        model = create_model(
            self.config['model']['type'],
            input_dim=data.x.shape[1],
            hidden_dim=self.config['model']['hidden_dim'],
            output_dim=2,
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            num_heads=self.config['model'].get('num_heads', 4)
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {self.config['model']['type'].upper()} with {num_params:,} parameters")
        
        # Initialize trainer
        trainer = GNNTrainer(
            model,
            device=device,
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            patience=self.config['training']['patience']
        )
        
        # Setup training
        class_weights = None
        if self.config['training']['use_class_weights']:
            class_weights = trainer.compute_class_weights(data.y)
        
        trainer.setup_training(
            class_weights=class_weights,
            scheduler_type=self.config['training']['scheduler_type']
        )
        
        # Train
        checkpoint_dir = Path(self.config['output']['model_dir']) / 'checkpoints'
        history = trainer.train(
            data,
            num_epochs=self.config['training']['num_epochs'],
            verbose=True,
            save_dir=checkpoint_dir
        )
        
        # Evaluate on test set
        test_metrics = trainer.test(data)
        
        # Save results
        results_dir = Path(self.config['output']['results_dir']) / f"{self.config['model']['type']}_evaluation"
        trainer.save_results(test_metrics, results_dir)
        
        logger.info(f"✅ Training complete. Results saved to {results_dir}")
        
        return {'history': history, 'test_metrics': test_metrics}
    
    def run_full_pipeline(self):
        """Execute complete pipeline."""
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("="*80)
        
        try:
            # Phase 1: Data preprocessing
            data_path = self.run_data_preprocessing()
            
            # Phase 2: Graph construction
            graph_path = self.run_graph_construction(data_path)
            
            # Phase 3: Training and evaluation
            results = self.run_training(graph_path)
            
            # Summary
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETE!")
            logger.info("="*80)
            logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
            logger.info(f"\nTest Results:")
            for key, value in results['test_metrics'].items():
                if key not in ['confusion_matrix', 'classification_report']:
                    logger.info(f"  {key}: {value:.4f}")
            
            # Save pipeline summary
            summary = {
                'config': self.config,
                'elapsed_time': elapsed_time,
                'test_metrics': {k: v for k, v in results['test_metrics'].items() 
                               if k not in ['confusion_matrix', 'classification_report']}
            }
            
            summary_path = Path(self.config['output']['results_dir']) / 'pipeline_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"\nPipeline summary saved to {summary_path}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def run_evaluation_only(self, checkpoint_path: str):
        """Run evaluation on a trained model."""
        logger.info("Running evaluation on trained model...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Load graph
        graph_path = Path(self.config['data']['graph_dir']) / 'cicids_pyg_data.pt'
        data = torch.load(graph_path)
        data = prepare_data_splits(data)
        
        # Create model and load weights
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_model(
            self.config['model']['type'],
            input_dim=data.x.shape[1],
            hidden_dim=self.config['model']['hidden_dim'],
            output_dim=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        trainer = GNNTrainer(model, device=device)
        trainer.criterion = torch.nn.CrossEntropyLoss()
        test_metrics = trainer.test(data)
        
        logger.info("Evaluation complete!")
        return test_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GNN-based Cyber Threat Prediction Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'preprocess', 'graph', 'train', 'evaluate'],
                       help='Pipeline mode')
    parser.add_argument('--model', type=str, default='gcn',
                       choices=['gcn', 'gat', 'sage', 'hybrid'],
                       help='Model type')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = Pipeline(args.config)
    
    # Override model type if specified
    if args.model:
        pipeline.config['model']['type'] = args.model
    
    # Execute based on mode
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
            logger.error("--checkpoint required for evaluation mode")
            sys.exit(1)
        pipeline.run_evaluation_only(args.checkpoint)


if __name__ == "__main__":
    main()