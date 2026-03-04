"""
Shared utility functions for the GNN Cyber Threat Prediction project.

Provides helpers for:
- Metric formatting and pretty-printing
- Graph statistics summarization
- Training result comparison
- Path resolution
- Reproducibility (seed setting)

Author: Mohamed salem eddah
Institution: Shandong University of Technology
Project: Predictive Cyber Behavior Modeling Using Graph Neural Networks
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def format_metrics(metrics: Dict[str, float], exclude: Optional[List[str]] = None) -> str:
    """
    Format a metrics dictionary into a human-readable string.

    Args:
        metrics: Dictionary of metric names to float values.
        exclude: Keys to exclude from output.

    Returns:
        Formatted multi-line string.
    """
    exclude = set(exclude or [])
    lines = []
    for key, value in metrics.items():
        if key in exclude:
            continue
        if isinstance(value, float):
            lines.append(f"  {key:<20}: {value:.4f}")
        else:
            lines.append(f"  {key:<20}: {value}")
    return "\n".join(lines)


def print_metrics_table(results: List[Dict[str, Any]],
                        key_col: str = 'model',
                        metric_cols: Optional[List[str]] = None) -> None:
    """
    Print a formatted comparison table of model results.

    Args:
        results: List of result dictionaries (one per model).
        key_col: Column name to use as row identifier.
        metric_cols: Columns to include. Defaults to common metrics.
    """
    if not results:
        return

    if metric_cols is None:
        metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'train_time_s']

    # Filter to columns that actually exist
    available = [c for c in metric_cols if c in results[0]]

    header = f"  {key_col:<15}" + "".join(f"{c:>12}" for c in available)
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in results:
        line = f"  {str(row.get(key_col, '')):<15}"
        for col in available:
            val = row.get(col, '')
            if isinstance(val, float):
                line += f"{val:>12.4f}"
            else:
                line += f"{str(val):>12}"
        print(line)

    print("=" * len(header))


# ------------------------------------------------------------------
# Graph statistics helpers
# ------------------------------------------------------------------

def summarize_graph_stats(stats: Dict) -> str:
    """
    Convert nested graph statistics dict into a readable summary string.

    Args:
        stats: Output of AdvancedGraphConstructor.compute_graph_statistics()

    Returns:
        Formatted summary string.
    """
    lines = []
    for section, values in stats.items():
        lines.append(f"[{section.upper()}]")
        if isinstance(values, dict):
            for k, v in values.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.6f}")
                else:
                    lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {values}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Path utilities
# ------------------------------------------------------------------

def resolve_path(path: str, base: Optional[str] = None) -> Path:
    """
    Resolve a path, optionally relative to a base directory.

    Args:
        path: Absolute or relative path string.
        base: Base directory for relative paths. Defaults to cwd.

    Returns:
        Resolved Path object.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base_dir = Path(base) if base else Path.cwd()
    return (base_dir / p).resolve()


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------
# JSON serialization helper
# ------------------------------------------------------------------

def safe_json_dump(obj: Any, path: str, indent: int = 2) -> None:
    """
    Save any object containing numpy types to a JSON file.

    Args:
        obj: Object to serialize (may contain np.integer, np.floating, np.ndarray).
        path: Output file path.
        indent: JSON indentation level.
    """
    def _convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return str(o)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent, default=_convert)
    logger.debug(f"Saved JSON to {path}")


# ------------------------------------------------------------------
# Training result loading
# ------------------------------------------------------------------

def load_benchmark_results(results_path: str) -> List[Dict]:
    """
    Load and display benchmark results from a saved JSON file.

    Args:
        results_path: Path to benchmark_results.json

    Returns:
        List of result dictionaries.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    print_metrics_table(results, key_col='model')
    return results
