# GNN Cyber Threat Prediction

Predictive cyber behavior modeling using Graph Neural Networks on the CICIDS2017 intrusion detection dataset. Network flows are converted into host-level graphs; GNN models classify nodes (IP hosts) as benign or malicious.

**Author:** Mohamed Salem Eddah | **Institution:** Shandong University of Technology

---

## Models

| Model | Architecture | Best Use |
|-------|-------------|----------|
| GCN | Graph Convolutional Network | Baseline, fast |
| GAT | Graph Attention Network | Weighted neighbor aggregation |
| GraphSAGE | Inductive sampling | Large graphs |
| HybridGNN | GCN + GAT + SAGE ensemble | Best accuracy |
| TemporalGNN | GCN + LSTM | Time-series captures |

---

## Dataset

Download CICIDS2017 CSV files from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and place them in `data/raw/CICIDS2017/`.

The dataset contains ~2.8M network flows across 15 attack categories (DDoS, PortScan, Brute Force, Botnet, etc.).

---

## Installation

```bash
git clone https://github.com/ze3tar/gnn-cyber-project.git
cd gnn-cyber-project

pip install -r requirements.txt
# PyTorch Geometric (match your CUDA/CPU version):
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

---

## Usage

### Train a single model

```bash
# GCN (default)
python main_pipeline.py --mode train --model gcn

# GAT
python main_pipeline.py --mode train --model gat

# GraphSAGE
python main_pipeline.py --mode train --model sage

# HybridGNN (recommended)
python main_pipeline.py --mode train --model hybrid

# TemporalGNN
python main_pipeline.py --mode temporal --model temporal
```

### Benchmark all architectures

```bash
python main_pipeline.py --mode benchmark
```

Trains all 4 non-temporal models and prints a comparison table with accuracy, F1, ROC-AUC, and training time.

### Evaluate a saved model

```bash
python main_pipeline.py --mode evaluate --model hybrid
```

---

## Configuration (`config.yaml`)

```yaml
data:
  raw_dir: data/raw/CICIDS2017
  processed_dir: data/processed
  sample_fraction: 0.1        # fraction of dataset to use (1.0 = full)
  deduplicate: true
  augment_rare: false

model:
  hidden_dim: 64
  num_layers: 3
  dropout: 0.3

training:
  epochs: 100
  lr: 0.001
  batch_size: 32
  early_stopping_patience: 15
  use_amp: false              # set true for GPU mixed precision
  scheduler: cosine           # cosine | plateau

graph:
  graph_type: host_based      # host_based | flow_based | hierarchical
  use_cache: true
```

---

## Pipeline Features

- Chunked CSV loading with deduplication and rare-class augmentation
- Host-based, flow-based, and hierarchical graph construction with disk caching
- AMP (mixed precision) training for GPU speedup
- Early stopping, gradient clipping, cosine/plateau LR scheduling
- TensorBoard logging (`tensorboard --logdir logs/tensorboard`)
- SHAP feature importance via `DeepExplainer`
- Interactive pyvis graph export to HTML
- Benchmark mode with multi-model comparison table

---

## Outputs

| Path | Contents |
|------|----------|
| `models/checkpoints/` | Best model `.pt` files |
| `results/` | `benchmark_results.json`, per-model metrics |
| `logs/tensorboard/` | TensorBoard event files |
| `data/processed/` | Cached PyG graph objects |
| `outputs/graph_visualization.html` | Interactive network graph |

---

## Citation

```
Mohamed Salem Eddah, "Predictive Cyber Behavior Modeling Using Graph Neural Networks",
Shandong University of Technology, 2025.
Dataset: Canadian Institute for Cybersecurity, CICIDS2017.
```
