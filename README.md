# GNN Cyber Project

![Python](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-under%20development-orange)
![CI](https://github.com/ze3tar/gnn-cyber-project/workflows/CI/badge.svg)
![Dataset Size](https://img.shields.io/badge/dataset-large-red)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What’s New / Key Features](#whats-new--key-features)
3. [Repository Structure](#repository-structure)
4. [Getting Started](#getting-started)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
   * [Configuration](#configuration)
   * [Running the Pipeline](#running-the-pipeline)
   * [Usage Examples](#usage-examples)
5. [Development Status](#development-status)
6. [Contribution](#contribution)
7. [License](#license)
8. [Contact](#contact)

---

## Project Overview

This project explores the use of **Graph Neural Networks (GNNs)** for cybersecurity, specifically for **network intrusion detection** using datasets such as CICIDS2017.

Unlike traditional ML models, this project **models network traffic as graphs**, where nodes represent entities (hosts, servers) and edges represent interactions (flows). This approach allows the detection of **multi-step attacks and complex intrusion patterns** that are difficult to capture with standard classifiers.

> ⚠ **Note:** The project is still under **active development**. The codebase, models, and pipeline are continuously being refined.

---

## What’s New / Key Features

* **Graph-based approach:** Captures relationships between network entities for enhanced threat detection.
* **Custom architectures:** Includes novel GNN layers designed specifically for cybersecurity datasets.
* **Modular pipeline:** Separates **preprocessing**, **graph construction**, **training**, and **evaluation**.
* **Configurable & scalable:** Supports different datasets and hyperparameters through `config.yaml`.
* **Lightweight repository:** Only code and configuration tracked; large datasets excluded.
* **Visualization support:** Generates graphs, metrics, and performance plots for analysis.

---

## Repository Structure

```
gnn-cyber-project/
├── main_pipeline.py          # Main pipeline entry point
├── requirements.txt          # Python dependencies
├── config.yaml               # Configuration file
├── data/                     # Raw & processed data (ignored in repo)
├── logs/                     # Logs (ignored in repo)
├── src/                      # Source code
│   ├── preprocessing/        # Data loading & graph construction
│   │   ├── cicids_loader.py
│   │   └── graph_constructor.py
│   ├── training/             # Training & evaluation
│   │   └── trainer.py
│   └── models/               # GNN model definitions
│       └── gnn_models.py
└── .gitignore                # Excludes large files, logs, checkpoints
```

---

## Getting Started

### Prerequisites

* Python 3.11
* pip / virtualenv
* PyTorch, DGL, NetworkX, and other dependencies in `requirements.txt`

### Installation

```bash
git clone https://github.com/ze3tar/gnn-cyber-project.git
cd gnn-cyber-project
pip install -r requirements.txt
```

### Configuration

Modify `config.yaml` to set dataset paths, model hyperparameters, and training options.

### Running the Pipeline

```bash
python main_pipeline.py
```

### Usage Examples

* Train a new GNN model on CICIDS2017:

```bash
python main_pipeline.py --mode train --dataset cicids2017
```

* Evaluate a saved model:

```bash
python main_pipeline.py --mode eval --model_path checkpoints/model.pth
```

---

## Development Status

> 🚧 Active development. Expect frequent updates, changes in APIs, and new features.

---

## Contribution

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Contact

* GitHub: [ze3tar](https://github.com/ze3tar)
* Project Repository: [gnn-cyber-project](https://github.com/ze3tar/gnn-cyber-project)
