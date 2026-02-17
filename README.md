# Optimizing in the Shadows: Addressing Class Imbalance in Fine-Tuning Deep Learners

*Beyond Balance* is a flexible framework for multi-label text classification in software engineering, supporting dynamic and static weighted loss strategies, and easy integration of custom models and metrics.

## Requirements

- Python 3.12
- PyTorch
- pandas, numpy
- datasets, transformers, setfit
- (see `req.txt` for full list)

## Installation

Clone the code repository, create a virtual environment, activate it, and install with pip:

```sh
git clone https://github.com/CybersecurityLab-unibz/beyondBalance.git
cd beyondBalance
python3 -m venv env
source env/bin/activate
pip install .
```

## Getting Started

### Quickstart

Run a dummy script:

```sh
beyondBalance --weighted-loss FAMO \
  --train ./data/NLBSE25/pharo/pharo_train.csv \
  --test ./data/NLBSE25/pharo/pharo_test.csv \
  --column-key-text combo \
  --column-key-label labels \
  --freeze-embeddings True \
  --model-path sentence-transformers/paraphrase-MiniLM-L6-v2 \
  --batch-size 8  \
  --lr 0.01 \
  --weight-decay 0  \
  --epoch 1
```

Run all tests:

```sh
python -m unittest discover -s tests
```

Check code coverage:

```sh
pip install coverage
coverage run -m unittest discover -s tests && coverage report
```

### Command-line Arguments

The main entrypoint is `beyondBalance` (or `python -m beyondBalance`).

Key arguments:

- `--train`, `--test`, `--val`: CSV file paths
- `--column-key-text`, `--column-key-label`: column names in CSV
- `--model-path`: HuggingFace model or local path
- `--weighted-loss-strategy`: EW, ICF, RBF, FAMO, COBA, or path to custom .py
- `--batch-size`, `--epoch`, `--lr`, `--weight-decay`: training hyperparameters
- `--freeze-embeddings`: freeze model embeddings (only for setfit)
- `--maximise-model`: grid search for best config
- `--generate-base-config`: create a config template

See `beyondBalance/utils.py:parse_args` for all options.

## API Reference

### beyondBalance.data_handler.DataHandler
Handles data loading, preprocessing, metrics, and results management.

**Constructor:**
```python
DataHandler(args)
```
**Key methods:**
- `set_seed(seed)`: Set random seed for reproducibility
- `create_paths()`: Prepare output/log/checkpoint paths
- `summarize_metrics(metrics)`: Log F1/precision/recall per class
- `update_csv(metric)`: Save run config and metrics to CSV
- `read_data()`: Load and preprocess train/val/test data
- `merge_dataset()`: Merge splits for training
- `find_best_args()`: Load best config from CSV
- `calculate_weights()`: Compute class weights

### beyondBalance.linear_weights.LinearWeights
Computes class weights for static strategies (EW, ICF, RBF) or custom callables.

**Constructor:**
```python
LinearWeights(type, labels)
```
**Key methods:**
- `get_class_weight_dicts()`: Returns list of class weight dicts

### beyondBalance.proxy.Proxy
Manages model selection, instantiation, and training/evaluation.

**Constructor:**
```python
Proxy(args)
```
**Key methods:**
- `create_model()`, `create_trainer()`, `train()`, `eval()`, `run_already_exists()`

### beyondBalance.utils
Utility functions for config parsing, metrics, device selection, and dynamic imports.

**Notable functions:**
- `parse_args()`: Argument parser for CLI
- `read_files_weighted_loss_strategy(args)`: Import custom loss functions
- `compute_metrics_setfit`, `compute_metrics_roberta`, `compute_metrics`: Evaluation metrics
- `generate_base_config(path)`: Write config template
- `get_device()`: Detects best device (MPS, CUDA, CPU)

See source code for full details and docstrings.

## How To Extend

### Adding a New Weighted Loss Strategy
1. Write a Python function that takes `labels` as input and returns class weight dicts.
2. Pass the path to your `.py` file via `--weighted-loss-strategy`.
3. The CLI will import and use your function automatically.

### Adding a New Model
1. Implement a new model class (see `roberta_models.py` or `setfit_models.py` for examples).
2. Update `Proxy` to recognize your model type and instantiate your class.
3. Add any required training/evaluation logic.

### Custom Metrics
1. Write a metric function and pass its path via `--costume-metrics`.
2. The CLI will import and use your function for evaluation.

## Weighted Loss

NLBSE supports both static and dynamic weighted loss strategies for multi-label classification:

- **EW**: Equal Weights
- **ICF**: Inverse Class Frequency
- **RBF**: Ranking-Based Frequency
- **FAMO, COBA**: Dynamic strategies (see code for details)

You can also provide a custom weighting function as a `.py` file.

See `linear_weights.py` and `dynamic_weights/` for implementation details.

## Example data

The folder [data](data/) contains example data.

<!--
## How to Cite

```
@inbook{key,
  title = "Beyond Balance - Addressing Class Imbalance in Fine-Tuning Deep Learners",
  author = "Moritz Mock, Thomas Borsani, Giuseppe di Fatta, and Barbara Russo",
  year = "2026",
  doi = "...",
  booktitle = "...",
}
```
-->

*Beyond Balance* will be maintained by the CyberSecurity Lab ([https://cslab.projects.unibz.it](https://cslab.projects.unibz.it)) to ensure its long-term accessibility.
