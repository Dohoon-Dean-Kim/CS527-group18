# CS527-group18

Course project for CS527 focused on using a simple graph neural network to predict failing builds from static call graphs and CI metadata.

## What is included

- A `uv`-managed Python package under `src/failure_gnn`
- A lightweight GCN-style encoder built with plain PyTorch sparse matrix ops
- A training CLI that reads the existing `graphs/*.csv` call graphs and `metadata/dataset_filter.csv`
- Temporal train/validation/test splitting and JSON metric export

## Model shape

Each Apache project contributes one static call graph. The model:

1. Builds structural node features from the call graph (`in-degree`, `out-degree`, `total degree`, bias term)
2. Runs two sparse graph-convolution layers over that project graph
3. Mean-pools the node states into a project embedding
4. Concatenates the project embedding with per-build metadata features
5. Predicts whether a build contains at least one failing test class

This is intentionally a simple baseline, but it gives you a clean place to iterate.

## Run with uv

The project is set up to run directly with `uv`:

```bash
uv sync
uv run failure-gnn --epochs 40
```

You can also pass custom paths:

```bash
uv run failure-gnn \
  --dataset-path metadata/dataset_filter.csv \
  --graphs-dir graphs \
  --output-dir outputs/simple_gnn
```

The training command writes:

- `outputs/simple_gnn/model.pt`
- `outputs/simple_gnn/metrics.json`

## Notes

- The default target is `num_fail_class > 0`
- The split is temporal, using build timestamps
- The graph encoder is static per project, while the classifier uses per-build metadata
