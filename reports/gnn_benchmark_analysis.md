# GNN Benchmark and Graph Structure Analysis

Generated on 2026-05-04 from the repository GNN baseline. Updated to include all three logical metadata datasets.

## Benchmark Run

Commands:

```bash
~/.local/bin/uv run failure-gnn --dataset-path outputs/all_datasets/dataset_init_compat.csv --epochs 40 --output-dir outputs/all_datasets/dataset_init
~/.local/bin/uv run failure-gnn --dataset-path metadata/dataset.csv --epochs 40 --output-dir outputs/all_datasets/dataset
~/.local/bin/uv run failure-gnn --epochs 40 --output-dir outputs/gnn_benchmark
```

Artifacts:

- `outputs/all_datasets/dataset_init/model.pt`
- `outputs/all_datasets/dataset_init/metrics.json`
- `outputs/all_datasets/dataset/model.pt`
- `outputs/all_datasets/dataset/metrics.json`
- `outputs/gnn_benchmark/model.pt`
- `outputs/gnn_benchmark/metrics.json`
- `outputs/all_datasets/diagnostics.json`

Note: `metadata/dataset_init.csv` uses an older schema, so it was normalized into `outputs/all_datasets/dataset_init_compat.csv`. The raw file has no `stage_id`, so `stage_id` was set to `unknown`; `ts_duration`, `passcount`, and `failcount` were mapped to the current training columns.

Configuration:

- Datasets: `metadata/dataset_init.csv`, `metadata/dataset.csv`, `metadata/dataset_filter.csv`
- Graphs: `graphs/*_callgraph.csv`
- Target: `num_fail_class > 0`
- Split: temporal, 70% train / 15% validation / 15% test
- Model: two-layer sparse GCN over each project call graph, mean pooled to a project embedding, concatenated with per-build metadata features
- Best epoch selected by validation loss
- Decision threshold selected on validation

## Cross-Dataset Results

| Dataset | Split | Rows | Positive rate | Best epoch | Threshold | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dataset_init` | Train | 2,334 | 0.891 | 40 | 0.20 | 0.919 | 0.973 | 0.934 | 0.953 |
| `dataset_init` | Validation | 500 | 0.434 | 40 | 0.20 | 0.732 | 0.799 | 0.512 | 0.624 |
| `dataset_init` | Test | 501 | 0.425 | 40 | 0.20 | 0.585 | 0.508 | 0.728 | 0.598 |
| `dataset` | Train | 5,157 | 0.902 | 34 | 0.20 | 0.926 | 0.963 | 0.954 | 0.959 |
| `dataset` | Validation | 1,105 | 0.702 | 34 | 0.20 | 0.837 | 0.906 | 0.857 | 0.881 |
| `dataset` | Test | 1,106 | 0.389 | 34 | 0.20 | 0.741 | 0.961 | 0.347 | 0.509 |
| `dataset_filter` | Train | 5,157 | 0.902 | 34 | 0.20 | 0.926 | 0.963 | 0.954 | 0.959 |
| `dataset_filter` | Validation | 1,105 | 0.702 | 34 | 0.20 | 0.837 | 0.906 | 0.857 | 0.881 |
| `dataset_filter` | Test | 1,106 | 0.389 | 34 | 0.20 | 0.741 | 0.961 | 0.347 | 0.509 |

`dataset.csv` and `dataset_filter.csv` produce identical benchmark behavior for this code path. The model only reads columns shared by both files, and rows are filtered to projects with available call graphs, so the extra filtered-only columns do not influence the current GNN.

## Baselines

All-positive test baseline:

| Dataset | Test accuracy | Test precision | Test recall | Test F1 | GNN test F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dataset_init` | 0.425 | 0.425 | 1.000 | 0.597 | 0.598 |
| `dataset` | 0.389 | 0.389 | 1.000 | 0.560 | 0.509 |
| `dataset_filter` | 0.389 | 0.389 | 1.000 | 0.560 | 0.509 |

On `dataset_init`, the GNN barely beats the all-positive baseline by F1, but only by 0.002. On `dataset` and `dataset_filter`, the GNN is worse than the all-positive baseline on test F1.

## Temporal Distribution Shift

The temporal split changes both the overall label rate and the project mix.

`dataset_init`:

| Split | Main project mix and failure rates |
| --- | --- |
| Train | `kafka`: 1,915 rows at 0.990; `hbase`: 339 rows at 0.546; `karaf`: 79 rows at 0.000 |
| Validation | `hbase`: 244 rows at 0.557; `james`: 97 rows at 0.021; `kafka`: 67 rows at 1.000; `karaf`: 51 rows at 0.000 |
| Test | `hadoop`: 266 rows at 0.459; `james`: 132 rows at 0.152; `hbase`: 78 rows at 0.590; `kafka`: 25 rows at 1.000 |

`dataset` and `dataset_filter`:

| Split | Main project mix and failure rates |
| --- | --- |
| Train | `kafka`: 4,589 rows at 0.964; `hbase`: 540 rows at 0.419; tiny amounts of `karaf` and `jackrabbit-oak` |
| Validation | `kafka`: 733 rows at 0.907; `hbase`: 242 rows at 0.459; several tiny zero-failure projects |
| Test | `james`: 422 rows at 0.052; `hadoop`: 287 rows at 0.467; `hbase`: 257 rows at 0.537; `kafka`: 137 rows at 0.993 |

This explains much of the benchmark behavior. The model sees mostly positive `kafka` builds during training and validation, then test contains many more `james` and `hadoop` builds with different failure rates. The validation threshold is tuned in a high-positive environment, but the test period has a much lower positive rate.

## Per-Project Test Behavior

`dataset_init` test metrics:

| Project | Rows | Positive rate | TP | TN | FP | FN | Precision | Recall | F1 | Mean probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hadoop` | 266 | 0.459 | 112 | 5 | 139 | 10 | 0.446 | 0.918 | 0.601 | 0.343 |
| `hbase` | 78 | 0.590 | 17 | 21 | 11 | 29 | 0.607 | 0.370 | 0.459 | 0.189 |
| `james` | 132 | 0.152 | 1 | 112 | 0 | 19 | 1.000 | 0.050 | 0.095 | 0.094 |
| `kafka` | 25 | 1.000 | 25 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.529 |

`dataset` and `dataset_filter` test metrics:

| Project | Rows | Positive rate | TP | TN | FP | FN | Precision | Recall | F1 | Mean probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hadoop` | 287 | 0.467 | 6 | 152 | 1 | 128 | 0.857 | 0.045 | 0.085 | 0.093 |
| `hbase` | 257 | 0.537 | 6 | 118 | 1 | 132 | 0.857 | 0.043 | 0.083 | 0.097 |
| `james` | 422 | 0.052 | 1 | 397 | 3 | 21 | 0.250 | 0.045 | 0.077 | 0.094 |
| `kafka` | 137 | 0.993 | 136 | 0 | 1 | 0 | 0.993 | 1.000 | 0.996 | 0.860 |
| `karaf` | 3 | 0.000 | 0 | 3 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.074 |

This table is the strongest evidence that the graph embedding is acting like project identity. On `dataset` and `dataset_filter`, the model predicts `kafka` as high-risk and nearly everything else as low-risk. That gives excellent `kafka` recall, but it misses almost all failures in `hadoop`, `hbase`, and `james`.

## Probability Shift

| Dataset | Split | Min | P25 | Median | P75 | Max | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dataset_init` | Train | 0.015 | 0.664 | 0.709 | 0.729 | 0.775 | 0.606 |
| `dataset_init` | Validation | 0.016 | 0.083 | 0.138 | 0.217 | 0.686 | 0.193 |
| `dataset_init` | Test | 0.034 | 0.126 | 0.254 | 0.370 | 0.694 | 0.263 |
| `dataset` / `dataset_filter` | Train | 0.010 | 0.647 | 0.753 | 0.797 | 0.935 | 0.663 |
| `dataset` / `dataset_filter` | Validation | 0.013 | 0.078 | 0.645 | 0.792 | 0.943 | 0.503 |
| `dataset` / `dataset_filter` | Test | 0.010 | 0.074 | 0.094 | 0.134 | 0.943 | 0.189 |

The model's test probabilities collapse downward outside `kafka`. In `dataset` and `dataset_filter`, the median test probability is only 0.094 despite a test positive rate of 0.389.

## Graph Structure Diagnostics

| Project | Nodes | Edges | Avg total degree | Weak components | Largest component | Zero in-degree | Zero out-degree | Max total degree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activemq | 15,951 | 43,270 | 5.43 | 332 | 94.6% | 61.8% | 38.2% | 904 |
| hadoop | 87,016 | 253,694 | 5.83 | 974 | 96.9% | 65.0% | 35.0% | 5,828 |
| hbase | 42,609 | 131,416 | 6.17 | 470 | 96.2% | 64.7% | 35.3% | 3,207 |
| hive | 16,060 | 44,522 | 5.54 | 511 | 91.5% | 53.0% | 47.0% | 1,178 |
| jackrabbit-oak | 34,245 | 94,337 | 5.51 | 426 | 96.1% | 66.0% | 34.0% | 2,122 |
| james | 23,723 | 69,061 | 5.82 | 337 | 95.9% | 65.0% | 35.0% | 2,062 |
| kafka | 29,757 | 87,855 | 5.90 | 577 | 95.0% | 63.6% | 36.4% | 2,889 |
| karaf | 7,870 | 21,794 | 5.54 | 166 | 94.3% | 54.0% | 46.0% | 597 |
| log4j | 9,597 | 22,041 | 4.59 | 157 | 95.3% | 65.7% | 34.3% | 554 |
| tvm | 299 | 471 | 3.15 | 15 | 83.6% | 40.1% | 59.9% | 29 |

## Why the GNN Fails

The model does not fail because PyTorch crashes; it fails because the graph signal is too coarse and mostly static for the prediction task.

1. The graph embedding is project-level, not build-level. Each project has one call graph, so every build from `kafka` gets the same graph embedding, every build from `hbase` gets the same graph embedding, and so on. The GNN cannot represent which methods changed, which tests ran, which files were touched, or which part of the graph was exercised by a specific build.

2. The graph node features are only structural degrees: `log(in_degree)`, `log(out_degree)`, `log(total_degree)`, and a bias term. There are no node identities, package/module labels, test coverage features, changed-method markers, historical failure features on nodes, or edge types. After two GCN layers and mean pooling, the embedding mostly summarizes graph size and degree distribution.

3. The graphs are huge sparse call graphs dominated by one giant weak component. Most projects have 91-97% of nodes in the largest weak component, average total degree around 4.6-6.2, and hundreds of weak components. Mean pooling over tens of thousands of nodes blurs local fault-relevant structure into a project fingerprint.

4. Directionality is discarded. The loader duplicates every directed call edge in reverse and then normalizes the result as an undirected adjacency matrix. For call graphs, caller-to-callee direction matters: failure propagation, dependency fan-in, and test reachability are asymmetric. The current encoder treats `A calls B` like `B calls A`.

5. The graph is confounded with project identity. Because there is one static graph per project, the model can use the graph embedding as a soft project ID. That works while train and validation are dominated by the same high-failure projects, especially `kafka`, but it generalizes poorly when the test period has a different project mix and much lower positive rate.

6. Several graph files are unused by the benchmark. The graph loader reads all graph CSVs, but `metadata/dataset_filter.csv` only contributes rows for projects present in the filtered CI metadata. Projects such as `activemq`, `hive`, `log4j`, and `tvm` have graph embeddings but no benchmark rows, so they do not help training or evaluation.

7. The model has no way to separate static project risk from current build risk. The per-project test metrics show that the learned decision surface mostly says "`kafka` is risky; `hadoop`, `hbase`, and `james` are not." That is not a graph-based failure predictor; it is a project prior with metadata around it.

## Conclusion

The GNN benchmark is not a reliable failure predictor in its current form. On the fully current datasets, test F1 is 0.509, below the all-positive baseline F1 of 0.560, and the failure mode is low recall on a shifted test distribution. On the initial dataset, test F1 is 0.598, effectively tied with the all-positive baseline at 0.597.

The graph structure is the main modeling problem. Static whole-project call graphs are too broad, too sparse, too weakly featured, and too disconnected from individual builds. To make the graph useful, the next version should construct build-specific subgraphs around changed files/methods and covered tests, preserve directed and typed edges, attach node-level signals, and pool over the relevant affected region rather than the entire project.
