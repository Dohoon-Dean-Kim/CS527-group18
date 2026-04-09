from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


NUMERIC_FEATURES = (
    "build_id",
    "build_duration",
    "build_timestamp",
    "test_suite_duration_s",
)


@dataclass(frozen=True)
class GraphData:
    project: str
    node_features: torch.Tensor
    adjacency: torch.Tensor


@dataclass(frozen=True)
class BuildRow:
    project: str
    build_timestamp: float
    feature_values: tuple[float, ...]
    label: float


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class SplitTensors:
    projects: tuple[str, ...]
    build_features: torch.Tensor
    labels: torch.Tensor


def load_graphs(graphs_dir: str | Path, projects: set[str] | None = None) -> dict[str, GraphData]:
    graphs_path = Path(graphs_dir)
    loaded: dict[str, GraphData] = {}
    for csv_path in sorted(graphs_path.glob("*_callgraph.csv")):
        project = csv_path.name.removesuffix("_callgraph.csv")
        if projects is not None and project not in projects:
            continue
        loaded[project] = _load_single_graph(project, csv_path)
    if not loaded:
        raise FileNotFoundError(f"No call graph CSVs found in {graphs_path}")
    return loaded


def _load_single_graph(project: str, csv_path: Path) -> GraphData:
    node_to_idx: dict[str, int] = {}
    directed_edges: list[tuple[int, int]] = []

    def node_idx(name: str) -> int:
        if name not in node_to_idx:
            node_to_idx[name] = len(node_to_idx)
        return node_to_idx[name]

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            source = row[0].strip()
            target = row[1].strip()
            if not source or not target:
                continue
            directed_edges.append((node_idx(source), node_idx(target)))

    num_nodes = len(node_to_idx)
    if num_nodes == 0:
        raise ValueError(f"Graph {csv_path} does not contain any nodes")

    in_degree = np.zeros(num_nodes, dtype=np.float32)
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    for source_idx, target_idx in directed_edges:
        out_degree[source_idx] += 1.0
        in_degree[target_idx] += 1.0

    total_degree = in_degree + out_degree
    node_features = np.stack(
        [
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(total_degree),
            np.ones(num_nodes, dtype=np.float32),
        ],
        axis=1,
    )

    undirected_edges = directed_edges + [(dst, src) for src, dst in directed_edges]
    undirected_edges.extend((idx, idx) for idx in range(num_nodes))

    indices = torch.tensor(undirected_edges, dtype=torch.long).t()
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    row_idx, col_idx = adjacency.indices()
    edge_values = adjacency.values()
    degree = torch.bincount(row_idx, weights=edge_values, minlength=num_nodes)
    inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
    norm_values = edge_values * inv_sqrt_degree[row_idx] * inv_sqrt_degree[col_idx]
    normalized_adjacency = torch.sparse_coo_tensor(
        adjacency.indices(),
        norm_values,
        adjacency.shape,
    ).coalesce()

    return GraphData(
        project=project,
        node_features=torch.tensor(node_features, dtype=torch.float32),
        adjacency=normalized_adjacency,
    )


def read_build_rows(
    dataset_path: str | Path,
    projects: set[str],
    target_mode: str = "fail",
) -> list[BuildRow]:
    rows: list[BuildRow] = []
    path = Path(dataset_path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            project = row["project"].strip()
            if project not in projects:
                continue
            label = _target_value(row, target_mode)
            features = tuple(float(row[column]) for column in NUMERIC_FEATURES)
            rows.append(
                BuildRow(
                    project=project,
                    build_timestamp=float(row["build_timestamp"]),
                    feature_values=features,
                    label=label,
                )
            )
    if not rows:
        raise ValueError(f"No matching build rows found in {path}")
    return rows


def _target_value(row: dict[str, str], target_mode: str) -> float:
    fail_count = float(row["num_fail_class"])
    transient_count = float(row["num_trans_class"])
    if target_mode == "fail":
        return 1.0 if fail_count > 0 else 0.0
    if target_mode == "fail_or_transient":
        return 1.0 if (fail_count + transient_count) > 0 else 0.0
    raise ValueError(f"Unsupported target mode: {target_mode}")


def temporal_split(rows: list[BuildRow], train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[list[BuildRow], list[BuildRow], list[BuildRow]]:
    ordered = sorted(rows, key=lambda row: row.build_timestamp)
    train_end = max(1, int(len(ordered) * train_ratio))
    val_end = max(train_end + 1, int(len(ordered) * (train_ratio + val_ratio)))
    train_rows = ordered[:train_end]
    val_rows = ordered[train_end:val_end]
    test_rows = ordered[val_end:]
    return train_rows, val_rows, test_rows


def fit_scaler(rows: list[BuildRow]) -> FeatureScaler:
    array = np.array([row.feature_values for row in rows], dtype=np.float32)
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std[std == 0] = 1.0
    return FeatureScaler(mean=mean, std=std)


def rows_to_tensors(rows: list[BuildRow], scaler: FeatureScaler) -> SplitTensors:
    features = np.array([row.feature_values for row in rows], dtype=np.float32)
    normalized = (features - scaler.mean) / scaler.std
    labels = np.array([row.label for row in rows], dtype=np.float32)
    return SplitTensors(
        projects=tuple(row.project for row in rows),
        build_features=torch.tensor(normalized, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.float32),
    )


def describe_split(split: SplitTensors) -> dict[str, float]:
    positive_count = float(split.labels.sum().item())
    total_count = float(split.labels.numel())
    return {
        "count": total_count,
        "positive_rate": positive_count / total_count if total_count else math.nan,
    }
