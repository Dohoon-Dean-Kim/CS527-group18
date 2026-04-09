from __future__ import annotations

import csv
import math
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message="Sparse invariant checks are implicitly disabled.*",
    category=UserWarning,
)

BASE_NUMERIC_FEATURES = (
    "log_build_id",
    "log_build_duration",
    "log_test_suite_duration",
    "log_num_pass_class",
    "build_age_days",
    "build_hour_sin",
    "build_hour_cos",
    "build_weekday_sin",
    "build_weekday_cos",
    "time_since_prev_project_build_hours",
    "prev_project_failed",
    "prev_project_fail_rate",
    "project_recent_fail_rate_5",
    "project_recent_fail_rate_20",
    "global_prev_fail_rate",
    "global_recent_fail_rate_20",
    "time_since_prev_pr_build_hours",
    "prev_pr_failed",
    "pr_prev_fail_rate",
    "pr_recent_fail_rate_5",
    "time_since_prev_stage_build_hours",
    "stage_prev_fail_rate",
    "stage_recent_fail_rate_10",
    "branch_prev_fail_rate",
    "project_stage_prev_fail_rate",
)
GRAPH_FEATURES = (
    "graph_log_num_nodes",
    "graph_log_num_edges",
    "graph_log_avg_degree",
)
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + GRAPH_FEATURES
TOP_STAGE_COUNT = 12
TOP_BRANCH_COUNT = 8


@dataclass(frozen=True)
class GraphData:
    project: str
    node_features: torch.Tensor
    adjacency: torch.Tensor
    graph_features: np.ndarray


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


@dataclass(frozen=True)
class FeatureSchema:
    stage_categories: tuple[str, ...]
    branch_categories: tuple[str, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            NUMERIC_FEATURES
            + tuple(f"stage::{value}" for value in self.stage_categories)
            + tuple(f"branch::{value}" for value in self.branch_categories)
        )


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
    num_edges = len(directed_edges)

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
        graph_features=np.array(
            [
                math.log1p(num_nodes),
                math.log1p(num_edges),
                math.log1p((2.0 * num_edges) / max(1, num_nodes)),
            ],
            dtype=np.float32,
        ),
    )


def read_build_rows(
    dataset_path: str | Path,
    graphs: dict[str, GraphData],
    target_mode: str = "fail",
) -> tuple[list[BuildRow], FeatureSchema]:
    rows: list[BuildRow] = []
    projects = set(graphs)
    path = Path(dataset_path)
    raw_rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            project = row["project"].strip()
            if project not in projects:
                continue
            raw_rows.append(row)
    if not raw_rows:
        raise ValueError(f"No matching build rows found in {path}")

    ordered_rows = sorted(raw_rows, key=lambda row: float(row["build_timestamp"]))
    schema = _build_feature_schema(ordered_rows)
    rows = _engineer_build_rows(ordered_rows, graphs, schema, target_mode)
    return rows, schema


def _build_feature_schema(rows: list[dict[str, str]]) -> FeatureSchema:
    stage_counts = Counter(row["stage_id"].strip() for row in rows)
    branch_counts = Counter(row["pr_base_branch"].strip() for row in rows)
    return FeatureSchema(
        stage_categories=tuple(value for value, _ in stage_counts.most_common(TOP_STAGE_COUNT)),
        branch_categories=tuple(value for value, _ in branch_counts.most_common(TOP_BRANCH_COUNT)),
    )


def _engineer_build_rows(
    ordered_rows: list[dict[str, str]],
    graphs: dict[str, GraphData],
    schema: FeatureSchema,
    target_mode: str,
) -> list[BuildRow]:
    first_timestamp = float(ordered_rows[0]["build_timestamp"])
    project_counts: defaultdict[str, int] = defaultdict(int)
    project_fail_counts: defaultdict[str, int] = defaultdict(int)
    project_last_timestamp: dict[str, float] = {}
    project_recent: defaultdict[str, deque[float]] = defaultdict(deque)
    pr_counts: defaultdict[str, int] = defaultdict(int)
    pr_fail_counts: defaultdict[str, int] = defaultdict(int)
    pr_last_timestamp: dict[str, float] = {}
    pr_recent: defaultdict[str, deque[float]] = defaultdict(deque)
    stage_counts: defaultdict[str, int] = defaultdict(int)
    stage_fail_counts: defaultdict[str, int] = defaultdict(int)
    stage_last_timestamp: dict[str, float] = {}
    stage_recent: defaultdict[str, deque[float]] = defaultdict(deque)
    branch_counts: defaultdict[str, int] = defaultdict(int)
    branch_fail_counts: defaultdict[str, int] = defaultdict(int)
    project_stage_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    project_stage_fail_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    global_recent: deque[float] = deque()
    global_count = 0
    global_fail_count = 0
    stage_index = {value: idx for idx, value in enumerate(schema.stage_categories)}
    branch_index = {value: idx for idx, value in enumerate(schema.branch_categories)}
    engineered_rows: list[BuildRow] = []

    for row in ordered_rows:
        project = row["project"].strip()
        pr_name = row["pr_name"].strip()
        stage_name = row["stage_id"].strip()
        branch_name = row["pr_base_branch"].strip()
        project_stage_key = (project, stage_name)
        timestamp = float(row["build_timestamp"])
        label = _target_value(row, target_mode)

        project_count = project_counts[project]
        project_fail_count = project_fail_counts[project]
        prev_project_failed = project_recent[project][-1] if project_recent[project] else 0.0
        prev_project_fail_rate = project_fail_count / project_count if project_count else 0.0
        recent_project_fail_rate_5 = (
            sum(list(project_recent[project])[-5:]) / min(5, len(project_recent[project]))
            if project_recent[project]
            else 0.0
        )
        recent_project_fail_rate_20 = (
            sum(project_recent[project]) / len(project_recent[project])
            if project_recent[project]
            else 0.0
        )
        prev_global_fail_rate = global_fail_count / global_count if global_count else 0.0
        recent_global_fail_rate_20 = sum(global_recent) / len(global_recent) if global_recent else 0.0
        time_since_prev = (
            max(0.0, (timestamp - project_last_timestamp[project]) / 3600.0)
            if project in project_last_timestamp
            else 0.0
        )
        time_since_prev_pr = (
            max(0.0, (timestamp - pr_last_timestamp[pr_name]) / 3600.0)
            if pr_name in pr_last_timestamp
            else 0.0
        )
        prev_pr_failed = pr_recent[pr_name][-1] if pr_recent[pr_name] else 0.0
        pr_prev_fail_rate = pr_fail_counts[pr_name] / pr_counts[pr_name] if pr_counts[pr_name] else 0.0
        pr_recent_fail_rate_5 = (
            sum(list(pr_recent[pr_name])[-5:]) / min(5, len(pr_recent[pr_name]))
            if pr_recent[pr_name]
            else 0.0
        )
        time_since_prev_stage = (
            max(0.0, (timestamp - stage_last_timestamp[stage_name]) / 3600.0)
            if stage_name in stage_last_timestamp
            else 0.0
        )
        stage_prev_fail_rate = stage_fail_counts[stage_name] / stage_counts[stage_name] if stage_counts[stage_name] else 0.0
        stage_recent_fail_rate_10 = (
            sum(stage_recent[stage_name]) / len(stage_recent[stage_name])
            if stage_recent[stage_name]
            else 0.0
        )
        branch_prev_fail_rate = branch_fail_counts[branch_name] / branch_counts[branch_name] if branch_counts[branch_name] else 0.0
        project_stage_prev_fail_rate = (
            project_stage_fail_counts[project_stage_key] / project_stage_counts[project_stage_key]
            if project_stage_counts[project_stage_key]
            else 0.0
        )

        build_hour = (timestamp % 86400.0) / 3600.0
        build_weekday = ((timestamp / 86400.0) + 4.0) % 7.0
        numeric_features = [
            math.log1p(float(row["build_id"])),
            math.log1p(max(0.0, float(row["build_duration"]))),
            math.log1p(max(0.0, float(row["test_suite_duration_s"]))),
            math.log1p(max(0.0, float(row["num_pass_class"]))),
            (timestamp - first_timestamp) / 86400.0,
            math.sin((2.0 * math.pi * build_hour) / 24.0),
            math.cos((2.0 * math.pi * build_hour) / 24.0),
            math.sin((2.0 * math.pi * build_weekday) / 7.0),
            math.cos((2.0 * math.pi * build_weekday) / 7.0),
            math.log1p(time_since_prev),
            prev_project_failed,
            prev_project_fail_rate,
            recent_project_fail_rate_5,
            recent_project_fail_rate_20,
            prev_global_fail_rate,
            recent_global_fail_rate_20,
            math.log1p(time_since_prev_pr),
            prev_pr_failed,
            pr_prev_fail_rate,
            pr_recent_fail_rate_5,
            math.log1p(time_since_prev_stage),
            stage_prev_fail_rate,
            stage_recent_fail_rate_10,
            branch_prev_fail_rate,
            project_stage_prev_fail_rate,
            *graphs[project].graph_features.tolist(),
        ]

        stage_features = [0.0] * len(schema.stage_categories)
        if stage_name in stage_index:
            stage_features[stage_index[stage_name]] = 1.0

        branch_features = [0.0] * len(schema.branch_categories)
        if branch_name in branch_index:
            branch_features[branch_index[branch_name]] = 1.0

        engineered_rows.append(
            BuildRow(
                project=project,
                build_timestamp=timestamp,
                feature_values=tuple(numeric_features + stage_features + branch_features),
                label=label,
            )
        )

        project_counts[project] += 1
        project_fail_counts[project] += int(label)
        project_last_timestamp[project] = timestamp
        project_recent[project].append(label)
        if len(project_recent[project]) > 20:
            project_recent[project].popleft()
        pr_counts[pr_name] += 1
        pr_fail_counts[pr_name] += int(label)
        pr_last_timestamp[pr_name] = timestamp
        pr_recent[pr_name].append(label)
        if len(pr_recent[pr_name]) > 5:
            pr_recent[pr_name].popleft()
        stage_counts[stage_name] += 1
        stage_fail_counts[stage_name] += int(label)
        stage_last_timestamp[stage_name] = timestamp
        stage_recent[stage_name].append(label)
        if len(stage_recent[stage_name]) > 10:
            stage_recent[stage_name].popleft()
        branch_counts[branch_name] += 1
        branch_fail_counts[branch_name] += int(label)
        project_stage_counts[project_stage_key] += 1
        project_stage_fail_counts[project_stage_key] += int(label)
        global_count += 1
        global_fail_count += int(label)
        global_recent.append(label)
        if len(global_recent) > 20:
            global_recent.popleft()

    return engineered_rows


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
