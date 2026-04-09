from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from failure_gnn.data import (
    FeatureSchema,
    describe_split,
    fit_scaler,
    load_graphs,
    read_build_rows,
    rows_to_tensors,
    temporal_split,
)
from failure_gnn.model import BuildClassifier, GraphEncoder


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple GNN baseline for CI failure prediction.")
    parser.add_argument("--dataset-path", default="metadata/dataset_filter.csv")
    parser.add_argument("--graphs-dir", default="graphs")
    parser.add_argument("--output-dir", default="outputs/simple_gnn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--target",
        choices=("fail", "fail_or_transient"),
        default="fail",
        help="Prediction target to learn from the metadata CSV.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_project_embeddings(encoder: GraphEncoder, graphs: dict[str, object]) -> dict[str, torch.Tensor]:
    return {project: encoder(graph) for project, graph in graphs.items()}


def stack_project_embeddings(projects: tuple[str, ...], embedding_map: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([embedding_map[project] for project in projects], dim=0)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor, threshold: float) -> Metrics:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    true_positive = float(((predictions == 1) & (labels == 1)).sum().item())
    true_negative = float(((predictions == 0) & (labels == 0)).sum().item())
    false_positive = float(((predictions == 1) & (labels == 0)).sum().item())
    false_negative = float(((predictions == 0) & (labels == 1)).sum().item())

    accuracy = (true_positive + true_negative) / max(1.0, float(labels.numel()))
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return Metrics(
        loss=float(loss.item()),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
    )


def select_best_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    thresholds = np.linspace(0.2, 0.8, 25)
    best_threshold = 0.5
    best_score = -1.0
    best_precision = -1.0
    probabilities = torch.sigmoid(logits)
    for threshold in thresholds:
        predictions = (probabilities >= float(threshold)).float()
        true_positive = float(((predictions == 1) & (labels == 1)).sum().item())
        false_positive = float(((predictions == 1) & (labels == 0)).sum().item())
        false_negative = float(((predictions == 0) & (labels == 1)).sum().item())
        precision = true_positive / max(1.0, true_positive + false_positive)
        recall = true_positive / max(1.0, true_positive + false_negative)
        score = 0.0 if precision + recall == 0 else (2.0 * precision * recall / (precision + recall))
        if score > best_score or (score == best_score and precision > best_precision):
            best_score = score
            best_precision = precision
            best_threshold = float(threshold)
    return best_threshold


def evaluate(
    encoder: GraphEncoder,
    classifier: BuildClassifier,
    graphs: dict[str, object],
    split,
    loss_fn: nn.Module,
    threshold: float,
) -> Metrics:
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        embeddings = build_project_embeddings(encoder, graphs)
        graph_embeddings = stack_project_embeddings(split.projects, embeddings)
        logits = classifier(graph_embeddings, split.build_features)
        loss = loss_fn(logits, split.labels)
        return compute_metrics(logits, split.labels, loss, threshold)


def save_outputs(
    output_dir: Path,
    encoder: GraphEncoder,
    classifier: BuildClassifier,
    metrics_payload: dict[str, object],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    feature_names: tuple[str, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "feature_mean": scaler_mean.tolist(),
            "feature_std": scaler_std.tolist(),
            "feature_names": list(feature_names),
        },
        output_dir / "model.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    graphs = load_graphs(args.graphs_dir)
    rows, schema = read_build_rows(args.dataset_path, graphs, target_mode=args.target)
    train_rows, val_rows, test_rows = temporal_split(rows)
    scaler = fit_scaler(train_rows)
    train_split = rows_to_tensors(train_rows, scaler)
    val_split = rows_to_tensors(val_rows, scaler)
    test_split = rows_to_tensors(test_rows, scaler)

    encoder = GraphEncoder(in_dim=4, hidden_dim=args.hidden_dim, dropout=args.dropout)
    classifier = BuildClassifier(
        graph_dim=args.hidden_dim,
        build_feature_dim=train_split.build_features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    positive_count = train_split.labels.sum().item()
    negative_count = float(train_split.labels.numel()) - positive_count
    pos_weight = torch.tensor([negative_count / max(1.0, positive_count)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        optimizer.zero_grad()

        embedding_map = build_project_embeddings(encoder, graphs)
        train_graph_embeddings = stack_project_embeddings(train_split.projects, embedding_map)
        train_logits = classifier(train_graph_embeddings, train_split.build_features)
        train_loss = loss_fn(train_logits, train_split.labels)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            encoder.eval()
            classifier.eval()
            val_embedding_map = build_project_embeddings(encoder, graphs)
            val_graph_embeddings = stack_project_embeddings(val_split.projects, val_embedding_map)
            val_logits = classifier(val_graph_embeddings, val_split.build_features)
            val_loss = loss_fn(val_logits, val_split.labels)
            epoch_threshold = select_best_threshold(val_logits, val_split.labels)
        train_metrics = compute_metrics(train_logits.detach(), train_split.labels, train_loss.detach(), epoch_threshold)
        val_metrics = compute_metrics(val_logits, val_split.labels, val_loss, epoch_threshold)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} train_f1={train_metrics.f1:.3f} "
            f"val_loss={val_metrics.loss:.4f} val_f1={val_metrics.f1:.3f} "
            f"threshold={epoch_threshold:.2f}"
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_state = {
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": epoch,
            }
            best_threshold = epoch_threshold
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    encoder.load_state_dict(best_state["encoder"])
    classifier.load_state_dict(best_state["classifier"])

    final_train_metrics = evaluate(encoder, classifier, graphs, train_split, loss_fn, best_threshold)
    final_val_metrics = evaluate(encoder, classifier, graphs, val_split, loss_fn, best_threshold)
    final_test_metrics = evaluate(encoder, classifier, graphs, test_split, loss_fn, best_threshold)

    metrics_payload = {
        "config": {
            "dataset_path": args.dataset_path,
            "graphs_dir": args.graphs_dir,
            "target": args.target,
            "epochs": args.epochs,
            "best_epoch": best_state["epoch"],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "decision_threshold": best_threshold,
            "feature_names": list(schema.feature_names),
        },
        "splits": {
            "train": describe_split(train_split),
            "val": describe_split(val_split),
            "test": describe_split(test_split),
        },
        "metrics": {
            "train": asdict(final_train_metrics),
            "val": asdict(final_val_metrics),
            "test": asdict(final_test_metrics),
        },
    }

    output_dir = Path(args.output_dir)
    save_outputs(output_dir, encoder, classifier, metrics_payload, scaler.mean, scaler.std, schema.feature_names)

    print(json.dumps(metrics_payload["metrics"], indent=2))
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
