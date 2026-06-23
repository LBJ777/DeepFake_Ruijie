from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np


def _validate_labels_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must be 1D arrays with matching length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    return y, s


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y, s = _validate_labels_scores(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    precision = tp / (np.arange(sorted_y.shape[0]) + 1)
    return float(np.sum(precision[sorted_y == 1]) / positives * 100.0)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y, s = _validate_labels_scores(labels, scores)
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            tied = inverse == index
            ranks[tied] = ranks[tied].mean()
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives) * 100.0)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y, s = _validate_labels_scores(labels, scores)
    pred = (s >= float(threshold)).astype(np.int64)
    real = y == 0
    fake = y == 1
    r_acc = float((pred[real] == 0).mean() * 100.0) if bool(real.any()) else 0.0
    f_acc = float((pred[fake] == 1).mean() * 100.0) if bool(fake.any()) else 0.0
    return {
        "acc": float((pred == y).mean() * 100.0),
        "ap": average_precision(y, s),
        "auc": roc_auc(y, s),
        "r_acc": r_acc,
        "f_acc": f_acc,
        "fpr": 100.0 - r_acc,
        "fnr": 100.0 - f_acc,
    }


def write_rows_csv(path: str | Path, rows: list[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    if "generator" in fieldnames:
        fieldnames = ["generator", *[field for field in fieldnames if field != "generator"]]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_target_report(
    output_dir: str | Path,
    packed_scores: Mapping[str, tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.5,
) -> dict[str, float]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for generator, (labels, scores) in sorted(packed_scores.items()):
        rows.append({"generator": generator, **binary_metrics(labels, scores, threshold=threshold)})
    mean = {
        f"mean_{key}": float(np.mean([float(row[key]) for row in rows]))
        for key in ("acc", "ap", "auc", "r_acc", "f_acc", "fpr", "fnr")
    }
    write_rows_csv(out / "per_generator.csv", rows)
    write_rows_csv(out / "overall.csv", [mean])
    return mean
