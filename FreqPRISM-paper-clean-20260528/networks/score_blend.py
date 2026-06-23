from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def probabilities_to_logits(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(values, float(eps), 1.0 - float(eps))
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))).astype(np.float32)


def logit_blend(primary_scores: np.ndarray, auxiliary_scores: np.ndarray, alpha: float) -> np.ndarray:
    primary = np.asarray(primary_scores, dtype=np.float32)
    auxiliary = np.asarray(auxiliary_scores, dtype=np.float32)
    if primary.shape != auxiliary.shape:
        raise ValueError("primary and auxiliary score arrays must have the same shape")
    return logits_to_probabilities(probabilities_to_logits(primary) + float(alpha) * probabilities_to_logits(auxiliary))


def select_first_per_label(labels: np.ndarray, scores: np.ndarray, per_label: int | None) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must be 1D arrays with matching length")
    if per_label is None or int(per_label) <= 0:
        return y, s
    selected: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)[: int(per_label)]
        if indices.shape[0] < int(per_label):
            raise ValueError(f"not enough label={label} rows: requested {per_label}, found {indices.shape[0]}")
        selected.append(indices)
    merged = np.concatenate(selected, axis=0)
    return y[merged].astype(np.int64), s[merged].astype(np.float32)


def load_score_npz(path: str | Path, per_label: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    cached = np.load(path, allow_pickle=False)
    labels = cached["labels"].astype(np.int64)
    scores = cached["scores"].astype(np.float32)
    return select_first_per_label(labels, scores, per_label=per_label)


def parse_alpha_grid(text: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(text, str):
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    else:
        values = tuple(float(item) for item in text)
    if not values:
        raise ValueError("alpha grid must contain at least one value")
    return values
