from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from networks.native_tiles import combine_whole_tile_aux_signed_delta_guard_scores
from networks.score_blend import logit_blend, logits_to_probabilities, probabilities_to_logits


COMPONENT_SCORE_KEYS = ("W", "T", "S", "R", "max_side", "final_fixed")


@dataclass(frozen=True)
class FusionParams:
    beta: float
    alpha_low_pos: float
    alpha_low_neg: float
    alpha_high_pos: float
    alpha_high_neg: float
    alpha_high_neg_guard: float
    tile_delta_threshold: float
    high_res_threshold: float
    gamma: float
    threshold: float = 0.50

    @classmethod
    def from_detector_config(cls, config: object) -> "FusionParams":
        return cls(
            beta=float(getattr(config, "beta")),
            alpha_low_pos=float(getattr(config, "alpha_low_pos")),
            alpha_low_neg=float(getattr(config, "alpha_low_neg")),
            alpha_high_pos=float(getattr(config, "alpha_high_pos")),
            alpha_high_neg=float(getattr(config, "alpha_high_neg")),
            alpha_high_neg_guard=float(getattr(config, "alpha_high_neg_guard")),
            tile_delta_threshold=float(getattr(config, "tile_delta_threshold")),
            high_res_threshold=float(getattr(config, "high_res_threshold")),
            gamma=float(getattr(config, "gamma")),
            threshold=float(getattr(config, "threshold")),
        )


@dataclass(frozen=True)
class WeightParams:
    tile_scale: float = 1.0
    semantic_pos_scale: float = 1.0
    semantic_neg_scale: float = 1.0
    residual_scale: float = 1.0

    @classmethod
    def default(cls) -> "WeightParams":
        return cls()

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "WeightParams":
        return cls(
            tile_scale=float(values.get("tile_scale", 1.0)),
            semantic_pos_scale=float(values.get("semantic_pos_scale", 1.0)),
            semantic_neg_scale=float(values.get("semantic_neg_scale", 1.0)),
            residual_scale=float(values.get("residual_scale", 1.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "tile_scale": float(self.tile_scale),
            "semantic_pos_scale": float(self.semantic_pos_scale),
            "semantic_neg_scale": float(self.semantic_neg_scale),
            "residual_scale": float(self.residual_scale),
        }


@dataclass(frozen=True)
class WeightSearchConfig:
    tile_scale_grid: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10)
    semantic_pos_scale_grid: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10)
    semantic_neg_scale_grid: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10)
    residual_scale_grid: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10)
    lambda_drift: float = 1.0
    lambda_flip: float = 1.0
    lambda_anchor: float = 0.25
    max_source_ba_drop: float = 0.2
    max_flip_rate: float = 0.01
    max_mean_score_drift: float = 0.01
    min_group_size: int = 25


@dataclass(frozen=True)
class WeightSearchResult:
    selected: WeightParams
    selected_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    candidates: list[dict[str, float | int | bool]]


def _component_array(components: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in components:
        raise KeyError(f"missing component score key: {key}")
    values = np.asarray(components[key], dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"{key} must be a 1D array")
    return values


def validate_component_scores(components: Mapping[str, np.ndarray], *, require_final_fixed: bool = False) -> int:
    required = ("W", "T", "S", "R", "max_side")
    if require_final_fixed:
        required = (*required, "final_fixed")
    lengths = {_component_array(components, key).shape[0] for key in required}
    if len(lengths) != 1:
        raise ValueError("component score arrays must have the same length")
    return int(next(iter(lengths)))


def compute_fixed_scores(components: Mapping[str, np.ndarray], params: FusionParams) -> np.ndarray:
    validate_component_scores(components)
    base = combine_whole_tile_aux_signed_delta_guard_scores(
        _component_array(components, "W"),
        _component_array(components, "T"),
        _component_array(components, "S"),
        high_res_mask=_component_array(components, "max_side") > float(params.high_res_threshold),
        beta=float(params.beta),
        alpha_low_pos=float(params.alpha_low_pos),
        alpha_low_neg=float(params.alpha_low_neg),
        alpha_high_pos=float(params.alpha_high_pos),
        alpha_high_neg=float(params.alpha_high_neg),
        alpha_high_neg_guard=float(params.alpha_high_neg_guard),
        tile_delta_threshold=float(params.tile_delta_threshold),
    )
    return logit_blend(base, _component_array(components, "R"), float(params.gamma)).astype(np.float32)


def compute_learned_weight_scores(
    components: Mapping[str, np.ndarray],
    params: FusionParams,
    weights: WeightParams,
) -> np.ndarray:
    validate_component_scores(components)
    whole_logit = probabilities_to_logits(_component_array(components, "W")).astype(np.float64)
    tile_logit = probabilities_to_logits(_component_array(components, "T")).astype(np.float64)
    sem_logit = probabilities_to_logits(_component_array(components, "S")).astype(np.float64)
    residual_logit = probabilities_to_logits(_component_array(components, "R")).astype(np.float64)
    max_side = _component_array(components, "max_side")

    tile_delta = np.maximum(0.0, tile_logit - whole_logit)
    high_res = max_side > float(params.high_res_threshold)
    relax_negative = high_res & (tile_delta > float(params.tile_delta_threshold))
    pos_alpha = np.where(high_res, float(params.alpha_high_pos), float(params.alpha_low_pos)).astype(np.float64)
    neg_alpha = np.where(
        relax_negative,
        float(params.alpha_high_neg),
        np.where(high_res, float(params.alpha_high_neg_guard), float(params.alpha_low_neg)),
    ).astype(np.float64)

    semantic_pos_term = pos_alpha * np.maximum(0.0, sem_logit)
    semantic_neg_term = neg_alpha * np.minimum(0.0, sem_logit)
    final_logit = (
        whole_logit
        + float(weights.tile_scale) * float(params.beta) * tile_delta
        + float(weights.semantic_pos_scale) * semantic_pos_term
        + float(weights.semantic_neg_scale) * semantic_neg_term
        + float(weights.residual_scale) * float(params.gamma) * residual_logit
    )
    return logits_to_probabilities(final_logit.astype(np.float32)).astype(np.float32)


def balanced_accuracy(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray, threshold: float = 0.5) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must be 1D arrays with matching length")
    pred = (s >= float(threshold)).astype(np.int64)
    pieces: list[float] = []
    for label in (0, 1):
        mask = y == label
        if bool(mask.any()):
            pieces.append(float((pred[mask] == label).mean() * 100.0))
    return float(np.mean(pieces)) if pieces else 0.0


def group_balanced_accuracies(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str] | np.ndarray | None,
    *,
    threshold: float,
    min_group_size: int,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    if groups is None:
        group_values = np.asarray(["all"] * y.shape[0], dtype=object)
    else:
        group_values = np.asarray(groups, dtype=object)
    if group_values.ndim != 1 or group_values.shape[0] != y.shape[0]:
        raise ValueError("groups must be 1D with one value per label")

    metrics: dict[str, float] = {}
    for group in sorted({str(item) for item in group_values.tolist()}):
        mask = group_values == group
        if int(mask.sum()) < int(min_group_size):
            continue
        metrics[group] = balanced_accuracy(y[mask], np.asarray(scores)[mask], threshold=threshold)
    if not metrics:
        metrics["all"] = balanced_accuracy(y, scores, threshold=threshold)
    return metrics


def make_source_diagnostic_groups(
    labels: Sequence[int] | np.ndarray,
    components: Mapping[str, np.ndarray],
    *,
    source_groups: Sequence[str] | np.ndarray | None = None,
    fixed_scores: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    n = validate_component_scores(components)
    if y.shape[0] != n:
        raise ValueError("labels length must match component scores")
    class_values = (
        np.asarray(source_groups, dtype=object)
        if source_groups is not None
        else np.asarray(["source"] * n, dtype=object)
    )
    if class_values.shape[0] != n:
        raise ValueError("source_groups length must match component scores")
    scores = np.asarray(fixed_scores if fixed_scores is not None else components.get("final_fixed"), dtype=np.float32)
    if scores.ndim != 1 or scores.shape[0] != n:
        scores = compute_fixed_scores(components, FusionParams(0.2, 0.15, 0.15, 0.2, 0.0, 0.2, 0.0, 960.0, 0.08))

    max_side = _component_array(components, "max_side")
    confidence = np.abs(scores - 0.5)
    w = _component_array(components, "W")
    t = _component_array(components, "T")
    s = _component_array(components, "S")
    r = _component_array(components, "R")
    top_component = np.asarray(["artifact"] * n, dtype=object)
    stacked = np.stack([np.abs(t - w), np.abs(s - 0.5), np.abs(r - 0.5)], axis=1)
    names = np.asarray(["tile", "semantic", "residual"], dtype=object)
    top_component = names[np.argmax(stacked, axis=1)]

    groups: list[str] = []
    for index in range(n):
        res_bin = "short" if max_side[index] < 512 else ("medium" if max_side[index] < 960 else "long")
        conf_bin = "low" if confidence[index] < 0.05 else ("medium" if confidence[index] < 0.20 else "high")
        groups.append(
            "class={};label={};res={};conf={};component={}".format(
                class_values[index],
                int(y[index]),
                res_bin,
                conf_bin,
                top_component[index],
            )
        )
    return np.asarray(groups, dtype=object)


def _score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    baseline_scores: np.ndarray,
    groups: Sequence[str] | np.ndarray | None,
    *,
    threshold: float,
    min_group_size: int,
) -> dict[str, float]:
    group_metrics = group_balanced_accuracies(
        labels,
        scores,
        groups,
        threshold=float(threshold),
        min_group_size=int(min_group_size),
    )
    pred = np.asarray(scores) >= float(threshold)
    baseline_pred = np.asarray(baseline_scores) >= float(threshold)
    return {
        "overall_ba": balanced_accuracy(labels, scores, threshold=threshold),
        "worst_group_ba": float(min(group_metrics.values())),
        "mean_score_drift": float(np.mean(np.abs(np.asarray(scores, dtype=np.float32) - baseline_scores))),
        "flip_rate": float(np.mean(pred != baseline_pred)),
    }


def _weight_distance(weights: WeightParams) -> float:
    default = WeightParams.default()
    return float(
        abs(float(weights.tile_scale) - default.tile_scale)
        + abs(float(weights.semantic_pos_scale) - default.semantic_pos_scale)
        + abs(float(weights.semantic_neg_scale) - default.semantic_neg_scale)
        + abs(float(weights.residual_scale) - default.residual_scale)
    )


def search_weight_params(
    labels: Sequence[int] | np.ndarray,
    components: Mapping[str, np.ndarray],
    params: FusionParams,
    *,
    groups: Sequence[str] | np.ndarray | None = None,
    config: WeightSearchConfig = WeightSearchConfig(),
) -> WeightSearchResult:
    y = np.asarray(labels, dtype=np.int64)
    n = validate_component_scores(components)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("labels must be 1D with one value per component score")

    baseline_scores = (
        _component_array(components, "final_fixed") if "final_fixed" in components else compute_fixed_scores(components, params)
    )
    baseline_metrics = _score_metrics(
        y,
        baseline_scores,
        baseline_scores,
        groups,
        threshold=float(params.threshold),
        min_group_size=int(config.min_group_size),
    )
    selected = WeightParams.default()
    selected_metrics = _score_metrics(
        y,
        baseline_scores,
        baseline_scores,
        groups,
        threshold=float(params.threshold),
        min_group_size=int(config.min_group_size),
    )
    best_objective = float("-inf")
    candidates: list[dict[str, float | int | bool]] = []

    for tile_scale, semantic_pos_scale, semantic_neg_scale, residual_scale in product(
        config.tile_scale_grid,
        config.semantic_pos_scale_grid,
        config.semantic_neg_scale_grid,
        config.residual_scale_grid,
    ):
        weights = WeightParams(
            tile_scale=float(tile_scale),
            semantic_pos_scale=float(semantic_pos_scale),
            semantic_neg_scale=float(semantic_neg_scale),
            residual_scale=float(residual_scale),
        )
        if weights == WeightParams.default():
            scores = baseline_scores
        else:
            scores = compute_learned_weight_scores(components, params, weights)
        metrics = _score_metrics(
            y,
            scores,
            baseline_scores,
            groups,
            threshold=float(params.threshold),
            min_group_size=int(config.min_group_size),
        )
        accepted = (
            metrics["overall_ba"] + 1e-9 >= baseline_metrics["overall_ba"] - float(config.max_source_ba_drop)
            and metrics["worst_group_ba"] + 1e-9 >= baseline_metrics["worst_group_ba"]
            and metrics["mean_score_drift"] <= float(config.max_mean_score_drift) + 1e-12
            and metrics["flip_rate"] <= float(config.max_flip_rate) + 1e-12
        )
        distance = _weight_distance(weights)
        objective = (
            metrics["worst_group_ba"]
            - float(config.lambda_drift) * metrics["mean_score_drift"] * 100.0
            - float(config.lambda_flip) * metrics["flip_rate"] * 100.0
            - float(config.lambda_anchor) * distance
        )
        row: dict[str, float | int | bool] = {
            **weights.to_dict(),
            **metrics,
            "anchor_distance": float(distance),
            "objective": float(objective),
            "accepted": bool(accepted),
        }
        candidates.append(row)
        if not accepted:
            continue
        better = objective > best_objective + 1e-9
        tied = abs(objective - best_objective) <= 1e-9
        if better or (
            tied
            and (
                metrics["mean_score_drift"],
                metrics["flip_rate"],
                distance,
            )
            < (
                selected_metrics["mean_score_drift"],
                selected_metrics["flip_rate"],
                _weight_distance(selected),
            )
        ):
            selected = weights
            selected_metrics = metrics
            best_objective = float(objective)

    return WeightSearchResult(
        selected=selected,
        selected_metrics=selected_metrics,
        baseline_metrics=baseline_metrics,
        candidates=candidates,
    )


def load_component_directory(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    root = Path(path)
    labels: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in COMPONENT_SCORE_KEYS}
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no component npz files found in {root}")
    for file_path in files:
        cached = np.load(file_path, allow_pickle=False)
        labels.append(cached["labels"].astype(np.int64))
        n = int(labels[-1].shape[0])
        if "paths" in cached.files:
            paths.append(cached["paths"].astype(str))
        else:
            paths.append(np.asarray([""] * n, dtype=str))
        if "groups" in cached.files:
            groups.append(cached["groups"].astype(str))
        else:
            groups.append(np.asarray([file_path.stem] * n, dtype=str))
        for key in COMPONENT_SCORE_KEYS:
            if key in cached.files:
                chunks[key].append(cached[key].astype(np.float32))
    merged = {key: np.concatenate(value, axis=0) for key, value in chunks.items() if value}
    return (
        np.concatenate(labels, axis=0),
        merged,
        np.concatenate(paths, axis=0),
        np.concatenate(groups, axis=0),
    )
