from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.component_scores import (
    FusionParams,
    WeightParams,
    balanced_accuracy,
    compute_fixed_scores,
    compute_learned_weight_scores,
    group_balanced_accuracies,
    make_source_diagnostic_groups,
    validate_component_scores,
)
from utils.metrics import write_rows_csv, write_target_report


@dataclass(frozen=True)
class FullFusionWeightParams:
    beta_scale: float = 1.0
    alpha_low_pos_scale: float = 1.0
    alpha_low_neg_scale: float = 1.0
    alpha_high_pos_scale: float = 1.0
    alpha_high_neg: float = 0.0
    alpha_high_neg_guard_scale: float = 1.0
    gamma_scale: float = 1.0

    @classmethod
    def default(cls) -> "FullFusionWeightParams":
        return cls()

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "FullFusionWeightParams":
        return cls(
            beta_scale=float(values.get("beta_scale", 1.0)),
            alpha_low_pos_scale=float(values.get("alpha_low_pos_scale", 1.0)),
            alpha_low_neg_scale=float(values.get("alpha_low_neg_scale", 1.0)),
            alpha_high_pos_scale=float(values.get("alpha_high_pos_scale", 1.0)),
            alpha_high_neg=float(values.get("alpha_high_neg", 0.0)),
            alpha_high_neg_guard_scale=float(values.get("alpha_high_neg_guard_scale", 1.0)),
            gamma_scale=float(values.get("gamma_scale", 1.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "beta_scale": float(self.beta_scale),
            "alpha_low_pos_scale": float(self.alpha_low_pos_scale),
            "alpha_low_neg_scale": float(self.alpha_low_neg_scale),
            "alpha_high_pos_scale": float(self.alpha_high_pos_scale),
            "alpha_high_neg": float(self.alpha_high_neg),
            "alpha_high_neg_guard_scale": float(self.alpha_high_neg_guard_scale),
            "gamma_scale": float(self.gamma_scale),
        }

    def to_fusion_params(self, anchor: FusionParams) -> FusionParams:
        return FusionParams(
            beta=float(anchor.beta) * float(self.beta_scale),
            alpha_low_pos=float(anchor.alpha_low_pos) * float(self.alpha_low_pos_scale),
            alpha_low_neg=float(anchor.alpha_low_neg) * float(self.alpha_low_neg_scale),
            alpha_high_pos=float(anchor.alpha_high_pos) * float(self.alpha_high_pos_scale),
            alpha_high_neg=float(self.alpha_high_neg),
            alpha_high_neg_guard=float(anchor.alpha_high_neg_guard) * float(self.alpha_high_neg_guard_scale),
            tile_delta_threshold=float(anchor.tile_delta_threshold),
            high_res_threshold=float(anchor.high_res_threshold),
            gamma=float(anchor.gamma) * float(self.gamma_scale),
            threshold=float(anchor.threshold),
        )


@dataclass(frozen=True)
class FullFusionWeightSearchConfig:
    beta_scale_grid: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)
    alpha_low_pos_scale_grid: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)
    alpha_low_neg_scale_grid: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)
    alpha_high_pos_scale_grid: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)
    alpha_high_neg_grid: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20)
    alpha_high_neg_guard_scale_grid: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)
    gamma_scale_grid: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)
    max_rounds: int = 2
    lambda_drift: float = 1.0
    lambda_flip: float = 1.0
    lambda_anchor: float = 0.25
    max_source_ba_drop: float = 0.2
    max_flip_rate: float = 0.01
    max_mean_score_drift: float = 0.01
    min_group_size: int = 25

    def grid_dict(self) -> dict[str, list[float]]:
        return {
            "beta_scale_grid": [float(value) for value in self.beta_scale_grid],
            "alpha_low_pos_scale_grid": [float(value) for value in self.alpha_low_pos_scale_grid],
            "alpha_low_neg_scale_grid": [float(value) for value in self.alpha_low_neg_scale_grid],
            "alpha_high_pos_scale_grid": [float(value) for value in self.alpha_high_pos_scale_grid],
            "alpha_high_neg_grid": [float(value) for value in self.alpha_high_neg_grid],
            "alpha_high_neg_guard_scale_grid": [float(value) for value in self.alpha_high_neg_guard_scale_grid],
            "gamma_scale_grid": [float(value) for value in self.gamma_scale_grid],
        }

    def constraint_dict(self) -> dict[str, float | int]:
        return {
            "max_rounds": int(self.max_rounds),
            "lambda_drift": float(self.lambda_drift),
            "lambda_flip": float(self.lambda_flip),
            "lambda_anchor": float(self.lambda_anchor),
            "max_source_ba_drop": float(self.max_source_ba_drop),
            "max_flip_rate": float(self.max_flip_rate),
            "max_mean_score_drift": float(self.max_mean_score_drift),
            "min_group_size": int(self.min_group_size),
        }


@dataclass(frozen=True)
class FullFusionWeightSearchResult:
    selected: FullFusionWeightParams
    selected_params: FusionParams
    selected_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    candidates: list[dict[str, float | int | bool | str]]
    target_labels_used_for_selection: bool = False


def score_full_fusion_weights(
    components: Mapping[str, np.ndarray],
    anchor_params: FusionParams,
    weights: FullFusionWeightParams,
) -> np.ndarray:
    return compute_learned_weight_scores(components, weights.to_fusion_params(anchor_params), WeightParams.default())


def _score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    baseline_scores: np.ndarray,
    groups: Sequence[str] | np.ndarray,
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


def _distance(weights: FullFusionWeightParams, anchor: FusionParams) -> float:
    distance = (
        abs(float(weights.beta_scale) - 1.0)
        + abs(float(weights.alpha_low_pos_scale) - 1.0)
        + abs(float(weights.alpha_low_neg_scale) - 1.0)
        + abs(float(weights.alpha_high_pos_scale) - 1.0)
        + abs(float(weights.alpha_high_neg_guard_scale) - 1.0)
        + abs(float(weights.gamma_scale) - 1.0)
    )
    high_neg_unit = max(abs(float(anchor.alpha_high_neg_guard)), 1e-6)
    distance += abs(float(weights.alpha_high_neg) - float(anchor.alpha_high_neg)) / high_neg_unit
    return float(distance)


def _objective(metrics: Mapping[str, float], distance: float, config: FullFusionWeightSearchConfig) -> float:
    return float(
        metrics["worst_group_ba"]
        - float(config.lambda_drift) * float(metrics["mean_score_drift"]) * 100.0
        - float(config.lambda_flip) * float(metrics["flip_rate"]) * 100.0
        - float(config.lambda_anchor) * float(distance)
    )


def _accepted(
    metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    config: FullFusionWeightSearchConfig,
) -> bool:
    return bool(
        metrics["overall_ba"] + 1e-9 >= baseline_metrics["overall_ba"] - float(config.max_source_ba_drop)
        and metrics["worst_group_ba"] + 1e-9 >= baseline_metrics["worst_group_ba"]
        and metrics["mean_score_drift"] <= float(config.max_mean_score_drift) + 1e-12
        and metrics["flip_rate"] <= float(config.max_flip_rate) + 1e-12
    )


def _candidate_row(
    *,
    weights: FullFusionWeightParams,
    params: FusionParams,
    metrics: Mapping[str, float],
    anchor: FusionParams,
    config: FullFusionWeightSearchConfig,
    baseline_metrics: Mapping[str, float],
    round_index: int,
    coordinate: str,
) -> dict[str, float | int | bool | str]:
    distance = _distance(weights, anchor)
    objective = _objective(metrics, distance, config)
    return {
        **weights.to_dict(),
        "beta": float(params.beta),
        "alpha_low_pos": float(params.alpha_low_pos),
        "alpha_low_neg": float(params.alpha_low_neg),
        "alpha_high_pos": float(params.alpha_high_pos),
        "alpha_high_neg_effective": float(params.alpha_high_neg),
        "alpha_high_neg_guard": float(params.alpha_high_neg_guard),
        "gamma": float(params.gamma),
        **{key: float(value) for key, value in metrics.items()},
        "anchor_distance": float(distance),
        "objective": float(objective),
        "accepted": _accepted(metrics, baseline_metrics, config),
        "round": int(round_index),
        "coordinate": coordinate,
    }


def _is_better(
    *,
    candidate_row: Mapping[str, float | int | bool | str],
    current_metrics: Mapping[str, float],
    current_distance: float,
    current_objective: float,
) -> bool:
    objective = float(candidate_row["objective"])
    if objective > current_objective + 1e-9:
        return True
    if abs(objective - current_objective) > 1e-9:
        return False
    return (
        float(candidate_row["mean_score_drift"]),
        float(candidate_row["flip_rate"]),
        float(candidate_row["anchor_distance"]),
    ) < (
        float(current_metrics["mean_score_drift"]),
        float(current_metrics["flip_rate"]),
        float(current_distance),
    )


def search_full_fusion_weights(
    labels: Sequence[int] | np.ndarray,
    components: Mapping[str, np.ndarray],
    anchor_params: FusionParams,
    *,
    groups: Sequence[str] | np.ndarray | None = None,
    config: FullFusionWeightSearchConfig = FullFusionWeightSearchConfig(),
) -> FullFusionWeightSearchResult:
    y = np.asarray(labels, dtype=np.int64)
    n = validate_component_scores(components)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("labels must be 1D with one value per component score")

    baseline_scores = (
        np.asarray(components["final_fixed"], dtype=np.float32)
        if "final_fixed" in components
        else compute_fixed_scores(components, anchor_params)
    )
    diagnostic_groups = make_source_diagnostic_groups(
        y,
        components,
        source_groups=groups,
        fixed_scores=baseline_scores,
    )
    baseline_metrics = _score_metrics(
        y,
        baseline_scores,
        baseline_scores,
        diagnostic_groups,
        threshold=float(anchor_params.threshold),
        min_group_size=int(config.min_group_size),
    )

    selected = FullFusionWeightParams.default()
    selected_metrics = baseline_metrics
    selected_distance = _distance(selected, anchor_params)
    best_objective = _objective(selected_metrics, selected_distance, config)
    candidates: list[dict[str, float | int | bool | str]] = []
    anchor_row = _candidate_row(
        weights=selected,
        params=selected.to_fusion_params(anchor_params),
        metrics=selected_metrics,
        anchor=anchor_params,
        config=config,
        baseline_metrics=baseline_metrics,
        round_index=0,
        coordinate="anchor",
    )
    candidates.append(anchor_row)

    coordinate_specs: tuple[tuple[str, tuple[float, ...]], ...] = (
        ("beta_scale", config.beta_scale_grid),
        ("alpha_low_pos_scale", config.alpha_low_pos_scale_grid),
        ("alpha_low_neg_scale", config.alpha_low_neg_scale_grid),
        ("alpha_high_pos_scale", config.alpha_high_pos_scale_grid),
        ("alpha_high_neg", config.alpha_high_neg_grid),
        ("alpha_high_neg_guard_scale", config.alpha_high_neg_guard_scale_grid),
        ("gamma_scale", config.gamma_scale_grid),
    )

    for round_index in range(1, int(config.max_rounds) + 1):
        changed = False
        for coordinate, grid in coordinate_specs:
            coordinate_best = selected
            coordinate_best_metrics = selected_metrics
            coordinate_best_objective = best_objective
            coordinate_best_distance = selected_distance
            for value in grid:
                candidate = replace(selected, **{coordinate: float(value)})
                candidate_params = candidate.to_fusion_params(anchor_params)
                scores = (
                    baseline_scores
                    if candidate == FullFusionWeightParams.default()
                    else score_full_fusion_weights(components, anchor_params, candidate)
                )
                metrics = _score_metrics(
                    y,
                    scores,
                    baseline_scores,
                    diagnostic_groups,
                    threshold=float(anchor_params.threshold),
                    min_group_size=int(config.min_group_size),
                )
                row = _candidate_row(
                    weights=candidate,
                    params=candidate_params,
                    metrics=metrics,
                    anchor=anchor_params,
                    config=config,
                    baseline_metrics=baseline_metrics,
                    round_index=round_index,
                    coordinate=coordinate,
                )
                candidates.append(row)
                if not bool(row["accepted"]):
                    continue
                if _is_better(
                    candidate_row=row,
                    current_metrics=coordinate_best_metrics,
                    current_distance=coordinate_best_distance,
                    current_objective=coordinate_best_objective,
                ):
                    coordinate_best = candidate
                    coordinate_best_metrics = metrics
                    coordinate_best_objective = float(row["objective"])
                    coordinate_best_distance = float(row["anchor_distance"])
            if coordinate_best != selected:
                selected = coordinate_best
                selected_metrics = coordinate_best_metrics
                best_objective = coordinate_best_objective
                selected_distance = coordinate_best_distance
                changed = True
        if not changed:
            break

    return FullFusionWeightSearchResult(
        selected=selected,
        selected_params=selected.to_fusion_params(anchor_params),
        selected_metrics=selected_metrics,
        baseline_metrics=baseline_metrics,
        candidates=candidates,
        target_labels_used_for_selection=False,
    )


def _pack_by_group(labels: np.ndarray, scores: np.ndarray, groups: Sequence[str] | np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    group_values = np.asarray(groups, dtype=str)
    for group in sorted(set(group_values.tolist())):
        mask = group_values == group
        packed[group] = (np.asarray(labels, dtype=np.int64)[mask], np.asarray(scores, dtype=np.float32)[mask])
    return packed


def _metrics_delta(selected: Mapping[str, float], anchor: Mapping[str, float]) -> dict[str, float]:
    return {key: float(selected[key]) - float(anchor[key]) for key in sorted(anchor)}


def _paper_rows(
    *,
    search: FullFusionWeightSearchResult,
    anchor_mean: Mapping[str, float],
    selected_mean: Mapping[str, float],
) -> list[dict[str, object]]:
    delta = _metrics_delta(selected_mean, anchor_mean)
    return [
        {
            "variant": "anchor",
            "selection_data": "none",
            **FullFusionWeightParams.default().to_dict(),
            **{f"source_{key}": value for key, value in search.baseline_metrics.items()},
            **{f"current17_{key}": value for key, value in anchor_mean.items()},
            "delta_mean_acc_vs_anchor": 0.0,
            "target_labels_used_for_selection": False,
        },
        {
            "variant": "source_calibrated_full_alpha_split",
            "selection_data": "source_gate_only",
            **search.selected.to_dict(),
            **{f"source_{key}": value for key, value in search.selected_metrics.items()},
            **{f"current17_{key}": value for key, value in selected_mean.items()},
            "delta_mean_acc_vs_anchor": float(delta.get("mean_acc", 0.0)),
            "target_labels_used_for_selection": False,
        },
    ]


def _protocol(
    *,
    search: FullFusionWeightSearchResult,
    config: FullFusionWeightSearchConfig,
    anchor_params: FusionParams,
    source_component_dir: str | Path,
    current_component_dir: str | Path,
) -> dict[str, Any]:
    return {
        "project": "FreqPRISM",
        "phase": "phase1w_full_alpha_split_calibration",
        "method_name": "FreqPRISM source-only calibrated full alpha-split fusion weights",
        "selection_data": "source_gate_only",
        "source_component_dir": str(Path(source_component_dir).resolve(strict=False)) if source_component_dir else "",
        "current_component_dir": str(Path(current_component_dir).resolve(strict=False)) if current_component_dir else "",
        "threshold": float(anchor_params.threshold),
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_report_only": True,
        "search_algorithm": "coordinate_search",
        "search_grid": config.grid_dict(),
        "constraints": config.constraint_dict(),
        "anchored_initial_weights": FullFusionWeightParams.default().to_dict(),
        "selected_weights": search.selected.to_dict(),
        "anchor_fusion_params": {
            "beta": float(anchor_params.beta),
            "alpha_low_pos": float(anchor_params.alpha_low_pos),
            "alpha_low_neg": float(anchor_params.alpha_low_neg),
            "alpha_high_pos": float(anchor_params.alpha_high_pos),
            "alpha_high_neg": float(anchor_params.alpha_high_neg),
            "alpha_high_neg_guard": float(anchor_params.alpha_high_neg_guard),
            "gamma": float(anchor_params.gamma),
        },
        "selected_fusion_params": {
            "beta": float(search.selected_params.beta),
            "alpha_low_pos": float(search.selected_params.alpha_low_pos),
            "alpha_low_neg": float(search.selected_params.alpha_low_neg),
            "alpha_high_pos": float(search.selected_params.alpha_high_pos),
            "alpha_high_neg": float(search.selected_params.alpha_high_neg),
            "alpha_high_neg_guard": float(search.selected_params.alpha_high_neg_guard),
            "gamma": float(search.selected_params.gamma),
        },
        "baseline_metrics": search.baseline_metrics,
        "selected_metrics": search.selected_metrics,
        "candidate_count": len(search.candidates),
        "accepted_candidate_count": int(sum(1 for row in search.candidates if bool(row["accepted"]))),
    }


def write_full_fusion_weight_artifacts(
    *,
    output_dir: str | Path,
    source_labels: Sequence[int] | np.ndarray,
    source_components: Mapping[str, np.ndarray],
    source_groups: Sequence[str] | np.ndarray,
    current_labels: Sequence[int] | np.ndarray,
    current_components: Mapping[str, np.ndarray],
    current_groups: Sequence[str] | np.ndarray,
    anchor_params: FusionParams,
    config: FullFusionWeightSearchConfig = FullFusionWeightSearchConfig(),
    source_component_dir: str | Path = "",
    current_component_dir: str | Path = "",
    selection_protocol_out: str | Path | None = None,
) -> dict[str, Any]:
    search = search_full_fusion_weights(
        labels=source_labels,
        components=source_components,
        anchor_params=anchor_params,
        groups=source_groups,
        config=config,
    )
    out = Path(output_dir)
    (out / "weight_search").mkdir(parents=True, exist_ok=True)

    source_payload = {
        **_protocol(
            search=search,
            config=config,
            anchor_params=anchor_params,
            source_component_dir=source_component_dir,
            current_component_dir=current_component_dir,
        ),
        "candidates": search.candidates,
    }
    (out / "weight_search" / "full_weight_search.json").write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n"
    )
    write_rows_csv(out / "weight_search" / "candidates.csv", search.candidates)

    anchor_scores = (
        np.asarray(current_components["final_fixed"], dtype=np.float32)
        if "final_fixed" in current_components
        else compute_fixed_scores(current_components, anchor_params)
    )
    selected_scores = score_full_fusion_weights(current_components, anchor_params, search.selected)
    anchor_mean = write_target_report(
        out / "current17_anchor",
        _pack_by_group(np.asarray(current_labels, dtype=np.int64), anchor_scores, current_groups),
        threshold=float(anchor_params.threshold),
    )
    selected_mean = write_target_report(
        out / "current17_source_calibrated",
        _pack_by_group(np.asarray(current_labels, dtype=np.int64), selected_scores, current_groups),
        threshold=float(anchor_params.threshold),
    )

    protocol = _protocol(
        search=search,
        config=config,
        anchor_params=anchor_params,
        source_component_dir=source_component_dir,
        current_component_dir=current_component_dir,
    )
    decision = {
        "phase": "phase1w_full_alpha_split_calibration",
        "decision": {
            "main_method_candidate": "FreqPRISM source-only calibrated full alpha-split fusion weights",
            "selected_weights_are_anchor_weights": search.selected == FullFusionWeightParams.default(),
            "selected_weights": search.selected.to_dict(),
            "selected_fusion_params": protocol["selected_fusion_params"],
        },
        "source_weight_search": {
            "baseline_metrics": search.baseline_metrics,
            "selected_metrics": search.selected_metrics,
            "candidate_count": len(search.candidates),
        },
        "current17_anchor_mean": anchor_mean,
        "current17_source_calibrated_mean": selected_mean,
        "current17_mean_delta_source_calibrated_minus_anchor": _metrics_delta(selected_mean, anchor_mean),
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_report_only": True,
    }
    paper_rows = _paper_rows(search=search, anchor_mean=anchor_mean, selected_mean=selected_mean)

    (out / "selection_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    write_rows_csv(out / "paper_table.csv", paper_rows)
    if selection_protocol_out is not None:
        selection_path = Path(selection_protocol_out)
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    return {
        "search": search,
        "protocol": protocol,
        "decision": decision,
        "paper_rows": paper_rows,
    }
