from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from networks.score_blend import probabilities_to_logits
from utils.component_scores import (
    FusionParams,
    WeightParams,
    balanced_accuracy,
    compute_fixed_scores,
    compute_learned_weight_scores,
    validate_component_scores,
)
from utils.metrics import average_precision, roc_auc, write_rows_csv


@dataclass(frozen=True)
class SourceStressConfig:
    scale_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    max_real_logloss: float = 0.0069
    max_flip_rate: float = 0.0001
    max_mean_score_drift: float = 0.01
    min_source_ba: float = 99.995
    min_source_ap: float = 99.999999
    min_source_auc: float = 99.999999

    def to_dict(self) -> dict[str, object]:
        return {
            "scale_grid": [float(value) for value in self.scale_grid],
            "max_real_logloss": float(self.max_real_logloss),
            "max_flip_rate": float(self.max_flip_rate),
            "max_mean_score_drift": float(self.max_mean_score_drift),
            "min_source_ba": float(self.min_source_ba),
            "min_source_ap": float(self.min_source_ap),
            "min_source_auc": float(self.min_source_auc),
        }


def _clip_scores(scores: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    s = _clip_scores(scores)
    return float(-(y * np.log(s) + (1.0 - y) * np.log(1.0 - s)).mean())


def _brier(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((np.asarray(scores, dtype=np.float64) - np.asarray(labels, dtype=np.float64)) ** 2))


def _margin(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((2.0 * y - 1.0) * probabilities_to_logits(scores)))


def _scale_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _anchor_distance(weights: WeightParams) -> float:
    return float(
        abs(float(weights.tile_scale) - 1.0)
        + abs(float(weights.semantic_pos_scale) - 1.0)
        + abs(float(weights.semantic_neg_scale) - 1.0)
        + abs(float(weights.residual_scale) - 1.0)
    )


def _effective_parameters(anchor: FusionParams, weights: WeightParams) -> dict[str, float]:
    return {
        "beta": float(anchor.beta) * float(weights.tile_scale),
        "alpha_low_pos": float(anchor.alpha_low_pos) * float(weights.semantic_pos_scale),
        "alpha_low_neg": float(anchor.alpha_low_neg) * float(weights.semantic_neg_scale),
        "alpha_high_pos": float(anchor.alpha_high_pos) * float(weights.semantic_pos_scale),
        "alpha_high_neg": float(anchor.alpha_high_neg),
        "alpha_high_neg_guard": float(anchor.alpha_high_neg_guard) * float(weights.semantic_neg_scale),
        "gamma": float(anchor.gamma) * float(weights.residual_scale),
    }


def _candidate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    real_mask = y == 0
    fake_mask = y == 1
    pred = np.asarray(scores) >= float(threshold)
    baseline_pred = np.asarray(baseline_scores) >= float(threshold)
    metrics = {
        "source_ba": balanced_accuracy(y, scores, threshold=threshold),
        "source_ap": average_precision(y, scores),
        "source_auc": roc_auc(y, scores),
        "source_logloss": _logloss(y, scores),
        "source_brier": _brier(y, scores),
        "source_margin": _margin(y, scores),
        "source_flip_rate": float(np.mean(pred != baseline_pred)),
        "source_mean_score_drift": float(np.mean(np.abs(np.asarray(scores, dtype=np.float32) - baseline_scores))),
    }
    if bool(real_mask.any()):
        metrics.update(
            {
                "real_logloss": _logloss(y[real_mask], np.asarray(scores)[real_mask]),
                "real_brier": _brier(y[real_mask], np.asarray(scores)[real_mask]),
                "real_margin": _margin(y[real_mask], np.asarray(scores)[real_mask]),
            }
        )
    else:
        metrics.update({"real_logloss": 0.0, "real_brier": 0.0, "real_margin": 0.0})
    if bool(fake_mask.any()):
        metrics.update(
            {
                "fake_logloss": _logloss(y[fake_mask], np.asarray(scores)[fake_mask]),
                "fake_brier": _brier(y[fake_mask], np.asarray(scores)[fake_mask]),
                "fake_margin": _margin(y[fake_mask], np.asarray(scores)[fake_mask]),
            }
        )
    else:
        metrics.update({"fake_logloss": 0.0, "fake_brier": 0.0, "fake_margin": 0.0})
    return metrics


def _is_accepted(metrics: Mapping[str, float], config: SourceStressConfig) -> bool:
    return bool(
        float(metrics["source_ba"]) + 1e-9 >= float(config.min_source_ba)
        and float(metrics["source_ap"]) + 1e-9 >= float(config.min_source_ap)
        and float(metrics["source_auc"]) + 1e-9 >= float(config.min_source_auc)
        and float(metrics["source_flip_rate"]) <= float(config.max_flip_rate) + 1e-12
        and float(metrics["source_mean_score_drift"]) <= float(config.max_mean_score_drift) + 1e-12
        and float(metrics["real_logloss"]) <= float(config.max_real_logloss) + 1e-12
    )


def select_source_stress_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    accepted = [dict(row) for row in candidates if bool(row.get("accepted", False))]
    if not accepted:
        raise ValueError("no accepted source stress calibration candidates")
    selected = sorted(
        accepted,
        key=lambda row: (
            float(row["fake_logloss"]),
            float(row["anchor_distance"]),
            float(row["real_logloss"]),
            float(row["source_logloss"]),
            str(row["variant"]),
        ),
    )[0]
    selected["target_labels_used_for_selection"] = False
    return selected


def run_source_stress_search(
    labels: Sequence[int] | np.ndarray,
    components: Mapping[str, np.ndarray],
    anchor_params: FusionParams,
    *,
    config: SourceStressConfig = SourceStressConfig(),
) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int64)
    n = validate_component_scores(components)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("labels must be 1D with one value per component score")

    baseline_scores = compute_fixed_scores(components, anchor_params)
    baseline_metrics = _candidate_metrics(
        y,
        baseline_scores,
        baseline_scores,
        threshold=float(anchor_params.threshold),
    )
    rows: list[dict[str, object]] = []
    for tile_scale, semantic_pos_scale, semantic_neg_scale, residual_scale in product(config.scale_grid, repeat=4):
        weights = WeightParams(
            tile_scale=float(tile_scale),
            semantic_pos_scale=float(semantic_pos_scale),
            semantic_neg_scale=float(semantic_neg_scale),
            residual_scale=float(residual_scale),
        )
        scores = compute_learned_weight_scores(components, anchor_params, weights)
        metrics = _candidate_metrics(
            y,
            scores,
            baseline_scores,
            threshold=float(anchor_params.threshold),
        )
        row: dict[str, object] = {
            "variant": "b{}_sp{}_sn{}_g{}".format(
                _scale_token(float(tile_scale)),
                _scale_token(float(semantic_pos_scale)),
                _scale_token(float(semantic_neg_scale)),
                _scale_token(float(residual_scale)),
            ),
            **weights.to_dict(),
            **_effective_parameters(anchor_params, weights),
            **metrics,
            "anchor_distance": _anchor_distance(weights),
            "accepted": _is_accepted(metrics, config),
            "target_labels_used_for_selection": False,
        }
        rows.append(row)

    selected = select_source_stress_candidate(rows)
    selected_weights = WeightParams.from_mapping(selected)
    return {
        "baseline_metrics": baseline_metrics,
        "candidates": rows,
        "selected": selected,
        "selected_weights": selected_weights.to_dict(),
        "selected_effective_parameters": _effective_parameters(anchor_params, selected_weights),
    }


def write_source_stress_artifacts(
    *,
    output_dir: str | Path,
    labels: Sequence[int] | np.ndarray,
    components: Mapping[str, np.ndarray],
    groups: Sequence[str] | np.ndarray | None,
    anchor_params: FusionParams,
    config: SourceStressConfig = SourceStressConfig(),
    source_component_dir: str | Path = "",
    selection_protocol_out: str | Path | None = None,
) -> dict[str, object]:
    result = run_source_stress_search(labels, components, anchor_params, config=config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = list(result["candidates"])
    write_rows_csv(out / "source_stress_candidates.csv", candidates)
    group_values = np.asarray(groups, dtype=str) if groups is not None else np.asarray([], dtype=str)
    protocol = {
        "project": "FreqPRISM",
        "phase": "phase1s_source_stress_calibration",
        "method_name": "FreqPRISM pure source-only stress-calibrated fusion",
        "selection_data": "source_gate_stress_only",
        "source_component_dir": str(Path(source_component_dir).resolve(strict=False)) if source_component_dir else "",
        "source_group_count": int(len(set(group_values.tolist()))) if group_values.size else 0,
        "candidate_count": int(len(candidates)),
        "accepted_candidate_count": int(sum(1 for row in candidates if bool(row["accepted"]))),
        "objective": "minimize fake-side source logloss among candidates satisfying source BA/AP/AUC, drift, flip-rate, and real-source logloss constraints",
        "constraints": config.to_dict(),
        "baseline_metrics": result["baseline_metrics"],
        "selected_metrics": result["selected"],
        "selected_weights": result["selected_weights"],
        "effective_parameters": result["selected_effective_parameters"],
        "threshold": float(anchor_params.threshold),
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_report_only": False,
    }
    protocol_path = out / "selection_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    if selection_protocol_out:
        destination = Path(selection_protocol_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if protocol_path.resolve(strict=False) != destination.resolve(strict=False):
            shutil.copyfile(protocol_path, destination)
    return {"protocol": protocol, **result}
