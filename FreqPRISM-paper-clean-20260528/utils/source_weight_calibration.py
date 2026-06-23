from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.metrics import write_rows_csv


TAIL_GROUPS = ("wukong", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "Midjourney", "gaugan", "biggan")
DIFFUSION_TEXT_GROUPS = (
    "ADM",
    "Glide",
    "VQDM",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "coco_sdxl_nw",
    "DALLE2",
    "Midjourney",
    "wukong",
)
GAN_FACE_TRANSLATION_GROUPS = (
    "progan",
    "stylegan",
    "stylegan2",
    "whichfaceisreal",
    "stargan",
    "cyclegan",
    "biggan",
    "gaugan",
)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_overall(report_dir: str | Path) -> dict[str, float]:
    rows = _read_csv_rows(Path(report_dir) / "overall.csv")
    if len(rows) != 1:
        raise ValueError(f"expected one row in {Path(report_dir) / 'overall.csv'}")
    return {key: float(value) for key, value in rows[0].items()}


def _read_per_generator(report_dir: str | Path) -> dict[str, dict[str, float]]:
    rows = _read_csv_rows(Path(report_dir) / "per_generator.csv")
    parsed: dict[str, dict[str, float]] = {}
    for row in rows:
        generator = row["generator"]
        parsed[generator] = {key: float(value) for key, value in row.items() if key != "generator"}
    return parsed


def _mean_metric(groups: Sequence[str], metric: str, rows: Mapping[str, Mapping[str, float]]) -> float | None:
    present = [group for group in groups if group in rows]
    if not present:
        return None
    return float(sum(float(rows[group][metric]) for group in present) / len(present))


def _metric_delta(learned: Mapping[str, float], fixed: Mapping[str, float]) -> dict[str, float]:
    return {metric: float(learned[metric]) - float(fixed[metric]) for metric in sorted(fixed)}


def _paper_row(
    *,
    variant: str,
    weights: Mapping[str, float],
    source_metrics: Mapping[str, float],
    current17_metrics: Mapping[str, float],
    deltas: Mapping[str, float],
    target_labels_used_for_selection: bool,
) -> dict[str, object]:
    return {
        "variant": variant,
        "selection_data": "source_gate_only" if variant == "source_calibrated_weights" else "anchor",
        "tile_scale": float(weights["tile_scale"]),
        "semantic_pos_scale": float(weights["semantic_pos_scale"]),
        "semantic_neg_scale": float(weights["semantic_neg_scale"]),
        "residual_scale": float(weights["residual_scale"]),
        "source_overall_ba": float(source_metrics["overall_ba"]),
        "source_worst_group_ba": float(source_metrics["worst_group_ba"]),
        "source_mean_score_drift": float(source_metrics["mean_score_drift"]),
        "source_flip_rate": float(source_metrics["flip_rate"]),
        "current17_mean_acc": float(current17_metrics["mean_acc"]),
        "current17_mean_ap": float(current17_metrics["mean_ap"]),
        "current17_mean_auc": float(current17_metrics["mean_auc"]),
        "current17_mean_f_acc": float(current17_metrics["mean_f_acc"]),
        "current17_mean_r_acc": float(current17_metrics["mean_r_acc"]),
        "delta_mean_acc_vs_fixed": float(deltas.get("mean_acc", 0.0)),
        "delta_mean_ap_vs_fixed": float(deltas.get("mean_ap", 0.0)),
        "delta_mean_auc_vs_fixed": float(deltas.get("mean_auc", 0.0)),
        "target_labels_used_for_selection": bool(target_labels_used_for_selection),
    }


def build_phase1w_artifacts(
    *,
    weight_search_json: str | Path,
    fixed_report_dir: str | Path,
    learned_report_dir: str | Path,
) -> dict[str, Any]:
    search = json.loads(Path(weight_search_json).read_text())
    fixed_mean = _read_overall(fixed_report_dir)
    learned_mean = _read_overall(learned_report_dir)
    fixed_groups = _read_per_generator(fixed_report_dir)
    learned_groups = _read_per_generator(learned_report_dir)
    deltas = _metric_delta(learned_mean, fixed_mean)
    default_weights = {
        "tile_scale": 1.0,
        "semantic_pos_scale": 1.0,
        "semantic_neg_scale": 1.0,
        "residual_scale": 1.0,
    }
    selected_weights = {key: float(value) for key, value in search["selected_weights"].items()}
    selected_are_anchor = all(abs(selected_weights[key] - default_weights[key]) <= 1e-12 for key in default_weights)

    group_slices = {
        "tail_fixed_f_acc": _mean_metric(TAIL_GROUPS, "f_acc", fixed_groups),
        "tail_learned_weight_f_acc": _mean_metric(TAIL_GROUPS, "f_acc", learned_groups),
        "diffusion_text_fixed_f_acc": _mean_metric(DIFFUSION_TEXT_GROUPS, "f_acc", fixed_groups),
        "diffusion_text_learned_weight_f_acc": _mean_metric(DIFFUSION_TEXT_GROUPS, "f_acc", learned_groups),
        "gan_face_translation_fixed_acc": _mean_metric(GAN_FACE_TRANSLATION_GROUPS, "acc", fixed_groups),
        "gan_face_translation_learned_weight_acc": _mean_metric(GAN_FACE_TRANSLATION_GROUPS, "acc", learned_groups),
    }
    for group in ("gaugan", "biggan"):
        if group in fixed_groups and group in learned_groups:
            group_slices[f"{group}_fixed_r_acc"] = float(fixed_groups[group]["r_acc"])
            group_slices[f"{group}_learned_weight_r_acc"] = float(learned_groups[group]["r_acc"])

    target_labels_used_for_selection = bool(search.get("target_labels_used", False))
    protocol = {
        "project": "FreqPRISM",
        "phase": "phase1w_source_weight_calibration",
        "method_name": "FreqPRISM source-only calibrated fusion weights",
        "selection_data": str(search.get("selection_data", "source_gate_only")),
        "component_dir": str(search.get("component_dir", "")),
        "config": str(search.get("config", "")),
        "threshold": float(search.get("threshold", 0.5)),
        "target_labels_used_for_selection": target_labels_used_for_selection,
        "target_labels_used_for_final_report_only": True,
        "anchored_initial_weights": default_weights,
        "learned_weight_parameterization": {
            "tile_scale": "multiplies beta * max(0, tile_logit - whole_logit)",
            "semantic_pos_scale": "multiplies positive semantic evidence",
            "semantic_neg_scale": "multiplies negative semantic evidence",
            "residual_scale": "multiplies gamma * residual_logit",
        },
        "search_grid": {
            "tile_scale_grid": [0.90, 0.95, 1.00, 1.05, 1.10],
            "semantic_pos_scale_grid": [0.90, 0.95, 1.00, 1.05, 1.10],
            "semantic_neg_scale_grid": [0.90, 0.95, 1.00, 1.05, 1.10],
            "residual_scale_grid": [0.90, 0.95, 1.00, 1.05, 1.10],
        },
        "constraints": search.get("constraints", {}),
        "objective": (
            "maximize source_gate worst-group BA with penalties for score drift, prediction flip rate, "
            "and distance from the anchored initialization"
        ),
        "selected_weights": selected_weights,
        "selected_metrics": search["selected_metrics"],
        "baseline_metrics": search["baseline_metrics"],
        "candidate_count": int(search["candidate_count"]),
        "accepted_candidate_count": int(search["accepted_candidate_count"]),
    }

    decision = {
        "phase": "phase1w_source_weight_calibration",
        "decision": {
            "main_method_after_phase1w": "FreqPRISM source-only calibrated fusion weights",
            "selected_weights_are_anchor_weights": selected_are_anchor,
            "selected_weights": selected_weights,
            "paper_interpretation": (
                "The fixed-looking fusion weights are retained because they are re-selected by a constrained "
                "source-only calibration protocol, not because target labels were used or because the final "
                "threshold was tuned."
            ),
        },
        "source_weight_search": search,
        "current17_fixed_mean": fixed_mean,
        "current17_learned_weight_mean": learned_mean,
        "current17_mean_delta_learned_minus_fixed": deltas,
        "current17_group_slices": group_slices,
        "target_labels_used_for_selection": target_labels_used_for_selection,
        "target_labels_used_for_final_report_only": True,
    }

    paper_rows = [
        _paper_row(
            variant="fixed_anchor",
            weights=default_weights,
            source_metrics=search["baseline_metrics"],
            current17_metrics=fixed_mean,
            deltas={},
            target_labels_used_for_selection=False,
        ),
        _paper_row(
            variant="source_calibrated_weights",
            weights=selected_weights,
            source_metrics=search["selected_metrics"],
            current17_metrics=learned_mean,
            deltas=deltas,
            target_labels_used_for_selection=target_labels_used_for_selection,
        ),
    ]

    return {
        "decision": decision,
        "protocol": protocol,
        "selection_protocol": protocol,
        "paper_rows": paper_rows,
    }


def write_phase1w_artifacts(
    *,
    weight_search_json: str | Path,
    fixed_report_dir: str | Path,
    learned_report_dir: str | Path,
    output_dir: str | Path,
    selection_protocol_out: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = build_phase1w_artifacts(
        weight_search_json=weight_search_json,
        fixed_report_dir=fixed_report_dir,
        learned_report_dir=learned_report_dir,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(artifacts["decision"], indent=2, sort_keys=True) + "\n")
    (out / "protocol.json").write_text(json.dumps(artifacts["protocol"], indent=2, sort_keys=True) + "\n")
    write_rows_csv(out / "paper_table.csv", artifacts["paper_rows"])
    if selection_protocol_out is not None:
        selection_path = Path(selection_protocol_out)
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(artifacts["selection_protocol"], indent=2, sort_keys=True) + "\n")
    return artifacts
