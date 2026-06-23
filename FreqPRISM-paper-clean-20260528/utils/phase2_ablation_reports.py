from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from networks.score_blend import logit_blend, logits_to_probabilities, probabilities_to_logits
from utils.component_scores import FusionParams, WeightParams, compute_learned_weight_scores, validate_component_scores
from utils.metrics import binary_metrics, write_rows_csv


@dataclass(frozen=True)
class AblationVariant:
    variant: str
    variant_name: str
    description: str


TILE_VARIANT_METADATA = {
    "RZ2_resized512_tile": (
        "Resized-512 tile",
        "resize long side to 512 before tile extraction; keep W/S/R fixed",
    ),
    "RZ3_center_crop_tile": (
        "Center crop tile",
        "replace native tile coverage with one center tile; keep W/S/R fixed",
    ),
    "RZ4_tile_mean_aggregation": (
        "Native tile mean aggregation",
        "aggregate native tile scores by mean instead of top1; keep W/S/R fixed",
    ),
}


def _array(components: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    values = np.asarray(components[key], dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"{key} must be a 1D array")
    return values


def _logits(components: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        probabilities_to_logits(_array(components, "W")).astype(np.float64),
        probabilities_to_logits(_array(components, "T")).astype(np.float64),
        probabilities_to_logits(_array(components, "S")).astype(np.float64),
        probabilities_to_logits(_array(components, "R")).astype(np.float64),
        _array(components, "max_side"),
    )


def _semantic_term(
    sem_logit: np.ndarray,
    tile_delta: np.ndarray,
    max_side: np.ndarray,
    params: FusionParams,
    weights: WeightParams,
) -> np.ndarray:
    high_res = max_side > float(params.high_res_threshold)
    relax_negative = high_res & (tile_delta > float(params.tile_delta_threshold))
    pos_alpha = np.where(high_res, float(params.alpha_high_pos), float(params.alpha_low_pos)).astype(np.float64)
    neg_alpha = np.where(
        relax_negative,
        float(params.alpha_high_neg),
        np.where(high_res, float(params.alpha_high_neg_guard), float(params.alpha_low_neg)),
    ).astype(np.float64)
    return (
        float(weights.semantic_pos_scale) * pos_alpha * np.maximum(0.0, sem_logit)
        + float(weights.semantic_neg_scale) * neg_alpha * np.minimum(0.0, sem_logit)
    )


def _prob(logit: np.ndarray) -> np.ndarray:
    return logits_to_probabilities(np.asarray(logit, dtype=np.float32)).astype(np.float32)


def compute_tile_resolution_ablation_scores(
    components: Mapping[str, np.ndarray],
    params: FusionParams,
    weights: WeightParams,
    tile_score_variants: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[AblationVariant]]:
    validate_component_scores(components)
    whole_logit, tile_logit, sem_logit, residual_logit, max_side = _logits(components)
    tile_delta = np.maximum(0.0, tile_logit - whole_logit)
    zero_tile_delta = np.zeros_like(tile_delta, dtype=np.float64)
    semantic_full = _semantic_term(sem_logit, tile_delta, max_side, params, weights)
    semantic_no_tile = _semantic_term(sem_logit, zero_tile_delta, max_side, params, weights)
    residual_term = float(weights.residual_scale) * float(params.gamma) * residual_logit
    full = compute_learned_weight_scores(components, params, weights)

    scores = {
        "RZ0_full_native_tile": full.astype(np.float32),
        "RZ1_whole_only_no_tile": _prob(whole_logit + semantic_no_tile + residual_term),
        "RZ5_current_top1_tile": _prob(
            whole_logit
            + float(weights.tile_scale) * float(params.beta) * tile_delta
            + semantic_full
            + residual_term
        ),
    }
    variants = [
        AblationVariant("RZ0_full_native_tile", "Full native tile", "current W + native top1 T + S + R"),
        AblationVariant("RZ1_whole_only_no_tile", "Whole only / no tile", "remove native tile evidence T"),
        AblationVariant("RZ5_current_top1_tile", "Current top1 tile", "explicit current top1 tile aggregation from component cache"),
    ]
    for variant_id, tile_scores in (tile_score_variants or {}).items():
        tile_values = np.asarray(tile_scores, dtype=np.float32)
        if tile_values.shape != _array(components, "W").shape:
            raise ValueError(f"tile score length mismatch for {variant_id}")
        variant_tile_logit = probabilities_to_logits(tile_values).astype(np.float64)
        variant_tile_delta = np.maximum(0.0, variant_tile_logit - whole_logit)
        variant_semantic = _semantic_term(sem_logit, variant_tile_delta, max_side, params, weights)
        scores[str(variant_id)] = _prob(
            whole_logit
            + float(weights.tile_scale) * float(params.beta) * variant_tile_delta
            + variant_semantic
            + residual_term
        )
        name, description = TILE_VARIANT_METADATA.get(
            str(variant_id),
            (str(variant_id), f"image-level tile score variant {variant_id}; keep W/S/R fixed"),
        )
        variants.append(AblationVariant(str(variant_id), name, description))
    return scores, variants


def compute_residual_npr_ablation_scores(
    components: Mapping[str, np.ndarray],
    params: FusionParams,
    weights: WeightParams,
    gamma_scales: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
) -> tuple[dict[str, np.ndarray], list[AblationVariant]]:
    validate_component_scores(components)
    whole_logit, tile_logit, sem_logit, residual_logit, max_side = _logits(components)
    tile_delta = np.maximum(0.0, tile_logit - whole_logit)
    artifact_logit = whole_logit + float(weights.tile_scale) * float(params.beta) * tile_delta
    semantic = _semantic_term(sem_logit, tile_delta, max_side, params, weights)
    residual_term = float(weights.residual_scale) * float(params.gamma) * residual_logit
    full = compute_learned_weight_scores(components, params, weights)

    scores: dict[str, np.ndarray] = {
        "RP0_full_residual_prior": full.astype(np.float32),
        "RP1_no_residual": _prob(artifact_logit + semantic),
        "RP2_residual_only": _array(components, "R").astype(np.float32),
        "RP3_artifact_residual": _prob(artifact_logit + residual_term),
        "RP4_semantic_residual": logit_blend(
            _array(components, "S"),
            _array(components, "R"),
            float(weights.residual_scale) * float(params.gamma),
        ).astype(np.float32),
    }
    variants = [
        AblationVariant("RP0_full_residual_prior", "Full residual prior", "current gamma * R correction"),
        AblationVariant("RP1_no_residual", "No residual", "set gamma to zero"),
        AblationVariant("RP2_residual_only", "Residual only", "use residual prior score R alone"),
        AblationVariant("RP3_artifact_residual", "Artifact + residual", "remove semantic prior S"),
        AblationVariant("RP4_semantic_residual", "Semantic + residual", "remove artifact prior W/T"),
    ]
    for scale in gamma_scales:
        key = f"RP6_gamma_scale_{float(scale):.2f}".replace(".", "p")
        scores[key] = _prob(artifact_logit + semantic + float(scale) * residual_term)
        variants.append(AblationVariant(key, f"Gamma scale {float(scale):.2f}", f"multiply residual correction by {float(scale):.2f}"))
    return scores, variants


def write_ablation_report(
    *,
    output_dir: str | Path,
    labels: np.ndarray,
    groups: Sequence[str] | np.ndarray,
    scores_by_variant: Mapping[str, np.ndarray],
    variants: Sequence[AblationVariant],
    threshold: float,
    protocol: Mapping[str, object],
) -> list[dict[str, object]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups, dtype=str)
    if y.ndim != 1 or group_values.ndim != 1 or y.shape[0] != group_values.shape[0]:
        raise ValueError("labels and groups must be 1D arrays with matching length")

    variant_lookup = {variant.variant: variant for variant in variants}
    per_generator_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    group_slice_rows: list[dict[str, object]] = []
    high_res_groups = ("Midjourney", "coco_sdxl_nw", "stable_diffusion_v_1_5", "wukong")
    tail_groups = ("wukong", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "Midjourney")
    diffusion_groups = ("ADM", "Glide", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "coco_sdxl_nw", "wukong")

    for variant in variants:
        scores = np.asarray(scores_by_variant[variant.variant], dtype=np.float32)
        if scores.shape[0] != y.shape[0]:
            raise ValueError(f"score length mismatch for {variant.variant}")
        generator_metrics: dict[str, dict[str, float]] = {}
        for group in sorted(set(group_values.tolist())):
            mask = group_values == group
            metrics = binary_metrics(y[mask], scores[mask], threshold=float(threshold))
            generator_metrics[group] = metrics
            per_generator_rows.append(
                {
                    "variant": variant.variant,
                    "variant_name": variant.variant_name,
                    "generator": group,
                    **metrics,
                }
            )
        mean = {
            f"mean_{metric}": float(np.mean([float(row[metric]) for row in generator_metrics.values()]))
            for metric in ("acc", "ap", "auc", "r_acc", "f_acc", "fpr", "fnr")
        }
        overall_rows.append({"variant": variant.variant, "variant_name": variant.variant_name, **mean})

        def mean_metric(subset: Sequence[str], metric: str) -> float | None:
            present = [group for group in subset if group in generator_metrics]
            if not present:
                return None
            return float(np.mean([generator_metrics[group][metric] for group in present]))

        group_slice_rows.append(
            {
                "variant": variant.variant,
                "variant_name": variant.variant_name,
                "high_res_acc": mean_metric(high_res_groups, "acc"),
                "tail_f_acc": mean_metric(tail_groups, "f_acc"),
                "diffusion_f_acc": mean_metric(diffusion_groups, "f_acc"),
                "gaugan_r_acc": generator_metrics.get("gaugan", {}).get("r_acc"),
                "biggan_r_acc": generator_metrics.get("biggan", {}).get("r_acc"),
            }
        )

    write_rows_csv(out / "overall.csv", overall_rows)
    write_rows_csv(out / "per_generator.csv", per_generator_rows)
    write_rows_csv(out / "group_slices.csv", group_slice_rows)
    payload = {
        **dict(protocol),
        "variants": {
            variant.variant: {
                "name": variant.variant_name,
                "description": variant.description,
            }
            for variant in variants
        },
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_report_only": True,
        "threshold": float(threshold),
    }
    (out / "protocol.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return overall_rows
