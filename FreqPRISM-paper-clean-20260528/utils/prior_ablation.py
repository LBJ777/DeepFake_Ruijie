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
class PriorAblationVariant:
    variant_id: str
    name: str
    description: str


PRIOR_ABLATION_VARIANTS = (
    PriorAblationVariant("A0_whole_artifact", "Whole artifact only", "whole-image artifact score W"),
    PriorAblationVariant("A1_artifact", "Artifact only", "whole artifact plus native tile delta"),
    PriorAblationVariant("A2_semantic", "Semantic only", "CLIP semantic prior score S"),
    PriorAblationVariant("A3_residual", "Residual only", "NPR residual prior score R"),
    PriorAblationVariant("A4_no_artifact", "No artifact", "semantic prior blended with residual prior"),
    PriorAblationVariant("A5_no_semantic", "No semantic", "artifact prior blended with residual prior"),
    PriorAblationVariant("A6_no_residual", "No residual", "artifact prior plus semantic prior"),
    PriorAblationVariant("A7_no_tile", "No tile", "whole artifact plus semantic and residual priors"),
    PriorAblationVariant("A8_full", "Full FreqPRISM", "source-only calibrated fusion weights"),
)

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
    *,
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


def compute_prior_ablation_scores(
    components: Mapping[str, np.ndarray],
    params: FusionParams,
    weights: WeightParams,
) -> dict[str, np.ndarray]:
    validate_component_scores(components)
    whole_logit, tile_logit, sem_logit, residual_logit, max_side = _logits(components)
    tile_delta = np.maximum(0.0, tile_logit - whole_logit)
    zero_tile_delta = np.zeros_like(tile_delta, dtype=np.float64)

    artifact_logit = whole_logit + float(weights.tile_scale) * float(params.beta) * tile_delta
    semantic_term = _semantic_term(
        sem_logit=sem_logit,
        tile_delta=tile_delta,
        max_side=max_side,
        params=params,
        weights=weights,
    )
    semantic_term_no_tile = _semantic_term(
        sem_logit=sem_logit,
        tile_delta=zero_tile_delta,
        max_side=max_side,
        params=params,
        weights=weights,
    )
    residual_term = float(weights.residual_scale) * float(params.gamma) * residual_logit

    semantic_residual = logit_blend(
        _array(components, "S"),
        _array(components, "R"),
        float(weights.residual_scale) * float(params.gamma),
    )
    full_scores = compute_learned_weight_scores(components, params, weights)

    return {
        "A0_whole_artifact": _array(components, "W").astype(np.float32),
        "A1_artifact": _prob(artifact_logit),
        "A2_semantic": _array(components, "S").astype(np.float32),
        "A3_residual": _array(components, "R").astype(np.float32),
        "A4_no_artifact": semantic_residual.astype(np.float32),
        "A5_no_semantic": _prob(artifact_logit + residual_term),
        "A6_no_residual": _prob(artifact_logit + semantic_term),
        "A7_no_tile": _prob(whole_logit + semantic_term_no_tile + residual_term),
        "A8_full": full_scores.astype(np.float32),
    }


def _mean_metric(groups: Sequence[str], metric: str, rows: Mapping[str, Mapping[str, float]]) -> float | None:
    present = [group for group in groups if group in rows]
    if not present:
        return None
    return float(sum(float(rows[group][metric]) for group in present) / len(present))


def _variant_name(variant_id: str) -> str:
    for variant in PRIOR_ABLATION_VARIANTS:
        if variant.variant_id == variant_id:
            return variant.name
    return variant_id


def write_prior_ablation_report(
    *,
    output_dir: str | Path,
    labels: np.ndarray,
    groups: Sequence[str] | np.ndarray,
    components: Mapping[str, np.ndarray],
    params: FusionParams,
    weights: WeightParams,
    component_dir: str | Path = "",
    weights_json: str | Path = "",
) -> list[dict[str, object]]:
    scores_by_variant = compute_prior_ablation_scores(components, params, weights)
    group_values = np.asarray(groups, dtype=str)
    y = np.asarray(labels, dtype=np.int64)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_generator_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    group_slice_rows: list[dict[str, object]] = []
    variant_metadata = {variant.variant_id: variant for variant in PRIOR_ABLATION_VARIANTS}

    for variant in PRIOR_ABLATION_VARIANTS:
        scores = scores_by_variant[variant.variant_id]
        generator_metrics: dict[str, dict[str, float]] = {}
        for group in sorted(set(group_values.tolist())):
            mask = group_values == group
            metrics = binary_metrics(y[mask], scores[mask], threshold=float(params.threshold))
            generator_metrics[group] = metrics
            per_generator_rows.append(
                {
                    "variant": variant.variant_id,
                    "variant_name": variant.name,
                    "generator": group,
                    **metrics,
                }
            )
        mean = {
            f"mean_{metric}": float(np.mean([float(row[metric]) for row in generator_metrics.values()]))
            for metric in ("acc", "ap", "auc", "r_acc", "f_acc", "fpr", "fnr")
        }
        overall_rows.append(
            {
                "variant": variant.variant_id,
                "variant_name": variant.name,
                **mean,
            }
        )
        group_slice_rows.append(
            {
                "variant": variant.variant_id,
                "variant_name": variant.name,
                "tail_f_acc": _mean_metric(TAIL_GROUPS, "f_acc", generator_metrics),
                "diffusion_text_f_acc": _mean_metric(DIFFUSION_TEXT_GROUPS, "f_acc", generator_metrics),
                "gan_face_translation_acc": _mean_metric(GAN_FACE_TRANSLATION_GROUPS, "acc", generator_metrics),
                "gaugan_r_acc": generator_metrics.get("gaugan", {}).get("r_acc"),
                "biggan_r_acc": generator_metrics.get("biggan", {}).get("r_acc"),
            }
        )

    write_rows_csv(out / "per_generator.csv", per_generator_rows)
    write_rows_csv(out / "overall.csv", overall_rows)
    write_rows_csv(out / "group_slices.csv", group_slice_rows)
    protocol = {
        "phase": "phase2_prior_ablation",
        "method_name": "FreqPRISM source-only calibrated fusion weights",
        "component_dir": str(Path(component_dir).resolve(strict=False)) if component_dir else "",
        "weights_json": str(Path(weights_json).resolve(strict=False)) if weights_json else "",
        "weights": weights.to_dict(),
        "threshold": float(params.threshold),
        "variants": {
            variant_id: {
                "name": variant_metadata[variant_id].name,
                "description": variant_metadata[variant_id].description,
            }
            for variant_id in scores_by_variant
        },
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_report_only": True,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return overall_rows
