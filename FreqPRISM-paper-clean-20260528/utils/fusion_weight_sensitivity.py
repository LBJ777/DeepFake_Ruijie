from __future__ import annotations

import json
from dataclasses import dataclass
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
from utils.metrics import binary_metrics, write_rows_csv


DEFAULT_SCALE_SWEEP_VALUES = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
DEFAULT_GAMMA_SCALE_SWEEP_VALUES = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
DEFAULT_ALPHA_HIGH_NEG_VALUES = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20)
PAPER_TABLE_SPECS = (
    ("table4a_beta_ablation", "beta_scale", 1.0),
    ("table4b_alpha_pos_ablation", "semantic_pos_scale", 1.0),
    ("table4c_alpha_neg_ablation", "semantic_neg_scale", 1.0),
    ("table4d_gamma_ablation", "gamma_scale", 1.0),
    ("tableS4d_alpha_low_pos_ablation", "alpha_low_pos_scale", 1.0),
    ("tableS4e_alpha_low_neg_ablation", "alpha_low_neg_scale", 1.0),
    ("tableS4f_alpha_high_pos_ablation", "alpha_high_pos_scale", 1.0),
    ("tableS4g_alpha_high_neg_guard_ablation", "alpha_high_neg_guard_scale", 1.0),
    ("tableS4h_alpha_high_neg_direct_ablation", "alpha_high_neg_direct", None),
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


@dataclass(frozen=True)
class FusionWeightVariant:
    variant: str
    family: str
    parameter: str
    value: float
    params: FusionParams
    weights: WeightParams
    description: str
    selection_data: str = "none"


def _fmt_value(value: float) -> str:
    return f"{float(value):.2f}".replace("-", "m").replace(".", "p")


def _replace_params(params: FusionParams, **updates: float) -> FusionParams:
    values = {
        "beta": params.beta,
        "alpha_low_pos": params.alpha_low_pos,
        "alpha_low_neg": params.alpha_low_neg,
        "alpha_high_pos": params.alpha_high_pos,
        "alpha_high_neg": params.alpha_high_neg,
        "alpha_high_neg_guard": params.alpha_high_neg_guard,
        "tile_delta_threshold": params.tile_delta_threshold,
        "high_res_threshold": params.high_res_threshold,
        "gamma": params.gamma,
        "threshold": params.threshold,
    }
    values.update({key: float(value) for key, value in updates.items()})
    return FusionParams(**values)


def _effective_weight_fields(params: FusionParams, weights: WeightParams, anchor: FusionParams) -> dict[str, float]:
    effective = {
        "beta": float(params.beta) * float(weights.tile_scale),
        "alpha_low_pos": float(params.alpha_low_pos) * float(weights.semantic_pos_scale),
        "alpha_low_neg": float(params.alpha_low_neg) * float(weights.semantic_neg_scale),
        "alpha_high_pos": float(params.alpha_high_pos) * float(weights.semantic_pos_scale),
        "alpha_high_neg": float(params.alpha_high_neg) * float(weights.semantic_neg_scale),
        "alpha_high_neg_guard": float(params.alpha_high_neg_guard) * float(weights.semantic_neg_scale),
        "gamma": float(params.gamma) * float(weights.residual_scale),
        "tile_scale": float(weights.tile_scale),
        "semantic_pos_scale": float(weights.semantic_pos_scale),
        "semantic_neg_scale": float(weights.semantic_neg_scale),
        "residual_scale": float(weights.residual_scale),
    }
    distance = 0.0
    for key in ("beta", "alpha_low_pos", "alpha_low_neg", "alpha_high_pos", "alpha_high_neg_guard", "gamma"):
        anchor_value = abs(float(getattr(anchor, key)))
        distance += abs(effective[key] - float(getattr(anchor, key))) / max(anchor_value, 1e-6)
    distance += abs(effective["alpha_high_neg"] - float(anchor.alpha_high_neg)) / max(abs(float(anchor.alpha_high_neg_guard)), 1e-6)
    effective["anchor_distance"] = float(distance)
    return effective


def _metadata_row(variant: FusionWeightVariant, anchor: FusionParams) -> dict[str, object]:
    return {
        "variant": variant.variant,
        "family": variant.family,
        "parameter": variant.parameter,
        "value": float(variant.value),
        "description": variant.description,
        "selection_data": variant.selection_data,
        "threshold": float(anchor.threshold),
        "target_labels_used_for_selection": False,
        **_effective_weight_fields(variant.params, variant.weights, anchor),
    }


def _with_weight(**updates: float) -> WeightParams:
    values = WeightParams.default().to_dict()
    values.update({key: float(value) for key, value in updates.items()})
    return WeightParams.from_mapping(values)


def make_fusion_weight_variants(
    params: FusionParams,
    *,
    selected_weights: WeightParams | None = None,
    compact_sweep_values: Sequence[float] = DEFAULT_SCALE_SWEEP_VALUES,
    gamma_sweep_values: Sequence[float] = DEFAULT_GAMMA_SCALE_SWEEP_VALUES,
    alpha_split_sweep_values: Sequence[float] = DEFAULT_SCALE_SWEEP_VALUES,
    alpha_high_neg_values: Sequence[float] = DEFAULT_ALPHA_HIGH_NEG_VALUES,
) -> list[FusionWeightVariant]:
    variants = [
        FusionWeightVariant(
            variant="W0_anchor",
            family="anchor",
            parameter="anchor",
            value=1.0,
            params=params,
            weights=WeightParams.default(),
            description="Fixed anchor beta/alpha/gamma fusion weights.",
            selection_data="anchor",
        )
    ]
    if selected_weights is not None:
        variants.append(
            FusionWeightVariant(
                variant="W1_source_selected_compact",
                family="learnable_weight",
                parameter="compact_selected",
                value=1.0,
                params=params,
                weights=selected_weights,
                description="Source-only selected compact 4-scale fusion weights.",
                selection_data="source_gate_only",
            )
        )

    compact_specs = (
        ("B0", "beta_scale", "tile_scale", "Tile artifact beta scale."),
        ("A0", "semantic_pos_scale", "semantic_pos_scale", "Semantic fake-side positive evidence scale."),
        ("A1", "semantic_neg_scale", "semantic_neg_scale", "Semantic real-side negative evidence scale."),
    )
    for prefix, parameter, weight_key, description in compact_specs:
        for value in compact_sweep_values:
            variants.append(
                FusionWeightVariant(
                    variant=f"{prefix}_{parameter}_{_fmt_value(float(value))}",
                    family="one_factor_compact",
                    parameter=parameter,
                    value=float(value),
                    params=params,
                    weights=_with_weight(**{weight_key: float(value)}),
                    description=description,
                    selection_data="none",
                )
            )
    for value in gamma_sweep_values:
        variants.append(
            FusionWeightVariant(
                variant=f"R0_gamma_scale_{_fmt_value(float(value))}",
                family="one_factor_compact",
                parameter="gamma_scale",
                value=float(value),
                params=params,
                weights=_with_weight(residual_scale=float(value)),
                description="Residual gamma scale.",
                selection_data="none",
            )
        )

    alpha_split_specs = (
        ("AS0", "alpha_low_pos_scale", "alpha_low_pos", params.alpha_low_pos, "Low-resolution semantic positive alpha."),
        ("AS1", "alpha_low_neg_scale", "alpha_low_neg", params.alpha_low_neg, "Low-resolution semantic negative alpha."),
        ("AS2", "alpha_high_pos_scale", "alpha_high_pos", params.alpha_high_pos, "High-resolution semantic positive alpha."),
        (
            "AS3",
            "alpha_high_neg_guard_scale",
            "alpha_high_neg_guard",
            params.alpha_high_neg_guard,
            "High-resolution semantic negative guard alpha.",
        ),
    )
    for prefix, parameter, param_key, anchor_value, description in alpha_split_specs:
        for value in alpha_split_sweep_values:
            variants.append(
                FusionWeightVariant(
                    variant=f"{prefix}_{parameter}_{_fmt_value(float(value))}",
                    family="one_factor_alpha_split",
                    parameter=parameter,
                    value=float(value),
                    params=_replace_params(params, **{param_key: float(anchor_value) * float(value)}),
                    weights=WeightParams.default(),
                    description=description,
                    selection_data="none",
                )
            )
    for value in alpha_high_neg_values:
        variants.append(
            FusionWeightVariant(
                variant=f"AS4_alpha_high_neg_direct_{_fmt_value(float(value))}",
                family="one_factor_alpha_split",
                parameter="alpha_high_neg_direct",
                value=float(value),
                params=_replace_params(params, alpha_high_neg=float(value)),
                weights=WeightParams.default(),
                description="High-resolution semantic negative alpha direct value.",
                selection_data="none",
            )
        )

    drop_specs = (
        ("D0_no_tile_weight", _with_weight(tile_scale=0.0), params, "Drop tile artifact beta contribution."),
        ("D1_no_semantic_pos_weight", _with_weight(semantic_pos_scale=0.0), params, "Drop semantic positive contribution."),
        ("D2_no_semantic_neg_weight", _with_weight(semantic_neg_scale=0.0), params, "Drop semantic negative contribution."),
        ("D3_no_semantic_weight", _with_weight(semantic_pos_scale=0.0, semantic_neg_scale=0.0), params, "Drop all semantic contribution."),
        ("D4_no_residual_weight", _with_weight(residual_scale=0.0), params, "Drop residual gamma contribution."),
    )
    for variant_id, weights, variant_params, description in drop_specs:
        variants.append(
            FusionWeightVariant(
                variant=variant_id,
                family="drop_weight",
                parameter=variant_id,
                value=0.0,
                params=variant_params,
                weights=weights,
                description=description,
                selection_data="none",
            )
        )

    deduped: list[FusionWeightVariant] = []
    seen: set[str] = set()
    for variant in variants:
        if variant.variant in seen:
            continue
        seen.add(variant.variant)
        deduped.append(variant)
    return deduped


def _variant_scores(components: Mapping[str, np.ndarray], variant: FusionWeightVariant) -> np.ndarray:
    return compute_learned_weight_scores(components, variant.params, variant.weights)


def _source_metrics(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    anchor_scores: np.ndarray,
    groups: Sequence[str] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    group_metrics = group_balanced_accuracies(
        labels,
        scores,
        groups,
        threshold=float(threshold),
        min_group_size=25,
    )
    return {
        "overall_ba": balanced_accuracy(labels, scores, threshold=threshold),
        "worst_group_ba": float(min(group_metrics.values())),
        "mean_score_drift": float(np.mean(np.abs(scores.astype(np.float32) - anchor_scores.astype(np.float32)))),
        "flip_rate": float(np.mean((scores >= float(threshold)) != (anchor_scores >= float(threshold)))),
    }


def _mean_metric(groups: Sequence[str], metric: str, rows: Mapping[str, Mapping[str, float]]) -> float | None:
    present = [group for group in groups if group in rows]
    if not present:
        return None
    return float(np.mean([float(rows[group][metric]) for group in present]))


def _current_rows(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str] | np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, object]]:
    y = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups, dtype=str)
    per_generator: list[dict[str, object]] = []
    generator_metrics: dict[str, dict[str, float]] = {}
    for group in sorted(set(group_values.tolist())):
        mask = group_values == group
        metrics = binary_metrics(y[mask], scores[mask], threshold=threshold)
        generator_metrics[group] = metrics
        per_generator.append({"generator": group, **metrics})

    overall = {
        f"mean_{metric}": float(np.mean([float(row[metric]) for row in generator_metrics.values()]))
        for metric in ("acc", "ap", "auc", "r_acc", "f_acc", "fpr", "fnr")
    }
    group_slices = {
        "tail_f_acc": _mean_metric(TAIL_GROUPS, "f_acc", generator_metrics),
        "diffusion_text_f_acc": _mean_metric(DIFFUSION_TEXT_GROUPS, "f_acc", generator_metrics),
        "gan_face_translation_acc": _mean_metric(GAN_FACE_TRANSLATION_GROUPS, "acc", generator_metrics),
        "gaugan_r_acc": generator_metrics.get("gaugan", {}).get("r_acc"),
        "biggan_r_acc": generator_metrics.get("biggan", {}).get("r_acc"),
    }
    return overall, per_generator, group_slices


def _row_by_variant(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    return {str(row["variant"]): row for row in rows}


def _per_generator_by_variant(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, Mapping[str, object]]]:
    by_variant: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), {})[str(row["generator"])] = row
    return by_variant


def _metric_delta(row: Mapping[str, object], reference: Mapping[str, object], metric: str) -> float | None:
    if metric not in row or metric not in reference:
        return None
    value = row[metric]
    ref_value = reference[metric]
    if value is None or ref_value is None:
        return None
    return float(value) - float(ref_value)


def _paired_generator_bootstrap_ci(
    variant_rows: Mapping[str, Mapping[str, object]],
    reference_rows: Mapping[str, Mapping[str, object]],
    *,
    metric: str,
    groups: Sequence[str] | None,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float | None, float | None]:
    common_groups = sorted(set(variant_rows) & set(reference_rows))
    if groups is not None:
        allowed = {str(group) for group in groups}
        common_groups = [group for group in common_groups if group in allowed]
    pairs: list[tuple[float, float]] = []
    for group in common_groups:
        value = variant_rows[group].get(metric)
        ref_value = reference_rows[group].get(metric)
        if value is None or ref_value is None:
            continue
        pairs.append((float(value), float(ref_value)))
    if not pairs:
        return None, None
    values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    reference = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    indices = rng.integers(0, len(pairs), size=(int(samples), len(pairs)))
    deltas = values[indices].mean(axis=1) - reference[indices].mean(axis=1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def _add_delta_columns(
    row: dict[str, object],
    reference: Mapping[str, object],
    *,
    variant_generator_rows: Mapping[str, Mapping[str, object]],
    reference_generator_rows: Mapping[str, Mapping[str, object]],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> None:
    delta_metrics = (
        "overall_ba",
        "worst_group_ba",
        "mean_score_drift",
        "flip_rate",
        "anchor_distance",
        "mean_acc",
        "mean_ap",
        "mean_auc",
        "mean_r_acc",
        "mean_f_acc",
        "mean_fpr",
        "mean_fnr",
        "tail_f_acc",
        "diffusion_text_f_acc",
        "gan_face_translation_acc",
        "gaugan_r_acc",
        "biggan_r_acc",
    )
    for metric in delta_metrics:
        row[f"delta_{metric}"] = _metric_delta(row, reference, metric)

    bootstrap_specs = {
        "delta_mean_acc": ("acc", None),
        "delta_mean_ap": ("ap", None),
        "delta_mean_auc": ("auc", None),
        "delta_mean_r_acc": ("r_acc", None),
        "delta_mean_f_acc": ("f_acc", None),
        "delta_mean_fpr": ("fpr", None),
        "delta_mean_fnr": ("fnr", None),
        "delta_tail_f_acc": ("f_acc", TAIL_GROUPS),
        "delta_diffusion_text_f_acc": ("f_acc", DIFFUSION_TEXT_GROUPS),
        "delta_gan_face_translation_acc": ("acc", GAN_FACE_TRANSLATION_GROUPS),
    }
    for delta_name, (metric, groups) in bootstrap_specs.items():
        low, high = _paired_generator_bootstrap_ci(
            variant_generator_rows,
            reference_generator_rows,
            metric=metric,
            groups=groups,
            rng=rng,
            samples=int(bootstrap_samples),
        )
        row[f"{delta_name}_ci95_low"] = low
        row[f"{delta_name}_ci95_high"] = high


def _build_paper_tables(
    *,
    source_rows: Sequence[Mapping[str, object]],
    current_overall_rows: Sequence[Mapping[str, object]],
    current_per_generator_rows: Sequence[Mapping[str, object]],
    current_group_slice_rows: Sequence[Mapping[str, object]],
    params: FusionParams,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, list[dict[str, object]]]:
    source_by_variant = _row_by_variant(source_rows)
    overall_by_variant = _row_by_variant(current_overall_rows)
    slices_by_variant = _row_by_variant(current_group_slice_rows)
    per_generator_by_variant = _per_generator_by_variant(current_per_generator_rows)
    reference_variant = "W0_anchor"
    reference = {
        **source_by_variant[reference_variant],
        **overall_by_variant[reference_variant],
        **slices_by_variant[reference_variant],
    }
    reference_generator_rows = per_generator_by_variant[reference_variant]
    rng = np.random.default_rng(int(bootstrap_seed))
    tables: dict[str, list[dict[str, object]]] = {}
    for table_name, parameter, reference_value in PAPER_TABLE_SPECS:
        actual_reference_value = float(params.alpha_high_neg) if reference_value is None else float(reference_value)
        rows: list[dict[str, object]] = []
        for source_row in source_rows:
            if str(source_row["parameter"]) != parameter:
                continue
            variant = str(source_row["variant"])
            row = {
                **source_row,
                **overall_by_variant[variant],
                **slices_by_variant[variant],
                "paper_table": table_name,
                "paper_parameter": parameter,
                "paper_value": float(source_row["value"]),
                "reference_value": actual_reference_value,
                "is_reference": bool(np.isclose(float(source_row["value"]), actual_reference_value)),
                "bootstrap_level": "current17_generator_paired",
                "bootstrap_samples": int(bootstrap_samples),
            }
            _add_delta_columns(
                row,
                reference,
                variant_generator_rows=per_generator_by_variant[variant],
                reference_generator_rows=reference_generator_rows,
                rng=rng,
                bootstrap_samples=int(bootstrap_samples),
            )
            rows.append(row)
        rows.sort(key=lambda item: float(item["paper_value"]))
        tables[table_name] = rows
    return tables


def build_fusion_weight_sensitivity_report(
    *,
    source_labels: Sequence[int] | np.ndarray,
    source_components: Mapping[str, np.ndarray],
    source_groups: Sequence[str] | np.ndarray,
    current_labels: Sequence[int] | np.ndarray,
    current_components: Mapping[str, np.ndarray],
    current_groups: Sequence[str] | np.ndarray,
    params: FusionParams,
    selected_weights: WeightParams | None = None,
    compact_sweep_values: Sequence[float] = DEFAULT_SCALE_SWEEP_VALUES,
    gamma_sweep_values: Sequence[float] = DEFAULT_GAMMA_SCALE_SWEEP_VALUES,
    alpha_split_sweep_values: Sequence[float] = DEFAULT_SCALE_SWEEP_VALUES,
    alpha_high_neg_values: Sequence[float] = DEFAULT_ALPHA_HIGH_NEG_VALUES,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260528,
    source_component_dir: str | Path = "",
    current_component_dir: str | Path = "",
    weights_json: str | Path = "",
) -> dict[str, Any]:
    validate_component_scores(source_components)
    validate_component_scores(current_components)
    source_y = np.asarray(source_labels, dtype=np.int64)
    current_y = np.asarray(current_labels, dtype=np.int64)
    anchor_source_scores = (
        np.asarray(source_components["final_fixed"], dtype=np.float32)
        if "final_fixed" in source_components
        else compute_fixed_scores(source_components, params)
    )
    diagnostic_groups = make_source_diagnostic_groups(
        source_y,
        source_components,
        source_groups=source_groups,
        fixed_scores=anchor_source_scores,
    )
    variants = make_fusion_weight_variants(
        params,
        selected_weights=selected_weights,
        compact_sweep_values=compact_sweep_values,
        gamma_sweep_values=gamma_sweep_values,
        alpha_split_sweep_values=alpha_split_sweep_values,
        alpha_high_neg_values=alpha_high_neg_values,
    )

    source_rows: list[dict[str, object]] = []
    current_overall_rows: list[dict[str, object]] = []
    current_per_generator_rows: list[dict[str, object]] = []
    current_group_slice_rows: list[dict[str, object]] = []

    for variant in variants:
        metadata = _metadata_row(variant, params)
        source_scores = _variant_scores(source_components, variant)
        source_rows.append(
            {
                **metadata,
                **_source_metrics(
                    labels=source_y,
                    scores=source_scores,
                    anchor_scores=anchor_source_scores,
                    groups=diagnostic_groups,
                    threshold=float(params.threshold),
                ),
            }
        )

        current_scores = _variant_scores(current_components, variant)
        overall, per_generator, group_slices = _current_rows(
            labels=current_y,
            scores=current_scores,
            groups=current_groups,
            threshold=float(params.threshold),
        )
        current_overall_rows.append({**metadata, **overall})
        for row in per_generator:
            current_per_generator_rows.append({**metadata, **row})
        current_group_slice_rows.append({**metadata, **group_slices})

    paper_tables = _build_paper_tables(
        source_rows=source_rows,
        current_overall_rows=current_overall_rows,
        current_per_generator_rows=current_per_generator_rows,
        current_group_slice_rows=current_group_slice_rows,
        params=params,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=int(bootstrap_seed),
    )

    protocol = {
        "phase": "phase2_fusion_weight_sensitivity",
        "method_name": "Prior fusion weight sensitivity and learnable-weight ablation",
        "source_component_dir": str(Path(source_component_dir).resolve(strict=False)) if source_component_dir else "",
        "current_component_dir": str(Path(current_component_dir).resolve(strict=False)) if current_component_dir else "",
        "weights_json": str(Path(weights_json).resolve(strict=False)) if weights_json else "",
        "threshold": float(params.threshold),
        "compact_sweep_values": [float(value) for value in compact_sweep_values],
        "gamma_sweep_values": [float(value) for value in gamma_sweep_values],
        "alpha_split_sweep_values": [float(value) for value in alpha_split_sweep_values],
        "alpha_high_neg_values": [float(value) for value in alpha_high_neg_values],
        "paper_tables": sorted(paper_tables),
        "paper_table_specs": [
            {
                "name": name,
                "parameter": parameter,
                "reference_value": float(params.alpha_high_neg) if reference is None else float(reference),
            }
            for name, parameter, reference in PAPER_TABLE_SPECS
        ],
        "bootstrap_level": "current17_generator_paired",
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "variant_count": len(variants),
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_report_only": True,
    }
    return {
        "source_gate_weight_sweep": source_rows,
        "current17_weight_sweep_overall": current_overall_rows,
        "current17_weight_sweep_per_generator": current_per_generator_rows,
        "current17_weight_sweep_group_slices": current_group_slice_rows,
        "paper_tables": paper_tables,
        "protocol": protocol,
    }


def write_fusion_weight_sensitivity_report(
    *,
    output_dir: str | Path,
    source_labels: Sequence[int] | np.ndarray,
    source_components: Mapping[str, np.ndarray],
    source_groups: Sequence[str] | np.ndarray,
    current_labels: Sequence[int] | np.ndarray,
    current_components: Mapping[str, np.ndarray],
    current_groups: Sequence[str] | np.ndarray,
    params: FusionParams,
    selected_weights: WeightParams | None = None,
    compact_sweep_values: Sequence[float] = DEFAULT_SCALE_SWEEP_VALUES,
    gamma_sweep_values: Sequence[float] = DEFAULT_GAMMA_SCALE_SWEEP_VALUES,
    alpha_split_sweep_values: Sequence[float] = DEFAULT_SCALE_SWEEP_VALUES,
    alpha_high_neg_values: Sequence[float] = DEFAULT_ALPHA_HIGH_NEG_VALUES,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260528,
    source_component_dir: str | Path = "",
    current_component_dir: str | Path = "",
    weights_json: str | Path = "",
) -> dict[str, Any]:
    report = build_fusion_weight_sensitivity_report(
        source_labels=source_labels,
        source_components=source_components,
        source_groups=source_groups,
        current_labels=current_labels,
        current_components=current_components,
        current_groups=current_groups,
        params=params,
        selected_weights=selected_weights,
        compact_sweep_values=compact_sweep_values,
        gamma_sweep_values=gamma_sweep_values,
        alpha_split_sweep_values=alpha_split_sweep_values,
        alpha_high_neg_values=alpha_high_neg_values,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        source_component_dir=source_component_dir,
        current_component_dir=current_component_dir,
        weights_json=weights_json,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_rows_csv(out / "source_gate_weight_sweep.csv", report["source_gate_weight_sweep"])
    write_rows_csv(out / "current17_weight_sweep_overall.csv", report["current17_weight_sweep_overall"])
    write_rows_csv(out / "current17_weight_sweep_per_generator.csv", report["current17_weight_sweep_per_generator"])
    write_rows_csv(out / "current17_weight_sweep_group_slices.csv", report["current17_weight_sweep_group_slices"])
    for table_name, rows in report["paper_tables"].items():
        write_rows_csv(out / "paper_tables" / f"{table_name}.csv", rows)
    (out / "protocol.json").write_text(json.dumps(report["protocol"], indent=2, sort_keys=True) + "\n")
    return report
