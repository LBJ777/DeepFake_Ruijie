#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import ImageSample, collect_labeled_images, pil_to_tensor
from data.datasets import apply_variant
from models.core import ResidualLogitCombiner
from models.hgb_parity import aggregate_probabilities
from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig
from networks.native_tiles import aggregate_tile_scores, extract_native_tiles, native_tile_boxes
from utils.component_scores import FusionParams, WeightParams, compute_learned_weight_scores, load_component_directory
from utils.metrics import write_rows_csv
from utils.phase2_ablation_reports import AblationVariant, write_ablation_report


ImageFile.LOAD_TRUNCATED_IMAGES = True

FAMILY_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("AF0_full_artifact_features", "Full artifact features", "current artifact prior"),
    AblationVariant("AF1_no_codec_block", "No codec/block", "mask codec_block family"),
    AblationVariant("AF2_no_chroma_luma", "No chroma-luma", "mask chroma_luma_coupling family"),
    AblationVariant("AF3_no_texture", "No texture", "mask texture_artifact family"),
    AblationVariant("AF4_no_recompression", "No recompression", "mask all recompression_* families"),
    AblationVariant("AF5_no_residual_spectrum", "No residual spectrum", "mask residual_spectrum family"),
    AblationVariant("AF6_no_residual_tail", "No residual tail", "mask residual_tail_shape family"),
    AblationVariant("AF7_no_patch_heterogeneity", "No patch heterogeneity", "mask patch_spectrum_heterogeneity family"),
    AblationVariant("AF8_codec_block_only", "Codec/block only", "keep codec_block and mask all other artifact families"),
    AblationVariant("AF9_spectrum_only", "Spectrum only", "keep spectrum-related families and mask all others"),
)


DROP_FAMILIES = {
    "AF1_no_codec_block": ("codec_block",),
    "AF2_no_chroma_luma": ("chroma_luma_coupling",),
    "AF3_no_texture": ("texture_artifact",),
    "AF4_no_recompression": ("recompression_*",),
    "AF5_no_residual_spectrum": ("residual_spectrum",),
    "AF6_no_residual_tail": ("residual_tail_shape",),
    "AF7_no_patch_heterogeneity": ("patch_spectrum_heterogeneity",),
}
KEEP_FAMILIES = {
    "AF8_codec_block_only": ("codec_block",),
    "AF9_spectrum_only": ("residual_spectrum", "patch_spectrum_heterogeneity"),
}


def _config_name(value: str) -> str:
    path = Path(value)
    if path.suffix in {".yaml", ".yml"}:
        return path.name
    return value


def _load_weights(path: str | Path) -> WeightParams:
    payload = json.loads(Path(path).read_text())
    values = payload.get("selected_weights", payload)
    return WeightParams.from_mapping(values)


def _feature_slices(feature_families: Mapping[str, object], patterns: Sequence[str]) -> list[slice]:
    slices: list[slice] = []
    for pattern in patterns:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            names = [name for name in feature_families if str(name).startswith(prefix)]
        else:
            names = [pattern] if pattern in feature_families else []
        if not names:
            raise KeyError(f"feature family pattern matched nothing: {pattern}")
        for name in names:
            slices.append(feature_families[name].slice)  # type: ignore[attr-defined]
    return slices


def _mask_features(
    features: np.ndarray,
    *,
    variant_id: str,
    family_slices: Mapping[str, list[slice]],
    median: np.ndarray,
) -> np.ndarray:
    if variant_id == "AF0_full_artifact_features":
        return features
    masked = np.array(features, copy=True)
    if variant_id in DROP_FAMILIES:
        for family_slice in family_slices[variant_id]:
            masked[:, family_slice] = median[family_slice]
        return masked
    if variant_id in KEEP_FAMILIES:
        keep = np.zeros(features.shape[1], dtype=bool)
        for family_slice in family_slices[variant_id]:
            keep[family_slice] = True
        masked[:, ~keep] = median[~keep]
        return masked
    raise KeyError(f"unknown artifact family variant: {variant_id}")


def _score_features(features: np.ndarray, artifact_payload: Mapping[str, object]) -> np.ndarray:
    codec_scores = artifact_payload["codec"].predict_proba(features)[:, 1]  # type: ignore[index]
    chroma_scores = artifact_payload["chroma"].predict_proba(features)[:, 1]  # type: ignore[index]
    return ResidualLogitCombiner(alpha=float(artifact_payload["alpha"])).predict_proba_from_scores(
        codec_scores,
        chroma_scores,
    ).astype(np.float32)


def _extract_features(detector: UnifiedArtifactDetector, tensors: Sequence[torch.Tensor]) -> np.ndarray:
    if not tensors:
        return np.empty((0, 0), dtype=np.float32)
    chunks: list[np.ndarray] = []
    batch_size = max(1, int(detector.config.artifact_forward_batch_size))
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(list(tensors[start : start + batch_size])).to(detector.device)
            chunks.append(detector.artifact_extractor(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _source_real_samples(source_root: str | Path, max_images: int) -> list[ImageSample]:
    samples = [sample for sample in collect_labeled_images(source_root) if int(sample.label) == 0]
    if int(max_images) > 0:
        samples = samples[: int(max_images)]
    if not samples:
        raise ValueError(f"no source-real samples found under {source_root}")
    return samples


def compute_source_real_median(
    detector: UnifiedArtifactDetector,
    *,
    source_root: str | Path,
    max_images: int,
    image_batch_size: int,
) -> np.ndarray:
    feature_chunks: list[np.ndarray] = []
    tensors: list[torch.Tensor] = []

    def flush_pending() -> None:
        if not tensors:
            return
        feature_chunks.append(_extract_features(detector, tensors))
        tensors.clear()

    max_tensors = max(1, int(image_batch_size)) * len(detector.config.artifact_variants)
    for sample in _source_real_samples(source_root, max_images=max_images):
        with Image.open(sample.path) as image:
            rgb = image.convert("RGB")
            for variant in detector.config.artifact_variants:
                tensors.append(
                    pil_to_tensor(
                        apply_variant(rgb, detector.config.artifact_image_size, variant),
                        detector.config.artifact_image_size,
                    )
                )
                if len(tensors) >= max_tensors:
                    flush_pending()
    flush_pending()
    features = np.concatenate(feature_chunks, axis=0)
    return np.median(features, axis=0).astype(np.float32)


def _aggregate_variant_scores(
    features: np.ndarray,
    *,
    variant_count: int,
    variant_id: str,
    family_slices: Mapping[str, list[slice]],
    median: np.ndarray,
    artifact_payload: Mapping[str, object],
) -> np.ndarray:
    masked = _mask_features(features, variant_id=variant_id, family_slices=family_slices, median=median)
    scores = _score_features(masked, artifact_payload)
    return aggregate_probabilities(scores.reshape(-1, int(variant_count)), "mean_logit").astype(np.float32)


def _score_tiles_by_variant(
    features: np.ndarray,
    tile_counts: Sequence[int],
    *,
    variant_id: str,
    family_slices: Mapping[str, list[slice]],
    median: np.ndarray,
    artifact_payload: Mapping[str, object],
    tile_mode: str,
) -> np.ndarray:
    masked = _mask_features(features, variant_id=variant_id, family_slices=family_slices, median=median)
    tile_scores = _score_features(masked, artifact_payload)
    image_scores: list[float] = []
    offset = 0
    for count in tile_counts:
        current = tile_scores[offset : offset + int(count)]
        image_scores.append(aggregate_tile_scores(current, tile_mode=tile_mode))
        offset += int(count)
    return np.asarray(image_scores, dtype=np.float32)


def score_group_family_components(
    detector: UnifiedArtifactDetector,
    paths: Sequence[str | Path],
    *,
    family_slices: Mapping[str, list[slice]],
    median: np.ndarray,
    image_batch_size: int,
    progress_label: str | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    batch_size = max(1, int(image_batch_size))
    outputs: dict[str, dict[str, list[np.ndarray]]] = {
        variant.variant: {"W": [], "T": [], "max_side": []}
        for variant in FAMILY_VARIANTS
    }

    for start in range(0, len(paths), batch_size):
        chunk_paths = paths[start : start + batch_size]
        whole_tensors: list[torch.Tensor] = []
        tile_tensors: list[torch.Tensor] = []
        tile_counts: list[int] = []
        max_sides: list[int] = []

        for path in chunk_paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                max_sides.append(max(int(width), int(height)))
                for artifact_variant in detector.config.artifact_variants:
                    whole_tensors.append(
                        pil_to_tensor(
                            apply_variant(rgb, detector.config.artifact_image_size, artifact_variant),
                            detector.config.artifact_image_size,
                        )
                    )
                if max(width, height) <= detector.config.tile_size:
                    tiles = [rgb]
                else:
                    boxes = native_tile_boxes(width, height, detector.config.tile_size, detector.config.tile_grid_size)
                    tiles = extract_native_tiles(rgb, boxes, tile_size=detector.config.tile_size)
                tile_counts.append(len(tiles))
                for tile in tiles:
                    tile_tensors.append(pil_to_tensor(tile, detector.config.artifact_image_size, "clean"))

        whole_features = _extract_features(detector, whole_tensors)
        tile_features = _extract_features(detector, tile_tensors)
        max_side_array = np.asarray(max_sides, dtype=np.float32)
        for variant in FAMILY_VARIANTS:
            outputs[variant.variant]["W"].append(
                _aggregate_variant_scores(
                    whole_features,
                    variant_count=len(detector.config.artifact_variants),
                    variant_id=variant.variant,
                    family_slices=family_slices,
                    median=median,
                    artifact_payload=detector.artifact_payload,
                )
            )
            outputs[variant.variant]["T"].append(
                _score_tiles_by_variant(
                    tile_features,
                    tile_counts,
                    variant_id=variant.variant,
                    family_slices=family_slices,
                    median=median,
                    artifact_payload=detector.artifact_payload,
                    tile_mode=detector.config.tile_mode,
                )
            )
            outputs[variant.variant]["max_side"].append(max_side_array)
        if progress_label:
            print(f"{progress_label}: chunk {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)

    merged_outputs: dict[str, dict[str, np.ndarray]] = {}
    for variant in FAMILY_VARIANTS:
        merged_outputs[variant.variant] = {
            key: np.concatenate(chunks, axis=0).astype(np.float32)
            for key, chunks in outputs[variant.variant].items()
        }
    return merged_outputs


def _family_slice_map(detector: UnifiedArtifactDetector) -> dict[str, list[slice]]:
    families = detector.artifact_extractor.feature_families
    mapping: dict[str, list[slice]] = {}
    for variant_id, patterns in DROP_FAMILIES.items():
        mapping[variant_id] = _feature_slices(families, patterns)
    for variant_id, patterns in KEEP_FAMILIES.items():
        mapping[variant_id] = _feature_slices(families, patterns)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser("Run Phase 2 artifact feature-family masking ablation")
    parser.add_argument("--component_dir", default="results/experiments/phase2_prior_ablation/current17_components")
    parser.add_argument("--output_dir", default="results/experiments/phase2_artifact_family_ablation")
    parser.add_argument("--config", default="configs/apfreq_train100k_full.yaml")
    parser.add_argument("--weights_json", default="results/main/source_weight_calibration/selection_protocol.json")
    parser.add_argument("--source_root", default="dataset/train_100k/progan_train")
    parser.add_argument("--source_median_max_images", type=int, default=512)
    parser.add_argument("--image_batch_size", type=int, default=128)
    parser.add_argument("--groups", default="", help="Optional comma-separated group subset for debugging/resume")
    parser.add_argument("--max_images_per_group", type=int, default=0, help="Debug limit; 0 uses every image")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    labels, components, paths, groups = load_component_directory(args.component_dir)
    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config))
    detector = UnifiedArtifactDetector(config, device=args.device, load_semantic=False)
    params = FusionParams.from_detector_config(config)
    weights = _load_weights(args.weights_json)
    out = Path(args.output_dir)
    cache_dir = out / (
        f"family_component_cache_first{int(args.max_images_per_group)}"
        if int(args.max_images_per_group) > 0
        else "family_component_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    median_path = out / "source_real_feature_median.npy"
    if median_path.exists():
        median = np.load(median_path).astype(np.float32)
    else:
        median = compute_source_real_median(
            detector,
            source_root=args.source_root,
            max_images=int(args.source_median_max_images),
            image_batch_size=int(args.image_batch_size),
        )
        out.mkdir(parents=True, exist_ok=True)
        np.save(median_path, median)

    family_slices = _family_slice_map(detector)
    scores_by_variant = {variant.variant: [] for variant in FAMILY_VARIANTS}
    labels_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []
    group_names = groups.astype(str)
    unique_groups = sorted(set(group_names.tolist()))
    if str(args.groups).strip():
        requested = {item.strip() for item in str(args.groups).split(",") if item.strip()}
        unique_groups = [group for group in unique_groups if group in requested]
        missing = sorted(requested.difference(unique_groups))
        if missing:
            raise ValueError(f"requested groups not found: {missing}")

    for group in unique_groups:
        cache_path = cache_dir / f"{group}.npz"
        group_indices = np.flatnonzero(group_names == group)
        if int(args.max_images_per_group) > 0:
            group_indices = group_indices[: int(args.max_images_per_group)]
        group_labels = labels[group_indices]
        group_components = {key: np.asarray(value[group_indices], dtype=np.float32) for key, value in components.items()}
        group_paths = paths[group_indices]
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            family_components = {
                variant.variant: {
                    "W": cached[f"{variant.variant}__W"].astype(np.float32),
                    "T": cached[f"{variant.variant}__T"].astype(np.float32),
                }
                for variant in FAMILY_VARIANTS
            }
        else:
            family_components = score_group_family_components(
                detector,
                group_paths,
                family_slices=family_slices,
                median=median,
                image_batch_size=int(args.image_batch_size),
                progress_label=group,
            )
            np.savez_compressed(
                cache_path,
                labels=group_labels,
                paths=group_paths,
                groups=np.asarray([group] * int(group_labels.shape[0]), dtype=str),
                **{
                    f"{variant_id}__{component_key}": component_values[component_key]
                    for variant_id, component_values in family_components.items()
                    for component_key in ("W", "T")
                },
            )
        labels_chunks.append(group_labels)
        group_chunks.append(np.asarray([group] * int(group_labels.shape[0]), dtype=str))
        for variant in FAMILY_VARIANTS:
            patched = {
                **group_components,
                "W": family_components[variant.variant]["W"],
                "T": family_components[variant.variant]["T"],
            }
            scores_by_variant[variant.variant].append(compute_learned_weight_scores(patched, params, weights))
        print(f"{group}: n={group_labels.shape[0]} cache={int(cache_path.exists())}", flush=True)

    merged_scores = {
        variant: np.concatenate(chunks, axis=0).astype(np.float32)
        for variant, chunks in scores_by_variant.items()
    }
    merged_labels = np.concatenate(labels_chunks, axis=0)
    merged_groups = np.concatenate(group_chunks, axis=0)
    rows = write_ablation_report(
        output_dir=out,
        labels=merged_labels,
        groups=merged_groups,
        scores_by_variant=merged_scores,
        variants=FAMILY_VARIANTS,
        threshold=float(params.threshold),
        protocol={
            "phase": "phase2_artifact_family_ablation",
            "component_dir": str(Path(args.component_dir).resolve(strict=False)),
            "config": str(args.config),
            "weights_json": str(Path(args.weights_json).resolve(strict=False)),
            "source_root": str(Path(args.source_root).resolve(strict=False)),
            "source_median_max_images": int(args.source_median_max_images),
            "image_batch_size": int(args.image_batch_size),
            "groups": list(unique_groups),
            "max_images_per_group": int(args.max_images_per_group),
            "family_component_cache": str(cache_dir.resolve(strict=False)),
            "feature_family_slices": {
                variant: [(int(item.start), int(item.stop)) for item in slices]
                for variant, slices in family_slices.items()
            },
        },
    )

    full = next(row for row in rows if row["variant"] == "AF0_full_artifact_features")
    contribution_rows = []
    for row in rows:
        contribution_rows.append(
            {
                "variant": row["variant"],
                "variant_name": row["variant_name"],
                "delta_mean_acc": float(row["mean_acc"]) - float(full["mean_acc"]),
                "delta_mean_ap": float(row["mean_ap"]) - float(full["mean_ap"]),
                "delta_mean_auc": float(row["mean_auc"]) - float(full["mean_auc"]),
            }
        )
    write_rows_csv(out / "family_contribution.csv", contribution_rows)
    print(json.dumps({"variants": len(rows), "output_dir": str(out.resolve(strict=False))}, indent=2))


if __name__ == "__main__":
    main()
