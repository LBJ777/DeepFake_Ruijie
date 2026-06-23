#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig
from utils.component_scores import FusionParams, WeightParams, load_component_directory
from utils.evaluation import safe_cache_token
from utils.phase2_ablation_reports import compute_tile_resolution_ablation_scores, write_ablation_report
from utils.progress import progress_iter
from utils.tile_resolution_rescore import score_tile_resolution_variants


def _config_name(value: str) -> str:
    path = Path(value)
    if path.suffix in {".yaml", ".yml"}:
        return path.name
    return value


def _load_weights(path: str | Path) -> WeightParams:
    payload = json.loads(Path(path).read_text())
    values = payload.get("selected_weights", payload)
    return WeightParams.from_mapping(values)


def main() -> None:
    parser = argparse.ArgumentParser("Run Phase 2 native tile/resolution ablation from component scores")
    parser.add_argument("--component_dir", default="results/experiments/phase2_prior_ablation/current17_components")
    parser.add_argument("--output_dir", default="results/experiments/phase2_tile_resolution_ablation")
    parser.add_argument("--config", default="configs/apfreq_train100k_full.yaml")
    parser.add_argument("--weights_json", default="results/main/source_weight_calibration/selection_protocol.json")
    parser.add_argument("--image_level_tile_variants", action="store_true")
    parser.add_argument("--tile_variant_cache_dir", default="")
    parser.add_argument("--resized_max_side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact_forward_batch_size", type=int, default=0)
    parser.add_argument("--artifact_tile_batch_size", type=int, default=0)
    args = parser.parse_args()

    labels, components, paths, groups = load_component_directory(args.component_dir)
    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config))
    runtime_updates = {}
    if int(args.artifact_forward_batch_size) > 0:
        runtime_updates["artifact_forward_batch_size"] = int(args.artifact_forward_batch_size)
    if int(args.artifact_tile_batch_size) > 0:
        runtime_updates["artifact_tile_batch_size"] = int(args.artifact_tile_batch_size)
    if runtime_updates:
        config = replace(config, **runtime_updates)
    params = FusionParams.from_detector_config(config)
    weights = _load_weights(args.weights_json)
    tile_score_variants: dict[str, np.ndarray] = {}
    if bool(args.image_level_tile_variants):
        cache_root = Path(args.tile_variant_cache_dir) if args.tile_variant_cache_dir else Path(args.output_dir) / "tile_variant_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        variant_ids = ("RZ2_resized512_tile", "RZ3_center_crop_tile", "RZ4_tile_mean_aggregation")
        tile_score_variants = {variant_id: np.empty(labels.shape[0], dtype=np.float32) for variant_id in variant_ids}
        group_values = np.asarray(groups, dtype=str)
        detector = UnifiedArtifactDetector(config, device=str(args.device), load_semantic=False)
        for group in progress_iter(
            sorted(set(group_values.tolist())),
            total=len(set(group_values.tolist())),
            desc="tile variant groups",
            unit="group",
        ):
            mask = group_values == group
            group_paths = np.asarray(paths[mask], dtype=str)
            cache_path = cache_root / f"{safe_cache_token(group)}.npz"
            cached: dict[str, np.ndarray] | None = None
            if cache_path.exists():
                loaded = np.load(cache_path, allow_pickle=False)
                if all(variant_id in loaded.files for variant_id in variant_ids) and "paths" in loaded.files:
                    cached_paths = loaded["paths"].astype(str)
                    if cached_paths.shape == group_paths.shape and bool(np.all(cached_paths == group_paths)):
                        cached = {variant_id: loaded[variant_id].astype(np.float32) for variant_id in variant_ids}
            if cached is None:
                cached = score_tile_resolution_variants(
                    group_paths.tolist(),
                    detector=detector,
                    resized_max_side=int(args.resized_max_side),
                )
                np.savez_compressed(
                    cache_path,
                    paths=group_paths,
                    **{variant_id: np.asarray(cached[variant_id], dtype=np.float32) for variant_id in variant_ids},
                )
            for variant_id in variant_ids:
                tile_score_variants[variant_id][mask] = np.asarray(cached[variant_id], dtype=np.float32)
            print(f"{group}: n={int(mask.sum())} tile_variant_cache={int(cache_path.exists())}", flush=True)
    scores, variants = compute_tile_resolution_ablation_scores(
        components,
        params,
        weights,
        tile_score_variants=tile_score_variants,
    )
    rows = write_ablation_report(
        output_dir=args.output_dir,
        labels=labels,
        groups=groups,
        scores_by_variant=scores,
        variants=variants,
        threshold=float(params.threshold),
        protocol={
            "phase": "phase2_tile_resolution_ablation",
            "component_dir": str(Path(args.component_dir).resolve(strict=False)),
            "config": str(args.config),
            "weights_json": str(Path(args.weights_json).resolve(strict=False)),
            "image_level_tile_variants": bool(args.image_level_tile_variants),
            "tile_variant_cache_dir": str(Path(args.tile_variant_cache_dir or Path(args.output_dir) / "tile_variant_cache").resolve(strict=False)),
            "resized_max_side": int(args.resized_max_side),
            "artifact_forward_batch_size": int(config.artifact_forward_batch_size),
            "artifact_tile_batch_size": int(config.artifact_tile_batch_size),
            "device": str(args.device),
            "note": "Image-level tile variants rescore only the artifact tile evidence T while keeping W/S/R fixed from the locked component cache. RZ6 downsample-before-full-pipeline requires full W/T/S/R rescoring and is not included in this run.",
        },
    )
    print(json.dumps({"variants": len(rows), "output_dir": str(Path(args.output_dir).resolve(strict=False))}, indent=2))


if __name__ == "__main__":
    main()
