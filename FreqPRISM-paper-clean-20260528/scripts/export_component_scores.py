#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import collect_labeled_images, limit_per_label
from data.manifests import load_image_samples_from_manifest
from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig
from utils.evaluation import score_component_group_with_cache
from utils.progress import progress_iter


def _config_name(value: str) -> str:
    path = Path(value)
    if path.suffix in {".yaml", ".yml"}:
        return path.name
    return value


def _config_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return PROJECT_ROOT / path
    return PROJECT_ROOT / "configs" / path


def main() -> None:
    parser = argparse.ArgumentParser("Export FreqPRISM component scores")
    parser.add_argument("--config", default="configs/apfreq_train100k_full.yaml")
    parser.add_argument("--target_root", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_label", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact_model", default="")
    parser.add_argument("--semantic_probe", default="")
    parser.add_argument("--residual_checkpoint", default="")
    parser.add_argument("--artifact_forward_batch_size", type=int, default=0)
    parser.add_argument("--artifact_tile_batch_size", type=int, default=0)
    parser.add_argument("--semantic_forward_batch_size", type=int, default=0)
    parser.add_argument("--residual_batch_size", type=int, default=32)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config)).with_artifact_overrides(
        artifact_model_path=args.artifact_model or None,
        semantic_probe_path=args.semantic_probe or None,
        residual_prior_path=args.residual_checkpoint or None,
    )
    runtime_overrides = {}
    if int(args.artifact_forward_batch_size) > 0:
        runtime_overrides["artifact_forward_batch_size"] = int(args.artifact_forward_batch_size)
    if int(args.artifact_tile_batch_size) > 0:
        runtime_overrides["artifact_tile_batch_size"] = int(args.artifact_tile_batch_size)
    if int(args.semantic_forward_batch_size) > 0:
        runtime_overrides["semantic_forward_batch_size"] = int(args.semantic_forward_batch_size)
    if runtime_overrides:
        config = replace(config, **runtime_overrides)
    detector = UnifiedArtifactDetector(config, device=args.device)
    target_root = args.target_root
    if not target_root:
        raw = yaml.safe_load(_config_path(args.config).read_text())
        target_root = str(raw["dataset"]["target_test_root"])
    if args.manifest:
        samples = load_image_samples_from_manifest(args.manifest)
    else:
        samples = collect_labeled_images(target_root)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = sorted({sample.group for sample in samples})
    cached_groups: list[str] = []
    counts: dict[str, int] = {}
    for group in progress_iter(
        groups,
        total=len(groups),
        desc="component groups",
        unit="group",
        enabled=not bool(args.no_progress),
    ):
        group_samples = [sample for sample in samples if sample.group == group]
        if int(args.per_label) > 0:
            group_samples = limit_per_label(group_samples, int(args.per_label))
        labels, components, used_cache = score_component_group_with_cache(
            group,
            group_samples,
            detector=detector,
            cache_dir=out,
            residual_batch_size=int(args.residual_batch_size),
        )
        if used_cache:
            cached_groups.append(group)
        counts[group] = int(labels.shape[0])
        print(
            f"{group}: n={labels.shape[0]} cache={int(used_cache)} "
            f"final_fixed_mean={float(np.mean(components['final_fixed'])):.6f}",
            flush=True,
        )

    protocol = {
        "config": str(args.config),
        "target_root": str(Path(target_root).resolve(strict=False)),
        "manifest": str(Path(args.manifest).resolve(strict=False)) if args.manifest else "",
        "output_dir": str(out.resolve(strict=False)),
        "per_label": int(args.per_label),
        "threshold": float(config.threshold),
        "component_keys": ["W", "T", "S", "R", "max_side", "final_fixed"],
        "artifact_model": str(config.artifact_model_path.resolve(strict=False)),
        "semantic_probe": str(config.semantic_probe_path.resolve(strict=False)),
        "residual_checkpoint": str(config.residual_prior_path.resolve(strict=False)),
        "residual_batch_size": int(args.residual_batch_size),
        "artifact_forward_batch_size": int(config.artifact_forward_batch_size),
        "artifact_tile_batch_size": int(config.artifact_tile_batch_size),
        "semantic_forward_batch_size": int(config.semantic_forward_batch_size),
        "counts": counts,
        "cached_groups": cached_groups,
        "target_labels_used_for_selection": False,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"groups": len(groups), "samples": int(sum(counts.values()))}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
