#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import ImageSample, collect_labeled_images, limit_per_label
from data.manifests import load_image_samples_from_manifest
from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig
from utils.evaluation import score_group_with_cache
from utils.metrics import write_target_report
from utils.progress import progress_iter
from utils.source_calibration import (
    fit_platt_affine,
    fit_real_fpr_logit_bias,
    load_score_cache_dir,
    score_calibration_manifest,
    write_calibrated_report,
)


def load_target_samples(*, target_root: str | Path, manifest: str | Path | None = None) -> list[ImageSample]:
    if manifest is not None and str(manifest):
        return load_image_samples_from_manifest(manifest)
    return collect_labeled_images(target_root)


def load_raw_config(config_name: str) -> dict:
    path = Path(config_name)
    if not path.is_absolute():
        path = PROJECT_ROOT / "configs" / path.name
    return yaml.safe_load(path.read_text())


@dataclass(frozen=True)
class SourceCalibrationSettings:
    enabled: bool
    dataset: str
    manifest: Path
    cache_dir: Path
    mode: str
    target_real_fpr_pct: float
    calibration_per_label: int
    calibration_seed: int


def build_source_calibration_settings(*, raw_config: dict, output_dir: str | Path) -> SourceCalibrationSettings:
    evaluation = dict(raw_config.get("evaluation") or {})
    calibration = dict(evaluation.get("source_probability_calibration") or {})
    enabled = bool(calibration.get("enabled", False))
    dataset = str(calibration.get("dataset", "")).strip().lower()
    source_manifest = calibration.get("calibration_manifest") or raw_config.get("dataset", {}).get("source_train_manifest")
    if not source_manifest:
        source_manifest = ""
    manifest_path = Path(str(source_manifest))
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    cache_path = calibration.get("calibration_score_cache_dir")
    if cache_path:
        cache_dir = Path(str(cache_path))
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir
    else:
        cache_dir = Path(output_dir) / "source_calibration_score_cache"
    return SourceCalibrationSettings(
        enabled=enabled and dataset == "genimage",
        dataset=dataset,
        manifest=manifest_path.resolve(strict=False),
        cache_dir=cache_dir.resolve(strict=False),
        mode=str(calibration.get("mode", "real_fpr_logit_bias")),
        target_real_fpr_pct=float(calibration.get("target_real_fpr_pct", 5.0)),
        calibration_per_label=int(calibration.get("calibration_per_label", 1000)),
        calibration_seed=int(calibration.get("calibration_seed", 20260529)),
    )


def main() -> None:
    parser = argparse.ArgumentParser("Frozen target report for FreqPRISM")
    parser.add_argument("--target_root", default=str(PROJECT_ROOT / "dataset" / "AIGCDetectBenchmark_test"))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "results" / "target_report"))
    parser.add_argument("--per_label", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config_name", default="apfreq_train100k_full.yaml")
    parser.add_argument(
        "--scoring_mode",
        choices=("config", "baseline", "equivalent_fast", "aggressive_fast"),
        default="config",
        help="Override runtime scoring mode without changing the YAML config.",
    )
    parser.add_argument("--artifact_model", default="")
    parser.add_argument("--semantic_probe", default="")
    parser.add_argument("--residual_checkpoint", default="")
    parser.add_argument("--residual_batch_size", type=int, default=32)
    parser.add_argument("--score_cache_dir", default="")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()
    raw_config = load_raw_config(args.config_name)

    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, args.config_name).with_artifact_overrides(
        artifact_model_path=args.artifact_model or None,
        semantic_probe_path=args.semantic_probe or None,
        residual_prior_path=args.residual_checkpoint or None,
    )
    if args.scoring_mode != "config":
        config = config.with_runtime_overrides(scoring_mode=args.scoring_mode)
    detector = UnifiedArtifactDetector(config, device=args.device)
    samples = load_target_samples(target_root=args.target_root, manifest=args.manifest)
    packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cache_dir = Path(args.score_cache_dir) if args.score_cache_dir else Path(args.output_dir) / "score_cache"
    cached_groups: list[str] = []
    groups = sorted({sample.group for sample in samples})
    for group in progress_iter(
        groups,
        total=len(groups),
        desc="target generators",
        unit="gen",
        enabled=not bool(args.no_progress),
    ):
        group_samples = [sample for sample in samples if sample.group == group]
        if int(args.per_label) > 0:
            group_samples = limit_per_label(group_samples, int(args.per_label))
        print(f"[target] scoring {group}: n={len(group_samples)}", flush=True)
        labels, scores, used_cache = score_group_with_cache(
            group,
            group_samples,
            detector=detector,
            cache_dir=cache_dir,
            residual_batch_size=int(args.residual_batch_size),
        )
        if used_cache:
            cached_groups.append(group)
        packed[group] = (labels, scores)
        print(f"{group}: n={len(group_samples)} cache={int(used_cache)}")

    out = Path(args.output_dir)
    calibration_settings = build_source_calibration_settings(raw_config=raw_config, output_dir=out)
    base_protocol = {
        "target_root": str(Path(args.target_root).resolve(strict=False)),
        "manifest": str(Path(args.manifest).resolve(strict=False)) if args.manifest else "",
        "per_label": int(args.per_label),
        "threshold": float(config.threshold),
        "artifact_model": str(config.artifact_model_path.resolve(strict=False)),
        "semantic_probe": str(config.semantic_probe_path.resolve(strict=False)),
        "residual_checkpoint": str(config.residual_prior_path.resolve(strict=False)),
        "score_cache_dir": str(cache_dir.resolve(strict=False)),
        "scoring_mode": str(config.scoring_mode),
        "gpu_preprocess": bool(config.gpu_preprocess),
        "artifact_forward_batch_size": int(config.artifact_forward_batch_size),
        "artifact_tile_batch_size": int(config.artifact_tile_batch_size),
        "semantic_forward_batch_size": int(config.semantic_forward_batch_size),
        "parity_required_for_optimized_reporting": bool(config.scoring_mode != "baseline"),
        "cached_groups": cached_groups,
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_report_only": True,
        "source_probability_calibration": {
            "enabled": bool(calibration_settings.enabled),
            "dataset": calibration_settings.dataset,
            "mode": calibration_settings.mode,
            "target_real_fpr_pct": float(calibration_settings.target_real_fpr_pct),
            "calibration_per_label": int(calibration_settings.calibration_per_label),
            "calibration_seed": int(calibration_settings.calibration_seed),
            "calibration_manifest": str(calibration_settings.manifest.resolve(strict=False)),
            "calibration_score_cache_dir": str(calibration_settings.cache_dir.resolve(strict=False)),
        },
    }
    if calibration_settings.enabled:
        if calibration_settings.cache_dir.exists() and list(calibration_settings.cache_dir.glob("*.npz")):
            calibration_packed = load_score_cache_dir(calibration_settings.cache_dir)
        else:
            calibration_packed = score_calibration_manifest(
                manifest=calibration_settings.manifest,
                cache_dir=calibration_settings.cache_dir,
                config_name=args.config_name,
                device=args.device,
                scoring_mode=args.scoring_mode,
                calibration_per_label=int(calibration_settings.calibration_per_label),
                calibration_seed=int(calibration_settings.calibration_seed),
                residual_batch_size=int(args.residual_batch_size),
                no_progress=bool(args.no_progress),
            )
        calibration_labels = np.concatenate([labels for labels, _ in calibration_packed.values()], axis=0)
        calibration_scores = np.concatenate([scores for _, scores in calibration_packed.values()], axis=0)
        if calibration_settings.mode == "real_fpr_logit_bias":
            calibration = fit_real_fpr_logit_bias(
                calibration_labels,
                calibration_scores,
                target_real_fpr_pct=float(calibration_settings.target_real_fpr_pct),
            )
        elif calibration_settings.mode == "platt_affine":
            calibration = fit_platt_affine(calibration_labels, calibration_scores)
        else:
            raise ValueError(f"unsupported source_probability_calibration.mode: {calibration_settings.mode}")
        mean = write_calibrated_report(
            out,
            target_packed=packed,
            calibration_packed=calibration_packed,
            calibration=calibration,
            protocol=base_protocol,
        )
        print(json.dumps(mean, indent=2, sort_keys=True))
        return
    else:
        mean = write_target_report(out, packed, threshold=config.threshold)
    protocol = {**base_protocol, "mean": mean}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(mean, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
