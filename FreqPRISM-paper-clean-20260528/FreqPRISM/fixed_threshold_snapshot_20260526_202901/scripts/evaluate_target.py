#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import collect_labeled_images, limit_per_label
from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig
from utils.evaluation import score_group_with_cache
from utils.metrics import write_target_report
from utils.progress import progress_iter


def main() -> None:
    parser = argparse.ArgumentParser("Frozen target report for FreqPRISM")
    parser.add_argument("--target_root", default=str(REPO_ROOT / "dataset" / "test" / "test"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "results" / "target_report"))
    parser.add_argument("--per_label", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config_name", default="apfreq_train100k_full.yaml")
    parser.add_argument("--artifact_model", default="")
    parser.add_argument("--semantic_probe", default="")
    parser.add_argument("--residual_checkpoint", default="")
    parser.add_argument("--residual_batch_size", type=int, default=32)
    parser.add_argument("--score_cache_dir", default="")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, args.config_name).with_artifact_overrides(
        artifact_model_path=args.artifact_model or None,
        semantic_probe_path=args.semantic_probe or None,
        residual_prior_path=args.residual_checkpoint or None,
    )
    detector = UnifiedArtifactDetector(config, device=args.device)
    samples = collect_labeled_images(args.target_root)
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
    mean = write_target_report(out, packed, threshold=config.threshold)
    protocol = {
        "target_root": str(Path(args.target_root).resolve(strict=False)),
        "per_label": int(args.per_label),
        "threshold": float(config.threshold),
        "artifact_model": str(config.artifact_model_path.resolve(strict=False)),
        "semantic_probe": str(config.semantic_probe_path.resolve(strict=False)),
        "residual_checkpoint": str(config.residual_prior_path.resolve(strict=False)),
        "score_cache_dir": str(cache_dir.resolve(strict=False)),
        "cached_groups": cached_groups,
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_report_only": True,
        "mean": mean,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(mean, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
