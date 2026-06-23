#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from networks.detector import UnifiedDetectorConfig
from utils.component_scores import FusionParams, WeightParams, load_component_directory
from utils.phase2_ablation_reports import compute_residual_npr_ablation_scores, write_ablation_report


def _config_name(value: str) -> str:
    path = Path(value)
    if path.suffix in {".yaml", ".yml"}:
        return path.name
    return value


def _load_weights(path: str | Path) -> WeightParams:
    payload = json.loads(Path(path).read_text())
    values = payload.get("selected_weights", payload)
    return WeightParams.from_mapping(values)


def _parse_gamma_scales(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("gamma scale list must contain at least one value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser("Run Phase 2 residual/NPR ablation from component scores")
    parser.add_argument("--component_dir", default="results/experiments/phase2_prior_ablation/current17_components")
    parser.add_argument("--output_dir", default="results/experiments/phase2_residual_npr_ablation")
    parser.add_argument("--config", default="configs/apfreq_train100k_full.yaml")
    parser.add_argument("--weights_json", default="results/main/source_weight_calibration/selection_protocol.json")
    parser.add_argument("--gamma_scales", default="0,0.5,1.0,1.5,2.0")
    args = parser.parse_args()

    labels, components, _paths, groups = load_component_directory(args.component_dir)
    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config))
    params = FusionParams.from_detector_config(config)
    weights = _load_weights(args.weights_json)
    scores, variants = compute_residual_npr_ablation_scores(
        components,
        params,
        weights,
        gamma_scales=_parse_gamma_scales(args.gamma_scales),
    )
    rows = write_ablation_report(
        output_dir=args.output_dir,
        labels=labels,
        groups=groups,
        scores_by_variant=scores,
        variants=variants,
        threshold=float(params.threshold),
        protocol={
            "phase": "phase2_residual_npr_ablation",
            "component_dir": str(Path(args.component_dir).resolve(strict=False)),
            "config": str(args.config),
            "weights_json": str(Path(args.weights_json).resolve(strict=False)),
            "gamma_scales": list(_parse_gamma_scales(args.gamma_scales)),
            "note": "Component-cache post-hoc ablation covers residual drop/only/combinations/gamma sweep. NPR energy-only requires image-level residual feature extraction.",
        },
    )
    print(json.dumps({"variants": len(rows), "output_dir": str(Path(args.output_dir).resolve(strict=False))}, indent=2))


if __name__ == "__main__":
    main()
