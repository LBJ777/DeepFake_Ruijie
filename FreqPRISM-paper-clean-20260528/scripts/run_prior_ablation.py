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
from utils.prior_ablation import write_prior_ablation_report


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
    parser = argparse.ArgumentParser("Run Phase 2 prior ablation from exported component scores")
    parser.add_argument("--component_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/freqprism_gpu_full.yaml")
    parser.add_argument("--weights_json", default="results/source_weight_selection/selection_protocol.json")
    args = parser.parse_args()

    labels, components, _paths, groups = load_component_directory(args.component_dir)
    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config))
    params = FusionParams.from_detector_config(config)
    weights = _load_weights(args.weights_json)
    rows = write_prior_ablation_report(
        output_dir=args.output_dir,
        labels=labels,
        groups=groups,
        components=components,
        params=params,
        weights=weights,
        component_dir=args.component_dir,
        weights_json=args.weights_json,
    )
    print(json.dumps({"variants": len(rows), "output_dir": str(Path(args.output_dir).resolve(strict=False))}, indent=2))


if __name__ == "__main__":
    main()
