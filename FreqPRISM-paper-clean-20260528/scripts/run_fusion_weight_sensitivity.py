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
from utils.fusion_weight_sensitivity import (
    DEFAULT_ALPHA_HIGH_NEG_VALUES,
    DEFAULT_GAMMA_SCALE_SWEEP_VALUES,
    DEFAULT_SCALE_SWEEP_VALUES,
    write_fusion_weight_sensitivity_report,
)


def _config_name(value: str) -> str:
    path = Path(value)
    if path.suffix in {".yaml", ".yml"}:
        return path.name
    return value


def _load_weights(path: str | Path) -> WeightParams:
    payload = json.loads(Path(path).read_text())
    values = payload.get("selected_weights", payload)
    return WeightParams.from_mapping(values)


def _parse_float_list(value: str) -> tuple[float, ...]:
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return tuple(float(piece) for piece in pieces)


def main() -> None:
    parser = argparse.ArgumentParser("Run prior fusion weight sensitivity from exported component scores")
    parser.add_argument(
        "--source_component_dir",
        default="results/experiments/phase1w_source_weight_calibration/source_gate_components",
    )
    parser.add_argument(
        "--current_component_dir",
        default="results/experiments/phase2_prior_ablation/current17_components",
    )
    parser.add_argument("--output_dir", default="results/experiments/phase2_fusion_weight_sensitivity")
    parser.add_argument("--config", default="configs/freqprism_gpu_full.yaml")
    parser.add_argument("--weights_json", default="results/source_weight_selection/selection_protocol.json")
    parser.add_argument("--compact_sweep_values", type=_parse_float_list, default=DEFAULT_SCALE_SWEEP_VALUES)
    parser.add_argument("--gamma_sweep_values", type=_parse_float_list, default=DEFAULT_GAMMA_SCALE_SWEEP_VALUES)
    parser.add_argument("--alpha_split_sweep_values", type=_parse_float_list, default=DEFAULT_SCALE_SWEEP_VALUES)
    parser.add_argument("--alpha_high_neg_values", type=_parse_float_list, default=DEFAULT_ALPHA_HIGH_NEG_VALUES)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260528)
    args = parser.parse_args()

    source_labels, source_components, _source_paths, source_groups = load_component_directory(args.source_component_dir)
    current_labels, current_components, _current_paths, current_groups = load_component_directory(args.current_component_dir)
    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config))
    params = FusionParams.from_detector_config(config)
    selected_weights = _load_weights(args.weights_json)

    report = write_fusion_weight_sensitivity_report(
        output_dir=args.output_dir,
        source_labels=source_labels,
        source_components=source_components,
        source_groups=source_groups,
        current_labels=current_labels,
        current_components=current_components,
        current_groups=current_groups,
        params=params,
        selected_weights=selected_weights,
        compact_sweep_values=args.compact_sweep_values,
        gamma_sweep_values=args.gamma_sweep_values,
        alpha_split_sweep_values=args.alpha_split_sweep_values,
        alpha_high_neg_values=args.alpha_high_neg_values,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        source_component_dir=args.source_component_dir,
        current_component_dir=args.current_component_dir,
        weights_json=args.weights_json,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve(strict=False)),
                "variant_count": int(report["protocol"]["variant_count"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
