#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from networks.detector import UnifiedDetectorConfig
from utils.component_scores import FusionParams, load_component_directory
from utils.full_fusion_weight_calibration import (
    FullFusionWeightSearchConfig,
    write_full_fusion_weight_artifacts,
)


def _config_name(value: str) -> str:
    path = Path(value)
    if path.suffix in {".yaml", ".yml"}:
        return path.name
    return value


def _parse_float_list(value: str) -> tuple[float, ...]:
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return tuple(float(piece) for piece in pieces)


def main() -> None:
    parser = argparse.ArgumentParser("Run source-only full alpha-split fusion weight calibration")
    parser.add_argument(
        "--source_component_dir",
        default="results/experiments/phase1w_source_weight_calibration/source_gate_components",
    )
    parser.add_argument(
        "--current_component_dir",
        default="results/experiments/phase2_prior_ablation/current17_components",
    )
    parser.add_argument("--output_dir", default="results/experiments/phase1w_full_alpha_split_calibration")
    parser.add_argument("--selection_protocol_out", default="")
    parser.add_argument("--config", default="configs/freqprism_gpu_full.yaml")
    parser.add_argument("--scale_grid", type=_parse_float_list, default=(0.50, 0.75, 1.00, 1.25, 1.50))
    parser.add_argument("--alpha_high_neg_grid", type=_parse_float_list, default=(0.00, 0.02, 0.05, 0.10, 0.15, 0.20))
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--lambda_drift", type=float, default=1.0)
    parser.add_argument("--lambda_flip", type=float, default=1.0)
    parser.add_argument("--lambda_anchor", type=float, default=0.25)
    parser.add_argument("--max_source_ba_drop", type=float, default=0.2)
    parser.add_argument("--max_flip_rate", type=float, default=0.01)
    parser.add_argument("--max_mean_score_drift", type=float, default=0.01)
    parser.add_argument("--min_group_size", type=int, default=25)
    args = parser.parse_args()

    source_labels, source_components, _source_paths, source_groups = load_component_directory(args.source_component_dir)
    current_labels, current_components, _current_paths, current_groups = load_component_directory(args.current_component_dir)
    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config))
    params = FusionParams.from_detector_config(config)
    search_config = FullFusionWeightSearchConfig(
        beta_scale_grid=args.scale_grid,
        alpha_low_pos_scale_grid=args.scale_grid,
        alpha_low_neg_scale_grid=args.scale_grid,
        alpha_high_pos_scale_grid=args.scale_grid,
        alpha_high_neg_grid=args.alpha_high_neg_grid,
        alpha_high_neg_guard_scale_grid=args.scale_grid,
        gamma_scale_grid=args.scale_grid,
        max_rounds=int(args.max_rounds),
        lambda_drift=float(args.lambda_drift),
        lambda_flip=float(args.lambda_flip),
        lambda_anchor=float(args.lambda_anchor),
        max_source_ba_drop=float(args.max_source_ba_drop),
        max_flip_rate=float(args.max_flip_rate),
        max_mean_score_drift=float(args.max_mean_score_drift),
        min_group_size=int(args.min_group_size),
    )
    artifacts = write_full_fusion_weight_artifacts(
        output_dir=args.output_dir,
        source_labels=source_labels,
        source_components=source_components,
        source_groups=source_groups,
        current_labels=current_labels,
        current_components=current_components,
        current_groups=current_groups,
        anchor_params=params,
        config=search_config,
        source_component_dir=args.source_component_dir,
        current_component_dir=args.current_component_dir,
        selection_protocol_out=args.selection_protocol_out or None,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve(strict=False)),
                "selected_weights": artifacts["protocol"]["selected_weights"],
                "selected_metrics": artifacts["protocol"]["selected_metrics"],
                "candidate_count": artifacts["protocol"]["candidate_count"],
                "target_labels_used_for_selection": artifacts["protocol"]["target_labels_used_for_selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
