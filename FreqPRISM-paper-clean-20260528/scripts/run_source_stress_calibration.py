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
from utils.source_stress_calibration import SourceStressConfig, write_source_stress_artifacts


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
    parser = argparse.ArgumentParser("Run pure source-only stress calibration for compact fusion weights")
    parser.add_argument(
        "--source_component_dir",
        default="results/experiments/phase1w_source_weight_calibration/source_gate_components",
    )
    parser.add_argument("--output_dir", default="results/experiments/phase1s_source_stress_calibration")
    parser.add_argument("--selection_protocol_out", default="")
    parser.add_argument("--config", default="configs/apfreq_train100k_source_gamma_anchor.yaml")
    parser.add_argument("--scale_grid", type=_parse_float_list, default=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0))
    parser.add_argument("--max_real_logloss", type=float, default=0.0069)
    parser.add_argument("--max_flip_rate", type=float, default=0.0001)
    parser.add_argument("--max_mean_score_drift", type=float, default=0.01)
    parser.add_argument("--min_source_ba", type=float, default=99.995)
    parser.add_argument("--min_source_ap", type=float, default=99.999999)
    parser.add_argument("--min_source_auc", type=float, default=99.999999)
    args = parser.parse_args()

    labels, components, _paths, groups = load_component_directory(args.source_component_dir)
    detector_config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config))
    params = FusionParams.from_detector_config(detector_config)
    search_config = SourceStressConfig(
        scale_grid=args.scale_grid,
        max_real_logloss=float(args.max_real_logloss),
        max_flip_rate=float(args.max_flip_rate),
        max_mean_score_drift=float(args.max_mean_score_drift),
        min_source_ba=float(args.min_source_ba),
        min_source_ap=float(args.min_source_ap),
        min_source_auc=float(args.min_source_auc),
    )
    artifacts = write_source_stress_artifacts(
        output_dir=args.output_dir,
        labels=labels,
        components=components,
        groups=groups,
        anchor_params=params,
        config=search_config,
        source_component_dir=args.source_component_dir,
        selection_protocol_out=args.selection_protocol_out or None,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve(strict=False)),
                "selected_weights": artifacts["protocol"]["selected_weights"],
                "effective_parameters": artifacts["protocol"]["effective_parameters"],
                "accepted_candidate_count": artifacts["protocol"]["accepted_candidate_count"],
                "target_labels_used_for_selection": artifacts["protocol"]["target_labels_used_for_selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
