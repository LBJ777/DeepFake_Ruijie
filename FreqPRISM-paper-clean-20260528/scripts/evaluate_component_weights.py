#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from networks.detector import UnifiedDetectorConfig
from utils.component_scores import (
    FusionParams,
    WeightParams,
    compute_fixed_scores,
    compute_learned_weight_scores,
    load_component_directory,
)
from utils.metrics import write_target_report


def _config_name(value: str) -> str:
    path = Path(value)
    if path.suffix in {".yaml", ".yml"}:
        return path.name
    return value


def _load_weights_payload(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _weights_from_payload(payload: dict[str, object]) -> WeightParams:
    values = payload.get("selected_weights", payload)
    return WeightParams.from_mapping(values)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate anchored learned fusion weights from exported component scores")
    parser.add_argument("--component_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/freqprism_gpu_full.yaml")
    parser.add_argument("--policy", choices=("fixed", "learned_weights"), default="learned_weights")
    parser.add_argument("--weights_json", default="")
    args = parser.parse_args()

    labels, components, _paths, groups = load_component_directory(args.component_dir)
    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, _config_name(args.config))
    params = FusionParams.from_detector_config(config)
    weight_protocol_payload: dict[str, object] = {}
    if args.policy == "fixed":
        scores = components["final_fixed"] if "final_fixed" in components else compute_fixed_scores(components, params)
        weights_payload: dict[str, object] = {}
    else:
        if not args.weights_json:
            raise ValueError("--weights_json is required when --policy learned_weights")
        weight_protocol_payload = _load_weights_payload(args.weights_json)
        weights = _weights_from_payload(weight_protocol_payload)
        scores = compute_learned_weight_scores(components, params, weights)
        weights_payload = weights.to_dict()
    target_labels_used_for_selection = bool(weight_protocol_payload.get("target_labels_used_for_selection", False))

    packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    group_values = np.asarray(groups, dtype=str)
    for group in sorted(set(group_values.tolist())):
        mask = group_values == group
        packed[group] = (labels[mask], scores[mask].astype(np.float32))

    out = Path(args.output_dir)
    mean = write_target_report(out, packed, threshold=float(params.threshold))
    protocol = {
        "component_dir": str(Path(args.component_dir).resolve(strict=False)),
        "config": str(args.config),
        "policy": str(args.policy),
        "weights_json": str(Path(args.weights_json).resolve(strict=False)) if args.weights_json else "",
        "weights": weights_payload,
        "threshold": float(params.threshold),
        "target_labels_used_for_selection": target_labels_used_for_selection,
        "target_labels_used_for_final_report_only": True,
        "mean": mean,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(mean, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
