from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from utils.component_scores import FusionParams
from utils.full_fusion_weight_calibration import (
    FullFusionWeightParams,
    FullFusionWeightSearchConfig,
    search_full_fusion_weights,
    write_full_fusion_weight_artifacts,
)


def _params() -> FusionParams:
    return FusionParams(
        beta=0.20,
        alpha_low_pos=0.15,
        alpha_low_neg=0.15,
        alpha_high_pos=0.20,
        alpha_high_neg=0.00,
        alpha_high_neg_guard=0.20,
        tile_delta_threshold=0.00,
        high_res_threshold=960.0,
        gamma=0.08,
        threshold=0.50,
    )


def _components() -> dict[str, np.ndarray]:
    return {
        "W": np.asarray([0.51, 0.51, 0.49, 0.49], dtype=np.float32),
        "T": np.asarray([0.51, 0.51, 0.49, 0.49], dtype=np.float32),
        "S": np.asarray([0.50, 0.50, 0.50, 0.50], dtype=np.float32),
        "R": np.asarray([0.40, 0.40, 0.60, 0.60], dtype=np.float32),
        "max_side": np.asarray([512, 512, 512, 512], dtype=np.float32),
    }


def _labels() -> np.ndarray:
    return np.asarray([0, 0, 1, 1], dtype=np.int64)


def _groups() -> np.ndarray:
    return np.asarray(["a", "a", "b", "b"], dtype=object)


def test_search_full_fusion_weights_can_select_non_anchor_gamma_from_source_only() -> None:
    result = search_full_fusion_weights(
        labels=_labels(),
        components=_components(),
        anchor_params=_params(),
        groups=_groups(),
        config=FullFusionWeightSearchConfig(
            beta_scale_grid=(1.0,),
            alpha_low_pos_scale_grid=(1.0,),
            alpha_low_neg_scale_grid=(1.0,),
            alpha_high_pos_scale_grid=(1.0,),
            alpha_high_neg_grid=(0.0,),
            alpha_high_neg_guard_scale_grid=(1.0,),
            gamma_scale_grid=(1.0, 2.0),
            max_mean_score_drift=0.10,
            max_flip_rate=1.0,
            min_group_size=1,
            lambda_anchor=0.0,
            lambda_flip=0.0,
        ),
    )

    assert result.selected.gamma_scale == 2.0
    assert result.selected_metrics["overall_ba"] > result.baseline_metrics["overall_ba"]
    assert result.selected_params.gamma == 0.16
    assert result.target_labels_used_for_selection is False


def test_write_full_fusion_weight_artifacts_writes_protocol_candidates_and_reports(tmp_path: Path) -> None:
    artifacts = write_full_fusion_weight_artifacts(
        output_dir=tmp_path,
        source_labels=_labels(),
        source_components=_components(),
        source_groups=_groups(),
        current_labels=_labels(),
        current_components=_components(),
        current_groups=_groups(),
        anchor_params=_params(),
        config=FullFusionWeightSearchConfig(
            beta_scale_grid=(1.0,),
            alpha_low_pos_scale_grid=(1.0,),
            alpha_low_neg_scale_grid=(1.0,),
            alpha_high_pos_scale_grid=(1.0,),
            alpha_high_neg_grid=(0.0,),
            alpha_high_neg_guard_scale_grid=(1.0,),
            gamma_scale_grid=(1.0, 2.0),
            max_mean_score_drift=0.10,
            max_flip_rate=1.0,
            min_group_size=1,
            lambda_anchor=0.0,
            lambda_flip=0.0,
        ),
        source_component_dir="source/cache",
        current_component_dir="current/cache",
    )

    expected_files = [
        "selection_protocol.json",
        "decision.json",
        "paper_table.csv",
        "weight_search/candidates.csv",
        "weight_search/full_weight_search.json",
        "current17_anchor/overall.csv",
        "current17_source_calibrated/overall.csv",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists(), filename

    protocol = json.loads((tmp_path / "selection_protocol.json").read_text())
    table = list(csv.DictReader((tmp_path / "paper_table.csv").open(newline="")))

    assert artifacts["search"].selected == FullFusionWeightParams(gamma_scale=2.0)
    assert protocol["target_labels_used_for_selection"] is False
    assert protocol["selected_weights"]["gamma_scale"] == 2.0
    assert table[0]["variant"] == "anchor"
    assert table[1]["variant"] == "source_calibrated_full_alpha_split"


def test_run_full_fusion_weight_calibration_cli_writes_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source_dir = tmp_path / "source_components"
    current_dir = tmp_path / "current_components"
    source_dir.mkdir()
    current_dir.mkdir()
    payload = {
        "labels": _labels(),
        "groups": _groups().astype(str),
        "paths": np.asarray([f"image_{index}.png" for index in range(len(_labels()))], dtype=str),
        **_components(),
    }
    np.savez(source_dir / "demo.npz", **payload)
    np.savez(current_dir / "demo.npz", **payload)

    output_dir = tmp_path / "phase1w_full"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_full_fusion_weight_calibration.py"),
            "--source_component_dir",
            str(source_dir),
            "--current_component_dir",
            str(current_dir),
            "--output_dir",
            str(output_dir),
            "--scale_grid",
            "1,2",
            "--alpha_high_neg_grid",
            "0",
            "--max_mean_score_drift",
            "0.10",
            "--max_flip_rate",
            "1.0",
            "--min_group_size",
            "1",
            "--lambda_anchor",
            "0.0",
            "--lambda_flip",
            "0.0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "selection_protocol.json").exists()
    assert (output_dir / "current17_source_calibrated" / "overall.csv").exists()
