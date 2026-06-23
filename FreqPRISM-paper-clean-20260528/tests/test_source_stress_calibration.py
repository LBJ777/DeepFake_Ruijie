from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from utils.component_scores import FusionParams
from utils.source_stress_calibration import (
    SourceStressConfig,
    select_source_stress_candidate,
    write_source_stress_artifacts,
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
        gamma=0.12,
        threshold=0.50,
    )


def _components() -> dict[str, np.ndarray]:
    return {
        "W": np.asarray([0.04, 0.08, 0.92, 0.96, 0.10, 0.90], dtype=np.float32),
        "T": np.asarray([0.05, 0.10, 0.96, 0.98, 0.14, 0.94], dtype=np.float32),
        "S": np.asarray([0.06, 0.12, 0.94, 0.97, 0.18, 0.93], dtype=np.float32),
        "R": np.asarray([0.08, 0.14, 0.95, 0.98, 0.20, 0.94], dtype=np.float32),
        "max_side": np.asarray([512, 512, 512, 512, 512, 512], dtype=np.float32),
    }


def _labels() -> np.ndarray:
    return np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)


def _groups() -> np.ndarray:
    return np.asarray(["cat", "dog", "cat", "dog", "cat", "dog"], dtype=str)


def test_select_source_stress_candidate_uses_fake_logloss_under_real_guard() -> None:
    rows = [
        {
            "variant": "overstrong_tile",
            "tile_scale": 1.50,
            "semantic_pos_scale": 2.0,
            "semantic_neg_scale": 1.25,
            "residual_scale": 1.75,
            "fake_logloss": 0.00090,
            "real_logloss": 0.00710,
            "source_logloss": 0.00400,
            "anchor_distance": 2.50,
            "accepted": False,
        },
        {
            "variant": "current_source_stress",
            "tile_scale": 1.25,
            "semantic_pos_scale": 2.0,
            "semantic_neg_scale": 1.25,
            "residual_scale": 1.75,
            "fake_logloss": 0.00095,
            "real_logloss": 0.00688,
            "source_logloss": 0.00392,
            "anchor_distance": 2.25,
            "accepted": True,
        },
        {
            "variant": "conservative_anchor_neighbor",
            "tile_scale": 1.00,
            "semantic_pos_scale": 1.50,
            "semantic_neg_scale": 1.25,
            "residual_scale": 1.50,
            "fake_logloss": 0.00120,
            "real_logloss": 0.00600,
            "source_logloss": 0.00380,
            "anchor_distance": 1.25,
            "accepted": True,
        },
    ]

    selected = select_source_stress_candidate(rows)

    assert selected["variant"] == "current_source_stress"
    assert selected["target_labels_used_for_selection"] is False


def test_write_source_stress_artifacts_records_source_only_protocol(tmp_path: Path) -> None:
    artifacts = write_source_stress_artifacts(
        output_dir=tmp_path,
        labels=_labels(),
        components=_components(),
        groups=_groups(),
        anchor_params=_params(),
        config=SourceStressConfig(
            scale_grid=(1.0, 1.25),
            max_real_logloss=0.50,
            max_flip_rate=1.0,
            max_mean_score_drift=1.0,
            min_source_ba=50.0,
            min_source_ap=50.0,
            min_source_auc=50.0,
        ),
        source_component_dir="source/cache",
    )

    assert (tmp_path / "source_stress_candidates.csv").exists()
    assert (tmp_path / "selection_protocol.json").exists()
    assert artifacts["protocol"]["selection_data"] == "source_gate_stress_only"
    assert artifacts["protocol"]["target_labels_used_for_selection"] is False
    assert artifacts["protocol"]["target_labels_used_for_final_report_only"] is False

    rows = list(csv.DictReader((tmp_path / "source_stress_candidates.csv").open(newline="")))
    assert len(rows) == 16
    assert artifacts["protocol"]["candidate_count"] == 16


def test_run_source_stress_calibration_cli_writes_protocol(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source_dir = tmp_path / "source_components"
    source_dir.mkdir()
    payload = {
        "labels": _labels(),
        "groups": _groups(),
        "paths": np.asarray([f"image_{index}.png" for index in range(len(_labels()))], dtype=str),
        **_components(),
    }
    np.savez(source_dir / "demo.npz", **payload)

    output_dir = tmp_path / "report"
    protocol_out = tmp_path / "selection_protocol.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_source_stress_calibration.py"),
            "--source_component_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--selection_protocol_out",
            str(protocol_out),
            "--config",
            "configs/apfreq_train100k_source_gamma_anchor.yaml",
            "--scale_grid",
            "1.0,1.25",
            "--max_real_logloss",
            "0.50",
            "--max_flip_rate",
            "1.0",
            "--max_mean_score_drift",
            "1.0",
            "--min_source_ba",
            "50",
            "--min_source_ap",
            "50",
            "--min_source_auc",
            "50",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    protocol = json.loads(protocol_out.read_text())
    assert protocol["phase"] == "phase1s_source_stress_calibration"
    assert protocol["target_labels_used_for_selection"] is False
    assert (output_dir / "source_stress_candidates.csv").exists()
