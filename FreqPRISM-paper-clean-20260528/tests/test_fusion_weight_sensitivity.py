from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from utils.component_scores import FusionParams, WeightParams
from utils.fusion_weight_sensitivity import (
    build_fusion_weight_sensitivity_report,
    write_fusion_weight_sensitivity_report,
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
        "W": np.asarray([0.12, 0.28, 0.76, 0.88, 0.42, 0.61], dtype=np.float32),
        "T": np.asarray([0.18, 0.54, 0.94, 0.82, 0.60, 0.73], dtype=np.float32),
        "S": np.asarray([0.22, 0.35, 0.81, 0.70, 0.30, 0.92], dtype=np.float32),
        "R": np.asarray([0.16, 0.40, 0.84, 0.67, 0.48, 0.78], dtype=np.float32),
        "max_side": np.asarray([512, 1024, 512, 1400, 768, 1600], dtype=np.float32),
    }


def _labels() -> np.ndarray:
    return np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)


def _groups() -> np.ndarray:
    return np.asarray(["gaugan", "gaugan", "gaugan", "biggan", "biggan", "biggan"], dtype=object)


def test_build_fusion_weight_sensitivity_report_contains_anchor_sweeps_and_drops() -> None:
    report = build_fusion_weight_sensitivity_report(
        source_labels=_labels(),
        source_components=_components(),
        source_groups=_groups(),
        current_labels=_labels(),
        current_components=_components(),
        current_groups=_groups(),
        params=_params(),
        selected_weights=WeightParams(tile_scale=1.25, semantic_pos_scale=1.0, semantic_neg_scale=1.0, residual_scale=1.0),
        compact_sweep_values=(0.0, 1.0),
        gamma_sweep_values=(0.0, 1.0, 2.0),
        alpha_split_sweep_values=(1.0,),
        alpha_high_neg_values=(0.0,),
    )

    source_ids = {row["variant"] for row in report["source_gate_weight_sweep"]}
    current_ids = {row["variant"] for row in report["current17_weight_sweep_overall"]}

    assert "W0_anchor" in source_ids
    assert "W1_source_selected_compact" in source_ids
    assert "B0_beta_scale_0p00" in source_ids
    assert "A0_semantic_pos_scale_0p00" in source_ids
    assert "R0_gamma_scale_0p00" in source_ids
    assert "R0_gamma_scale_2p00" in source_ids
    assert "D0_no_tile_weight" in source_ids
    assert "D4_no_residual_weight" in source_ids
    assert source_ids == current_ids

    selected = next(row for row in report["source_gate_weight_sweep"] if row["variant"] == "W1_source_selected_compact")
    assert selected["selection_data"] == "source_gate_only"
    assert selected["tile_scale"] == 1.25
    assert selected["target_labels_used_for_selection"] is False
    assert selected["threshold"] == 0.5

    protocol = report["protocol"]
    assert protocol["phase"] == "phase2_fusion_weight_sensitivity"
    assert protocol["target_labels_used_for_selection"] is False
    assert protocol["target_labels_used_for_final_report_only"] is True
    assert protocol["gamma_sweep_values"] == [0.0, 1.0, 2.0]

    paper_tables = report["paper_tables"]
    assert "table4a_beta_ablation" in paper_tables
    assert "table4d_gamma_ablation" in paper_tables
    assert "tableS4h_alpha_high_neg_direct_ablation" in paper_tables
    assert [row["variant"] for row in paper_tables["table4d_gamma_ablation"]] == [
        "R0_gamma_scale_0p00",
        "R0_gamma_scale_1p00",
        "R0_gamma_scale_2p00",
    ]
    reference = next(row for row in paper_tables["table4d_gamma_ablation"] if row["variant"] == "R0_gamma_scale_1p00")
    assert reference["is_reference"] is True
    assert reference["delta_mean_acc"] == 0.0
    assert reference["delta_tail_f_acc"] == 0.0


def test_write_fusion_weight_sensitivity_report_writes_expected_files(tmp_path: Path) -> None:
    write_fusion_weight_sensitivity_report(
        output_dir=tmp_path,
        source_labels=_labels(),
        source_components=_components(),
        source_groups=_groups(),
        current_labels=_labels(),
        current_components=_components(),
        current_groups=_groups(),
        params=_params(),
        selected_weights=WeightParams.default(),
        compact_sweep_values=(0.0, 1.0),
        gamma_sweep_values=(0.0, 1.0, 2.0),
        alpha_split_sweep_values=(1.0,),
        alpha_high_neg_values=(0.0,),
        source_component_dir="source/cache",
        current_component_dir="current/cache",
        weights_json="weights.json",
    )

    expected = [
        "source_gate_weight_sweep.csv",
        "current17_weight_sweep_overall.csv",
        "current17_weight_sweep_per_generator.csv",
        "current17_weight_sweep_group_slices.csv",
        "paper_tables/table4a_beta_ablation.csv",
        "paper_tables/table4b_alpha_pos_ablation.csv",
        "paper_tables/table4c_alpha_neg_ablation.csv",
        "paper_tables/table4d_gamma_ablation.csv",
        "paper_tables/tableS4d_alpha_low_pos_ablation.csv",
        "paper_tables/tableS4e_alpha_low_neg_ablation.csv",
        "paper_tables/tableS4f_alpha_high_pos_ablation.csv",
        "paper_tables/tableS4g_alpha_high_neg_guard_ablation.csv",
        "paper_tables/tableS4h_alpha_high_neg_direct_ablation.csv",
        "protocol.json",
    ]
    for filename in expected:
        assert (tmp_path / filename).exists(), filename

    source_rows = list(csv.DictReader((tmp_path / "source_gate_weight_sweep.csv").open(newline="")))
    table4d_rows = list(csv.DictReader((tmp_path / "paper_tables" / "table4d_gamma_ablation.csv").open(newline="")))
    protocol = json.loads((tmp_path / "protocol.json").read_text())

    assert source_rows[0]["variant"] == "W0_anchor"
    assert {row["variant"] for row in table4d_rows} == {
        "R0_gamma_scale_0p00",
        "R0_gamma_scale_1p00",
        "R0_gamma_scale_2p00",
    }
    assert "delta_mean_acc" in table4d_rows[0]
    assert protocol["source_component_dir"].endswith("source/cache")
    assert protocol["current_component_dir"].endswith("current/cache")


def test_run_fusion_weight_sensitivity_cli_writes_report_from_component_cache(tmp_path: Path) -> None:
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
    payload["final_fixed"] = np.asarray([0.10, 0.20, 0.82, 0.86, 0.44, 0.78], dtype=np.float32)
    np.savez(source_dir / "demo.npz", **payload)
    np.savez(current_dir / "demo.npz", **payload)
    weights_json = tmp_path / "weights.json"
    weights_json.write_text(json.dumps({"selected_weights": WeightParams.default().to_dict()}) + "\n")

    output_dir = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_fusion_weight_sensitivity.py"),
            "--source_component_dir",
            str(source_dir),
            "--current_component_dir",
            str(current_dir),
            "--output_dir",
            str(output_dir),
            "--weights_json",
            str(weights_json),
            "--compact_sweep_values",
            "0,1",
            "--gamma_sweep_values",
            "0,1,2",
            "--alpha_split_sweep_values",
            "1",
            "--alpha_high_neg_values",
            "0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "source_gate_weight_sweep.csv").exists()
    assert (output_dir / "current17_weight_sweep_overall.csv").exists()
    assert (output_dir / "paper_tables" / "table4d_gamma_ablation.csv").exists()
