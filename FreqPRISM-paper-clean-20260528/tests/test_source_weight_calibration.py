from __future__ import annotations

import csv
import json
from pathlib import Path

from utils.source_weight_calibration import build_phase1w_artifacts, write_phase1w_artifacts


FIELDNAMES = ["mean_acc", "mean_ap", "mean_auc", "mean_f_acc", "mean_fnr", "mean_fpr", "mean_r_acc"]
PER_GENERATOR_FIELDNAMES = ["generator", "acc", "ap", "auc", "f_acc", "fnr", "fpr", "r_acc"]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, *, mean_acc: float, mean_ap: float, group_acc: float) -> None:
    _write_csv(
        path / "overall.csv",
        [
            {
                "mean_acc": mean_acc,
                "mean_ap": mean_ap,
                "mean_auc": 97.0,
                "mean_f_acc": 93.0,
                "mean_fnr": 7.0,
                "mean_fpr": 5.0,
                "mean_r_acc": 95.0,
            }
        ],
        FIELDNAMES,
    )
    _write_csv(
        path / "per_generator.csv",
        [
            {
                "generator": "gaugan",
                "acc": group_acc,
                "ap": 98.0,
                "auc": 97.0,
                "f_acc": 92.0,
                "fnr": 8.0,
                "fpr": 3.0,
                "r_acc": 97.0,
            },
            {
                "generator": "biggan",
                "acc": group_acc - 1.0,
                "ap": 98.0,
                "auc": 97.0,
                "f_acc": 91.0,
                "fnr": 9.0,
                "fpr": 4.0,
                "r_acc": 96.0,
            },
        ],
        PER_GENERATOR_FIELDNAMES,
    )


def _write_search(path: Path) -> None:
    payload = {
        "accepted_candidate_count": 4,
        "baseline_metrics": {
            "flip_rate": 0.0,
            "mean_score_drift": 0.0,
            "overall_ba": 99.0,
            "worst_group_ba": 98.0,
        },
        "candidate_count": 4,
        "component_dir": "source/components",
        "config": "configs/freqprism_gpu_full.yaml",
        "constraints": {
            "lambda_anchor": 0.25,
            "lambda_drift": 1.0,
            "lambda_flip": 1.0,
            "max_flip_rate": 0.01,
            "max_mean_score_drift": 0.01,
            "max_source_ba_drop": 0.2,
            "min_group_size": 25,
        },
        "selected_metrics": {
            "flip_rate": 0.0,
            "mean_score_drift": 0.0,
            "overall_ba": 99.0,
            "worst_group_ba": 98.0,
        },
        "selected_weights": {
            "residual_scale": 1.0,
            "semantic_neg_scale": 1.0,
            "semantic_pos_scale": 1.0,
            "tile_scale": 1.0,
        },
        "selection_data": "source_gate_only",
        "target_labels_used": False,
        "threshold": 0.5,
    }
    path.write_text(json.dumps(payload) + "\n")


def test_build_phase1w_artifacts_records_source_only_selection(tmp_path: Path) -> None:
    search_path = tmp_path / "weight_search.json"
    fixed_dir = tmp_path / "fixed"
    learned_dir = tmp_path / "learned"
    _write_search(search_path)
    _write_report(fixed_dir, mean_acc=93.5, mean_ap=99.1, group_acc=94.0)
    _write_report(learned_dir, mean_acc=93.5, mean_ap=99.1, group_acc=94.0)

    artifacts = build_phase1w_artifacts(
        weight_search_json=search_path,
        fixed_report_dir=fixed_dir,
        learned_report_dir=learned_dir,
    )

    assert artifacts["protocol"]["phase"] == "phase1w_source_weight_calibration"
    assert artifacts["protocol"]["target_labels_used_for_selection"] is False
    assert artifacts["decision"]["decision"]["selected_weights_are_anchor_weights"] is True
    assert artifacts["decision"]["current17_mean_delta_learned_minus_fixed"]["mean_acc"] == 0.0
    assert artifacts["paper_rows"][1]["variant"] == "source_calibrated_weights"
    assert artifacts["paper_rows"][1]["target_labels_used_for_selection"] is False


def test_write_phase1w_artifacts_writes_decision_protocol_and_table(tmp_path: Path) -> None:
    search_path = tmp_path / "weight_search.json"
    fixed_dir = tmp_path / "fixed"
    learned_dir = tmp_path / "learned"
    output_dir = tmp_path / "phase1w"
    selection_protocol = tmp_path / "selection_protocol.json"
    _write_search(search_path)
    _write_report(fixed_dir, mean_acc=93.0, mean_ap=99.0, group_acc=94.0)
    _write_report(learned_dir, mean_acc=93.25, mean_ap=99.05, group_acc=94.25)

    write_phase1w_artifacts(
        weight_search_json=search_path,
        fixed_report_dir=fixed_dir,
        learned_report_dir=learned_dir,
        output_dir=output_dir,
        selection_protocol_out=selection_protocol,
    )

    decision = json.loads((output_dir / "decision.json").read_text())
    protocol = json.loads(selection_protocol.read_text())
    table_rows = list(csv.DictReader((output_dir / "paper_table.csv").open(newline="")))

    assert decision["current17_mean_delta_learned_minus_fixed"]["mean_acc"] == 0.25
    assert protocol["selected_weights"]["tile_scale"] == 1.0
    assert table_rows[0]["variant"] == "fixed_anchor"
    assert table_rows[1]["variant"] == "source_calibrated_weights"
