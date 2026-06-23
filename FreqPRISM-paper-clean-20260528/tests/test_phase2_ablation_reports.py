from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from networks.score_blend import logits_to_probabilities, probabilities_to_logits
from utils.component_scores import FusionParams, WeightParams
from utils.phase2_ablation_reports import (
    compute_residual_npr_ablation_scores,
    compute_tile_resolution_ablation_scores,
    write_ablation_report,
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
        "W": np.asarray([0.20, 0.60, 0.80], dtype=np.float32),
        "T": np.asarray([0.70, 0.40, 0.90], dtype=np.float32),
        "S": np.asarray([0.65, 0.25, 0.55], dtype=np.float32),
        "R": np.asarray([0.52, 0.45, 0.75], dtype=np.float32),
        "max_side": np.asarray([512, 1200, 1500], dtype=np.float32),
    }


def test_tile_resolution_ablation_includes_full_and_no_tile() -> None:
    scores, variants = compute_tile_resolution_ablation_scores(_components(), _params(), WeightParams.default())

    assert [variant.variant for variant in variants] == [
        "RZ0_full_native_tile",
        "RZ1_whole_only_no_tile",
        "RZ5_current_top1_tile",
    ]
    assert scores["RZ0_full_native_tile"].shape == (3,)
    np.testing.assert_allclose(scores["RZ0_full_native_tile"], scores["RZ5_current_top1_tile"], rtol=1e-6, atol=1e-6)


def test_tile_resolution_ablation_accepts_image_level_tile_variants() -> None:
    components = _components()
    resized_tile = np.asarray([0.40, 0.45, 0.50], dtype=np.float32)
    mean_tile = np.asarray([0.60, 0.55, 0.70], dtype=np.float32)

    scores, variants = compute_tile_resolution_ablation_scores(
        components,
        _params(),
        WeightParams.default(),
        tile_score_variants={
            "RZ2_resized512_tile": resized_tile,
            "RZ4_tile_mean_aggregation": mean_tile,
        },
    )

    variant_ids = [variant.variant for variant in variants]
    assert "RZ2_resized512_tile" in variant_ids
    assert "RZ4_tile_mean_aggregation" in variant_ids
    assert scores["RZ2_resized512_tile"].shape == components["W"].shape
    assert scores["RZ4_tile_mean_aggregation"].shape == components["W"].shape
    assert not np.allclose(scores["RZ2_resized512_tile"], scores["RZ0_full_native_tile"])


def test_residual_ablation_no_residual_matches_gamma_zero_formula() -> None:
    components = _components()
    params = _params()
    scores, variants = compute_residual_npr_ablation_scores(components, params, WeightParams.default(), gamma_scales=(0.0, 1.0))

    assert "RP1_no_residual" in scores
    assert "RP6_gamma_scale_0p00" in scores
    np.testing.assert_allclose(scores["RP1_no_residual"], scores["RP6_gamma_scale_0p00"], rtol=1e-6, atol=1e-6)
    assert [variant.variant for variant in variants][-2:] == ["RP6_gamma_scale_0p00", "RP6_gamma_scale_1p00"]


def test_residual_only_uses_residual_component() -> None:
    components = _components()
    scores, _variants = compute_residual_npr_ablation_scores(components, _params(), WeightParams.default())

    np.testing.assert_allclose(scores["RP2_residual_only"], components["R"])


def test_write_ablation_report_outputs_expected_files(tmp_path: Path) -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    groups = np.asarray(["g1", "g1", "g2", "g2"], dtype=str)
    scores = {
        "demo": np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
    }
    from utils.phase2_ablation_reports import AblationVariant

    rows = write_ablation_report(
        output_dir=tmp_path,
        labels=labels,
        groups=groups,
        scores_by_variant=scores,
        variants=[AblationVariant("demo", "Demo", "demo variant")],
        threshold=0.5,
        protocol={"phase": "test"},
    )

    assert rows[0]["variant"] == "demo"
    assert (tmp_path / "overall.csv").exists()
    assert (tmp_path / "per_generator.csv").exists()
    assert (tmp_path / "group_slices.csv").exists()
    assert json.loads((tmp_path / "protocol.json").read_text())["phase"] == "test"
    with (tmp_path / "per_generator.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
