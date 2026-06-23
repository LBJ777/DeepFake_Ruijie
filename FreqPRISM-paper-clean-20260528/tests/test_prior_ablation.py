from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from networks.score_blend import logits_to_probabilities, probabilities_to_logits
from utils.component_scores import FusionParams, WeightParams, compute_learned_weight_scores
from utils.prior_ablation import PRIOR_ABLATION_VARIANTS, compute_prior_ablation_scores, write_prior_ablation_report


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


def test_prior_ablation_variants_include_expected_matrix() -> None:
    assert [variant.variant_id for variant in PRIOR_ABLATION_VARIANTS] == [
        "A0_whole_artifact",
        "A1_artifact",
        "A2_semantic",
        "A3_residual",
        "A4_no_artifact",
        "A5_no_semantic",
        "A6_no_residual",
        "A7_no_tile",
        "A8_full",
    ]


def test_prior_ablation_full_matches_calibrated_weight_scores() -> None:
    components = _components()
    params = _params()
    weights = WeightParams.default()

    scores = compute_prior_ablation_scores(components, params, weights)

    np.testing.assert_allclose(scores["A8_full"], compute_learned_weight_scores(components, params, weights))
    np.testing.assert_allclose(scores["A0_whole_artifact"], components["W"])
    np.testing.assert_allclose(scores["A2_semantic"], components["S"])
    np.testing.assert_allclose(scores["A3_residual"], components["R"])


def test_prior_ablation_artifact_variant_uses_tile_delta_only() -> None:
    components = _components()
    params = _params()
    weights = WeightParams.default()
    scores = compute_prior_ablation_scores(components, params, weights)

    w = probabilities_to_logits(components["W"]).astype(np.float64)
    t = probabilities_to_logits(components["T"]).astype(np.float64)
    expected = logits_to_probabilities((w + params.beta * np.maximum(0.0, t - w)).astype(np.float32))

    np.testing.assert_allclose(scores["A1_artifact"], expected, rtol=1e-6, atol=1e-6)


def test_write_prior_ablation_report_writes_overall_and_per_generator(tmp_path: Path) -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    groups = np.asarray(["g1", "g1", "g2", "g2"], dtype=str)
    components = {
        "W": np.asarray([0.10, 0.90, 0.20, 0.80], dtype=np.float32),
        "T": np.asarray([0.10, 0.95, 0.25, 0.85], dtype=np.float32),
        "S": np.asarray([0.20, 0.80, 0.30, 0.70], dtype=np.float32),
        "R": np.asarray([0.40, 0.60, 0.45, 0.55], dtype=np.float32),
        "max_side": np.asarray([512, 512, 512, 512], dtype=np.float32),
    }

    mean_rows = write_prior_ablation_report(
        output_dir=tmp_path,
        labels=labels,
        groups=groups,
        components=components,
        params=_params(),
        weights=WeightParams.default(),
    )

    assert {row["variant"] for row in mean_rows} == {variant.variant_id for variant in PRIOR_ABLATION_VARIANTS}
    assert (tmp_path / "overall.csv").exists()
    assert (tmp_path / "per_generator.csv").exists()
    assert (tmp_path / "group_slices.csv").exists()
    assert json.loads((tmp_path / "protocol.json").read_text())["target_labels_used_for_selection"] is False

    with (tmp_path / "per_generator.csv").open(newline="") as handle:
        per_generator = list(csv.DictReader(handle))
    assert len(per_generator) == len(PRIOR_ABLATION_VARIANTS) * 2
