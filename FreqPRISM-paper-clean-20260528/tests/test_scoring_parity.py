from __future__ import annotations

import numpy as np

from data.datasets import ImageSample
from scripts.check_scoring_parity import (
    ENABLED_SCORE_DRIFT_ATOL,
    compare_component_scores,
    compare_score_outputs,
    load_baseline_score_cache,
)


def _components(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "W": values.astype(np.float32),
        "T": values.astype(np.float32),
        "S": values.astype(np.float32),
        "R": values.astype(np.float32),
        "max_side": np.asarray([256, 1024], dtype=np.float32),
        "final_fixed": values.astype(np.float32),
    }


def test_enabled_score_drift_tolerance_matches_accepted_parity_run() -> None:
    assert ENABLED_SCORE_DRIFT_ATOL == 1e-3


def test_compare_component_scores_accepts_identical_outputs() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline = _components(np.asarray([0.2, 0.8], dtype=np.float32))
    candidate = _components(np.asarray([0.2, 0.8], dtype=np.float32))

    result = compare_component_scores(labels, baseline, candidate, threshold=0.5, atol=1e-6, groups=["a", "a"])

    assert result["passed"] is True
    assert result["label_flips"] == 0
    assert result["max_abs_diff"]["final_fixed"] == 0.0
    assert result["metric_csv_exact"] is True


def test_compare_component_scores_rejects_label_flip() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline = _components(np.asarray([0.49, 0.8], dtype=np.float32))
    candidate = _components(np.asarray([0.51, 0.8], dtype=np.float32))

    result = compare_component_scores(labels, baseline, candidate, threshold=0.5, atol=1e-6, groups=["a", "a"])

    assert result["passed"] is False
    assert result["label_flips"] == 1
    assert result["max_abs_diff"]["final_fixed"] > 1e-6
    assert result["metric_csv_exact"] is False


def test_compare_component_scores_rejects_max_side_change() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline = _components(np.asarray([0.2, 0.8], dtype=np.float32))
    candidate = _components(np.asarray([0.2, 0.8], dtype=np.float32))
    candidate["max_side"] = np.asarray([255, 1024], dtype=np.float32)

    result = compare_component_scores(labels, baseline, candidate, threshold=0.5, atol=1e-6, groups=["a", "a"])

    assert result["passed"] is False
    assert result["max_side_exact"] is False


def test_compare_score_outputs_accepts_cached_baseline_scores() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline_scores = np.asarray([0.2, 0.8], dtype=np.float32)
    candidate_scores = np.asarray([0.2, 0.8], dtype=np.float32)

    result = compare_score_outputs(
        labels,
        baseline_scores,
        candidate_scores,
        threshold=0.5,
        atol=1e-6,
        groups=["biggan", "biggan"],
    )

    assert result["passed"] is True
    assert result["label_flips"] == 0
    assert result["metric_csv_exact"] is True
    assert result["max_abs_diff"]["final_fixed"] == 0.0


def test_compare_score_outputs_accepts_small_score_drift_with_stable_metrics() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline_scores = np.asarray([0.20, 0.80], dtype=np.float32)
    candidate_scores = np.asarray([0.2007, 0.7993], dtype=np.float32)

    result = compare_score_outputs(
        labels,
        baseline_scores,
        candidate_scores,
        threshold=0.5,
        atol=1e-3,
        groups=["biggan", "biggan"],
    )

    assert result["passed"] is True
    assert result["label_flips"] == 0
    assert result["metric_csv_exact"] is True
    assert result["max_abs_diff"]["final_fixed"] > 1e-6


def test_compare_score_outputs_rejects_score_drift_above_enabled_tolerance() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline_scores = np.asarray([0.20, 0.80], dtype=np.float32)
    candidate_scores = np.asarray([0.202, 0.798], dtype=np.float32)

    result = compare_score_outputs(
        labels,
        baseline_scores,
        candidate_scores,
        threshold=0.5,
        atol=1e-3,
        groups=["biggan", "biggan"],
    )

    assert result["passed"] is False
    assert result["label_flips"] == 0
    assert result["metric_csv_exact"] is True
    assert result["max_abs_diff"]["final_fixed"] > 1e-3


def test_compare_score_outputs_rejects_cached_baseline_metric_change() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline_scores = np.asarray([0.49, 0.8], dtype=np.float32)
    candidate_scores = np.asarray([0.51, 0.8], dtype=np.float32)

    result = compare_score_outputs(
        labels,
        baseline_scores,
        candidate_scores,
        threshold=0.5,
        atol=1e-6,
        groups=["biggan", "biggan"],
    )

    assert result["passed"] is False
    assert result["label_flips"] == 1
    assert result["metric_csv_exact"] is False


def test_load_baseline_score_cache_matches_samples_across_dataset_roots(tmp_path) -> None:
    cache_dir = tmp_path / "score_cache"
    cache_dir.mkdir()
    cached_paths = np.asarray(
        [
            "/old/root/dataset/test/test/biggan/0_real/a.png",
            "/old/root/dataset/test/test/biggan/1_fake/b.png",
        ]
    )
    np.savez_compressed(
        cache_dir / "biggan.npz",
        labels=np.asarray([0, 1], dtype=np.int64),
        scores=np.asarray([0.2, 0.8], dtype=np.float32),
        paths=cached_paths,
    )
    samples = [
        ImageSample(path=tmp_path / "dataset" / "AIGCDetectBenchmark_test" / "biggan" / "0_real" / "a.png", label=0, group="biggan"),
        ImageSample(path=tmp_path / "dataset" / "AIGCDetectBenchmark_test" / "biggan" / "1_fake" / "b.png", label=1, group="biggan"),
    ]

    labels, scores = load_baseline_score_cache(cache_dir, samples)

    np.testing.assert_array_equal(labels, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(scores, np.asarray([0.2, 0.8], dtype=np.float32))
