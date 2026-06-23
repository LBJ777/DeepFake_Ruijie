from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from utils.source_calibration import (
    apply_affine_calibration,
    fit_real_fpr_logit_bias,
    load_score_cache_dir,
    write_calibrated_report,
)


def test_real_fpr_bias_makes_point_five_match_source_real_quantile() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.01, 0.02, 0.03, 0.04, 0.80, 0.90], dtype=np.float32)

    calibration = fit_real_fpr_logit_bias(labels, scores, target_real_fpr_pct=25.0)
    calibrated = apply_affine_calibration(scores, calibration)

    expected_threshold = float(np.percentile(scores[labels == 0], 75.0))
    assert calibration.raw_threshold_equivalent == expected_threshold
    np.testing.assert_array_equal(calibrated >= 0.5, scores >= expected_threshold)


def test_load_score_cache_and_write_calibrated_report(tmp_path: Path) -> None:
    cache = tmp_path / "score_cache"
    cache.mkdir()
    np.savez_compressed(
        cache / "GenA.npz",
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        scores=np.asarray([0.01, 0.02, 0.20, 0.30], dtype=np.float32),
    )
    np.savez_compressed(
        cache / "GenB.npz",
        labels=np.asarray([0, 1], dtype=np.int64),
        scores=np.asarray([0.04, 0.60], dtype=np.float32),
    )
    packed = load_score_cache_dir(cache)
    calibration = fit_real_fpr_logit_bias(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0.02, 0.04, 0.90, 0.95], dtype=np.float32),
        target_real_fpr_pct=50.0,
    )

    mean = write_calibrated_report(
        tmp_path / "out",
        target_packed=packed,
        calibration_packed={"source": (np.asarray([0, 0, 1, 1]), np.asarray([0.02, 0.04, 0.90, 0.95]))},
        calibration=calibration,
        protocol={"target_labels_used_for_calibration": False},
    )

    assert (tmp_path / "out" / "per_generator.csv").exists()
    assert (tmp_path / "out" / "overall.csv").exists()
    assert (tmp_path / "out" / "calibration.csv").exists()
    assert (tmp_path / "out" / "protocol.json").exists()
    assert mean["mean_acc"] > 0.0
    with (tmp_path / "out" / "calibration.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["mode"] == "real_fpr_logit_bias"
    assert row["target_real_fpr_pct"] == "50.0"
    protocol = json.loads((tmp_path / "out" / "protocol.json").read_text())
    assert protocol["target_labels_used_for_calibration"] is False
    assert protocol["threshold"] == 0.5
