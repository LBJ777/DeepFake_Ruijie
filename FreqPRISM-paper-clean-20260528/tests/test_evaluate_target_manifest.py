from __future__ import annotations

import csv
from pathlib import Path

from scripts.evaluate_target import build_source_calibration_settings, load_target_samples


def test_load_target_samples_prefers_manifest_groups(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"path": str(tmp_path / "fake.png"), "label": 1, "class_name": "dalle3", "split": "test"},
        {"path": str(tmp_path / "real.png"), "label": 0, "class_name": "dalle3", "split": "test"},
    ]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)

    samples = load_target_samples(target_root=tmp_path / "unused", manifest=manifest)

    assert [sample.label for sample in samples] == [1, 0]
    assert [sample.group for sample in samples] == ["dalle3", "dalle3"]


def test_source_calibration_settings_disabled_by_default(tmp_path: Path) -> None:
    raw_config = {
        "dataset": {
            "source_train_manifest": str(tmp_path / "source.csv"),
            "target_test_manifest": str(tmp_path / "target.csv"),
        },
        "evaluation": {"per_label": 0},
    }

    settings = build_source_calibration_settings(
        raw_config=raw_config,
        output_dir=tmp_path / "results",
    )

    assert settings.enabled is False


def test_genimage_source_calibration_settings_use_source_manifest_and_output_cache(tmp_path: Path) -> None:
    source_manifest = tmp_path / "sd14_train_manifest.csv"
    raw_config = {
        "dataset": {
            "source_train_manifest": str(source_manifest),
            "target_test_manifest": str(tmp_path / "val_manifest.csv"),
        },
        "evaluation": {
            "source_probability_calibration": {
                "enabled": True,
                "dataset": "genimage",
                "mode": "real_fpr_logit_bias",
                "target_real_fpr_pct": 5.0,
                "calibration_per_label": 1000,
                "calibration_seed": 20260529,
            }
        },
    }

    settings = build_source_calibration_settings(
        raw_config=raw_config,
        output_dir=tmp_path / "results",
    )

    assert settings.enabled is True
    assert settings.dataset == "genimage"
    assert settings.manifest == source_manifest.resolve(strict=False)
    assert settings.cache_dir == (tmp_path / "results" / "source_calibration_score_cache").resolve(strict=False)
    assert settings.mode == "real_fpr_logit_bias"
    assert settings.target_real_fpr_pct == 5.0
    assert settings.calibration_per_label == 1000
    assert settings.calibration_seed == 20260529
