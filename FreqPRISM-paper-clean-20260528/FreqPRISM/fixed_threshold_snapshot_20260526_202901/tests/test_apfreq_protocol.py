from __future__ import annotations

from pathlib import Path

import yaml

from data.datasets import count_by_label
from networks.detector import UnifiedDetectorConfig


ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve(strict=False)


def test_dffreq_style_project_layout_exists() -> None:
    for relative in [
        "data/datasets.py",
        "data/manifests.py",
        "networks/detector.py",
        "networks/artifact_prior.py",
        "networks/residual_prior.py",
        "options/base_options.py",
        "options/train_options.py",
        "options/test_options.py",
        "train.py",
        "test.py",
        "validate.py",
    ]:
        assert (ROOT / relative).exists(), relative


def test_full_train100k_config_has_no_sample_caps() -> None:
    config = yaml.safe_load((ROOT / "configs" / "apfreq_train100k_full.yaml").read_text())

    assert config["project"] == "FreqPRISM"
    assert _resolve_project_path(config["dataset"]["source_train_root"]).exists()
    assert _resolve_project_path(config["dataset"]["target_test_root"]).exists()
    assert config["artifact_prior"]["train_per_label"] == 0
    assert config["semantic_prior"]["train_per_label"] == 0
    assert config["semantic_prior"]["holdout_per_label"] == 0
    assert config["residual_prior"]["max_samples_per_label"] == 0
    assert config["evaluation"]["per_label"] == 0
    assert config["evaluation"]["full_target"] is True


def test_train100k_symlink_dataset_is_counted() -> None:
    config = yaml.safe_load((ROOT / "configs" / "apfreq_train100k_full.yaml").read_text())
    counts = count_by_label(_resolve_project_path(config["dataset"]["source_train_root"]))

    assert counts == {"real": 50000, "fake": 50000, "total": 100000}


def test_default_detector_config_is_freqprism_full_protocol() -> None:
    config = UnifiedDetectorConfig.from_root(ROOT)

    assert config.artifact_model_path == ROOT / "checkpoints" / "artifact_prior" / "artifact_prior_models.joblib"
    assert config.semantic_probe_path == ROOT / "checkpoints" / "semantic_prior" / "semantic_probe.joblib"
    assert config.residual_prior_path == ROOT / "checkpoints" / "residual_prior" / "checkpoint-1.pth"
    assert config.residual_inference_image_size is None
    assert config.gamma == 0.08


def test_public_docs_use_freqprism_name() -> None:
    assert (ROOT / "README.md").read_text().startswith("# FreqPRISM\n")
    assert (ROOT / "docs" / "PROTOCOL.md").read_text().startswith("# FreqPRISM Protocol\n")
