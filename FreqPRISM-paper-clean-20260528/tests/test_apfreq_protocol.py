from __future__ import annotations

import json
from pathlib import Path

import yaml

from data.datasets import count_by_label
from networks.detector import UnifiedDetectorConfig


ROOT = Path(__file__).resolve().parents[1]
MAIN_FUSION_PROTOCOL = ROOT / "results" / "main" / "pure_source_stress_calibration" / "selection_protocol.json"


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


def test_full_train100k_configs_use_repo_local_dataset_roots() -> None:
    for config_name in ("apfreq_train100k_full.yaml", "freqprism_gpu_full.yaml"):
        config = yaml.safe_load((ROOT / "configs" / config_name).read_text())

        assert config["project"] == "FreqPRISM"
        assert config["dataset"]["source_train_root"] == "dataset/train_100k/progan_train"
        assert config["dataset"]["target_test_root"] == "dataset/AIGCDetectBenchmark_test"
        assert _resolve_project_path(config["dataset"]["source_train_root"]).exists()
        assert _resolve_project_path(config["dataset"]["target_test_root"]).exists()
        assert config["artifact_prior"]["train_per_label"] == 0
        assert config["semantic_prior"]["train_per_label"] == 0
        assert config["semantic_prior"]["holdout_per_label"] == 0
        assert config["residual_prior"]["max_samples_per_label"] == 0
        assert config["composition"]["beta"] == 0.25
        assert config["composition"]["alpha_low_pos"] == 0.30
        assert config["composition"]["alpha_low_neg"] == 0.1875
        assert config["composition"]["alpha_high_pos"] == 0.40
        assert config["composition"]["alpha_high_neg"] == 0.00
        assert config["composition"]["alpha_high_neg_guard"] == 0.25
        assert config["composition"]["gamma"] == 0.21
        assert config["selection"]["source_only"] is True
        assert config["selection"]["protocol"] == "pure_source_stress_calibration"
        assert config["selection"]["target_labels_used"] is False
        assert config["evaluation"]["per_label"] == 0
        assert config["evaluation"]["full_target"] is True


def test_train100k_local_dataset_is_counted() -> None:
    config = yaml.safe_load((ROOT / "configs" / "apfreq_train100k_full.yaml").read_text())
    counts = count_by_label(_resolve_project_path(config["dataset"]["source_train_root"]))

    assert counts == {"real": 50000, "fake": 50000, "total": 100000}


def test_default_detector_config_is_freqprism_full_protocol() -> None:
    config = UnifiedDetectorConfig.from_root(ROOT)

    assert config.artifact_model_path == ROOT / "checkpoints" / "artifact_prior" / "artifact_prior_models.joblib"
    assert config.semantic_probe_path == ROOT / "checkpoints" / "semantic_prior" / "semantic_probe.joblib"
    assert config.residual_prior_path == ROOT / "checkpoints" / "residual_prior" / "checkpoint-1.pth"
    assert config.residual_inference_image_size is None
    assert config.beta == 0.25
    assert config.alpha_low_pos == 0.30
    assert config.alpha_low_neg == 0.1875
    assert config.alpha_high_pos == 0.40
    assert config.alpha_high_neg == 0.00
    assert config.alpha_high_neg_guard == 0.25
    assert config.gamma == 0.21


def test_main_fusion_source_stress_protocol_records_evidence_and_selection_role() -> None:
    protocol = json.loads(MAIN_FUSION_PROTOCOL.read_text())

    assert protocol["method_name"] == "FreqPRISM pure source-only stress-calibrated fusion"
    assert protocol["target_labels_used_for_selection"] is False
    assert protocol["target_labels_used_for_final_report_only"] is False
    assert protocol["selection_data"] == "source_gate_stress_only"
    assert protocol["selected_weights"] == {
        "residual_scale": 1.75,
        "semantic_neg_scale": 1.25,
        "semantic_pos_scale": 2.0,
        "tile_scale": 1.25,
    }
    assert protocol["effective_parameters"] == {
        "beta": 0.25,
        "alpha_low_pos": 0.30,
        "alpha_low_neg": 0.1875,
        "alpha_high_pos": 0.40,
        "alpha_high_neg": 0.0,
        "alpha_high_neg_guard": 0.25,
        "gamma": 0.21,
    }


def test_public_docs_use_freqprism_name() -> None:
    assert (ROOT / "README.md").read_text().startswith("# FreqPRISM\n")
    assert (ROOT / "docs" / "PROTOCOL.md").read_text().startswith("# FreqPRISM Protocol\n")
