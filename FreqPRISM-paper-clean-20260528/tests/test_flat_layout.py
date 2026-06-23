from __future__ import annotations

from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]


def test_src_tree_has_been_flattened() -> None:
    assert not (ROOT / "src").exists()
    assert not (ROOT / "unified_detector").exists()


def test_legacy_full_pipeline_helper_is_absent() -> None:
    assert not (ROOT / "utils" / "full_pipeline.py").exists()


def test_flat_project_packages_are_primary_imports() -> None:
    from data.datasets import ImageSample, collect_labeled_images
    from data.manifests import prepare_source_manifests
    from models.core import ResidualLogitCombiner
    from models.hgb_parity import aggregate_probabilities
    from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig
    from networks.semantic_prior import ClipLinearProbe, load_openai_clip
    from utils.metrics import binary_metrics
    from utils.progress import progress_iter

    assert ImageSample is not None
    assert collect_labeled_images is not None
    assert prepare_source_manifests is not None
    assert ResidualLogitCombiner is not None
    assert aggregate_probabilities is not None
    assert UnifiedArtifactDetector is not None
    assert UnifiedDetectorConfig is not None
    assert ClipLinearProbe is not None
    assert load_openai_clip is not None
    assert binary_metrics is not None
    assert list(progress_iter([1], total=1, enabled=False)) == [1]


def test_existing_joblib_artifacts_remain_loadable_after_flattening() -> None:
    semantic_probe = joblib.load(ROOT / "checkpoints" / "semantic_prior" / "semantic_probe.joblib")
    artifact_payload = joblib.load(ROOT / "checkpoints" / "artifact_prior" / "artifact_prior_models.joblib")

    assert semantic_probe.__class__.__name__ == "ClipLinearProbe"
    assert semantic_probe.__class__.__module__ == "networks.semantic_prior"
    assert set(artifact_payload) >= {"codec", "chroma", "feature_dim", "image_size"}
    assert artifact_payload["codec"].__class__.__module__ == "models.core"
    assert artifact_payload["chroma"].__class__.__module__ == "models.core"
