from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data.datasets import ImageSample
from data.manifests import load_image_samples_from_manifest, prepare_source_gate_split
from networks.native_tiles import combine_whole_tile_aux_signed_delta_guard_scores
from networks.score_blend import logit_blend
from utils.component_scores import (
    FusionParams,
    WeightParams,
    WeightSearchConfig,
    compute_fixed_scores,
    compute_learned_weight_scores,
    search_weight_params,
)
from utils.evaluation import score_component_group_with_cache


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


def test_default_weight_scales_match_fixed_fusion_formula() -> None:
    components = {
        "W": np.asarray([0.20, 0.55, 0.81, 0.40], dtype=np.float32),
        "T": np.asarray([0.75, 0.50, 0.93, 0.45], dtype=np.float32),
        "S": np.asarray([0.65, 0.30, 0.20, 0.88], dtype=np.float32),
        "R": np.asarray([0.52, 0.45, 0.91, 0.10], dtype=np.float32),
        "max_side": np.asarray([512, 1200, 1500, 800], dtype=np.float32),
    }
    params = _params()

    base = combine_whole_tile_aux_signed_delta_guard_scores(
        components["W"],
        components["T"],
        components["S"],
        high_res_mask=components["max_side"] > params.high_res_threshold,
        beta=params.beta,
        alpha_low_pos=params.alpha_low_pos,
        alpha_low_neg=params.alpha_low_neg,
        alpha_high_pos=params.alpha_high_pos,
        alpha_high_neg=params.alpha_high_neg,
        alpha_high_neg_guard=params.alpha_high_neg_guard,
        tile_delta_threshold=params.tile_delta_threshold,
    )
    expected = logit_blend(base, components["R"], params.gamma)

    np.testing.assert_allclose(compute_fixed_scores(components, params), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        compute_learned_weight_scores(components, params, WeightParams.default()),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

def test_weight_search_uses_fixed_weights_when_drift_is_not_allowed() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    components = {
        "W": np.asarray([0.10, 0.20, 0.80, 0.90], dtype=np.float32),
        "T": np.asarray([0.10, 0.20, 0.95, 0.95], dtype=np.float32),
        "S": np.asarray([0.10, 0.20, 0.90, 0.90], dtype=np.float32),
        "R": np.asarray([0.10, 0.20, 0.80, 0.90], dtype=np.float32),
        "max_side": np.asarray([512, 512, 512, 512], dtype=np.float32),
    }
    groups = np.asarray(["a", "a", "b", "b"], dtype=object)

    result = search_weight_params(
        labels,
        components,
        _params(),
        groups=groups,
        config=WeightSearchConfig(
            tile_scale_grid=(1.0, 1.1),
            semantic_pos_scale_grid=(1.0, 1.1),
            semantic_neg_scale_grid=(1.0,),
            residual_scale_grid=(1.0, 1.1),
            max_mean_score_drift=0.0,
            max_flip_rate=0.0,
            min_group_size=1,
        ),
    )

    assert result.selected == WeightParams.default()
    assert result.selected_metrics["mean_score_drift"] == 0.0
    assert result.selected_metrics["flip_rate"] == 0.0


def test_score_component_group_with_cache_round_trips_component_arrays(tmp_path: Path) -> None:
    samples = [
        ImageSample(path=tmp_path / "a.jpg", label=0, group="demo"),
        ImageSample(path=tmp_path / "b.jpg", label=1, group="demo"),
    ]

    class FakeDetector:
        calls = 0

        def score_component_paths(self, paths, *, residual_batch_size: int):
            self.calls += 1
            assert [Path(path).name for path in paths] == ["a.jpg", "b.jpg"]
            assert residual_batch_size == 7
            return {
                "W": np.asarray([0.1, 0.9], dtype=np.float32),
                "T": np.asarray([0.2, 0.8], dtype=np.float32),
                "S": np.asarray([0.3, 0.7], dtype=np.float32),
                "R": np.asarray([0.4, 0.6], dtype=np.float32),
                "max_side": np.asarray([256, 1024], dtype=np.float32),
                "final_fixed": np.asarray([0.12, 0.88], dtype=np.float32),
            }

    detector = FakeDetector()
    labels, components, used_cache = score_component_group_with_cache(
        "demo",
        samples,
        detector=detector,
        cache_dir=tmp_path / "cache",
        residual_batch_size=7,
    )

    assert used_cache is False
    assert detector.calls == 1
    np.testing.assert_array_equal(labels, np.asarray([0, 1]))
    np.testing.assert_allclose(components["final_fixed"], np.asarray([0.12, 0.88], dtype=np.float32))

    labels_cached, components_cached, used_cache = score_component_group_with_cache(
        "demo",
        samples,
        detector=detector,
        cache_dir=tmp_path / "cache",
        residual_batch_size=7,
    )

    assert used_cache is True
    assert detector.calls == 1
    np.testing.assert_array_equal(labels_cached, labels)
    np.testing.assert_allclose(components_cached["W"], components["W"])


def test_score_component_group_with_cache_invalidates_when_detector_fingerprint_changes(tmp_path: Path) -> None:
    samples = [
        ImageSample(path=tmp_path / "a.jpg", label=0, group="demo"),
        ImageSample(path=tmp_path / "b.jpg", label=1, group="demo"),
    ]

    class FakeDetector:
        def __init__(self, fingerprint: str, score: float) -> None:
            self.fingerprint = fingerprint
            self.score = score
            self.calls = 0

        def cache_fingerprint(self) -> str:
            return self.fingerprint

        def score_component_paths(self, paths, *, residual_batch_size: int):
            self.calls += 1
            return {
                "W": np.asarray([self.score, 0.9], dtype=np.float32),
                "T": np.asarray([0.2, 0.8], dtype=np.float32),
                "S": np.asarray([0.3, 0.7], dtype=np.float32),
                "R": np.asarray([0.4, 0.6], dtype=np.float32),
                "max_side": np.asarray([256, 1024], dtype=np.float32),
                "final_fixed": np.asarray([self.score, 0.88], dtype=np.float32),
            }

    first = FakeDetector("v1", 0.12)
    _, first_components, first_used_cache = score_component_group_with_cache(
        "demo",
        samples,
        detector=first,
        cache_dir=tmp_path / "cache",
        residual_batch_size=7,
    )
    assert first_used_cache is False
    assert first.calls == 1
    np.testing.assert_allclose(first_components["final_fixed"], np.asarray([0.12, 0.88], dtype=np.float32))

    second = FakeDetector("v2", 0.34)
    _, second_components, second_used_cache = score_component_group_with_cache(
        "demo",
        samples,
        detector=second,
        cache_dir=tmp_path / "cache",
        residual_batch_size=7,
    )
    assert second_used_cache is False
    assert second.calls == 1
    np.testing.assert_allclose(second_components["final_fixed"], np.asarray([0.34, 0.88], dtype=np.float32))


def test_prepare_source_gate_split_is_stratified_by_class_and_label(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for class_name in ("cat", "dog"):
        for label_dir in ("0_real", "1_fake"):
            directory = source / class_name / label_dir
            directory.mkdir(parents=True)
            for index in range(4):
                (directory / f"{index}.jpg").write_text("placeholder")

    counts = prepare_source_gate_split(source_root=source, output_dir=tmp_path / "split", gate_fraction=0.25, seed=3)

    assert counts == {"source_fit": 12, "source_gate": 4}
    with (tmp_path / "split" / "source_gate_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert sorted((row["class_name"], int(row["label"])) for row in rows) == [
        ("cat", 0),
        ("cat", 1),
        ("dog", 0),
        ("dog", 1),
    ]


def test_load_image_samples_from_manifest_uses_class_name_as_group(tmp_path: Path) -> None:
    image_path = tmp_path / "source" / "cat" / "0_real" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("placeholder")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "path": str(image_path),
                "label": 0,
                "class_name": "cat",
                "split": "source_gate",
            }
        )

    samples = load_image_samples_from_manifest(manifest)

    assert samples == [ImageSample(path=image_path.resolve(strict=False), label=0, group="cat")]
