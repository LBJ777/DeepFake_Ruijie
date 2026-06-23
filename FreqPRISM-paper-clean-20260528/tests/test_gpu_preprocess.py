from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from PIL import Image

from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig


ROOT_CONFIG = "apfreq_train100k_full.yaml"
GPU_CONFIG = "freqprism_gpu_full.yaml"


def test_gpu_preprocess_runtime_flag_is_parsed() -> None:
    config = UnifiedDetectorConfig.from_root(".", ROOT_CONFIG)

    assert config.gpu_preprocess is False
    assert config.semantic_forward_batch_size == 32
    assert config.artifact_tile_batch_size == config.artifact_forward_batch_size


def test_apfreq_config_enables_equivalent_fast_runtime() -> None:
    config = UnifiedDetectorConfig.from_root(".", ROOT_CONFIG)

    assert config.scoring_mode == "equivalent_fast"
    assert config.gpu_preprocess is False


def test_gpu_config_enables_fast_preprocess_and_reuses_default_weights() -> None:
    config = UnifiedDetectorConfig.from_root(".", GPU_CONFIG)

    assert config.gpu_preprocess is True
    assert config.artifact_model_path.name == "artifact_prior_models.joblib"
    assert "checkpoints" in str(config.artifact_model_path)
    assert "checkpoints" in str(config.semantic_probe_path)
    assert "checkpoints" in str(config.residual_prior_path)
    assert config.artifact_forward_batch_size == 128
    assert config.artifact_tile_batch_size == 128
    assert config.semantic_forward_batch_size == 128


def test_gpu_config_maps_to_aggressive_fast_for_backward_compatibility() -> None:
    config = UnifiedDetectorConfig.from_root(".", GPU_CONFIG)

    assert config.scoring_mode == "aggressive_fast"
    assert config.gpu_preprocess is True


def test_runtime_override_can_select_baseline_without_mutating_source_config() -> None:
    config = UnifiedDetectorConfig.from_root(".", ROOT_CONFIG)
    updated = config.with_runtime_overrides(scoring_mode="baseline")

    assert config.scoring_mode == "equivalent_fast"
    assert updated.scoring_mode == "baseline"
    assert updated.gpu_preprocess is False


def test_gpu_tile_path_batches_tiles_without_pil_scoring(tmp_path) -> None:
    image_path = tmp_path / "large.png"
    array = np.zeros((512, 512, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(512, dtype=np.uint8)[None, :]
    array[:, :, 1] = np.arange(512, dtype=np.uint8)[:, None]
    Image.fromarray(array).save(image_path)

    config = replace(
        UnifiedDetectorConfig.from_root(".", ROOT_CONFIG),
        gpu_preprocess=True,
        artifact_forward_batch_size=4,
        artifact_tile_batch_size=4,
    )
    detector = UnifiedArtifactDetector.__new__(UnifiedArtifactDetector)
    detector.config = config
    detector.device = torch.device("cpu")
    seen_batch_shapes: list[tuple[int, ...]] = []

    def fake_tensor_batch_scores(batch: torch.Tensor) -> np.ndarray:
        seen_batch_shapes.append(tuple(batch.shape))
        start = sum(shape[0] for shape in seen_batch_shapes[:-1])
        return np.arange(start, start + batch.shape[0], dtype=np.float32)

    def fail_pil_scoring(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("GPU tile preprocessing should not call the PIL scoring path")

    detector._score_artifact_tensor_batch = fake_tensor_batch_scores
    detector._score_rgb_images = fail_pil_scoring

    scores, max_sides = detector.score_artifact_tile_paths([image_path])

    assert max_sides.tolist() == [512.0]
    assert scores.tolist() == [8.0]
    assert seen_batch_shapes == [(4, 3, 256, 256), (4, 3, 256, 256), (1, 3, 256, 256)]


def test_equivalent_fast_tile_path_batches_tiles_with_pil_semantics(tmp_path) -> None:
    image_path = tmp_path / "large.png"
    array = np.zeros((512, 512, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(512, dtype=np.uint8)[None, :]
    array[:, :, 1] = np.arange(512, dtype=np.uint8)[:, None]
    Image.fromarray(array).save(image_path)

    config = replace(
        UnifiedDetectorConfig.from_root(".", ROOT_CONFIG),
        scoring_mode="equivalent_fast",
        artifact_forward_batch_size=5,
        artifact_tile_batch_size=5,
    )
    detector = UnifiedArtifactDetector.__new__(UnifiedArtifactDetector)
    detector.config = config
    detector.device = torch.device("cpu")
    seen_batch_shapes: list[tuple[int, ...]] = []

    def fake_tensor_batch_scores(batch: torch.Tensor) -> np.ndarray:
        seen_batch_shapes.append(tuple(batch.shape))
        start = sum(shape[0] for shape in seen_batch_shapes[:-1])
        return np.arange(start, start + batch.shape[0], dtype=np.float32)

    def fail_gpu_reader(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("equivalent_fast must preserve PIL preprocessing semantics")

    detector._score_artifact_tensor_batch = fake_tensor_batch_scores
    detector._read_rgb_tensor = fail_gpu_reader

    scores, max_sides = detector.score_artifact_tile_paths([image_path])

    assert max_sides.tolist() == [512.0]
    assert scores.tolist() == [8.0]
    assert seen_batch_shapes == [(5, 3, 256, 256), (4, 3, 256, 256)]


def test_whole_artifact_path_scoring_streams_image_chunks(tmp_path) -> None:
    paths = []
    for index in range(5):
        image_path = tmp_path / f"image_{index}.png"
        Image.fromarray(np.full((16, 16, 3), index, dtype=np.uint8)).save(image_path)
        paths.append(image_path)

    config = replace(
        UnifiedDetectorConfig.from_root(".", ROOT_CONFIG),
        artifact_forward_batch_size=2,
    )
    detector = UnifiedArtifactDetector.__new__(UnifiedArtifactDetector)
    detector.config = config
    seen_chunk_sizes: list[int] = []
    start = 0

    def fake_rgb_scores(images, _variants=None) -> np.ndarray:
        nonlocal start
        seen_chunk_sizes.append(len(images))
        values = np.arange(start, start + len(images), dtype=np.float32)
        start += len(images)
        return values

    detector._score_rgb_images = fake_rgb_scores

    scores = detector.score_artifact_whole_paths(paths)

    assert seen_chunk_sizes == [2, 2, 1]
    np.testing.assert_allclose(scores, np.asarray([0, 1, 2, 3, 4], dtype=np.float32))


def test_equivalent_fast_uses_batched_semantic_path() -> None:
    config = replace(UnifiedDetectorConfig.from_root(".", ROOT_CONFIG), scoring_mode="equivalent_fast")
    detector = UnifiedArtifactDetector.__new__(UnifiedArtifactDetector)
    detector.config = config
    detector.semantic_model = object()
    detector.semantic_preprocess = object()
    calls: list[str] = []

    def fake_batched(paths) -> np.ndarray:
        calls.append("batched")
        return np.asarray([0.25 for _ in paths], dtype=np.float32)

    detector._score_semantic_paths_batched = fake_batched

    scores = detector.score_semantic_paths(["a.png", "b.png"])

    assert calls == ["batched"]
    np.testing.assert_allclose(scores, np.asarray([0.25, 0.25], dtype=np.float32))
