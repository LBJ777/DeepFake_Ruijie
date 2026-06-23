from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from PIL import Image

from networks.detector import UnifiedDetectorConfig
from utils.tile_resolution_rescore import score_tile_resolution_variants


def test_score_tile_resolution_variants_streams_expected_tile_variants(tmp_path) -> None:
    image_path = tmp_path / "large.png"
    array = np.zeros((512, 512, 3), dtype=np.uint8)
    array[128:384, 128:384, :] = 255
    Image.fromarray(array).save(image_path)

    config = replace(
        UnifiedDetectorConfig.from_root(".", "apfreq_train100k_full.yaml"),
        artifact_tile_batch_size=4,
        artifact_forward_batch_size=4,
    )

    class FakeDetector:
        def __init__(self) -> None:
            self.config = config
            self.batch_sizes: list[int] = []

        def _score_artifact_tensor_batch(self, batch: torch.Tensor) -> np.ndarray:
            self.batch_sizes.append(int(batch.shape[0]))
            return batch.float().mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)

    detector = FakeDetector()
    scores = score_tile_resolution_variants([image_path], detector=detector, resized_max_side=512)

    assert set(scores) == {"RZ2_resized512_tile", "RZ3_center_crop_tile", "RZ4_tile_mean_aggregation"}
    for values in scores.values():
        assert values.shape == (1,)
    assert all(size <= 4 for size in detector.batch_sizes)
    assert scores["RZ3_center_crop_tile"][0] > scores["RZ4_tile_mean_aggregation"][0]
