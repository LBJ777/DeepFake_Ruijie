from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from PIL import Image

from data.datasets import pil_to_tensor
from networks.native_tiles import aggregate_tile_scores, extract_native_tiles, native_tile_boxes


class ArtifactTileScorer(Protocol):
    config: object

    def _score_artifact_tensor_batch(self, tensors: torch.Tensor) -> np.ndarray:
        ...


def _resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    current_max = max(int(width), int(height))
    if current_max <= int(max_side):
        return rgb.copy()
    scale = float(max_side) / float(current_max)
    new_width = max(1, int(round(float(width) * scale)))
    new_height = max(1, int(round(float(height) * scale)))
    return rgb.resize((new_width, new_height), Image.BICUBIC)


def _center_box(width: int, height: int, tile_size: int) -> tuple[int, int, int, int]:
    crop_width = min(int(width), int(tile_size))
    crop_height = min(int(height), int(tile_size))
    left = max(0, (int(width) - crop_width) // 2)
    top = max(0, (int(height) - crop_height) // 2)
    return int(left), int(top), int(left + crop_width), int(top + crop_height)


def _tile_images(image: Image.Image, *, tile_size: int, grid_size: int) -> list[Image.Image]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    boxes = native_tile_boxes(width, height, tile_size=tile_size, grid_size=grid_size)
    return extract_native_tiles(rgb, boxes, tile_size=tile_size)


def _center_tile_image(image: Image.Image, *, tile_size: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    return extract_native_tiles(rgb, [_center_box(width, height, tile_size)], tile_size=tile_size)[0]


def score_tile_resolution_variants(
    paths: Sequence[str | Path],
    *,
    detector: ArtifactTileScorer,
    resized_max_side: int = 512,
) -> dict[str, np.ndarray]:
    config = detector.config
    tile_size = int(getattr(config, "tile_size"))
    grid_size = int(getattr(config, "tile_grid_size"))
    artifact_image_size = int(getattr(config, "artifact_image_size"))
    batch_size = max(1, int(getattr(config, "artifact_tile_batch_size", getattr(config, "artifact_forward_batch_size", 64))))
    variant_tile_mode = {
        "RZ2_resized512_tile": "top1",
        "RZ3_center_crop_tile": "top1",
        "RZ4_tile_mean_aggregation": "mean",
    }
    pending_tensors: list[torch.Tensor] = []
    pending_items: list[tuple[str, int]] = []
    scores_by_variant: dict[str, list[list[float]]] = {
        variant: [[] for _ in paths] for variant in variant_tile_mode
    }

    def flush_pending() -> None:
        if not pending_tensors:
            return
        batch_scores = detector._score_artifact_tensor_batch(torch.stack(pending_tensors))
        for (variant, image_index), score in zip(pending_items, batch_scores):
            scores_by_variant[variant][int(image_index)].append(float(score))
        pending_tensors.clear()
        pending_items.clear()

    def add_tile(variant: str, image_index: int, tile: Image.Image) -> None:
        pending_tensors.append(pil_to_tensor(tile, artifact_image_size, "clean"))
        pending_items.append((variant, int(image_index)))
        if len(pending_tensors) >= batch_size:
            flush_pending()

    for image_index, path in enumerate(paths):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            resized = _resize_max_side(rgb, int(resized_max_side))
            for tile in _tile_images(resized, tile_size=tile_size, grid_size=grid_size):
                add_tile("RZ2_resized512_tile", image_index, tile)
            add_tile("RZ3_center_crop_tile", image_index, _center_tile_image(rgb, tile_size=tile_size))
            for tile in _tile_images(rgb, tile_size=tile_size, grid_size=grid_size):
                add_tile("RZ4_tile_mean_aggregation", image_index, tile)
    flush_pending()

    outputs: dict[str, np.ndarray] = {}
    for variant, tile_mode in variant_tile_mode.items():
        outputs[variant] = np.asarray(
            [aggregate_tile_scores(image_scores, tile_mode=tile_mode) for image_scores in scores_by_variant[variant]],
            dtype=np.float32,
        )
    return outputs
