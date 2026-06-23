from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .score_blend import logits_to_probabilities, probabilities_to_logits


def _grid_starts(side: int, tile_size: int, grid_size: int) -> list[int]:
    if side <= tile_size:
        return [0]
    max_start = int(side) - int(tile_size)
    values = np.linspace(0.0, float(max_start), num=int(grid_size), dtype=np.float64)
    starts = [int(round(value)) for value in values]
    unique: list[int] = []
    for start in starts:
        clipped = min(max(0, start), max_start)
        if not unique or unique[-1] != clipped:
            unique.append(clipped)
    if unique[0] != 0:
        unique.insert(0, 0)
    if unique[-1] != max_start:
        unique.append(max_start)
    return unique


def native_tile_boxes(width: int, height: int, tile_size: int = 256, grid_size: int = 3) -> list[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if width <= tile_size and height <= tile_size:
        return [(0, 0, int(width), int(height))]

    x_starts = _grid_starts(int(width), int(tile_size), int(grid_size))
    y_starts = _grid_starts(int(height), int(tile_size), int(grid_size))
    boxes: list[tuple[int, int, int, int]] = []
    for top in y_starts:
        for left in x_starts:
            right = min(int(left) + int(tile_size), int(width))
            bottom = min(int(top) + int(tile_size), int(height))
            if right <= left or bottom <= top:
                continue
            boxes.append((int(left), int(top), int(right), int(bottom)))

    unique: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if box not in unique:
            unique.append(box)
    return unique


def _pad_crop_to_square_rgb(image: Image.Image, tile_size: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected RGB image")
    height, width = array.shape[:2]
    pad_bottom = max(0, int(tile_size) - height)
    pad_right = max(0, int(tile_size) - width)
    if pad_bottom == 0 and pad_right == 0:
        return image
    padded = np.pad(array, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    return Image.fromarray(padded[: int(tile_size), : int(tile_size)])


def extract_native_tiles(image: Image.Image, boxes: Sequence[tuple[int, int, int, int]], tile_size: int = 256) -> list[Image.Image]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    rgb = image.convert("RGB")
    tiles: list[Image.Image] = []
    for box in boxes:
        crop = rgb.crop(tuple(int(value) for value in box))
        tiles.append(_pad_crop_to_square_rgb(crop, int(tile_size)))
    return tiles


def aggregate_tile_scores(tile_scores: Iterable[float], tile_mode: str = "top1") -> float:
    values = np.asarray(list(tile_scores), dtype=np.float32)
    if values.size == 0:
        raise ValueError("tile_scores must contain at least one value")
    mode = str(tile_mode)
    if mode == "top1":
        return float(np.max(values))
    if mode == "top2mean":
        if values.size == 1:
            return float(values[0])
        top2 = np.sort(values)[-2:]
        return float(np.mean(top2))
    if mode == "q75":
        return float(np.quantile(values, 0.75))
    raise ValueError(f"unsupported tile_mode: {tile_mode}")


def combine_whole_tile_scores(
    whole_scores: Sequence[float] | np.ndarray,
    tile_scores_by_image: Sequence[Iterable[float]],
    beta: float,
    tile_mode: str = "top1",
) -> np.ndarray:
    whole = np.asarray(whole_scores, dtype=np.float32)
    if whole.ndim != 1:
        raise ValueError("whole_scores must be 1D")
    if len(tile_scores_by_image) != whole.shape[0]:
        raise ValueError("tile_scores_by_image length must match whole_scores")

    combined: list[float] = []
    for score, tile_scores in zip(whole, tile_scores_by_image):
        tile = aggregate_tile_scores(tile_scores, tile_mode=tile_mode)
        whole_logit = float(probabilities_to_logits(np.asarray([score], dtype=np.float32))[0])
        tile_logit = float(probabilities_to_logits(np.asarray([tile], dtype=np.float32))[0])
        blended_logit = whole_logit + float(beta) * max(0.0, tile_logit - whole_logit)
        combined.append(float(logits_to_probabilities(np.asarray([blended_logit], dtype=np.float32))[0]))
    return np.asarray(combined, dtype=np.float32)


def combine_whole_tile_aux_scores(
    whole_scores: Sequence[float] | np.ndarray,
    tile_scores: Sequence[float] | np.ndarray,
    auxiliary_scores: Sequence[float] | np.ndarray,
    beta: float,
    alpha: float,
) -> np.ndarray:
    whole = np.asarray(whole_scores, dtype=np.float32)
    tile = np.asarray(tile_scores, dtype=np.float32)
    auxiliary = np.asarray(auxiliary_scores, dtype=np.float32)
    if whole.ndim != 1 or tile.ndim != 1 or auxiliary.ndim != 1:
        raise ValueError("score arrays must be 1D")
    if whole.shape != tile.shape or whole.shape != auxiliary.shape:
        raise ValueError("whole, tile, and auxiliary scores must have the same shape")

    whole_logit = probabilities_to_logits(whole).astype(np.float64)
    tile_logit = probabilities_to_logits(tile).astype(np.float64)
    aux_logit = probabilities_to_logits(auxiliary).astype(np.float64)
    combined = whole_logit + float(beta) * np.maximum(0.0, tile_logit - whole_logit) + float(alpha) * aux_logit
    return logits_to_probabilities(combined.astype(np.float32))


def combine_whole_tile_aux_conditional_scores(
    whole_scores: Sequence[float] | np.ndarray,
    tile_scores: Sequence[float] | np.ndarray,
    auxiliary_scores: Sequence[float] | np.ndarray,
    *,
    high_res_mask: Sequence[bool] | np.ndarray,
    beta: float,
    alpha_low: float,
    alpha_high: float,
) -> np.ndarray:
    whole = np.asarray(whole_scores, dtype=np.float32)
    tile = np.asarray(tile_scores, dtype=np.float32)
    auxiliary = np.asarray(auxiliary_scores, dtype=np.float32)
    high_res = np.asarray(high_res_mask, dtype=bool)
    if whole.ndim != 1 or tile.ndim != 1 or auxiliary.ndim != 1 or high_res.ndim != 1:
        raise ValueError("score arrays and high_res_mask must be 1D")
    if whole.shape != tile.shape or whole.shape != auxiliary.shape or whole.shape != high_res.shape:
        raise ValueError("whole, tile, auxiliary, and high_res_mask must have the same shape")

    whole_logit = probabilities_to_logits(whole).astype(np.float64)
    tile_logit = probabilities_to_logits(tile).astype(np.float64)
    aux_logit = probabilities_to_logits(auxiliary).astype(np.float64)
    alpha = np.where(high_res, float(alpha_high), float(alpha_low)).astype(np.float64)
    combined = whole_logit + float(beta) * np.maximum(0.0, tile_logit - whole_logit) + alpha * aux_logit
    return logits_to_probabilities(combined.astype(np.float32))


def combine_whole_tile_aux_signed_conditional_scores(
    whole_scores: Sequence[float] | np.ndarray,
    tile_scores: Sequence[float] | np.ndarray,
    auxiliary_scores: Sequence[float] | np.ndarray,
    *,
    high_res_mask: Sequence[bool] | np.ndarray,
    beta: float,
    alpha_low_pos: float,
    alpha_low_neg: float,
    alpha_high_pos: float,
    alpha_high_neg: float,
) -> np.ndarray:
    whole = np.asarray(whole_scores, dtype=np.float32)
    tile = np.asarray(tile_scores, dtype=np.float32)
    auxiliary = np.asarray(auxiliary_scores, dtype=np.float32)
    high_res = np.asarray(high_res_mask, dtype=bool)
    if whole.ndim != 1 or tile.ndim != 1 or auxiliary.ndim != 1 or high_res.ndim != 1:
        raise ValueError("score arrays and high_res_mask must be 1D")
    if whole.shape != tile.shape or whole.shape != auxiliary.shape or whole.shape != high_res.shape:
        raise ValueError("whole, tile, auxiliary, and high_res_mask must have the same shape")

    whole_logit = probabilities_to_logits(whole).astype(np.float64)
    tile_logit = probabilities_to_logits(tile).astype(np.float64)
    aux_logit = probabilities_to_logits(auxiliary).astype(np.float64)
    pos_alpha = np.where(high_res, float(alpha_high_pos), float(alpha_low_pos)).astype(np.float64)
    neg_alpha = np.where(high_res, float(alpha_high_neg), float(alpha_low_neg)).astype(np.float64)
    aux_term = pos_alpha * np.maximum(0.0, aux_logit) + neg_alpha * np.minimum(0.0, aux_logit)
    combined = whole_logit + float(beta) * np.maximum(0.0, tile_logit - whole_logit) + aux_term
    return logits_to_probabilities(combined.astype(np.float32))


def combine_whole_tile_aux_signed_delta_guard_scores(
    whole_scores: Sequence[float] | np.ndarray,
    tile_scores: Sequence[float] | np.ndarray,
    auxiliary_scores: Sequence[float] | np.ndarray,
    *,
    high_res_mask: Sequence[bool] | np.ndarray,
    beta: float,
    alpha_low_pos: float,
    alpha_low_neg: float,
    alpha_high_pos: float,
    alpha_high_neg: float,
    alpha_high_neg_guard: float,
    tile_delta_threshold: float,
) -> np.ndarray:
    whole = np.asarray(whole_scores, dtype=np.float32)
    tile = np.asarray(tile_scores, dtype=np.float32)
    auxiliary = np.asarray(auxiliary_scores, dtype=np.float32)
    high_res = np.asarray(high_res_mask, dtype=bool)
    if whole.ndim != 1 or tile.ndim != 1 or auxiliary.ndim != 1 or high_res.ndim != 1:
        raise ValueError("score arrays and high_res_mask must be 1D")
    if whole.shape != tile.shape or whole.shape != auxiliary.shape or whole.shape != high_res.shape:
        raise ValueError("whole, tile, auxiliary, and high_res_mask must have the same shape")

    whole_logit = probabilities_to_logits(whole).astype(np.float64)
    tile_logit = probabilities_to_logits(tile).astype(np.float64)
    aux_logit = probabilities_to_logits(auxiliary).astype(np.float64)
    tile_delta = np.maximum(0.0, tile_logit - whole_logit)
    relax_negative = high_res & (tile_delta > float(tile_delta_threshold))
    pos_alpha = np.where(high_res, float(alpha_high_pos), float(alpha_low_pos)).astype(np.float64)
    neg_alpha = np.where(
        relax_negative,
        float(alpha_high_neg),
        np.where(high_res, float(alpha_high_neg_guard), float(alpha_low_neg)),
    ).astype(np.float64)
    aux_term = pos_alpha * np.maximum(0.0, aux_logit) + neg_alpha * np.minimum(0.0, aux_logit)
    combined = whole_logit + float(beta) * tile_delta + aux_term
    return logits_to_probabilities(combined.astype(np.float32))
