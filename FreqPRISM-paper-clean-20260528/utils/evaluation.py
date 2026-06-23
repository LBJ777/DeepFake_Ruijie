from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from data.datasets import ImageSample
from utils.component_scores import COMPONENT_SCORE_KEYS, validate_component_scores


class PathScorer(Protocol):
    def score_paths(self, paths: Sequence[Path], *, residual_batch_size: int) -> np.ndarray:
        ...


class PathComponentScorer(Protocol):
    def score_component_paths(self, paths: Sequence[Path], *, residual_batch_size: int) -> dict[str, np.ndarray]:
        ...


def safe_cache_token(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _detector_cache_fingerprint(detector: object) -> str | None:
    fingerprint = getattr(detector, "cache_fingerprint", None)
    if not callable(fingerprint):
        return None
    value = fingerprint()
    return str(value) if value else None


def _cached_fingerprint(cached: np.lib.npyio.NpzFile) -> str | None:
    if "cache_fingerprint" not in cached.files:
        return None
    value = cached["cache_fingerprint"]
    if value.shape == ():
        return str(value.item())
    return str(value.tolist())


def _cache_matches(cached: np.lib.npyio.NpzFile, expected_fingerprint: str | None) -> bool:
    if expected_fingerprint is None:
        return True
    return _cached_fingerprint(cached) == str(expected_fingerprint)


def _cache_metadata(expected_fingerprint: str | None) -> dict[str, np.ndarray]:
    if expected_fingerprint is None:
        return {}
    return {"cache_fingerprint": np.asarray(str(expected_fingerprint))}


def _load_component_cache(
    cache_path: Path,
    *,
    expected_fingerprint: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool] | None:
    cached = np.load(cache_path, allow_pickle=False)
    if not _cache_matches(cached, expected_fingerprint):
        return None
    components = {
        key: cached[key].astype(np.float32)
        for key in COMPONENT_SCORE_KEYS
        if key in cached.files
    }
    validate_component_scores(components, require_final_fixed=True)
    return cached["labels"].astype(np.int64), components, True


def score_group_with_cache(
    group: str,
    samples: Sequence[ImageSample],
    *,
    detector: PathScorer,
    cache_dir: str | Path,
    residual_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{safe_cache_token(group)}.npz"
    expected_fingerprint = _detector_cache_fingerprint(detector)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if _cache_matches(cached, expected_fingerprint):
            return cached["labels"].astype(np.int64), cached["scores"].astype(np.float32), True

    labels = np.asarray([int(sample.label) for sample in samples], dtype=np.int64)
    paths = [sample.path for sample in samples]
    scores = detector.score_paths(paths, residual_batch_size=int(residual_batch_size)).astype(np.float32)
    np.savez_compressed(
        cache_path,
        labels=labels,
        scores=scores,
        paths=np.asarray([str(path.resolve(strict=False)) for path in paths]),
        **_cache_metadata(expected_fingerprint),
    )
    return labels, scores, False


def score_component_group_with_cache(
    group: str,
    samples: Sequence[ImageSample],
    *,
    detector: PathComponentScorer,
    cache_dir: str | Path,
    residual_batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{safe_cache_token(group)}.npz"
    expected_fingerprint = _detector_cache_fingerprint(detector)
    if cache_path.exists():
        cached = _load_component_cache(cache_path, expected_fingerprint=expected_fingerprint)
        if cached is not None:
            return cached

    labels = np.asarray([int(sample.label) for sample in samples], dtype=np.int64)
    paths = [sample.path for sample in samples]
    components = detector.score_component_paths(paths, residual_batch_size=int(residual_batch_size))
    validate_component_scores(components, require_final_fixed=True)
    np.savez_compressed(
        cache_path,
        labels=labels,
        paths=np.asarray([str(path.resolve(strict=False)) for path in paths]),
        groups=np.asarray([str(sample.group) for sample in samples]),
        **{key: np.asarray(components[key], dtype=np.float32) for key in COMPONENT_SCORE_KEYS},
        **_cache_metadata(expected_fingerprint),
    )
    return labels, {key: np.asarray(components[key], dtype=np.float32) for key in COMPONENT_SCORE_KEYS}, False
