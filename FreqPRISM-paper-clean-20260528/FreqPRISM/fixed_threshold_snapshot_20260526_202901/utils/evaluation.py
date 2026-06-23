from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from data.datasets import ImageSample


class PathScorer(Protocol):
    def score_paths(self, paths: Sequence[Path], *, residual_batch_size: int) -> np.ndarray:
        ...


def safe_cache_token(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


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
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["labels"].astype(np.int64), cached["scores"].astype(np.float32), True

    labels = np.asarray([int(sample.label) for sample in samples], dtype=np.int64)
    paths = [sample.path for sample in samples]
    scores = detector.score_paths(paths, residual_batch_size=int(residual_batch_size)).astype(np.float32)
    np.savez_compressed(
        cache_path,
        labels=labels,
        scores=scores,
        paths=np.asarray([str(path.resolve(strict=False)) for path in paths]),
    )
    return labels, scores, False
