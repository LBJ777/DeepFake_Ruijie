from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.progress import progress_iter


def normalize_feature_rows(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must be a 2D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, float(eps)), out=np.zeros_like(values), where=norms > 0.0).astype(np.float32)


def parse_clip_variant_spec(spec: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(spec, str):
        text = spec.removeprefix("expand:")
        variants = tuple(item.strip() for item in text.split(",") if item.strip())
    else:
        variants = tuple(str(item).strip() for item in spec if str(item).strip())
    if not variants:
        raise ValueError("at least one CLIP variant is required")
    return variants


def _validate_xy(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("labels must be 1D and match features")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    return x, y


@dataclass
class ClipLinearProbe:
    model: Any
    clip_model_name: str
    feature_dim: int
    normalize_features: bool = True
    train_config: dict[str, Any] = field(default_factory=dict)
    target_labels_used: bool = False

    def prepare_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("features must be 2D")
        if values.shape[1] != int(self.feature_dim):
            raise ValueError(f"expected {self.feature_dim} CLIP features, got {values.shape[1]}")
        if self.normalize_features:
            return normalize_feature_rows(values)
        return values

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        prepared = self.prepare_features(features)
        return np.asarray(self.model.predict_proba(prepared)[:, 1], dtype=np.float32)


def fit_clip_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    clip_model_name: str,
    c: float = 1.0,
    random_state: int = 20260519,
    normalize_features: bool = True,
    train_config: dict[str, Any] | None = None,
) -> ClipLinearProbe:
    x, y = _validate_xy(features, labels)
    prepared = normalize_feature_rows(x) if bool(normalize_features) else x
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=float(c), class_weight="balanced", random_state=int(random_state)),
    )
    model.fit(prepared, y)
    return ClipLinearProbe(
        model=model,
        clip_model_name=str(clip_model_name),
        feature_dim=int(x.shape[1]),
        normalize_features=bool(normalize_features),
        train_config=dict(train_config or {}),
        target_labels_used=False,
    )


def resolve_torch_device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(name))


def load_openai_clip(
    model_name: str,
    *,
    device: str | torch.device,
    download_root: str | Path = "/data/lizihao/.cache/clip",
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    import clip

    torch_device = resolve_torch_device(str(device))
    model, preprocess = clip.load(str(model_name), device=str(torch_device), download_root=str(download_root), jit=False)
    model.eval()
    return model, preprocess


class ClipImageDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Any],
        *,
        image_size: int,
        variants: str | Sequence[str],
        preprocess: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.variants = parse_clip_variant_spec(variants)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples) * len(self.variants)

    def __getitem__(self, index: int) -> dict[str, object]:
        from data.datasets import apply_variant

        sample = self.samples[index // len(self.variants)]
        variant = self.variants[index % len(self.variants)]
        with Image.open(sample.path) as image:
            prepared = apply_variant(image, self.image_size, variant)
            tensor = self.preprocess(prepared)
        return {
            "image": tensor,
            "label": int(sample.label),
            "group": str(sample.group),
            "path": str(sample.path.resolve(strict=False)),
            "variant": variant,
        }


def clip_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": torch.stack([item["image"] for item in batch]),  # type: ignore[list-item]
        "labels": torch.tensor([int(item["label"]) for item in batch], dtype=torch.int64),
        "groups": [str(item["group"]) for item in batch],
        "paths": [str(item["path"]) for item in batch],
        "variants": [str(item["variant"]) for item in batch],
    }


def _load_feature_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = np.load(path, allow_pickle=False)
    return (
        cached["features"].astype(np.float32),
        cached["labels"].astype(np.int64),
        cached["groups"].astype(str),
        cached["paths"].astype(str),
    )


def _save_feature_npz(path: Path, features: np.ndarray, labels: np.ndarray, groups: np.ndarray, paths: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=str),
        paths=np.asarray(paths, dtype=str),
    )
    tmp_path.replace(path)


def extract_clip_features(
    *,
    model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    samples: Sequence[Any],
    image_size: int,
    variants: str | Sequence[str],
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
    show_progress: bool = True,
    progress_desc: str = "CLIP features",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch_device = resolve_torch_device(str(device))
    dataset = ClipImageDataset(samples, image_size=int(image_size), variants=variants, preprocess=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=clip_collate,
    )
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    groups: list[str] = []
    paths: list[str] = []
    with torch.no_grad():
        for batch in progress_iter(
            loader,
            total=len(loader),
            desc=str(progress_desc),
            unit="batch",
            enabled=bool(show_progress),
        ):
            images = batch["images"].to(torch_device)
            encoded = model.encode_image(images).detach().float().cpu().numpy().astype(np.float32)
            feature_chunks.append(encoded)
            label_chunks.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
            groups.extend([str(group) for group in batch["groups"]])
            paths.extend([str(path) for path in batch["paths"]])
    if not feature_chunks:
        raise ValueError("no CLIP features extracted")
    return (
        np.concatenate(feature_chunks, axis=0),
        np.concatenate(label_chunks, axis=0),
        np.asarray(groups),
        np.asarray(paths),
    )


def extract_clip_features_resumable(
    *,
    cache_path: str | Path,
    model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    samples: Sequence[Any],
    image_size: int,
    variants: str | Sequence[str],
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
    cache_chunk_samples: int = 512,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(cache_path)
    if path.exists():
        return _load_feature_npz(path)
    sample_list = list(samples)
    chunk_size = int(cache_chunk_samples)
    if chunk_size <= 0:
        features, labels, groups, paths = extract_clip_features(
            model=model,
            preprocess=preprocess,
            samples=sample_list,
            image_size=int(image_size),
            variants=variants,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            device=device,
            show_progress=bool(show_progress),
            progress_desc=str(progress_desc or f"CLIP {path.stem}"),
        )
        _save_feature_npz(path, features, labels, groups, paths)
        return features, labels, groups, paths

    chunk_dir = path.with_name(f"{path.stem}_chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    starts = list(range(0, len(sample_list), chunk_size))
    outer_desc = str(progress_desc or f"CLIP chunks {path.stem}")
    for part_index, start in progress_iter(
        list(enumerate(starts)),
        total=len(starts),
        desc=outer_desc,
        unit="chunk",
        enabled=bool(show_progress),
    ):
        stop = min(start + chunk_size, len(sample_list))
        part_path = chunk_dir / f"part_{part_index:06d}.npz"
        part_paths.append(part_path)
        if part_path.exists():
            continue
        features, labels, groups, paths = extract_clip_features(
            model=model,
            preprocess=preprocess,
            samples=sample_list[start:stop],
            image_size=int(image_size),
            variants=variants,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            device=device,
            show_progress=bool(show_progress),
            progress_desc=f"{outer_desc} part {part_index + 1}/{len(starts)}",
        )
        _save_feature_npz(part_path, features, labels, groups, paths)
    if not part_paths:
        raise ValueError("no CLIP features extracted")

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []
    path_chunks: list[np.ndarray] = []
    for part_path in part_paths:
        features, labels, groups, paths = _load_feature_npz(part_path)
        feature_chunks.append(features)
        label_chunks.append(labels)
        group_chunks.append(groups)
        path_chunks.append(paths)
    merged_features = np.concatenate(feature_chunks, axis=0)
    merged_labels = np.concatenate(label_chunks, axis=0)
    merged_groups = np.concatenate(group_chunks, axis=0)
    merged_paths = np.concatenate(path_chunks, axis=0)
    _save_feature_npz(path, merged_features, merged_labels, merged_groups, merged_paths)
    return merged_features, merged_labels, merged_groups, merged_paths
