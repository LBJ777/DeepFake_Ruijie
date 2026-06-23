from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from PIL import ImageFilter
import io
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    group: str


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    generator: str


def infer_label(path: Path) -> int | None:
    parts = set(path.parts)
    if "0_real" in parts and "1_fake" in parts:
        raise ValueError(f"path contains both labels: {path}")
    if "0_real" in parts:
        return 0
    if "1_fake" in parts:
        return 1
    return None


def collect_labeled_images(root: str | Path) -> list[ImageSample]:
    base = Path(root).expanduser().resolve(strict=False)
    if not base.exists():
        raise FileNotFoundError(f"source root does not exist: {base}")
    samples: list[ImageSample] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label(path)
        if label is None:
            continue
        relative_parts = path.relative_to(base).parts
        group = relative_parts[0] if relative_parts else base.name
        samples.append(ImageSample(path=path, label=int(label), group=group))
    if not samples:
        raise ValueError(f"no labeled images found under: {base}")
    return samples


def select_per_label(samples: list[ImageSample], max_per_label: int, skip_per_label: int = 0) -> list[ImageSample]:
    if max_per_label <= 0:
        return samples
    selected: list[ImageSample] = []
    for label in (0, 1):
        by_group: dict[str, list[ImageSample]] = {}
        for sample in samples:
            if sample.label != label:
                continue
            by_group.setdefault(sample.group, []).append(sample)
        group_items = [by_group[key] for key in sorted(by_group)]
        depth = 0
        label_count = 0
        label_limit = max_per_label + max(0, int(skip_per_label))
        label_selected: list[ImageSample] = []
        while label_count < label_limit:
            added_at_depth = False
            for items in group_items:
                if depth >= len(items):
                    continue
                label_selected.append(items[depth])
                label_count += 1
                added_at_depth = True
                if label_count >= label_limit:
                    break
            if not added_at_depth:
                break
            depth += 1
        selected.extend(label_selected[max(0, int(skip_per_label)):])
    return sorted(selected, key=lambda sample: str(sample.path))


def limit_per_label(samples: list[ImageSample], max_per_label: int) -> list[ImageSample]:
    return select_per_label(samples, max_per_label=max_per_label, skip_per_label=0)


def apply_variant(image: Image.Image, image_size: int, variant: str) -> Image.Image:
    image = image.convert("RGB")
    if variant == "clean":
        return image.resize((int(image_size), int(image_size)), Image.BICUBIC)
    if variant.startswith("jpeg"):
        quality = int(variant.removeprefix("jpeg"))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").resize((int(image_size), int(image_size)), Image.BICUBIC)
    if variant == "resize50":
        small = max(8, int(image_size) // 2)
        return image.resize((small, small), Image.BICUBIC).resize((int(image_size), int(image_size)), Image.BICUBIC)
    if variant == "blur1":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0)).resize((int(image_size), int(image_size)), Image.BICUBIC)
    raise ValueError(f"unsupported source variant: {variant}")


def parse_variant_spec(variant: str) -> tuple[str, tuple[str, ...]]:
    if variant.startswith("expand:"):
        variants = tuple(item.strip() for item in variant.removeprefix("expand:").split(",") if item.strip())
        if not variants:
            raise ValueError("expand variant requires at least one item")
        return "expand", variants
    return "single", (variant,)


def pil_to_tensor(image: Image.Image, image_size: int, variant: str = "clean") -> torch.Tensor:
    resized = apply_variant(image, image_size, variant)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class ArtifactPriorImageDataset(Dataset):
    def __init__(self, samples: list[ImageSample], image_size: int, variant: str = "clean") -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.variant_mode, self.variants = parse_variant_spec(variant)

    def __len__(self) -> int:
        if self.variant_mode == "expand":
            return len(self.samples) * len(self.variants)
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        if self.variant_mode == "expand":
            sample = self.samples[index // len(self.variants)]
            variant = self.variants[index % len(self.variants)]
        else:
            sample = self.samples[index]
            variant = self.variants[0]
        with Image.open(sample.path) as image:
            tensor = pil_to_tensor(image, self.image_size, variant=variant)
        return {
            "image": tensor,
            "label": int(sample.label),
            "path": str(sample.path.resolve(strict=False)),
            "group": sample.group,
            "variant": variant,
        }


def artifact_prior_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": torch.stack([item["image"] for item in batch]),  # type: ignore[list-item]
        "labels": torch.tensor([int(item["label"]) for item in batch], dtype=torch.float32),
        "paths": [str(item["path"]) for item in batch],
        "groups": [str(item["group"]) for item in batch],
    }


def collect_images(root: str | Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for sample in collect_labeled_images(root):
        records.append(ImageRecord(path=sample.path, label=int(sample.label), generator=sample.group))
    return records


def _iter_names_under_label_dirs(root: Path, label_dir: str) -> Iterable[str]:
    for class_entry in os.scandir(root):
        if not class_entry.is_dir(follow_symlinks=False):
            continue
        target = Path(class_entry.path) / label_dir
        if not target.is_dir():
            continue
        for image_entry in os.scandir(target):
            yield image_entry.name


def count_by_label(root: str | Path) -> dict[str, int]:
    counts = {"real": 0, "fake": 0, "total": 0}
    base = Path(root).expanduser().resolve(strict=False)
    if not base.exists():
        raise FileNotFoundError(f"source root does not exist: {base}")
    for name in _iter_names_under_label_dirs(base, "0_real"):
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
            counts["real"] += 1
    for name in _iter_names_under_label_dirs(base, "1_fake"):
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
            counts["fake"] += 1
    counts["total"] = counts["real"] + counts["fake"]
    if counts["total"] == 0:
        raise ValueError(f"no labeled images found under: {base}")
    return counts


def to_unified_samples(records: list[ImageRecord]) -> list[ImageSample]:
    return [ImageSample(path=record.path, label=record.label, group=record.generator) for record in records]
