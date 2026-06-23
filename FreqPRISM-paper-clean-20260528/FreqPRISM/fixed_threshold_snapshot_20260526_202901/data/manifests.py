from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from data.datasets import IMAGE_EXTENSIONS, ImageSample


MANIFEST_FIELDS = ["path", "label", "class_name", "split"]


def read_manifest_path_set(manifest_path: str | Path | None) -> set[Path]:
    if manifest_path is None or str(manifest_path) == "":
        return set()
    paths: set[Path] = set()
    with Path(manifest_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("path") or row.get("filepath") or row.get("image_path")
            if value:
                paths.add(Path(value).resolve(strict=False))
    if not paths:
        raise ValueError(f"no paths loaded from manifest {manifest_path}")
    return paths


def iter_source_images(source_root: str | Path) -> Iterable[Path]:
    root = Path(source_root).resolve(strict=False)
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.parent.name in {"0_real", "1_fake"}:
            yield path


def source_record(path: Path, *, source_root: Path, split: str) -> dict[str, str | int]:
    rel = path.relative_to(source_root)
    label_name = rel.parts[-2]
    if label_name == "0_real":
        label = 0
    elif label_name == "1_fake":
        label = 1
    else:
        raise ValueError(f"cannot infer label from {path}")
    return {
        "path": str(path.resolve(strict=False)),
        "label": int(label),
        "class_name": str(rel.parts[-3] if len(rel.parts) >= 3 else ""),
        "split": str(split),
    }


def write_manifest(path: str | Path, rows: list[dict[str, str | int]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_source_manifests(
    *,
    source_root: str | Path,
    holdout_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    source = Path(source_root).resolve(strict=False)
    holdout_paths = read_manifest_path_set(holdout_manifest)
    train_rows: list[dict[str, str | int]] = []
    holdout_rows: list[dict[str, str | int]] = []
    for path in iter_source_images(source):
        resolved = path.resolve(strict=False)
        if resolved in holdout_paths:
            holdout_rows.append(source_record(resolved, source_root=source, split="holdout"))
        else:
            train_rows.append(source_record(resolved, source_root=source, split="train"))
    out = Path(output_dir)
    write_manifest(out / "source_train_manifest.csv", train_rows)
    write_manifest(out / "source_holdout_manifest.csv", holdout_rows)
    counts = {"train": len(train_rows), "holdout": len(holdout_rows)}
    with (out / "source_split_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "count"])
        writer.writeheader()
        writer.writerow({"split": "train", "count": counts["train"]})
        writer.writerow({"split": "holdout", "count": counts["holdout"]})
    return counts


def filter_image_samples_by_manifest(samples: list[ImageSample], manifest_path: str | Path | None) -> list[ImageSample]:
    if manifest_path is None or str(manifest_path) == "":
        return list(samples)
    include = read_manifest_path_set(manifest_path)
    return [sample for sample in samples if sample.path.resolve(strict=False) in include]
