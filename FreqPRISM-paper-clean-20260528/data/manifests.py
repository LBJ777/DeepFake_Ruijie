from __future__ import annotations

import csv
import hashlib
import json
import random
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


def load_image_samples_from_manifest(manifest_path: str | Path) -> list[ImageSample]:
    samples: list[ImageSample] = []
    with Path(manifest_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("path") or row.get("filepath") or row.get("image_path")
            if not value:
                continue
            label_value = row.get("label")
            if label_value is None or label_value == "":
                label = 1 if row.get("label_name") == "1_fake" else 0
            else:
                label = int(label_value)
            group = row.get("class_name") or row.get("generator") or row.get("group") or Path(value).parent.parent.name
            samples.append(
                ImageSample(
                    path=Path(value).expanduser().resolve(strict=False),
                    label=int(label),
                    group=str(group),
                )
            )
    if not samples:
        raise ValueError(f"no samples loaded from manifest {manifest_path}")
    return samples


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


def _stable_shuffle(paths: list[Path], *, seed: int, key: tuple[str, int]) -> list[Path]:
    digest = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = list(paths)
    rng.shuffle(shuffled)
    return shuffled


def prepare_source_gate_split(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    gate_fraction: float = 0.20,
    seed: int = 100,
) -> dict[str, int]:
    if not 0.0 < float(gate_fraction) < 1.0:
        raise ValueError("gate_fraction must be between 0 and 1")
    source = Path(source_root).resolve(strict=False)
    by_key: dict[tuple[str, int], list[Path]] = {}
    for path in iter_source_images(source):
        record = source_record(path, source_root=source, split="source")
        by_key.setdefault((str(record["class_name"]), int(record["label"])), []).append(path)
    if not by_key:
        raise ValueError(f"no source images found under: {source}")

    fit_rows: list[dict[str, str | int]] = []
    gate_rows: list[dict[str, str | int]] = []
    split_protocol: list[dict[str, str | int]] = []
    for key in sorted(by_key):
        paths = _stable_shuffle(sorted(by_key[key]), seed=int(seed), key=key)
        gate_count = int(round(len(paths) * float(gate_fraction)))
        if len(paths) > 1:
            gate_count = min(max(1, gate_count), len(paths) - 1)
        else:
            gate_count = 0
        gate_paths = set(paths[:gate_count])
        for path in sorted(paths, key=lambda item: item.as_posix()):
            split = "source_gate" if path in gate_paths else "source_fit"
            row = source_record(path, source_root=source, split=split)
            if split == "source_gate":
                gate_rows.append(row)
            else:
                fit_rows.append(row)
        split_protocol.append(
            {
                "class_name": key[0],
                "label": int(key[1]),
                "total": int(len(paths)),
                "source_gate": int(gate_count),
                "source_fit": int(len(paths) - gate_count),
            }
        )

    out = Path(output_dir)
    write_manifest(out / "source_fit_manifest.csv", fit_rows)
    write_manifest(out / "source_gate_manifest.csv", gate_rows)
    counts = {"source_fit": len(fit_rows), "source_gate": len(gate_rows)}
    with (out / "source_split_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "count"])
        writer.writeheader()
        writer.writerow({"split": "source_fit", "count": counts["source_fit"]})
        writer.writerow({"split": "source_gate", "count": counts["source_gate"]})
    (out / "source_gate_split_protocol.json").write_text(
        json.dumps(
            {
                "source_root": str(source),
                "gate_fraction": float(gate_fraction),
                "seed": int(seed),
                "target_labels_used": False,
                "by_class_label": split_protocol,
                "counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return counts


def filter_image_samples_by_manifest(samples: list[ImageSample], manifest_path: str | Path | None) -> list[ImageSample]:
    if manifest_path is None or str(manifest_path) == "":
        return list(samples)
    include = read_manifest_path_set(manifest_path)
    return [sample for sample in samples if sample.path.resolve(strict=False) in include]
