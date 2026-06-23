from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml

from data.datasets import ImageSample
from scripts.train_detector import load_source_training_samples

ROOT = Path(__file__).resolve().parents[1]


def test_load_source_training_samples_uses_manifest_without_scanning_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "GenImage"
    image_path = source_root / "stable_diffusion_v_1_4" / "imagenet_ai_0419_sdv4" / "train" / "ai" / "a.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("placeholder")
    manifest = source_root / "sd14_train_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "path": str(image_path),
                "label": 1,
                "class_name": "stable_diffusion_v_1_4",
                "split": "train",
            }
        )

    samples = load_source_training_samples(source_root=source_root, train_manifest=manifest)

    assert samples == [
        ImageSample(path=image_path.resolve(strict=False), label=1, group="stable_diffusion_v_1_4")
    ]


def _write_manifest_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = yaml.safe_load((ROOT / "configs" / "apfreq_train100k_full.yaml").read_text())
    train_manifest = tmp_path / "sd14_train_manifest.csv"
    target_manifest = tmp_path / "val_manifest.csv"
    train_manifest.write_text("path,label,class_name,split\n")
    target_manifest.write_text("path,label,class_name,split\n")
    config["dataset"]["source_train_root"] = "dataset/GenImage"
    config["dataset"]["source_train_manifest"] = str(train_manifest)
    config["dataset"]["target_test_root"] = "dataset/GenImage"
    config["dataset"]["target_test_manifest"] = str(target_manifest)
    config_path = tmp_path / "genimage.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, train_manifest, target_manifest


def test_train_wrapper_passes_configured_source_manifest_in_dry_run(tmp_path: Path) -> None:
    config_path, train_manifest, _ = _write_manifest_config(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--config",
            str(config_path),
            "--stage",
            "artifact",
            "--dry_run",
            "--output_dir",
            str(tmp_path / "checkpoints"),
            "--device",
            "cpu",
        ],
        cwd=str(ROOT),
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--train_manifest" in completed.stdout
    assert str(train_manifest.resolve(strict=False)) in completed.stdout


def test_test_wrapper_passes_configured_target_manifest_in_dry_run(tmp_path: Path) -> None:
    config_path, _, target_manifest = _write_manifest_config(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "test.py"),
            "--config",
            str(config_path),
            "--dry_run",
            "--output_dir",
            str(tmp_path / "results"),
            "--device",
            "cpu",
        ],
        cwd=str(ROOT),
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--manifest" in completed.stdout
    assert str(target_manifest.resolve(strict=False)) in completed.stdout
