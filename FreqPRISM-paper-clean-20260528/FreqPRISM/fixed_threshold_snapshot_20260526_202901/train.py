#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from options.base_options import list_to_variant_spec, load_config
from options.train_options import create_train_parser


def _run(argv: list[str], *, dry_run: bool = False) -> None:
    print("$ " + " ".join(str(item) for item in argv), flush=True)
    if dry_run:
        return
    completed = subprocess.run(argv, cwd=str(PROJECT_ROOT.parent))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _with_progress_flag(argv: list[str], no_progress: bool) -> list[str]:
    if no_progress:
        return [*argv, "--no_progress"]
    return argv


def main() -> None:
    args = create_train_parser().parse_args()
    config = load_config(args.config)
    raw = config.raw
    runtime = raw.get("runtime", {})
    stages = ("artifact", "semantic", "residual") if args.stage == "all" else (args.stage,)
    checkpoint_root = PROJECT_ROOT / args.output_dir
    python = sys.executable
    train_script = PROJECT_ROOT / "scripts" / "train_detector.py"
    num_workers = int(args.num_workers if args.num_workers is not None else runtime.get("num_workers", 4))

    if "artifact" in stages:
        artifact = raw["artifact_prior"]
        _run(
            _with_progress_flag(
                [
                    python,
                    str(train_script),
                    "--stage",
                    "artifact",
                    "--source_root",
                    str(config.source_root),
                    "--output_dir",
                    str(checkpoint_root / "artifact_prior"),
                    "--device",
                    str(args.device),
                    "--batch_size",
                    str(args.artifact_batch_size or runtime.get("artifact_batch_size", 64)),
                    "--num_workers",
                    str(num_workers),
                    "--train_per_label",
                    str(int(artifact["train_per_label"])),
                    "--train_variant",
                    list_to_variant_spec(list(artifact["train_variants"])),
                    "--codec_max_iter",
                    str(int(artifact["codec_max_iter"])),
                    "--artifact_alpha",
                    str(float(artifact["chroma_alpha"])),
                ],
                bool(args.no_progress),
            ),
            dry_run=bool(args.dry_run),
        )

    if "semantic" in stages:
        semantic = raw["semantic_prior"]
        _run(
            _with_progress_flag(
                [
                    python,
                    str(train_script),
                    "--stage",
                    "semantic",
                    "--source_root",
                    str(config.source_root),
                    "--target_root",
                    str(config.target_root),
                    "--output_dir",
                    str(checkpoint_root / "semantic_prior"),
                    "--device",
                    str(args.device),
                    "--batch_size",
                    str(args.semantic_batch_size or runtime.get("semantic_batch_size", 32)),
                    "--num_workers",
                    str(num_workers),
                    "--train_per_label",
                    str(int(semantic["train_per_label"])),
                    "--holdout_per_label",
                    str(int(semantic["holdout_per_label"])),
                    "--clip_model",
                    str(semantic["model_name"]),
                    "--clip_download_root",
                    str(semantic["download_root"]),
                    "--semantic_train_variants",
                    ",".join(str(item) for item in semantic["train_variants"]),
                    "--semantic_holdout_variants",
                    ",".join(str(item) for item in semantic["holdout_variants"]),
                    "--semantic_eval_variants",
                    ",".join(str(item) for item in semantic["eval_variants"]),
                    "--linear_c",
                    str(float(semantic["linear_c"])),
                    "--skip_target_report",
                ],
                bool(args.no_progress),
            ),
            dry_run=bool(args.dry_run),
        )

    if "residual" in stages:
        residual = raw["residual_prior"]
        _run(
            _with_progress_flag(
                [
                    python,
                    str(train_script),
                    "--stage",
                    "residual",
                    "--source_root",
                    str(config.source_root),
                    "--output_dir",
                    str(checkpoint_root / "residual_prior"),
                    "--device",
                    str(args.device),
                    "--batch_size",
                    str(args.residual_batch_size or runtime.get("residual_batch_size", 32)),
                    "--num_workers",
                    str(num_workers),
                    "--epochs",
                    str(int(args.residual_epochs or residual["epochs"])),
                    "--residual_train_image_size",
                    str(int(residual["train_image_size"])),
                    "--residual_lr",
                    str(float(residual["lr"])),
                    "--weight_decay",
                    str(float(residual["weight_decay"])),
                    "--max_samples_per_label",
                    str(int(residual["max_samples_per_label"])),
                    "--random_state",
                    str(int(residual["random_state"])),
                ],
                bool(args.no_progress),
            ),
            dry_run=bool(args.dry_run),
        )


if __name__ == "__main__":
    main()
