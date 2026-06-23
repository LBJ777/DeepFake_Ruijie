#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from options.base_options import load_config
from options.test_options import create_test_parser


def main() -> None:
    args = create_test_parser().parse_args()
    config = load_config(args.config)
    runtime = config.raw.get("runtime", {})
    config_name = Path(args.config).name
    per_label = config.per_label if args.per_label is None else int(args.per_label)
    residual_batch_size = int(args.residual_batch_size or runtime.get("residual_eval_batch_size", 64))
    argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_target.py"),
        "--target_root",
        str(config.target_root),
        "--output_dir",
        str(PROJECT_ROOT / args.output_dir),
        "--device",
        str(args.device),
        "--config_name",
        config_name,
        "--per_label",
        str(per_label),
        "--residual_batch_size",
        str(residual_batch_size),
    ]
    if args.score_cache_dir:
        argv.extend(["--score_cache_dir", str(args.score_cache_dir)])
    if args.no_progress:
        argv.append("--no_progress")
    print("$ " + " ".join(argv), flush=True)
    if args.dry_run:
        return
    completed = subprocess.run(argv, cwd=str(PROJECT_ROOT.parent))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
