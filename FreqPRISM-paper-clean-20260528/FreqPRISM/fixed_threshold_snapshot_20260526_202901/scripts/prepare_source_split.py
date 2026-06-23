#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.manifests import prepare_source_manifests


def main() -> None:
    parser = argparse.ArgumentParser("Prepare source train/holdout manifests for unified detector reproduction")
    parser.add_argument("--source_root", default=str(REPO_ROOT / "dataset" / "train" / "progan_train"))
    parser.add_argument("--holdout_manifest", default=str(PROJECT_ROOT / "manifests" / "source_holdout_seed_20260506.csv"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "manifests" / "source_split"))
    args = parser.parse_args()

    counts = prepare_source_manifests(
        source_root=args.source_root,
        holdout_manifest=args.holdout_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
