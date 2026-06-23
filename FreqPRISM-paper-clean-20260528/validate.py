#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from options.base_options import load_config


def main() -> None:
    parser = argparse.ArgumentParser("Record FreqPRISM source-only weight selection grid")
    parser.add_argument("--config", default="configs/apfreq_train100k_full.yaml")
    parser.add_argument("--output_dir", default="results/source_weight_selection")
    args = parser.parse_args()
    config = load_config(args.config)
    raw = config.raw
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "project": raw["project"],
        "selection": raw["selection"],
        "initial_composition": raw["composition"],
        "source_root": str(config.source_root),
        "target_labels_used": False,
        "note": (
            "This entrypoint records the source-only grid for FreqPRISM. "
            "Run score extraction on source validation before changing fixed composition weights."
        ),
    }
    (out / "selection_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
