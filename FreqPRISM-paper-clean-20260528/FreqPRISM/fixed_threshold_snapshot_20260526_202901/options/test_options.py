from __future__ import annotations

import argparse


def create_test_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate FreqPRISM on all target generators")
    parser.add_argument("--config", default="configs/apfreq_train100k_full.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/apfreq_full_target")
    parser.add_argument("--per_label", type=int, default=None, help="0 means full generator split")
    parser.add_argument("--residual_batch_size", type=int, default=None)
    parser.add_argument("--score_cache_dir", default="")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser
