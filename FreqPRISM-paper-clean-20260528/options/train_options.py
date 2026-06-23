from __future__ import annotations

import argparse


def create_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train FreqPRISM source-only components")
    parser.add_argument("--config", default="configs/apfreq_train100k_full.yaml")
    parser.add_argument("--stage", choices=("artifact", "semantic", "residual", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--artifact_batch_size", type=int, default=None)
    parser.add_argument("--semantic_batch_size", type=int, default=None)
    parser.add_argument("--residual_batch_size", type=int, default=None)
    parser.add_argument("--residual_epochs", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser
