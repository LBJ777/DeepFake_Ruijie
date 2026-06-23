#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py \
  --config configs/apfreq_train100k_full.yaml \
  --stage all \
  --device "${1:-cuda:0}"
