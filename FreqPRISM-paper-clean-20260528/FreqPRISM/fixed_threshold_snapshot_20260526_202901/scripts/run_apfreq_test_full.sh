#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python test.py \
  --config configs/apfreq_train100k_full.yaml \
  --device "${1:-cuda:0}"
