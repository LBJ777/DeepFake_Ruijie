#!/usr/bin/env bash
set -euo pipefail

cd /data/lizihao/FreqPRISM-main

WAIT_PID="${WAIT_PID:-}"
DEVICE="${DEVICE:-cuda:3}"
PYTHON_BIN="${PYTHON_BIN:-/data/lizihao/.conda/envs/aigc/bin/python}"
MAIN_WEIGHTS_JSON="results/main/main_fusion_parameters/folded_weights.json"
ANCHOR_WEIGHTS_JSON="results/main/source_weight_calibration/selection_protocol.json"
CONFIG="configs/apfreq_train100k_full.yaml"
ANCHOR_CONFIG="configs/apfreq_train100k_source_gamma_anchor.yaml"
ROOT_OUT="results/experiments/phase3_external_benchmarks"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    log "waiting for artifact-family PID ${WAIT_PID} before UniversalFakeDetect strong-candidate benchmark"
    sleep 60
  done
fi

log "exporting UniversalFakeDetect components"
"${PYTHON_BIN}" -u scripts/export_component_scores.py \
  --config "${CONFIG}" \
  --target_root "dataset/UniversalFakeDetect official benchmark" \
  --output_dir "${ROOT_OUT}/universalfakedetect_strong_multimetric_components" \
  --device "${DEVICE}" \
  --residual_batch_size 64 \
  --no_progress

log "evaluating UniversalFakeDetect promoted main"
"${PYTHON_BIN}" -u scripts/evaluate_component_weights.py \
  --component_dir "${ROOT_OUT}/universalfakedetect_strong_multimetric_components" \
  --output_dir "${ROOT_OUT}/universalfakedetect_strong_multimetric_candidate" \
  --config "${CONFIG}" \
  --policy learned_weights \
  --weights_json "${MAIN_WEIGHTS_JSON}"

log "evaluating UniversalFakeDetect source-gamma anchor"
"${PYTHON_BIN}" -u scripts/evaluate_component_weights.py \
  --component_dir "${ROOT_OUT}/universalfakedetect_strong_multimetric_components" \
  --output_dir "${ROOT_OUT}/universalfakedetect_main_anchor_from_components" \
  --config "${ANCHOR_CONFIG}" \
  --policy learned_weights \
  --weights_json "${ANCHOR_WEIGHTS_JSON}"

log "UniversalFakeDetect promoted-main benchmark finished"
