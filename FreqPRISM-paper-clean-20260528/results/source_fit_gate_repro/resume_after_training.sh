#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/lizihao/FreqPRISM-main"
PY="/data/lizihao/.conda/envs/aigc/bin/python"
OUT="$ROOT/results/source_fit_gate_repro"
SPLIT="$OUT/source_split"
LOGS="$OUT/logs"
SOURCE_ROOT="$ROOT/dataset/train_100k/progan_train"
GATE_MANIFEST="$SPLIT/source_gate_manifest.csv"
ARTIFACT_OUT="$OUT/checkpoints/artifact_prior"
SEMANTIC_OUT="$OUT/checkpoints/semantic_prior"
RESIDUAL_OUT="$OUT/checkpoints/residual_prior"
SOURCE_GATE_COMPONENTS="$OUT/component_cache/source_gate_components"
CURRENT17_COMPONENTS="$OUT/component_cache/current17_components"
STRESS_OUT="$OUT/source_stress_calibration"
CURRENT17_REPORT="$OUT/current17_report"

mkdir -p "$LOGS" "$OUT/component_cache"
cd "$ROOT"

echo "[strict-repro-resume] start $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/export_component_scores.py \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  --target_root "$SOURCE_ROOT" \
  --manifest "$GATE_MANIFEST" \
  --output_dir "$SOURCE_GATE_COMPONENTS" \
  --device cuda:1 \
  --artifact_model "$ARTIFACT_OUT/artifact_prior_models.joblib" \
  --semantic_probe "$SEMANTIC_OUT/semantic_probe.joblib" \
  --residual_checkpoint "$RESIDUAL_OUT/checkpoint-1.pth" \
  --artifact_forward_batch_size 8 \
  --artifact_tile_batch_size 8 \
  --semantic_forward_batch_size 32 \
  --residual_batch_size 64 \
  > "$LOGS/source_gate_components_resume.log" 2>&1
echo "[strict-repro-resume] source_gate components done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/run_source_stress_calibration.py \
  --source_component_dir "$SOURCE_GATE_COMPONENTS" \
  --output_dir "$STRESS_OUT" \
  --selection_protocol_out "$STRESS_OUT/selection_protocol.json" \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  > "$LOGS/source_stress_calibration.log" 2>&1
echo "[strict-repro-resume] source stress calibration done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/export_component_scores.py \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  --target_root "$ROOT/dataset/AIGCDetectBenchmark_test" \
  --output_dir "$CURRENT17_COMPONENTS" \
  --device cuda:1 \
  --artifact_model "$ARTIFACT_OUT/artifact_prior_models.joblib" \
  --semantic_probe "$SEMANTIC_OUT/semantic_probe.joblib" \
  --residual_checkpoint "$RESIDUAL_OUT/checkpoint-1.pth" \
  --artifact_forward_batch_size 8 \
  --artifact_tile_batch_size 8 \
  --semantic_forward_batch_size 32 \
  --residual_batch_size 64 \
  > "$LOGS/current17_components.log" 2>&1
echo "[strict-repro-resume] current17 components done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/evaluate_component_weights.py \
  --component_dir "$CURRENT17_COMPONENTS" \
  --output_dir "$CURRENT17_REPORT" \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  --policy learned_weights \
  --weights_json "$STRESS_OUT/selection_protocol.json" \
  > "$LOGS/current17_report.log" 2>&1
echo "[strict-repro-resume] current17 report done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

echo "[strict-repro-resume] complete $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
