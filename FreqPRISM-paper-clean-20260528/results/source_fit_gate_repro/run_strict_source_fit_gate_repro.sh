#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/lizihao/FreqPRISM-main"
PY="/data/lizihao/.conda/envs/aigc/bin/python"
OUT="$ROOT/results/source_fit_gate_repro"
SPLIT="$OUT/source_split"
LOGS="$OUT/logs"
SOURCE_ROOT="$ROOT/dataset/train_100k/progan_train"
FIT_MANIFEST="$SPLIT/source_fit_manifest.csv"
GATE_MANIFEST="$SPLIT/source_gate_manifest.csv"
ARTIFACT_OUT="$OUT/checkpoints/artifact_prior"
SEMANTIC_OUT="$OUT/checkpoints/semantic_prior"
RESIDUAL_OUT="$OUT/checkpoints/residual_prior"
SOURCE_GATE_COMPONENTS="$OUT/component_cache/source_gate_components"
CURRENT17_COMPONENTS="$OUT/component_cache/current17_components"
STRESS_OUT="$OUT/source_stress_calibration"
CURRENT17_REPORT="$OUT/current17_report"

mkdir -p "$LOGS" "$OUT/checkpoints" "$OUT/component_cache"

cd "$ROOT"

echo "[strict-repro] start $(date --iso-8601=seconds)" | tee "$LOGS/pipeline.log"

"$PY" scripts/train_detector.py \
  --stage artifact \
  --source_root "$SOURCE_ROOT" \
  --train_manifest "$FIT_MANIFEST" \
  --output_dir "$ARTIFACT_OUT" \
  --device cuda:1 \
  --image_size 256 \
  --batch_size 64 \
  --num_workers 4 \
  --train_per_label 0 \
  --train_variant "expand:clean,jpeg50,jpeg50,resize50,blur1" \
  --codec_max_iter 200 \
  --artifact_alpha -0.40 \
  > "$LOGS/artifact_train.log" 2>&1 &
ARTIFACT_PID=$!
echo "[strict-repro] artifact pid=$ARTIFACT_PID" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/train_detector.py \
  --stage semantic \
  --source_root "$SOURCE_ROOT" \
  --train_manifest "$FIT_MANIFEST" \
  --holdout_manifest "$GATE_MANIFEST" \
  --output_dir "$SEMANTIC_OUT" \
  --device cuda:3 \
  --image_size 256 \
  --batch_size 64 \
  --num_workers 4 \
  --train_per_label 0 \
  --holdout_per_label 0 \
  --clip_model "ViT-L/14" \
  --clip_download_root "/data/lizihao/.cache/clip" \
  --semantic_train_variants clean \
  --semantic_holdout_variants clean \
  --semantic_eval_variants clean \
  --tta_aggregation mean_logit \
  --linear_c 1.0 \
  --skip_target_report \
  > "$LOGS/semantic_train.log" 2>&1 &
SEMANTIC_PID=$!
echo "[strict-repro] semantic pid=$SEMANTIC_PID" | tee -a "$LOGS/pipeline.log"

wait "$ARTIFACT_PID"
echo "[strict-repro] artifact done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/train_detector.py \
  --stage residual \
  --source_root "$SOURCE_ROOT" \
  --train_manifest "$FIT_MANIFEST" \
  --output_dir "$RESIDUAL_OUT" \
  --device cuda:1 \
  --batch_size 64 \
  --num_workers 4 \
  --epochs 2 \
  --residual_train_image_size 256 \
  --residual_lr 0.0002 \
  --weight_decay 0.0 \
  --max_samples_per_label 0 \
  --random_state 100 \
  > "$LOGS/residual_train.log" 2>&1
echo "[strict-repro] residual done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

wait "$SEMANTIC_PID"
echo "[strict-repro] semantic done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/export_component_scores.py \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  --target_root "$SOURCE_ROOT" \
  --manifest "$GATE_MANIFEST" \
  --output_dir "$SOURCE_GATE_COMPONENTS" \
  --device cuda:3 \
  --artifact_model "$ARTIFACT_OUT/artifact_prior_models.joblib" \
  --semantic_probe "$SEMANTIC_OUT/semantic_probe.joblib" \
  --residual_checkpoint "$RESIDUAL_OUT/checkpoint-1.pth" \
  --artifact_forward_batch_size 8 \
  --artifact_tile_batch_size 8 \
  --semantic_forward_batch_size 32 \
  --residual_batch_size 64 \
  > "$LOGS/source_gate_components.log" 2>&1
echo "[strict-repro] source_gate components done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/run_source_stress_calibration.py \
  --source_component_dir "$SOURCE_GATE_COMPONENTS" \
  --output_dir "$STRESS_OUT" \
  --selection_protocol_out "$STRESS_OUT/selection_protocol.json" \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  > "$LOGS/source_stress_calibration.log" 2>&1
echo "[strict-repro] source stress calibration done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/export_component_scores.py \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  --target_root "$ROOT/dataset/AIGCDetectBenchmark_test" \
  --output_dir "$CURRENT17_COMPONENTS" \
  --device cuda:3 \
  --artifact_model "$ARTIFACT_OUT/artifact_prior_models.joblib" \
  --semantic_probe "$SEMANTIC_OUT/semantic_probe.joblib" \
  --residual_checkpoint "$RESIDUAL_OUT/checkpoint-1.pth" \
  --artifact_forward_batch_size 8 \
  --artifact_tile_batch_size 8 \
  --semantic_forward_batch_size 32 \
  --residual_batch_size 64 \
  > "$LOGS/current17_components.log" 2>&1
echo "[strict-repro] current17 components done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

"$PY" scripts/evaluate_component_weights.py \
  --component_dir "$CURRENT17_COMPONENTS" \
  --output_dir "$CURRENT17_REPORT" \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  --policy learned_weights \
  --weights_json "$STRESS_OUT/selection_protocol.json" \
  > "$LOGS/current17_report.log" 2>&1
echo "[strict-repro] current17 report done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

echo "[strict-repro] complete $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
