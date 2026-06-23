# FreqPRISM Protocol

FreqPRISM is the in-place, DFFreq-style organization of the strict source-only single detector that integrates APSD artifact, semantic, and residual priors.

## Data

Training root:

```text
dataset/train_100k/progan_train
```

The train_100k tree contains:

```text
50000 real
50000 fake
100000 total
```

FreqPRISM training uses the full tree. No `max_sample` or per-label cap is set in the default full protocol.

Testing root:

```text
dataset/AIGCDetectBenchmark_test
```

Full target evaluation uses all images under each generator. The default `evaluation.per_label` is `0`, which means no per-generator/per-label truncation.

## Training

```bash
python train.py --config configs/apfreq_train100k_full.yaml --stage all --device cuda:0
```

Progress bars are enabled by default for feature extraction and residual training. Add `--no_progress` to suppress them.

Stages:

1. `artifact`: train APSD codec/chroma artifact prior.
2. `semantic`: train OpenAI CLIP ViT-L/14 source-only linear probe.
3. `residual`: train internal NPR-style residual prior.

Target labels are not used by training.

## Pure Source-Only Stress-Calibrated Fusion

The current default method uses pure source-only stress-calibrated fusion parameters in
`configs/apfreq_train100k_full.yaml`. The final decision threshold is fixed:

```text
threshold=0.50
```

The promoted fusion constants are:

```text
beta=0.25
alpha_low_pos=0.30
alpha_low_neg=0.1875
alpha_high_pos=0.40
alpha_high_neg=0.00
alpha_high_neg_guard=0.25
gamma=0.21
```

These constants fold the selected compact fusion scales directly into the
runtime config:

```text
tile_scale=1.25
semantic_pos_scale=2.00
semantic_neg_scale=1.25
residual_scale=1.75
```

The promoted parameters are selected using only source-gate component scores and
source labels. The source-only stress objective minimizes fake-side source
logloss under source BA/AP/AUC, drift, flip-rate, and real-source logloss
constraints. The canonical selection record is:

```text
results/main/pure_source_stress_calibration/selection_protocol.json
```

The source-only gamma anchor remains available as the target-label-free baseline:

```text
results/main/source_gamma_selection/selection_protocol.json
```

The removed trigger-threshold path is not part of the runtime protocol. Target
labels are not used by prior training, threshold tuning, or fusion-parameter
selection. Current17 and UniversalFakeDetect are report-only diagnostics.

## Fusion Weight Sensitivity

Prior fusion weight sensitivity is run from cached component scores, so it does not retrain priors or rerun image inference:

```bash
python scripts/run_fusion_weight_sensitivity.py \
  --source_component_dir results/experiments/phase1w_source_weight_calibration/source_gate_components \
  --current_component_dir results/experiments/phase2_prior_ablation/current17_components \
  --output_dir results/experiments/phase2_fusion_weight_sensitivity \
  --weights_json results/source_weight_selection/selection_protocol.json
```

This writes source-side sensitivity metrics and locked current17 diagnostic
reports. Current17 labels are report-only and are not used for selecting fusion
weights.

## Source-Only Anchor Calibration

The source-only full alpha-split calibration is retained as the clean
target-label-free anchor. It learns the anchored scales for `beta`, each
semantic alpha, and `gamma` using only source-gate component scores:

```bash
python scripts/run_full_fusion_weight_calibration.py \
  --source_component_dir results/experiments/phase1w_source_weight_calibration/source_gate_components \
  --current_component_dir results/experiments/phase2_prior_ablation/current17_components \
  --output_dir results/experiments/phase1w_full_alpha_split_calibration \
  --selection_protocol_out results/main/full_fusion_weight_calibration/selection_protocol.json
```

That run re-selects the non-residual anchor weights. The residual coefficient is
then selected by the source-only gamma sweep:

```text
beta_scale=1.0
alpha_low_pos_scale=1.0
alpha_low_neg_scale=1.0
alpha_high_pos_scale=1.0
alpha_high_neg=0.0
alpha_high_neg_guard_scale=1.0
gamma_scale=1.5
effective_gamma=0.12
```

The promoted main method starts from this source-gamma anchor and then applies
the source-stress-selected compact scales listed above.

## Full Evaluation

```bash
python test.py --config configs/apfreq_train100k_full.yaml --device cuda:0
```

The target runner reports generator-level progress by default. Add `--no_progress` for compact batch logs.

This runs full-generator target evaluation with the promoted fusion constants
and writes:

```text
results/apfreq_full_target/overall.csv
results/apfreq_full_target/per_generator.csv
results/apfreq_full_target/protocol.json
```

The `apfreq_*` config and result path names are retained as stable demo artifact names. The canonical fixed-fusion current17 report is:

```text
results/apfreq_full_target/
```

UniversalFakeDetect and Synthbuster are reported from:

```text
results/experiments/phase3_external_benchmarks/universalfakedetect_learned_gates/
results/experiments/phase3_external_benchmarks/synthbuster_learned_gates/
```
