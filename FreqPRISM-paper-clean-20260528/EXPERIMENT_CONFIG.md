# FreqPRISM Experiment Configuration Notes

This note summarizes the experiment configuration contained in this clean paper
package. It is intended as a writing aid for the paper experiment section and
for checking which files support each reported experiment.

The clean package keeps code, configs, protocol records, CSV summaries, and
paper-facing tables. It intentionally does not include datasets, model weights,
feature caches, score caches, or other binary intermediate files. Paths such as
`dataset/...` and `checkpoints/.../*.pth` in the configs are the canonical
runtime paths used by the original experiment environment.

## 1. Main Protocol

The main paper method is the source-only, stress-calibrated FreqPRISM detector.
The default configuration is:

```text
configs/apfreq_train100k_full.yaml
```

GPU preprocessing uses the same promoted method constants:

```text
configs/freqprism_gpu_full.yaml
```

Core settings:

| item | value |
| --- | --- |
| random seed | `100` |
| source training root | `dataset/train_100k/progan_train` |
| source training size | `50,000` real + `50,000` fake |
| target current17 root | `dataset/AIGCDetectBenchmark_test` |
| target truncation | `per_label=0`, meaning full target evaluation |
| decision threshold | fixed `0.50` |
| target labels for training/selection | `false` |

Training and evaluation entry points:

```bash
python train.py --config configs/apfreq_train100k_full.yaml --stage all --device cuda:0
python test.py --config configs/apfreq_train100k_full.yaml --device cuda:0
python validate.py --config configs/apfreq_train100k_full.yaml
```

Main current17 report:

```text
results/apfreq_full_target/
  overall.csv
  per_generator.csv
  protocol.json
```

The protocol record reports mean current17 Acc/AP/AUC as:

```text
94.8231 / 99.3441 / 99.3029
```

## 2. Prior Training Configuration

FreqPRISM combines three priors.

Artifact prior:

| field | value |
| --- | --- |
| image size | `256` |
| train variants | `clean`, `jpeg50`, `jpeg50`, `resize50`, `blur1` |
| eval variants | `clean`, `jpeg35`, `jpeg50`, `resize50`, `blur1` |
| chroma coefficient | `chroma_alpha=-0.40` |
| codec max iterations | `200` |

Semantic prior:

| field | value |
| --- | --- |
| backbone | OpenAI CLIP `ViT-L/14` |
| image size | `256` |
| train/eval variant | `clean` |
| linear probe C | `1.0` |

Residual prior:

| field | value |
| --- | --- |
| train image size | `256` |
| selected epoch | `1` |
| epochs | `2` |
| learning rate | `0.0002` |
| weight decay | `0.0` |

Lightweight training records are retained under:

```text
checkpoints/artifact_prior/training_protocol.json
checkpoints/semantic_prior/training_protocol.json
checkpoints/semantic_prior/summary.csv
checkpoints/residual_prior/progress.jsonl
```

Actual model files are omitted from the clean package.

## 3. Fusion Parameters

The promoted main constants are folded directly into
`configs/apfreq_train100k_full.yaml` and `configs/freqprism_gpu_full.yaml`.

```text
beta=0.25
alpha_low_pos=0.30
alpha_low_neg=0.1875
alpha_high_pos=0.40
alpha_high_neg=0.00
alpha_high_neg_guard=0.25
gamma=0.21
tile_mode=top1
tile_size=256
tile_grid_size=3
high_res_threshold=960.0
threshold=0.50
```

The corresponding compact selected scales are:

```text
tile_scale=1.25
semantic_pos_scale=2.00
semantic_neg_scale=1.25
residual_scale=1.75
```

Selection record:

```text
results/main/pure_source_stress_calibration/selection_protocol.json
results/main/pure_source_stress_calibration/source_stress_candidates.csv
results/main/main_fusion_parameters/folded_weights.json
```

The selection objective is source-only: choose the candidate minimizing
fake-side source logloss while satisfying source BA/AP/AUC, score drift,
flip-rate, and real-source logloss constraints. The recorded candidate count is
`1296`, with `231` accepted candidates. The selected protocol explicitly records:

```text
target_labels_used_for_selection=false
selection_data=source_gate_stress_only
```

The source split used for fitting and gate-side selection is:

```text
results/main/source_gate_split/
```

Its protocol records `80,000` source-fit samples and `20,000` source-gate
samples, with a gate fraction of `0.2` and seed `100`.

## 4. Baselines and Anchors

The simpler source-only gamma anchor is retained for comparison:

```text
configs/apfreq_train100k_source_gamma_anchor.yaml
results/main/source_gamma_selection/selection_protocol.json
results/main/source_gamma_selection/candidates.csv
```

That anchor uses:

```text
beta=0.20
alpha_low_pos=0.15
alpha_low_neg=0.15
alpha_high_pos=0.20
alpha_high_neg=0.00
alpha_high_neg_guard=0.20
gamma=0.12
threshold=0.50
```

When writing the paper, describe this as a target-label-free source anchor, not
as a target-tuned baseline.

## 5. External Benchmark Configurations

UniversalFakeDetect report:

```text
results/experiments/phase3_external_benchmarks/universalfakedetect_learned_gates/
  overall.csv
  per_generator.csv
  protocol.json
```

The protocol records Acc/AP/AUC:

```text
89.3867 / 99.2438 / 99.1876
```

Synthbuster report:

```text
results/experiments/phase3_external_benchmarks/synthbuster_learned_gates/
  overall.csv
  per_generator.csv
  protocol.json
```

The protocol records Acc/AP/AUC:

```text
40.6389 / 45.2020 / 35.5055
```

GenImage SD1.4 source-trained report:

```text
configs/genimage_sd14_full.yaml
results/experiments/phase3_external_benchmarks/genimage_sd14_source_logit_bias_p1k/
  calibration.csv
  overall.csv
  per_generator.csv
  protocol.json
```

GenImage uses:

```text
source_train_root=dataset/GenImage
source_train_manifest=dataset/GenImage/sd14_train_manifest.csv
target_test_manifest=dataset/GenImage/val_manifest.csv
```

It enables source probability calibration:

```text
mode=real_fpr_logit_bias
target_real_fpr_pct=5.0
calibration_per_label=1000
calibration_seed=20260529
```

The GenImage protocol records the calibration bias as `4.416184902191162` and
reports Acc/AP/AUC:

```text
93.8359 / 98.5026 / 98.5572
```

All external benchmark protocol files record
`target_labels_used_for_selection=false`; labels are used only for final
reporting.

## 6. Ablation and Sensitivity Experiments

Prior ablation:

```text
results/phase2_current_params/prior_ablation/
  overall.csv
  per_generator.csv
  group_slices.csv
  protocol.json
```

Residual NPR ablation:

```text
results/phase2_current_params/residual_npr_ablation/
```

Artifact family ablation:

```text
results/phase2_current_params/artifact_family_ablation/
```

Tile resolution ablation:

```text
results/phase2_current_params/tile_resolution_ablation/
  overall.csv
  per_generator.csv
  group_slices.csv
  protocol.json
```

Tile variants in the protocol:

```text
RZ0_full_native_tile
RZ1_whole_only_no_tile
RZ2_resized512_tile
RZ3_center_crop_tile
RZ4_tile_mean_aggregation
RZ5_current_top1_tile
```

Fusion weight sensitivity and paper tables:

```text
results/phase2_current_params/fusion_weight_sensitivity/
  source_gate_weight_sweep.csv
  current17_weight_sweep_overall.csv
  current17_weight_sweep_per_generator.csv
  current17_weight_sweep_group_slices.csv
  protocol.json
  paper_tables/
```

Paper-facing tables:

```text
table4a_beta_ablation.csv
table4b_alpha_pos_ablation.csv
table4c_alpha_neg_ablation.csv
table4d_gamma_ablation.csv
tableS4d_alpha_low_pos_ablation.csv
tableS4e_alpha_low_neg_ablation.csv
tableS4f_alpha_high_pos_ablation.csv
tableS4g_alpha_high_neg_guard_ablation.csv
tableS4h_alpha_high_neg_direct_ablation.csv
```

The sensitivity protocol uses `1000` paired bootstrap samples at the current17
generator level with seed `20260528`. Current17 labels are locked diagnostics
only and are not used to select fusion weights.

## 7. Recommended Paper Wording Checks

Use the following constraints when drafting the experiment section:

- State that prior training, threshold choice, and fusion-parameter selection
  are source-only.
- State that the final threshold is fixed at `0.50`.
- Do not describe current17, UniversalFakeDetect, Synthbuster, or GenImage
  target labels as parameter-selection data.
- Distinguish source-gate selection metrics from final report metrics.
- For external benchmarks, cite each `protocol.json` as the run-level record
  and each `overall.csv` / `per_generator.csv` as the table source.
- Mention that this clean package omits weights, datasets, and caches; it is a
  paper/reproducibility metadata package, not a self-contained runnable archive.

