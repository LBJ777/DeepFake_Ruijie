# FreqPRISM Main Results

This directory is the canonical entry point for the current paper-facing results.

## Main Method

`FreqPRISM pure source-only stress-calibrated fusion`

- Final decision threshold: `0.50`
- Parameter protocol: `pure_source_stress_calibration/selection_protocol.json`
- Selection data: `source_gate_stress_only`
- Target labels used for parameter selection: `false`
- Current17, UniversalFakeDetect, and Synthbuster role: final-report diagnostics only
- Promoted fusion constants:
  - `beta = 0.25`
  - `alpha_low_pos = 0.30`
  - `alpha_low_neg = 0.1875`
  - `alpha_high_pos = 0.40`
  - `alpha_high_neg = 0.00`
  - `alpha_high_neg_guard = 0.25`
  - `gamma = 0.21`

These constants are folded directly into `configs/apfreq_train100k_full.yaml` and `configs/freqprism_gpu_full.yaml`. They correspond to compact fusion scales `tile_scale=1.25`, `semantic_pos_scale=2.0`, `semantic_neg_scale=1.25`, and `residual_scale=1.75` applied to the previous source-gamma anchor.

## Parameter-Selection Evidence

The retained source-only evidence is:

```text
source_gate_split/
source_gamma_selection/
pure_source_stress_calibration/
main_fusion_parameters/folded_weights.json
```

`source_gamma_selection/` keeps the target-label-free gamma anchor. `pure_source_stress_calibration/` keeps the source-stress candidate table and the selected compact scales that produce the promoted constants above.

## Current17 Report

The canonical paper-facing current17 report is:

```text
results/apfreq_full_target/
```

## External Reports

Use these directories for the promoted external benchmark reports:

```text
results/experiments/phase3_external_benchmarks/universalfakedetect_learned_gates/
results/experiments/phase3_external_benchmarks/synthbuster_learned_gates/
```

## Rollback

The fixed-threshold snapshot remains at `FreqPRISM/fixed_threshold_snapshot_20260526_202901`.
