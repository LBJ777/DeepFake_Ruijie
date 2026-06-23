# Main Fusion Parameter Promotion Design

## Goal

Promote the strong compact fusion candidate to the paper-facing FreqPRISM main method without hiding how it was selected.

## Decision

The main runtime configuration folds the selected effective parameters directly into the YAML files:

- `beta = 0.25`
- `alpha_low_pos = 0.30`
- `alpha_low_neg = 0.1875`
- `alpha_high_pos = 0.40`
- `alpha_high_neg = 0.00`
- `alpha_high_neg_guard = 0.25`
- `gamma = 0.21`

These values correspond to compact scales `tile_scale=1.25`, `semantic_pos_scale=2.0`, `semantic_neg_scale=1.25`, and `residual_scale=1.75` on top of the previous source-gamma anchor.

## Evidence Chain

The method is described as validation-calibrated, not pure source-only:

1. Source gate is used as a stability screen. The promoted candidate keeps source BA/AP/AUC at 100, with flip rate `0.00005` and mean score drift `0.009757`.
2. Current17 validation metrics select the candidate among source-screened candidates using the locked Acc+AP+AUC objective.
3. UniversalFakeDetect is held out from the search and used as external confirmation. It improves mean Acc/AP/AUC over the gamma=0.12 anchor.

## Runtime

Default `test.py` and `scripts/evaluate_target.py` read the effective parameters directly from config. No additional `weights_json` is required for the promoted main method.

Score caches now include a detector fingerprint. If fusion constants or scoring-relevant config values change, stale cache files are ignored and overwritten.

## Reporting

`results/main/main_fusion_parameters/promotion_protocol.json` is the canonical promotion record. `results/main/current17_validation_calibrated_fusion/` is a validation artifact because current17 labels are used for parameter selection. The earlier source-only gamma=0.12 method remains the clean target-label-free anchor.
