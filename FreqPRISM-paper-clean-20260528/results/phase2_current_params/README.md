# Phase 2 Current-Parameter Rerun

This folder is the clean Phase 2 result root. It uses:

- Config: `configs/apfreq_train100k_full.yaml`
- Weights: `results/main/main_fusion_parameters/folded_weights.json`
- Threshold: `0.50`
- Component cache: `component_cache/current17_components/`

Key tables:

- `main_current17_report/overall.csv`: main current17 table.
- `prior_ablation/overall.csv`: A0-A8 prior ablation.
- `fusion_weight_sensitivity/current17_weight_sweep_overall.csv`: beta/alpha/gamma and drop-weight sensitivity.
- `tile_resolution_ablation/overall.csv`: native tile / no-tile ablation.
- `artifact_family_ablation/overall.csv`: artifact feature-family masking ablation.
- `residual_npr_ablation/overall.csv`: residual / NPR-specific ablation.
- `bootstrap_ci/*.csv`: paired bootstrap CI for threshold metrics. `mean_ap` and `mean_auc` are observed deltas in these CSVs; their full observed means are in each experiment `overall.csv`.

Headline current17 means:

| Report | Variant | Mean Acc | Mean AP | Mean AUC |
| --- | --- | ---: | ---: | ---: |
| Main | Full | 94.8231 | 99.3441 | 99.3029 |
| Prior ablation | No artifact | 84.0615 | 94.1588 | 94.1008 |
| Prior ablation | No semantic | 91.6210 | 95.9848 | 96.3019 |
| Prior ablation | No residual | 90.6164 | 97.5137 | 97.5413 |
| Prior ablation | No tile | 93.0469 | 99.1736 | 99.1146 |

Cleanup note:

Older Phase 2 outputs were removed from `results/` after this current-parameter rerun was consolidated.
