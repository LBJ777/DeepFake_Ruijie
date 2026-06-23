# FreqPRISM Results Index

Primary current result folders:

- `phase2_current_params/`: rerun Phase 2 reports using the current folded main parameters in `configs/apfreq_train100k_full.yaml` and `results/main/main_fusion_parameters/folded_weights.json`.
- `main/`: locked main-method parameters and source-only parameter-selection evidence.
- `experiments/phase3_external_benchmarks/`: promoted external benchmark reports and required component cache.
- `apfreq_full_target/`: canonical current17 main report.

Current Phase 2 headline:

- Main current17 mean Acc/AP/AUC: `94.8231 / 99.3441 / 99.3029`.
- Main report: `phase2_current_params/main_current17_report/overall.csv`.
- Prior ablation: `phase2_current_params/prior_ablation/overall.csv`.
- Bootstrap CI: `phase2_current_params/bootstrap_ci/`.

Cleanup note:

- Older temporary outputs were removed after the current-parameter cleanup.
