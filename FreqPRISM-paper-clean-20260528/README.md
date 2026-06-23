# FreqPRISM

FreqPRISM is a pure source-only AI-generated image detector that integrates frequency-domain artifact, semantic, and residual priors. Prior training and fusion-parameter selection use source data only; target benchmarks are report-only.

It uses a DFFreq-style root layout:

```text
data/
networks/
options/
train.py
test.py
validate.py
```

Method note:

```text
docs/FreqPRISM_方法说明与实验设计.md
```

Current paper-facing result entry point:

```text
results/main/
```

The default method uses the promoted pure source-only stress-calibrated fusion constants in `configs/apfreq_train100k_full.yaml`. The final decision threshold remains fixed at `0.50`; the earlier source-only gamma anchor is kept as a simpler baseline.

The training protocol is full `train_100k` source coverage:

```text
dataset/train_100k/progan_train
50000 real + 50000 fake
```

The default test protocol evaluates every image under every target generator:

```text
dataset/AIGCDetectBenchmark_test
per_label = 0
```

Key entrypoints:

```bash
python train.py --config configs/apfreq_train100k_full.yaml --stage all
python test.py --config configs/apfreq_train100k_full.yaml
python validate.py --config configs/apfreq_train100k_full.yaml
python scripts/run_prior_ablation.py --component_dir results/experiments/phase2_prior_ablation/current17_components --output_dir results/experiments/phase2_prior_ablation/report
```

The `apfreq_*` config and result paths are kept as stable demo artifact names for reproducibility.

Long training and evaluation loops show progress by default. Use `--no_progress` only when writing very compact logs is preferred.

This repository does not use target labels for prior training, threshold tuning, or fusion-parameter selection. The promoted fusion constants are selected by `results/main/pure_source_stress_calibration/selection_protocol.json`.

Canonical final reports:

```text
results/apfreq_full_target/
results/experiments/phase3_external_benchmarks/universalfakedetect_learned_gates/
results/experiments/phase3_external_benchmarks/synthbuster_learned_gates/
```
