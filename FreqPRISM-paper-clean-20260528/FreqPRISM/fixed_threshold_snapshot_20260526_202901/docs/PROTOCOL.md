# FreqPRISM Protocol

FreqPRISM is the in-place, DFFreq-style organization of the strict source-only single detector that integrates APSD artifact, semantic, and residual priors.

## Data

Training root:

```text
../dataset/train_100k/progan_train
```

The train_100k tree is symlink-based. It contains:

```text
50000 real
50000 fake
100000 total
```

FreqPRISM training uses the full tree. No `max_sample` or per-label cap is set in the default full protocol.

Testing root:

```text
../dataset/test/test
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

## Weight Selection

The default config keeps the validated fixed composition from the previous strict detector:

```text
beta=0.20
gamma=0.08
threshold=0.50
```

`validate.py` records the source-only grid that should be used if these weights are reselected. Target labels must not be used for selecting weights, thresholds, epochs, or generator tails.

## Full Evaluation

```bash
python test.py --config configs/apfreq_train100k_full.yaml --device cuda:0
```

The target runner reports generator-level progress by default. Add `--no_progress` for compact batch logs.

This runs full-generator target evaluation and writes:

```text
results/apfreq_full_target/overall.csv
results/apfreq_full_target/per_generator.csv
results/apfreq_full_target/protocol.json
```

The `apfreq_*` config and result path names are retained as stable demo artifact names.
