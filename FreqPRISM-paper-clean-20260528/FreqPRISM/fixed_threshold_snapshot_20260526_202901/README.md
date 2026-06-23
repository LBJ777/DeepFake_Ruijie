# FreqPRISM

FreqPRISM is a strict source-only detector for AI-generated image detection. It integrates frequency-domain artifact, semantic, and residual priors without using target labels for training or selection.

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

The training protocol is full `train_100k` source coverage:

```text
dataset/train_100k/progan_train
50000 real + 50000 fake
```

The default test protocol evaluates every image under every target generator:

```text
dataset/test/test
per_label = 0
```

Key entrypoints:

```bash
python train.py --config configs/apfreq_train100k_full.yaml --stage all
python test.py --config configs/apfreq_train100k_full.yaml
python validate.py --config configs/apfreq_train100k_full.yaml
```

The `apfreq_*` config and result paths are kept as stable demo artifact names for reproducibility.

Long training and evaluation loops show progress by default. Use `--no_progress` only when writing very compact logs is preferred.

This repository does not use target labels for training, threshold tuning, or weight selection.
