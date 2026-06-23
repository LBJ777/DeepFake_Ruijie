from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from data.datasets import ImageSample
from data.manifests import load_image_samples_from_manifest
from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig
from networks.score_blend import logits_to_probabilities, probabilities_to_logits
from utils.evaluation import score_group_with_cache
from utils.metrics import binary_metrics, write_target_report
from utils.progress import progress_iter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AffineCalibration:
    mode: str
    slope: float
    bias: float
    raw_threshold_equivalent: float
    target_real_fpr_pct: float


def _validate_labels_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must be 1D arrays with matching length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    return y, s


def fit_real_fpr_logit_bias(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    target_real_fpr_pct: float,
) -> AffineCalibration:
    y, s = _validate_labels_scores(labels, scores)
    if not 0.0 <= float(target_real_fpr_pct) < 100.0:
        raise ValueError("target_real_fpr_pct must be in [0, 100)")
    real_scores = s[y == 0]
    if real_scores.size == 0:
        raise ValueError("real_fpr_logit_bias requires at least one real calibration sample")
    raw_threshold = float(np.percentile(real_scores, 100.0 - float(target_real_fpr_pct)))
    raw_threshold = float(np.clip(raw_threshold, 1e-6, 1.0 - 1e-6))
    bias = float(-probabilities_to_logits(np.asarray([raw_threshold], dtype=np.float32))[0])
    return AffineCalibration(
        mode="real_fpr_logit_bias",
        slope=1.0,
        bias=bias,
        raw_threshold_equivalent=raw_threshold,
        target_real_fpr_pct=float(target_real_fpr_pct),
    )


def fit_platt_affine(labels: np.ndarray, scores: np.ndarray) -> AffineCalibration:
    y, s = _validate_labels_scores(labels, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("platt calibration requires both real and fake labels")
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - sklearn is a project dependency in normal runs
        raise RuntimeError("platt calibration requires scikit-learn") from exc

    logits = probabilities_to_logits(s).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    model.fit(logits, y)
    slope = float(model.coef_[0, 0])
    bias = float(model.intercept_[0])
    raw_threshold = float(logits_to_probabilities(np.asarray([-bias / slope], dtype=np.float32))[0]) if slope != 0 else 0.5
    return AffineCalibration(
        mode="platt_affine",
        slope=slope,
        bias=bias,
        raw_threshold_equivalent=float(np.clip(raw_threshold, 1e-6, 1.0 - 1e-6)),
        target_real_fpr_pct=float("nan"),
    )


def apply_affine_calibration(scores: np.ndarray, calibration: AffineCalibration) -> np.ndarray:
    raw_logits = probabilities_to_logits(np.asarray(scores, dtype=np.float32))
    calibrated_logits = float(calibration.slope) * raw_logits + float(calibration.bias)
    return logits_to_probabilities(calibrated_logits).astype(np.float32)


def load_score_cache_dir(cache_dir: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = Path(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"score cache dir does not exist: {root}")
    packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(root.glob("*.npz")):
        cached = np.load(path, allow_pickle=False)
        if "labels" not in cached.files or "scores" not in cached.files:
            continue
        labels, scores = _validate_labels_scores(cached["labels"], cached["scores"])
        packed[path.stem] = (labels, scores)
    if not packed:
        raise ValueError(f"no label/score npz files found under: {root}")
    return packed


def _concat_packed(packed: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([np.asarray(item[0], dtype=np.int64) for item in packed.values()], axis=0)
    scores = np.concatenate([np.asarray(item[1], dtype=np.float32) for item in packed.values()], axis=0)
    return _validate_labels_scores(labels, scores)


def _calibration_summary_rows(
    *,
    calibration_packed: Mapping[str, tuple[np.ndarray, np.ndarray]],
    calibration: AffineCalibration,
) -> list[dict[str, object]]:
    labels, raw_scores = _concat_packed(calibration_packed)
    calibrated_scores = apply_affine_calibration(raw_scores, calibration)
    raw_metrics = binary_metrics(labels, raw_scores, threshold=0.5)
    calibrated_metrics = binary_metrics(labels, calibrated_scores, threshold=0.5)
    return [
        {
            **asdict(calibration),
            "threshold": 0.5,
            "calibration_count": int(labels.shape[0]),
            "calibration_real_count": int((labels == 0).sum()),
            "calibration_fake_count": int((labels == 1).sum()),
            **{f"raw_{key}": value for key, value in raw_metrics.items()},
            **{f"calibrated_{key}": value for key, value in calibrated_metrics.items()},
        }
    ]


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_calibrated_report(
    output_dir: str | Path,
    *,
    target_packed: Mapping[str, tuple[np.ndarray, np.ndarray]],
    calibration_packed: Mapping[str, tuple[np.ndarray, np.ndarray]],
    calibration: AffineCalibration,
    protocol: Mapping[str, object],
) -> dict[str, float]:
    out = Path(output_dir)
    calibrated_packed = {
        group: (labels, apply_affine_calibration(scores, calibration))
        for group, (labels, scores) in target_packed.items()
    }
    mean = write_target_report(out, calibrated_packed, threshold=0.5)
    _write_rows(out / "calibration.csv", _calibration_summary_rows(calibration_packed=calibration_packed, calibration=calibration))
    full_protocol = {
        **dict(protocol),
        "threshold": 0.5,
        "calibration": asdict(calibration),
        "mean": mean,
    }
    (out / "protocol.json").write_text(json.dumps(full_protocol, indent=2, sort_keys=True) + "\n")
    return mean


def _stable_sample_key(sample: ImageSample, *, seed: int) -> str:
    encoded = f"{int(seed)}|{sample.label}|{sample.path.resolve(strict=False)}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_balanced_samples(
    samples: Sequence[ImageSample],
    *,
    per_label: int,
    seed: int,
) -> list[ImageSample]:
    if int(per_label) <= 0:
        return list(samples)
    selected: list[ImageSample] = []
    for label in (0, 1):
        label_samples = [sample for sample in samples if int(sample.label) == label]
        if len(label_samples) < int(per_label):
            raise ValueError(f"not enough label={label} samples: requested {per_label}, found {len(label_samples)}")
        label_samples = sorted(label_samples, key=lambda sample: _stable_sample_key(sample, seed=int(seed)))
        selected.extend(label_samples[: int(per_label)])
    return sorted(selected, key=lambda sample: str(sample.path.resolve(strict=False)))


def score_calibration_manifest(
    *,
    manifest: str | Path,
    cache_dir: str | Path,
    config_name: str,
    device: str,
    scoring_mode: str,
    calibration_per_label: int,
    calibration_seed: int,
    residual_batch_size: int,
    no_progress: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, config_name)
    if scoring_mode != "config":
        config = config.with_runtime_overrides(scoring_mode=scoring_mode)
    detector = UnifiedArtifactDetector(config, device=device)
    samples = select_balanced_samples(
        load_image_samples_from_manifest(manifest),
        per_label=int(calibration_per_label),
        seed=int(calibration_seed),
    )
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for group in progress_iter(["source_calibration"], total=1, desc="calibration", unit="set", enabled=not no_progress):
        group_labels, group_scores, _ = score_group_with_cache(
            group,
            samples,
            detector=detector,
            cache_dir=cache_dir,
            residual_batch_size=int(residual_batch_size),
        )
        labels.append(group_labels)
        scores.append(group_scores)
    return {"source_calibration": (np.concatenate(labels), np.concatenate(scores))}
