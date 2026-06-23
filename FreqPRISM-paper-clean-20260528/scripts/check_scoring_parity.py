#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import ImageSample, collect_labeled_images
from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig
from utils.component_scores import COMPONENT_SCORE_KEYS
from utils.metrics import binary_metrics


ENABLED_SCORE_DRIFT_ATOL = 1e-3


def _relative_key_for_group(path: str | Path, group: str) -> tuple[str, ...]:
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part == group and any(item in {"0_real", "1_fake"} for item in parts[index + 1 :]):
            return tuple(str(item) for item in parts[index:])
    raise ValueError(f"path does not contain group-relative labeled key for {group}: {path}")


def _metric_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    group_values = np.asarray(list(groups), dtype=object)
    if y.ndim != 1 or s.ndim != 1 or group_values.ndim != 1:
        raise ValueError("labels, scores, and groups must be 1D")
    if y.shape[0] != s.shape[0] or y.shape[0] != group_values.shape[0]:
        raise ValueError("labels, scores, and groups must have matching lengths")

    rows: list[dict[str, object]] = []
    for group in sorted({str(item) for item in group_values.tolist()}):
        mask = group_values == group
        rows.append({"generator": group, **binary_metrics(y[mask], s[mask], threshold=threshold)})
    mean = {
        f"mean_{key}": float(np.mean([float(row[key]) for row in rows]))
        for key in ("acc", "ap", "auc", "r_acc", "f_acc", "fpr", "fnr")
    }
    rows.append({"generator": "__overall__", **mean})
    return rows


def _serialize_metric_rows(rows: Sequence[Mapping[str, object]]) -> str:
    fieldnames = sorted({key for row in rows for key in row})
    if "generator" in fieldnames:
        fieldnames = ["generator", *[field for field in fieldnames if field != "generator"]]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def compare_component_scores(
    labels: np.ndarray,
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    threshold: float,
    atol: float,
    groups: Sequence[str] | None = None,
) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int64)
    group_values = list(groups) if groups is not None else ["all"] * int(y.shape[0])
    max_abs_diff: dict[str, float] = {}
    for key in COMPONENT_SCORE_KEYS:
        base = np.asarray(baseline[key], dtype=np.float32)
        cand = np.asarray(candidate[key], dtype=np.float32)
        if base.shape != cand.shape:
            max_abs_diff[key] = float("inf")
        else:
            max_abs_diff[key] = float(np.max(np.abs(base - cand))) if base.size else 0.0

    baseline_pred = (np.asarray(baseline["final_fixed"], dtype=np.float32) >= float(threshold)).astype(np.int64)
    candidate_pred = (np.asarray(candidate["final_fixed"], dtype=np.float32) >= float(threshold)).astype(np.int64)
    label_flips = int(np.sum(baseline_pred != candidate_pred))
    max_side_exact = bool(np.array_equal(np.asarray(baseline["max_side"]), np.asarray(candidate["max_side"])))
    score_keys_pass = all(max_abs_diff[key] <= float(atol) for key in ("W", "T", "S", "R", "final_fixed"))
    baseline_metric_csv = _serialize_metric_rows(
        _metric_rows(y, np.asarray(baseline["final_fixed"]), group_values, threshold=float(threshold))
    )
    candidate_metric_csv = _serialize_metric_rows(
        _metric_rows(y, np.asarray(candidate["final_fixed"]), group_values, threshold=float(threshold))
    )
    metric_csv_exact = bool(baseline_metric_csv == candidate_metric_csv)
    passed = bool(max_side_exact and label_flips == 0 and score_keys_pass and metric_csv_exact)
    return {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "label_flips": label_flips,
        "max_side_exact": max_side_exact,
        "metric_csv_exact": metric_csv_exact,
        "threshold": float(threshold),
        "atol": float(atol),
    }


def compare_score_outputs(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    threshold: float,
    atol: float,
    groups: Sequence[str] | None = None,
) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int64)
    baseline = np.asarray(baseline_scores, dtype=np.float32)
    candidate = np.asarray(candidate_scores, dtype=np.float32)
    group_values = list(groups) if groups is not None else ["all"] * int(y.shape[0])
    if y.ndim != 1 or baseline.ndim != 1 or candidate.ndim != 1:
        raise ValueError("labels and scores must be 1D")
    if y.shape[0] != baseline.shape[0] or y.shape[0] != candidate.shape[0]:
        raise ValueError("labels and scores must have matching lengths")

    max_abs_diff = {"final_fixed": float(np.max(np.abs(baseline - candidate))) if baseline.size else 0.0}
    baseline_pred = (baseline >= float(threshold)).astype(np.int64)
    candidate_pred = (candidate >= float(threshold)).astype(np.int64)
    label_flips = int(np.sum(baseline_pred != candidate_pred))
    score_pass = bool(max_abs_diff["final_fixed"] <= float(atol))
    baseline_metric_csv = _serialize_metric_rows(_metric_rows(y, baseline, group_values, threshold=float(threshold)))
    candidate_metric_csv = _serialize_metric_rows(_metric_rows(y, candidate, group_values, threshold=float(threshold)))
    metric_csv_exact = bool(baseline_metric_csv == candidate_metric_csv)
    passed = bool(label_flips == 0 and score_pass and metric_csv_exact)
    return {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "label_flips": label_flips,
        "metric_csv_exact": metric_csv_exact,
        "threshold": float(threshold),
        "atol": float(atol),
    }


def load_baseline_score_cache(cache_dir: str | Path, samples: Sequence[ImageSample]) -> tuple[np.ndarray, np.ndarray]:
    cache_root = Path(cache_dir)
    labels: list[int] = []
    scores: list[float] = []
    cache_by_group: dict[str, dict[tuple[str, ...], tuple[int, float]]] = {}

    for sample in samples:
        group = str(sample.group)
        if group not in cache_by_group:
            cache_path = cache_root / f"{group}.npz"
            if not cache_path.exists():
                raise FileNotFoundError(f"baseline score cache missing for group {group}: {cache_path}")
            cached = np.load(cache_path, allow_pickle=False)
            cached_labels = np.asarray(cached["labels"], dtype=np.int64)
            cached_scores = np.asarray(cached["scores"], dtype=np.float32)
            cached_paths = np.asarray(cached["paths"])
            if cached_labels.shape[0] != cached_scores.shape[0] or cached_labels.shape[0] != cached_paths.shape[0]:
                raise ValueError(f"baseline cache arrays have mismatched lengths: {cache_path}")
            group_cache: dict[tuple[str, ...], tuple[int, float]] = {}
            for label, score, cached_path in zip(cached_labels, cached_scores, cached_paths):
                group_cache[_relative_key_for_group(str(cached_path), group)] = (int(label), float(score))
            cache_by_group[group] = group_cache

        key = _relative_key_for_group(sample.path, group)
        if key not in cache_by_group[group]:
            raise KeyError(f"sample is missing from baseline cache for group {group}: {'/'.join(key)}")
        cached_label, cached_score = cache_by_group[group][key]
        if cached_label != int(sample.label):
            raise ValueError(f"baseline label mismatch for {'/'.join(key)}: cache={cached_label} sample={sample.label}")
        labels.append(cached_label)
        scores.append(cached_score)

    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def select_samples(samples: Sequence[ImageSample], *, groups: Sequence[str], per_label: int) -> list[ImageSample]:
    selected: list[ImageSample] = []
    for group in groups:
        group_samples = [sample for sample in samples if sample.group == group]
        if not group_samples:
            raise ValueError(f"group not found in target root: {group}")
        for label in (0, 1):
            label_samples = [sample for sample in group_samples if int(sample.label) == label]
            if not label_samples:
                raise ValueError(f"group {group} has no label {label} samples")
            selected.extend(label_samples[: int(per_label)])
    return selected


def score_components(
    *,
    config: UnifiedDetectorConfig,
    device: str,
    paths: Sequence[Path],
    residual_batch_size: int,
) -> dict[str, np.ndarray]:
    detector = UnifiedArtifactDetector(config, device=device)
    return detector.score_component_paths(paths, residual_batch_size=int(residual_batch_size))


def main() -> None:
    parser = argparse.ArgumentParser("Check FreqPRISM scoring-mode equivalence")
    parser.add_argument("--target_root", default=str(PROJECT_ROOT / "dataset" / "AIGCDetectBenchmark_test"))
    parser.add_argument("--baseline_score_cache_dir", default=str(PROJECT_ROOT / "results" / "apfreq_full_target" / "score_cache"))
    parser.add_argument("--config_name", default="apfreq_train100k_full.yaml")
    parser.add_argument("--candidate_mode", choices=("equivalent_fast", "aggressive_fast"), default="equivalent_fast")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--groups", default="biggan,stylegan,whichfaceisreal")
    parser.add_argument("--per_label", type=int, default=2)
    parser.add_argument("--residual_batch_size", type=int, default=16)
    parser.add_argument("--atol", type=float, default=ENABLED_SCORE_DRIFT_ATOL)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    base_config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, args.config_name).with_runtime_overrides(
        scoring_mode="baseline"
    )
    candidate_config = UnifiedDetectorConfig.from_root(PROJECT_ROOT, args.config_name).with_runtime_overrides(
        scoring_mode=args.candidate_mode
    )
    groups = [item.strip() for item in str(args.groups).split(",") if item.strip()]
    samples = select_samples(
        collect_labeled_images(args.target_root),
        groups=groups,
        per_label=max(1, int(args.per_label)),
    )
    labels, baseline_scores = load_baseline_score_cache(args.baseline_score_cache_dir, samples)
    paths = [sample.path for sample in samples]

    candidate = score_components(
        config=candidate_config,
        device=str(args.device),
        paths=paths,
        residual_batch_size=int(args.residual_batch_size),
    )
    result = compare_score_outputs(
        labels,
        baseline_scores,
        candidate["final_fixed"],
        threshold=float(base_config.threshold),
        atol=float(args.atol),
        groups=[sample.group for sample in samples],
    )
    result.update(
        {
            "candidate_mode": str(args.candidate_mode),
            "config_name": str(args.config_name),
            "groups": groups,
            "num_samples": int(labels.shape[0]),
            "baseline_score_cache_dir": str(Path(args.baseline_score_cache_dir).resolve(strict=False)),
            "baseline_source": "score_cache",
        }
    )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n")
    if not bool(result["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
