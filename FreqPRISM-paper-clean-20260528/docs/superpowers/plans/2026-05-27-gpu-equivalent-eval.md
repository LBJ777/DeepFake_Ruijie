# GPU-Equivalent Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add result-preserving GPU/batched evaluation modes and a parity gate so future benchmark runs can use faster scoring only after proving outputs are unchanged against `results/apfreq_full_target`.

**Architecture:** `UnifiedDetectorConfig` gains an explicit scoring mode. `baseline` keeps the current path, `equivalent_fast` batches GPU forward passes while preserving PIL preprocessing semantics, and `aggressive_fast` exposes the existing GPU-preprocess path behind a parity check. A standalone parity script compares candidate final scores against the completed `results/apfreq_full_target/score_cache` baseline before optimized runs are trusted.

**Tech Stack:** Python 3.9, NumPy, PyTorch, PIL/Pillow, pytest, existing FreqPRISM detector/evaluation modules.

**Repository Note:** The user clarified this workspace should be treated as not-a-Git workflow. Do not run commit steps. Use verification checkpoints instead.

**Enablement Update:** The user accepted the observed `equivalent_fast` final-score drift (`max_abs_diff=0.0007184744`) because threshold labels and serialized CSV metrics were unchanged. `apfreq_train100k_full.yaml` now defaults to `scoring_mode: equivalent_fast`, while `baseline` remains available via runtime override. The parity gate default absolute score tolerance is `1e-3`.

---

## File Structure

- Modify `networks/detector.py`: add scoring mode parsing, runtime override helper, equivalent-fast tile batching, and mode-aware routing.
- Modify `scripts/evaluate_target.py`: add a `--scoring_mode` CLI override and write scoring runtime metadata to `protocol.json`.
- Create `scripts/check_scoring_parity.py`: compare completed baseline cache scores and candidate scoring mode outputs on deterministic samples and fail unless the equivalence gate passes.
- Modify `tests/test_gpu_preprocess.py`: cover scoring mode parsing and the new equivalent-fast tile route.
- Create `tests/test_scoring_parity.py`: cover parity summary logic without loading real checkpoints.

---

### Task 1: Add Explicit Scoring Mode To Detector Config

**Files:**
- Modify: `networks/detector.py`
- Test: `tests/test_gpu_preprocess.py`

- [ ] **Step 1: Write the failing config tests**

Append these tests to `tests/test_gpu_preprocess.py`:

```python
def test_default_scoring_mode_preserves_baseline_runtime() -> None:
    config = UnifiedDetectorConfig.from_root(".", ROOT_CONFIG)

    assert config.scoring_mode == "baseline"
    assert config.gpu_preprocess is False


def test_gpu_config_maps_to_aggressive_fast_for_backward_compatibility() -> None:
    config = UnifiedDetectorConfig.from_root(".", GPU_CONFIG)

    assert config.scoring_mode == "aggressive_fast"
    assert config.gpu_preprocess is True


def test_runtime_override_can_select_equivalent_fast_without_mutating_source_config() -> None:
    config = UnifiedDetectorConfig.from_root(".", ROOT_CONFIG)
    updated = config.with_runtime_overrides(scoring_mode="equivalent_fast")

    assert config.scoring_mode == "baseline"
    assert updated.scoring_mode == "equivalent_fast"
    assert updated.gpu_preprocess is False
```

- [ ] **Step 2: Run the config tests and verify they fail**

Run:

```bash
pytest tests/test_gpu_preprocess.py::test_default_scoring_mode_preserves_baseline_runtime tests/test_gpu_preprocess.py::test_gpu_config_maps_to_aggressive_fast_for_backward_compatibility tests/test_gpu_preprocess.py::test_runtime_override_can_select_equivalent_fast_without_mutating_source_config -q
```

Expected: FAIL with `AttributeError` for `scoring_mode` or `with_runtime_overrides`.

- [ ] **Step 3: Implement scoring mode parsing**

In `networks/detector.py`, add this near `ImageFile.LOAD_TRUNCATED_IMAGES = True`:

```python
SCORING_MODES = {"baseline", "equivalent_fast", "aggressive_fast"}


def _normalize_scoring_mode(value: object, *, gpu_preprocess: bool) -> str:
    if value is None:
        return "aggressive_fast" if bool(gpu_preprocess) else "baseline"
    mode = str(value).strip()
    if not mode:
        return "aggressive_fast" if bool(gpu_preprocess) else "baseline"
    if mode not in SCORING_MODES:
        raise ValueError(f"unsupported scoring_mode: {mode}")
    return mode
```

Add this field to `UnifiedDetectorConfig` after `gpu_preprocess: bool`:

```python
    scoring_mode: str
```

In `UnifiedDetectorConfig.from_root`, replace the direct `return cls(...)` with a local `gpu_preprocess` variable and pass `scoring_mode`:

```python
        gpu_preprocess = bool(runtime.get("gpu_preprocess", False))
        scoring_mode = _normalize_scoring_mode(runtime.get("scoring_mode"), gpu_preprocess=gpu_preprocess)
        return cls(
```

Inside the `cls(...)` call, replace:

```python
            gpu_preprocess=bool(runtime.get("gpu_preprocess", False)),
```

with:

```python
            gpu_preprocess=gpu_preprocess,
            scoring_mode=scoring_mode,
```

Add this method to `UnifiedDetectorConfig` after `with_artifact_overrides`:

```python
    def with_runtime_overrides(
        self,
        *,
        scoring_mode: str | None = None,
        gpu_preprocess: bool | None = None,
    ) -> "UnifiedDetectorConfig":
        updated_gpu_preprocess = self.gpu_preprocess if gpu_preprocess is None else bool(gpu_preprocess)
        updated_scoring_mode = (
            self.scoring_mode
            if scoring_mode is None
            else _normalize_scoring_mode(scoring_mode, gpu_preprocess=updated_gpu_preprocess)
        )
        return replace(
            self,
            gpu_preprocess=updated_gpu_preprocess,
            scoring_mode=updated_scoring_mode,
        )
```

- [ ] **Step 4: Run the config tests and verify they pass**

Run:

```bash
pytest tests/test_gpu_preprocess.py::test_default_scoring_mode_preserves_baseline_runtime tests/test_gpu_preprocess.py::test_gpu_config_maps_to_aggressive_fast_for_backward_compatibility tests/test_gpu_preprocess.py::test_runtime_override_can_select_equivalent_fast_without_mutating_source_config -q
```

Expected: PASS.

- [ ] **Step 5: Verification checkpoint**

Run:

```bash
pytest tests/test_gpu_preprocess.py -q
```

Expected: existing tests pass. If this checkpoint fails, fix the scoring mode compatibility before starting Task 2.

---

### Task 2: Add Equivalent-Fast Batched Tile And Semantic Routing

**Files:**
- Modify: `networks/detector.py`
- Test: `tests/test_gpu_preprocess.py`

- [ ] **Step 1: Write failing equivalent-fast tile test**

Append this test to `tests/test_gpu_preprocess.py`:

```python
def test_equivalent_fast_tile_path_batches_tiles_with_pil_semantics(tmp_path) -> None:
    image_path = tmp_path / "large.png"
    array = np.zeros((512, 512, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(512, dtype=np.uint8)[None, :]
    array[:, :, 1] = np.arange(512, dtype=np.uint8)[:, None]
    Image.fromarray(array).save(image_path)

    config = replace(
        UnifiedDetectorConfig.from_root(".", ROOT_CONFIG),
        scoring_mode="equivalent_fast",
        artifact_forward_batch_size=5,
        artifact_tile_batch_size=5,
    )
    detector = UnifiedArtifactDetector.__new__(UnifiedArtifactDetector)
    detector.config = config
    detector.device = torch.device("cpu")
    seen_batch_shapes: list[tuple[int, ...]] = []

    def fake_tensor_batch_scores(batch: torch.Tensor) -> np.ndarray:
        seen_batch_shapes.append(tuple(batch.shape))
        start = sum(shape[0] for shape in seen_batch_shapes[:-1])
        return np.arange(start, start + batch.shape[0], dtype=np.float32)

    def fail_gpu_reader(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("equivalent_fast must preserve PIL preprocessing semantics")

    detector._score_artifact_tensor_batch = fake_tensor_batch_scores
    detector._read_rgb_tensor = fail_gpu_reader

    scores, max_sides = detector.score_artifact_tile_paths([image_path])

    assert max_sides.tolist() == [512.0]
    assert scores.tolist() == [8.0]
    assert seen_batch_shapes == [(5, 3, 256, 256), (4, 3, 256, 256)]
```

- [ ] **Step 2: Write failing semantic routing test**

Append this test to `tests/test_gpu_preprocess.py`:

```python
def test_equivalent_fast_uses_batched_semantic_path() -> None:
    config = replace(UnifiedDetectorConfig.from_root(".", ROOT_CONFIG), scoring_mode="equivalent_fast")
    detector = UnifiedArtifactDetector.__new__(UnifiedArtifactDetector)
    detector.config = config
    detector.semantic_model = object()
    detector.semantic_preprocess = object()
    calls: list[str] = []

    def fake_batched(paths) -> np.ndarray:
        calls.append("batched")
        return np.asarray([0.25 for _ in paths], dtype=np.float32)

    detector._score_semantic_paths_batched = fake_batched

    scores = detector.score_semantic_paths(["a.png", "b.png"])

    assert calls == ["batched"]
    np.testing.assert_allclose(scores, np.asarray([0.25, 0.25], dtype=np.float32))
```

- [ ] **Step 3: Run the new routing tests and verify they fail**

Run:

```bash
pytest tests/test_gpu_preprocess.py::test_equivalent_fast_tile_path_batches_tiles_with_pil_semantics tests/test_gpu_preprocess.py::test_equivalent_fast_uses_batched_semantic_path -q
```

Expected: FAIL because equivalent-fast routing is not implemented.

- [ ] **Step 4: Implement PIL-semantics tile batching**

In `networks/detector.py`, add this method before `score_artifact_tile_paths`:

```python
    def _score_artifact_tile_paths_pil_batched(self, paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        scores_by_image: list[list[float]] = [[] for _ in paths]
        max_sides: list[int] = []
        pending_tensors: list[torch.Tensor] = []
        pending_indices: list[int] = []
        tile_batch_size = max(1, int(self.config.artifact_tile_batch_size))

        def flush_pending() -> None:
            if not pending_tensors:
                return
            batch_scores = self._score_artifact_tensor_batch(torch.stack(pending_tensors))
            for image_index, score in zip(pending_indices, batch_scores):
                scores_by_image[int(image_index)].append(float(score))
            pending_tensors.clear()
            pending_indices.clear()

        for image_index, path in enumerate(paths):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                max_sides.append(max(int(width), int(height)))
                if max(width, height) <= self.config.tile_size:
                    tiles = [rgb]
                else:
                    boxes = native_tile_boxes(width, height, self.config.tile_size, self.config.tile_grid_size)
                    tiles = extract_native_tiles(rgb, boxes, tile_size=self.config.tile_size)
                for tile in tiles:
                    pending_tensors.append(pil_to_tensor(tile, self.config.artifact_image_size, "clean"))
                    pending_indices.append(image_index)
                    if len(pending_tensors) >= tile_batch_size:
                        flush_pending()
        flush_pending()

        scores = [aggregate_tile_scores(image_scores, self.config.tile_mode) for image_scores in scores_by_image]
        return np.asarray(scores, dtype=np.float32), np.asarray(max_sides, dtype=np.float32)
```

Replace `score_artifact_tile_paths` routing with:

```python
    def score_artifact_tile_paths(self, paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        if self.config.scoring_mode == "aggressive_fast":
            return self._score_artifact_tile_paths_gpu(paths)
        if self.config.scoring_mode == "equivalent_fast":
            return self._score_artifact_tile_paths_pil_batched(paths)
        scores: list[float] = []
        max_sides: list[int] = []
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                max_sides.append(max(int(width), int(height)))
                if max(width, height) <= self.config.tile_size:
                    scores.append(float(self._score_rgb_images([rgb], ("clean",))[0]))
                    continue
                boxes = native_tile_boxes(width, height, self.config.tile_size, self.config.tile_grid_size)
                tiles = extract_native_tiles(rgb, boxes, tile_size=self.config.tile_size)
                tile_scores = self._score_rgb_images(tiles, ("clean",))
                scores.append(aggregate_tile_scores(tile_scores, self.config.tile_mode))
        return np.asarray(scores, dtype=np.float32), np.asarray(max_sides, dtype=np.float32)
```

Replace the first routing condition in `score_semantic_paths` with:

```python
        if self.config.scoring_mode in {"equivalent_fast", "aggressive_fast"}:
            return self._score_semantic_paths_batched(paths)
```

- [ ] **Step 5: Run routing tests and verify they pass**

Run:

```bash
pytest tests/test_gpu_preprocess.py::test_equivalent_fast_tile_path_batches_tiles_with_pil_semantics tests/test_gpu_preprocess.py::test_equivalent_fast_uses_batched_semantic_path -q
```

Expected: PASS.

- [ ] **Step 6: Run existing GPU preprocess tests**

Run:

```bash
pytest tests/test_gpu_preprocess.py -q
```

Expected: PASS.

---

### Task 3: Add Evaluation CLI Override And Protocol Metadata

**Files:**
- Modify: `scripts/evaluate_target.py`
- Test: command-line smoke check with `--help`

- [ ] **Step 1: Add CLI argument**

In `scripts/evaluate_target.py`, add this parser argument after `--config_name`:

```python
    parser.add_argument(
        "--scoring_mode",
        choices=("config", "baseline", "equivalent_fast", "aggressive_fast"),
        default="config",
        help="Override runtime scoring mode without changing the YAML config.",
    )
```

- [ ] **Step 2: Apply the runtime override**

After `config = UnifiedDetectorConfig.from_root(...).with_artifact_overrides(...)`, add:

```python
    if args.scoring_mode != "config":
        config = config.with_runtime_overrides(scoring_mode=args.scoring_mode)
```

- [ ] **Step 3: Write protocol metadata**

Add these fields to the `protocol` dictionary:

```python
        "scoring_mode": str(config.scoring_mode),
        "gpu_preprocess": bool(config.gpu_preprocess),
        "artifact_forward_batch_size": int(config.artifact_forward_batch_size),
        "artifact_tile_batch_size": int(config.artifact_tile_batch_size),
        "semantic_forward_batch_size": int(config.semantic_forward_batch_size),
        "parity_required_for_optimized_reporting": bool(config.scoring_mode != "baseline"),
```

- [ ] **Step 4: Run CLI smoke check**

Run:

```bash
python scripts/evaluate_target.py --help | rg -- '--scoring_mode'
```

Expected: output contains `--scoring_mode`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_gpu_preprocess.py -q
```

Expected: PASS.

---

### Task 4: Add Scoring Parity Gate Script

**Files:**
- Create: `scripts/check_scoring_parity.py`
- Test: `tests/test_scoring_parity.py`

- [ ] **Step 1: Write parity summary tests**

Create `tests/test_scoring_parity.py` with:

```python
from __future__ import annotations

import numpy as np

from scripts.check_scoring_parity import compare_component_scores


def _components(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "W": values.astype(np.float32),
        "T": values.astype(np.float32),
        "S": values.astype(np.float32),
        "R": values.astype(np.float32),
        "max_side": np.asarray([256, 1024], dtype=np.float32),
        "final_fixed": values.astype(np.float32),
    }


def test_compare_component_scores_accepts_identical_outputs() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline = _components(np.asarray([0.2, 0.8], dtype=np.float32))
    candidate = _components(np.asarray([0.2, 0.8], dtype=np.float32))

    result = compare_component_scores(labels, baseline, candidate, threshold=0.5, atol=1e-6, groups=["a", "a"])

    assert result["passed"] is True
    assert result["label_flips"] == 0
    assert result["max_abs_diff"]["final_fixed"] == 0.0
    assert result["metric_csv_exact"] is True


def test_compare_component_scores_rejects_label_flip() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline = _components(np.asarray([0.49, 0.8], dtype=np.float32))
    candidate = _components(np.asarray([0.51, 0.8], dtype=np.float32))

    result = compare_component_scores(labels, baseline, candidate, threshold=0.5, atol=1e-6, groups=["a", "a"])

    assert result["passed"] is False
    assert result["label_flips"] == 1
    assert result["max_abs_diff"]["final_fixed"] > 1e-6
    assert result["metric_csv_exact"] is False


def test_compare_component_scores_rejects_max_side_change() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    baseline = _components(np.asarray([0.2, 0.8], dtype=np.float32))
    candidate = _components(np.asarray([0.2, 0.8], dtype=np.float32))
    candidate["max_side"] = np.asarray([255, 1024], dtype=np.float32)

    result = compare_component_scores(labels, baseline, candidate, threshold=0.5, atol=1e-6, groups=["a", "a"])

    assert result["passed"] is False
    assert result["max_side_exact"] is False
```

- [ ] **Step 2: Run parity tests and verify they fail**

Run:

```bash
pytest tests/test_scoring_parity.py -q
```

Expected: FAIL because `scripts/check_scoring_parity.py` does not exist.

- [ ] **Step 3: Create parity script**

Create `scripts/check_scoring_parity.py` with:

```python
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
    parser.add_argument("--atol", type=float, default=1e-6)
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
    labels = np.asarray([int(sample.label) for sample in samples], dtype=np.int64)
    paths = [sample.path for sample in samples]

    baseline = score_components(
        config=base_config,
        device=str(args.device),
        paths=paths,
        residual_batch_size=int(args.residual_batch_size),
    )
    candidate = score_components(
        config=candidate_config,
        device=str(args.device),
        paths=paths,
        residual_batch_size=int(args.residual_batch_size),
    )
    result = compare_component_scores(
        labels,
        baseline,
        candidate,
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
```

- [ ] **Step 4: Run parity summary tests**

Run:

```bash
pytest tests/test_scoring_parity.py -q
```

Expected: PASS.

- [ ] **Step 5: Run parity script help**

Run:

```bash
python scripts/check_scoring_parity.py --help | rg -- '--candidate_mode'
```

Expected: output contains `--candidate_mode`.

---

### Task 5: Full Local Verification Without Touching Current Experiment Output

**Files:**
- Read-only: current experiment output directory
- No code files changed in this task

- [ ] **Step 1: Run fast unit tests**

Run:

```bash
pytest tests/test_gpu_preprocess.py tests/test_scoring_parity.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broader regression tests**

Run:

```bash
pytest tests -q
```

Expected: PASS. If unrelated legacy tests fail, record the exact failing test names and error lines before proceeding.

- [ ] **Step 3: Confirm the in-flight experiment is not modified by this work**

Run:

```bash
find results/experiments/phase3_external_benchmarks/universalfakedetect_learned_gates -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected before the current run finishes: no files are created by this optimization work. Expected after the current run finishes: only the experiment's own report files, such as `overall.csv`, `per_generator.csv`, and `protocol.json`.

- [ ] **Step 4: Run the AIGCDetectBenchmark equivalence gate**

Run:

```bash
python scripts/check_scoring_parity.py \
  --target_root "dataset/AIGCDetectBenchmark_test" \
  --baseline_score_cache_dir results/apfreq_full_target/score_cache \
  --config_name apfreq_train100k_full.yaml \
  --candidate_mode equivalent_fast \
  --device cuda:1 \
  --groups biggan,stylegan,whichfaceisreal \
  --per_label 2 \
  --residual_batch_size 16 \
  --atol 1e-6 \
  --output_json results/experiments/phase3_external_benchmarks/parity_equivalent_fast.json
```

Expected: exit code 0 and JSON contains `"passed": true`, `"baseline_source": "score_cache"`, and the resolved baseline cache path under `results/apfreq_full_target/score_cache`.

- [ ] **Step 5: Test aggressive mode only if equivalent mode passes**

Run:

```bash
python scripts/check_scoring_parity.py \
  --target_root "dataset/AIGCDetectBenchmark_test" \
  --baseline_score_cache_dir results/apfreq_full_target/score_cache \
  --config_name apfreq_train100k_full.yaml \
  --candidate_mode aggressive_fast \
  --device cuda:1 \
  --groups biggan,stylegan,whichfaceisreal \
  --per_label 2 \
  --residual_batch_size 16 \
  --atol 1e-6 \
  --output_json results/experiments/phase3_external_benchmarks/parity_aggressive_fast.json
```

Expected for use in benchmark reporting: exit code 0 and JSON contains `"passed": true`. If this fails, do not use `aggressive_fast` for benchmark reporting.

- [ ] **Step 6: Run a separate optimized smoke output directory**

Run only after the current baseline run has completed:

```bash
python scripts/evaluate_target.py \
  --target_root "dataset/AIGCDetectBenchmark_test" \
  --output_dir results/experiments/phase3_external_benchmarks/universalfakedetect_equivalent_fast_smoke \
  --config_name apfreq_train100k_full.yaml \
  --scoring_mode equivalent_fast \
  --device cuda:1 \
  --per_label 1 \
  --residual_batch_size 16 \
  --no_progress
```

Expected: new smoke directory is separate from `universalfakedetect_learned_gates`; its `protocol.json` contains `"scoring_mode": "equivalent_fast"` and `"parity_required_for_optimized_reporting": true`.
