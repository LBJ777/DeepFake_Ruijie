# Source Stress Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure source-only stress calibration experiment that can select the currently promoted fusion parameters without using target labels.

**Architecture:** Implement a small utility that searches compact fusion scales from cached source component scores. The selector applies source-only hard constraints, then minimizes fake-side source logloss under a real-side calibration guard. A CLI writes candidates and a selection protocol.

**Tech Stack:** Python, NumPy, existing `utils.component_scores`, existing CSV/protocol helpers, pytest.

---

### Task 1: Source Stress Calibration Unit Tests

**Files:**
- Create: `tests/test_source_stress_calibration.py`
- Create: `utils/source_stress_calibration.py`

- [ ] **Step 1: Write failing tests**

Add tests for metric calculation, candidate acceptance, and protocol fields. The synthetic test data should verify that the selector chooses scales `(1.25, 2.0, 1.25, 1.75)` when those scales minimize fake logloss while satisfying the real-source guard.

- [ ] **Step 2: Run tests**

Run:

```bash
/data/lizihao/.conda/envs/aigc/bin/python -m pytest tests/test_source_stress_calibration.py -q
```

Expected: fail because `utils.source_stress_calibration` does not exist.

### Task 2: Utility Implementation

**Files:**
- Create: `utils/source_stress_calibration.py`

- [ ] **Step 1: Implement `SourceStressConfig`, `run_source_stress_search`, and `write_source_stress_artifacts`**

The search must:

- use only source labels and component scores
- compute source BA/AP/AUC, logloss, Brier, fake logloss, real logloss, fake/real margins, score drift, flip rate, and anchor distance
- accept candidates only if source BA/AP/AUC, flip, drift, and real logloss constraints pass
- select by `(fake_logloss, anchor_distance, real_logloss, source_logloss)`
- record `target_labels_used_for_selection=false`

- [ ] **Step 2: Run the unit tests**

Run:

```bash
/data/lizihao/.conda/envs/aigc/bin/python -m pytest tests/test_source_stress_calibration.py -q
```

Expected: pass.

### Task 3: CLI Entrypoint

**Files:**
- Create: `scripts/run_source_stress_calibration.py`

- [ ] **Step 1: Add CLI arguments**

Arguments:

- `--source_component_dir`
- `--output_dir`
- `--selection_protocol_out`
- `--config`
- `--scale_grid`
- `--max_real_logloss`
- `--max_flip_rate`
- `--max_mean_score_drift`
- `--min_source_ba`
- `--min_source_ap`
- `--min_source_auc`

- [ ] **Step 2: Add CLI test**

Extend `tests/test_source_stress_calibration.py` with a temporary component cache and subprocess call.

- [ ] **Step 3: Run CLI test**

Run:

```bash
/data/lizihao/.conda/envs/aigc/bin/python -m pytest tests/test_source_stress_calibration.py -q
```

Expected: pass.

### Task 4: Run Source-Only Experiment

**Files:**
- Write: `results/experiments/phase1s_source_stress_calibration/`
- Write: `results/main/pure_source_stress_calibration/selection_protocol.json`

- [ ] **Step 1: Run the source-only search**

Run:

```bash
/data/lizihao/.conda/envs/aigc/bin/python scripts/run_source_stress_calibration.py \
  --source_component_dir results/experiments/phase1w_source_weight_calibration/source_gate_components \
  --output_dir results/experiments/phase1s_source_stress_calibration \
  --selection_protocol_out results/main/pure_source_stress_calibration/selection_protocol.json \
  --config configs/apfreq_train100k_source_gamma_anchor.yaml \
  --scale_grid 0.75,1.0,1.25,1.5,1.75,2.0 \
  --max_real_logloss 0.0069 \
  --max_flip_rate 0.0001 \
  --max_mean_score_drift 0.01
```

Expected: selected weights equal `tile_scale=1.25`, `semantic_pos_scale=2.0`, `semantic_neg_scale=1.25`, `residual_scale=1.75`.

### Task 5: Main Protocol Reconciliation

**Files:**
- Modify: `configs/apfreq_train100k_full.yaml`
- Modify: `configs/freqprism_gpu_full.yaml`
- Modify: `results/main/main_fusion_parameters/folded_weights.json`
- Modify: `results/main/manifest.json`
- Modify: `results/main/README.md`
- Modify: `README.md`
- Modify: `docs/PROTOCOL.md`
- Modify: `docs/FreqPRISM_方法说明与实验设计.md`
- Modify: `FreqPRISM_experiment_design.md`
- Modify: `tests/test_apfreq_protocol.py`

- [ ] **Step 1: Update configs**

Keep the effective composition parameters unchanged. Change selection metadata to source-only stress calibration and `target_labels_used=false`.

- [ ] **Step 2: Update docs**

Describe the evidence chain as pure source-only stress calibration, with current17 and UniversalFakeDetect as final-report external diagnostics only.

- [ ] **Step 3: Update tests**

Update protocol assertions to require pure source-only selection protocol.

### Task 6: Verification

**Files:**
- No new files.

- [ ] **Step 1: Compile changed Python files**

Run:

```bash
/data/lizihao/.conda/envs/aigc/bin/python -m py_compile scripts/run_source_stress_calibration.py utils/source_stress_calibration.py
```

- [ ] **Step 2: Run full tests**

Run:

```bash
/data/lizihao/.conda/envs/aigc/bin/python -m pytest tests -q
```

Expected: all tests pass.
