# Phase 1-W Source Weight Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize the existing anchored source-only learned-weight result as Phase 1-W so the paper can describe fusion weights as calibrated rather than hand-picked.

**Architecture:** Reuse the existing component-score cache and learned-weight search/evaluation scripts. Add a small summary utility and CLI that reads locked search/evaluation outputs and writes reproducible Phase 1-W artifacts: protocol JSON, decision JSON, and paper-ready CSV.

**Tech Stack:** Python standard library, existing `utils.metrics.write_rows_csv`, existing Phase 0b component-score outputs.

---

### Task 1: Add Phase 1-W Summary Utility

**Files:**
- Create: `utils/source_weight_calibration.py`
- Test: `tests/test_source_weight_calibration.py`

- [ ] **Step 1: Write tests**

Create temp `weight_search.json`, `overall.csv`, and `per_generator.csv` inputs. Assert that the summary reports selected weights, source-only selection, metric deltas, and target-label usage flags.

- [ ] **Step 2: Implement utility**

Implement CSV/JSON loading, mean delta calculation, group-slice calculation, protocol payload construction, decision payload construction, and paper-row construction.

- [ ] **Step 3: Verify tests**

Run: `/data/lizihao/.conda/envs/aigc/bin/python -m pytest tests/test_source_weight_calibration.py -q`

Expected: all tests pass.

### Task 2: Add Phase 1-W Summary CLI

**Files:**
- Create: `scripts/summarize_phase1w_weights.py`

- [ ] **Step 1: Implement CLI**

Arguments:
- `--weight_search_json`
- `--fixed_report_dir`
- `--learned_report_dir`
- `--output_dir`
- `--selection_protocol_out`

Write:
- `decision.json`
- `paper_table.csv`
- `protocol.json`
- updated `results/source_weight_selection/selection_protocol.json`

- [ ] **Step 2: Compile-check CLI**

Run: `/data/lizihao/.conda/envs/aigc/bin/python -m py_compile scripts/summarize_phase1w_weights.py utils/source_weight_calibration.py`

Expected: exit code 0.

### Task 3: Run Phase 1-W Artifacts

**Files:**
- Create under: `results/experiments/phase1w_source_weight_calibration/`
- Modify: `results/source_weight_selection/selection_protocol.json`

- [ ] **Step 1: Run source-only weight search into Phase 1-W directory**

Run `scripts/learn_phase0_weights.py` on `results/experiments/phase0_source_gate_components` with the same anchored grids and constraints.

- [ ] **Step 2: Run fixed and learned-weight current17 reports**

Run `scripts/evaluate_component_weights.py` twice on `results/experiments/phase0_current17_components`, once with `--policy fixed` and once with `--policy learned_weights`.

- [ ] **Step 3: Run summary CLI**

Use the Phase 1-W weight search and report directories to produce the formal decision/protocol/table outputs.

### Task 4: Update Experiment Design Document

**Files:**
- Modify: `FreqPRISM_experiment_design.md`

- [ ] **Step 1: Update execution order**

Replace the old Phase 1 gate-threshold route with Phase 1-W as the main-method confirmation route.

- [ ] **Step 2: Update paper table list**

Rename Table 4 to source-only calibrated fusion weights.

### Task 5: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run full tests**

Run: `/data/lizihao/.conda/envs/aigc/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Inspect key artifacts**

Read `results/experiments/phase1w_source_weight_calibration/decision.json` and `results/source_weight_selection/selection_protocol.json` to confirm they state `target_labels_used_for_selection=false`.
