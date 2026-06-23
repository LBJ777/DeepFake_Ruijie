# Fixed Threshold Snapshot

Created: 2026-05-26 20:29 Asia/Shanghai

Purpose: rollback point for the current fixed-gates / fixed-threshold FreqPRISM state before Phase 0 learned-gates diagnostics.

Included:

- Project scripts and Python modules: `scripts/`, `models/`, `networks/`, `data/`, `utils/`, `options/`
- Configs and docs: `configs/`, `docs/`, `FreqPRISM_experiment_design.md`, `README.md`, `pyproject.toml`
- Current tests: `tests/`
- Current weights and model artifacts: `checkpoints/`
- Current results and score caches: `results/`

Excluded:

- `dataset/` and other raw image data
- Future Phase 0 outputs created after this snapshot

Rollback note:

Restore the copied directories/files from this snapshot into the repository root to return to the fixed-gates state captured here. Use `checksums.sha256` to verify file integrity after copying.
