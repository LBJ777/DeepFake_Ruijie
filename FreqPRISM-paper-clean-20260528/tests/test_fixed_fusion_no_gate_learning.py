from __future__ import annotations

from pathlib import Path

import utils.component_scores as component_scores


ROOT = Path(__file__).resolve().parents[1]


def test_tau_gate_learning_entrypoints_are_removed_from_runtime_code() -> None:
    removed_paths = [
        ROOT / "scripts" / "learn_component_gates.py",
        ROOT / "scripts" / "evaluate_component_gates.py",
        ROOT / "scripts" / "run_gate_sensitivity.py",
        ROOT / "scripts" / "summarize_phase1g_gates.py",
        ROOT / "utils" / "gate_sensitivity.py",
        ROOT / "utils" / "learned_gate_calibration.py",
    ]

    for path in removed_paths:
        assert not path.exists(), str(path)


def test_component_scores_exports_fixed_and_weight_calibration_only() -> None:
    for removed_name in [
        "GateParams",
        "GateSearchConfig",
        "GateSearchResult",
        "compute_learned_gate_scores",
        "search_gate_params",
        "search_component_gates",
    ]:
        assert not hasattr(component_scores, removed_name), removed_name

    assert hasattr(component_scores, "compute_fixed_scores")
    assert hasattr(component_scores, "compute_learned_weight_scores")
    assert hasattr(component_scores, "search_weight_params")
