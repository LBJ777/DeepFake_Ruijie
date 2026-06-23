from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from utils.component_scores import WeightParams


def test_evaluate_component_weights_marks_target_report_as_final_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    component_dir = tmp_path / "components"
    component_dir.mkdir()
    np.savez(
        component_dir / "demo.npz",
        labels=np.asarray([0, 1], dtype=np.int64),
        groups=np.asarray(["demo", "demo"], dtype=str),
        paths=np.asarray(["real.png", "fake.png"], dtype=str),
        W=np.asarray([0.1, 0.9], dtype=np.float32),
        T=np.asarray([0.1, 0.9], dtype=np.float32),
        S=np.asarray([0.1, 0.9], dtype=np.float32),
        R=np.asarray([0.1, 0.9], dtype=np.float32),
        max_side=np.asarray([512, 512], dtype=np.float32),
        final_fixed=np.asarray([0.1, 0.9], dtype=np.float32),
    )
    weights_json = tmp_path / "weights.json"
    weights_json.write_text(
        json.dumps(
            {
                "selected_weights": WeightParams.default().to_dict(),
                "target_labels_used_for_selection": False,
                "target_labels_used_for_final_report_only": False,
            }
        )
        + "\n"
    )
    output_dir = tmp_path / "report"

    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate_component_weights.py"),
            "--component_dir",
            str(component_dir),
            "--output_dir",
            str(output_dir),
            "--config",
            "configs/apfreq_train100k_full.yaml",
            "--policy",
            "learned_weights",
            "--weights_json",
            str(weights_json),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    protocol = json.loads((output_dir / "protocol.json").read_text())
    assert protocol["target_labels_used_for_selection"] is False
    assert protocol["target_labels_used_for_final_report_only"] is True
