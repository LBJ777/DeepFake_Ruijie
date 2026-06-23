from __future__ import annotations

from pathlib import Path


FORBIDDEN_SNIPPETS = {
    "ARTIFACT_PRIOR_SINGLE_DETECTOR/results",
    "ARTIFACT_PRIOR_SINGLE_DETECTOR_V2/results",
    "STRICT_SINGLE_DETECTOR_APSD_NPR/artifacts",
    "STRICT_SINGLE_DETECTOR_APSD_NPR/results",
    "STRICT_SINGLE_DETECTOR_UNIFIED_NPR/results",
    "AIGC_SOTA_AIDE_BASELINE_REPRO/results",
}


def test_freqprism_does_not_reference_prior_result_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    scan_roots = [
        root / "data",
        root / "models",
        root / "networks",
        root / "options",
        root / "scripts",
        root / "unified_detector",
        root / "utils",
        root / "configs",
        root / "docs",
        root / "README.md",
        root / "train.py",
        root / "test.py",
        root / "validate.py",
    ]
    offenders: list[str] = []
    for scan_root in scan_roots:
        paths = [scan_root] if scan_root.is_file() else list(scan_root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in {".py", ".md", ".yaml", ".toml"}:
                continue
            text = path.read_text(errors="ignore")
            for snippet in FORBIDDEN_SNIPPETS:
                if snippet in text:
                    offenders.append(f"{path}: {snippet}")
    assert offenders == []
