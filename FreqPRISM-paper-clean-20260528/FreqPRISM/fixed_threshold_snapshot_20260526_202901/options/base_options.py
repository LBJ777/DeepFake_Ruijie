from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class FreqPRISMConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def source_root(self) -> Path:
        return (PROJECT_ROOT / self.raw["dataset"]["source_train_root"]).resolve(strict=False)

    @property
    def target_root(self) -> Path:
        return (PROJECT_ROOT / self.raw["dataset"]["target_test_root"]).resolve(strict=False)

    @property
    def artifact_model(self) -> Path:
        return PROJECT_ROOT / self.raw["artifacts"]["artifact_model"]

    @property
    def semantic_probe(self) -> Path:
        return PROJECT_ROOT / self.raw["artifacts"]["semantic_probe"]

    @property
    def residual_checkpoint(self) -> Path:
        return PROJECT_ROOT / self.raw["artifacts"]["residual_prior"]

    @property
    def per_label(self) -> int:
        return int(self.raw.get("evaluation", {}).get("per_label", 0))


def load_config(config: str | Path) -> FreqPRISMConfig:
    path = Path(config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text())
    return FreqPRISMConfig(path=path, raw=raw)


def list_to_variant_spec(values: list[str]) -> str:
    return "expand:" + ",".join(str(item) for item in values)
