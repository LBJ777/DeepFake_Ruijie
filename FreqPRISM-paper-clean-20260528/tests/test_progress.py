from __future__ import annotations

import builtins

from utils.progress import progress_iter


def test_progress_iter_disabled_preserves_items() -> None:
    assert list(progress_iter([1, 2, 3], total=3, desc="disabled", enabled=False)) == [1, 2, 3]


def test_progress_iter_falls_back_when_tqdm_is_unavailable(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("no tqdm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert list(progress_iter(["a", "b"], total=2, desc="fallback", enabled=True, min_interval=0.0)) == ["a", "b"]
