from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def _infer_total(iterable: Iterable[T], total: int | None) -> int | None:
    if total is not None:
        return int(total)
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None


def _fallback_progress(
    iterable: Iterable[T],
    *,
    total: int | None,
    desc: str,
    unit: str,
    min_interval: float,
) -> Iterator[T]:
    expected = _infer_total(iterable, total)
    start = time.monotonic()
    last_print = 0.0
    count = 0
    label = str(desc or "progress")
    for item in iterable:
        yield item
        count += 1
        now = time.monotonic()
        should_print = now - last_print >= float(min_interval)
        is_done = expected is not None and count >= expected
        if should_print or is_done:
            elapsed = max(now - start, 1e-9)
            rate = count / elapsed
            if expected is None:
                message = f"[{label}] {count} {unit} ({rate:.2f} {unit}/s)"
            else:
                message = f"[{label}] {count}/{expected} {unit} ({rate:.2f} {unit}/s)"
            print(message, file=sys.stderr, flush=True)
            last_print = now


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    enabled: bool = True,
    min_interval: float = 5.0,
) -> Iterable[T]:
    if not enabled:
        return iterable
    expected = _infer_total(iterable, total)
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=expected, desc=desc or None, unit=unit, dynamic_ncols=True)
    except Exception:
        return _fallback_progress(iterable, total=expected, desc=desc, unit=unit, min_interval=float(min_interval))
