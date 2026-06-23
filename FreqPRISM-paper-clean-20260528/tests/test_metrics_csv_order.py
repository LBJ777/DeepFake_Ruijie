from __future__ import annotations

from pathlib import Path

from utils.metrics import write_rows_csv


def test_write_rows_csv_places_generator_first(tmp_path: Path) -> None:
    output = tmp_path / "per_generator.csv"
    write_rows_csv(output, [{"acc": 1.0, "generator": "foo", "r_acc": 2.0}])

    assert output.read_text().splitlines()[0] == "generator,acc,r_acc"
