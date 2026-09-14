from __future__ import annotations

import csv
from pathlib import Path

from fly_sniff.odor_authority import build_authority_from_long_csv


def test_sparse_panel_uses_only_common_measured_receptors(tmp_path: Path) -> None:
    path = tmp_path / "responses.csv"
    rows = [
        ("R1", "A", 1.0),
        ("R2", "A", 0.4),
        ("R3", "A", 0.2),
        ("R1", "B", 0.3),
        ("R2", "B", 0.9),
        ("R4", "B", 0.7),
        ("R1", "C", 0.1),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["receptor", "odorant", "response"])
        writer.writeheader()
        for receptor, odorant, response in rows:
            writer.writerow({"receptor": receptor, "odorant": odorant, "response": response})

    authority = build_authority_from_long_csv(
        path,
        source_name="toy",
        source_version="1",
        selected_odorants=["A", "B"],
    )
    assert authority["receptors"] == ["R1", "R2"]
    assert set(authority["odorants"]) == {"A", "B"}
    assert authority["selection"]["no_imputation"] is True
    assert authority["selection"]["retained_receptor_count"] == 2
