#!/usr/bin/env python3
"""Validate the public replication deposit without licensed CRSP inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def performance_row(period: str, strategy: str) -> dict[str, str]:
    path = ROOT / "outputs" / "performance_all_periods.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["period"] == period and row["strategy"] == strategy:
                return row
    raise AssertionError(f"Missing performance row: {period} / {strategy}")


def verify_manifest() -> None:
    manifest = ROOT / "SHA256SUMS.txt"
    require(manifest.exists(), "repository SHA-256 manifest exists")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"manifest file exists: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == expected, f"SHA-256 matches: {relative}")


def main() -> int:
    required_docs = [
        "README.md",
        "DATA_AVAILABILITY.md",
        "REPRODUCIBILITY.md",
        "CODEBOOK.md",
        "CITATION.cff",
        "LICENSE",
    ]
    for relative in required_docs:
        require((ROOT / relative).is_file(), f"required document exists: {relative}")

    momentum = performance_row(
        "Combined OOS 2019-May 2026", "momentum_6m_top2_invvol"
    )
    ppo = performance_row(
        "Combined OOS 2019-May 2026", "ppo_10_seed_ensemble"
    )
    require(close(float(momentum["cagr"]), 0.18564294289404737), "momentum CAGR")
    require(
        close(float(momentum["excess_sharpe"]), 1.1117217256242347),
        "momentum excess Sharpe",
    )
    require(close(float(ppo["cagr"]), 0.09558414552385641), "PPO CAGR")
    require(
        close(float(ppo["excess_sharpe"]), 0.43779292418280724),
        "PPO excess Sharpe",
    )

    summary = json.loads(
        (ROOT / "outputs" / "final_decision_summary.json").read_text(
            encoding="utf-8"
        )
    )
    require(summary["publication_sample"]["months"] == 89, "89 OOS months")
    require(
        close(summary["march_2026"]["ppo_net_return"], 0.2891550448110064),
        "March 2026 PPO net return",
    )

    model_archive = ROOT / "models" / "ppo_fixed_results.zip"
    with zipfile.ZipFile(model_archive) as archive:
        require(archive.testzip() is None, "PPO model archive integrity")
        names = set(archive.namelist())
        for seed in range(1001, 1011):
            require(
                f"models/ppo_seed_{seed}.zip" in names,
                f"frozen PPO seed {seed}",
            )

    pdf_figures = sorted((ROOT / "figures").glob("fig*.pdf"))
    png_figures = sorted((ROOT / "figures").glob("fig*.png"))
    require(len(pdf_figures) == 18, "18 vector publication figures")
    require(len(png_figures) == 18, "18 raster publication figures")

    notebook = json.loads(
        (ROOT / "notebooks" / "WRDS_reconstruction.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    require(all(not cell.get("outputs") for cell in code_cells), "notebook outputs stripped")
    require(
        all(cell.get("execution_count") is None for cell in code_cells),
        "notebook execution counts stripped",
    )

    verify_manifest()
    print("\nRepository validation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
