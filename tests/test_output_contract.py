from __future__ import annotations

import csv
import json
import math
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicDepositContract(unittest.TestCase):
    def setUp(self) -> None:
        path = ROOT / "outputs" / "performance_all_periods.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            self.performance = list(csv.DictReader(stream))

    def row(self, period: str, strategy: str) -> dict[str, str]:
        return next(
            row
            for row in self.performance
            if row["period"] == period and row["strategy"] == strategy
        )

    def test_unified_sample_has_89_months(self) -> None:
        row = self.row(
            "Combined OOS 2019-May 2026", "momentum_6m_top2_invvol"
        )
        self.assertEqual(int(row["months"]), 89)

    def test_headline_performance(self) -> None:
        momentum = self.row(
            "Combined OOS 2019-May 2026", "momentum_6m_top2_invvol"
        )
        ppo = self.row(
            "Combined OOS 2019-May 2026", "ppo_10_seed_ensemble"
        )
        self.assertTrue(
            math.isclose(float(momentum["cagr"]), 0.18564294289404737)
        )
        self.assertTrue(math.isclose(float(ppo["cagr"]), 0.09558414552385641))

    def test_temporal_extension_has_17_months(self) -> None:
        row = self.row(
            "Temporal extension 2025-May 2026", "ppo_10_seed_ensemble"
        )
        self.assertEqual(int(row["months"]), 17)

    def test_no_security_level_crsp_panel_is_deposited(self) -> None:
        forbidden = {
            "verified_monthly_total_return_panel_NO_FILL.parquet",
            "crsp_30day_tbill_monthly_return.parquet",
            "model_ready_primary_with_crsp_cash_2008_2024.parquet",
        }
        deposited = {path.name for path in ROOT.rglob("*") if path.is_file()}
        self.assertTrue(forbidden.isdisjoint(deposited))

    def test_model_archive_has_ten_seeds(self) -> None:
        with zipfile.ZipFile(ROOT / "models" / "ppo_fixed_results.zip") as archive:
            names = set(archive.namelist())
        for seed in range(1001, 1011):
            self.assertIn(f"models/ppo_seed_{seed}.zip", names)

    def test_wrds_notebook_is_output_stripped(self) -> None:
        notebook = json.loads(
            (ROOT / "notebooks" / "WRDS_reconstruction.ipynb").read_text(
                encoding="utf-8"
            )
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs", []), [])
                self.assertIsNone(cell.get("execution_count"))


if __name__ == "__main__":
    unittest.main()
