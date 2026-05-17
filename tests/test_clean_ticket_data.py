from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clean_ticket_data import clean_ticket_data, normalize_priority, normalize_status  # noqa: E402


class CleanTicketDataTests(unittest.TestCase):
    def test_normalize_priority_and_status(self) -> None:
        self.assertEqual(normalize_priority("high"), "High")
        self.assertEqual(normalize_priority("CRITICAL"), "Critical")
        self.assertEqual(normalize_status("waiting customer"), "Waiting Customer")

    def test_clean_ticket_data_standardizes_core_fields(self) -> None:
        raw_df = pd.DataFrame(
            [
                {
                    "ticket_id": " T1001 ",
                    "title": " VPN Issue ",
                    "description": " User cannot connect ",
                    "account": " Example Co ",
                    "location": "unknown",
                    "status": "waiting customer",
                    "priority": "high",
                    "source": "Portal",
                    "primary_resource": "unassigned",
                    "role": "Support",
                    "queue": "IT:Support",
                    "issue_type": "SW:Support",
                    "sub_issue_type": "",
                    "work_type": "Incident",
                    "contract_name": "none",
                    "sla": "Gold",
                    "created_by": "Dispatcher",
                    "resolution": "",
                    "completed_by": "",
                    "created_at": "30/04/2026 10:00",
                    "completed_at": "",
                    "due_at": "30/04/2026 14:00",
                    "first_response_at": "30/04/2026 10:30",
                    "estimated_hours": "2.5",
                    "priority_numeric": "3",
                    "is_legacy": "0",
                }
            ]
        )

        temp_dir = PROJECT_ROOT / "tests" / f"tmp_{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            csv_path = temp_dir / "raw.csv"
            raw_df.to_csv(csv_path, index=False)

            cleaned_df, summary = clean_ticket_data(csv_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        row = cleaned_df.iloc[0]
        self.assertEqual(row["priority"], "High")
        self.assertEqual(row["status"], "Waiting Customer")
        self.assertTrue(pd.isna(row["primary_resource"]))
        self.assertTrue(pd.isna(row["location"]))
        self.assertEqual(row["ticket_text"], "VPN Issue User cannot connect")
        self.assertAlmostEqual(float(row["first_response_minutes"]), 30.0, places=1)
        self.assertAlmostEqual(float(row["hours_until_due"]), 4.0, places=2)
        self.assertEqual(summary["output_rows"], 1)


if __name__ == "__main__":
    unittest.main()
