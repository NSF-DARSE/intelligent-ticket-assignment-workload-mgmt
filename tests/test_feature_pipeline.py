from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clean_employee_skills import build_profile_dataset  # noqa: E402
from feature_engineering import engineer_features  # noqa: E402


class FeaturePipelineIntegrationTests(unittest.TestCase):
    def test_engineer_features_creates_training_open_and_summary_outputs(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "ticket_id": "T1",
                    "title": "VPN access issue",
                    "description": "Remote user cannot sign in",
                    "account": "Acct A",
                    "location": "HQ",
                    "status": "In Progress",
                    "priority": "High",
                    "source": "Portal",
                    "primary_resource": pd.NA,
                    "role": "Support",
                    "queue": "IT:Support",
                    "issue_type": "SW:Support",
                    "sub_issue_type": pd.NA,
                    "work_type": "Incident",
                    "contract_name": pd.NA,
                    "sla": "Gold",
                    "created_by": "Dispatcher",
                    "resolution": pd.NA,
                    "completed_by": pd.NA,
                    "created_at": pd.Timestamp("2026-04-01 09:00:00"),
                    "completed_at": pd.NaT,
                    "due_at": pd.Timestamp("2026-04-01 13:00:00"),
                    "first_response_at": pd.Timestamp("2026-04-01 09:15:00"),
                    "estimated_hours": 2.0,
                    "estimated_hours_clean": 2.0,
                    "priority_numeric": 3,
                    "is_legacy": False,
                    "is_completed": False,
                    "resolution_hours": np.nan,
                    "resolution_days": np.nan,
                    "first_response_minutes": 15.0,
                    "hours_until_due": 4.0,
                    "hours_past_due": np.nan,
                    "title_word_count": 3,
                    "description_word_count": 5,
                    "ticket_text": "VPN access issue Remote user cannot sign in",
                },
                {
                    "ticket_id": "T2",
                    "title": "Printer fixed",
                    "description": "Office printer restored",
                    "account": "Acct B",
                    "location": "Branch",
                    "status": "Complete",
                    "priority": "Low",
                    "source": "Phone",
                    "primary_resource": "J Moore",
                    "role": "Support",
                    "queue": "IT:Hardware",
                    "issue_type": "IT:Hardware",
                    "sub_issue_type": pd.NA,
                    "work_type": "Incident",
                    "contract_name": pd.NA,
                    "sla": "Silver",
                    "created_by": "Dispatcher",
                    "resolution": "Resolved",
                    "completed_by": "jmoore",
                    "created_at": pd.Timestamp("2026-04-01 08:00:00"),
                    "completed_at": pd.Timestamp("2026-04-01 10:00:00"),
                    "due_at": pd.Timestamp("2026-04-01 17:00:00"),
                    "first_response_at": pd.Timestamp("2026-04-01 08:20:00"),
                    "estimated_hours": 1.0,
                    "estimated_hours_clean": 1.0,
                    "priority_numeric": 1,
                    "is_legacy": False,
                    "is_completed": True,
                    "resolution_hours": 2.0,
                    "resolution_days": 0.08,
                    "first_response_minutes": 20.0,
                    "hours_until_due": 9.0,
                    "hours_past_due": np.nan,
                    "title_word_count": 2,
                    "description_word_count": 3,
                    "ticket_text": "Printer fixed Office printer restored",
                },
            ]
        )

        feature_df, training_df, open_df, profiles_df, summary = engineer_features(df)

        self.assertEqual(len(feature_df), 2)
        self.assertEqual(len(training_df), 1)
        self.assertEqual(len(open_df), 1)
        self.assertEqual(summary["active_ticket_count"], 1)
        self.assertIn("sla_priority_class", feature_df.columns)
        self.assertIn("ticket_text_has_vpn", feature_df.columns)
        self.assertFalse(profiles_df.empty)

    def test_build_profile_dataset_creates_dashboard_friendly_skill_profile(self) -> None:
        source_df = pd.DataFrame(
            {
                "Employee Name": ["J Moore"],
                "Role": ["Support Specialist"],
                "Skills": ["VPN Client Software Support; Remote Access Tools; Incident Management"],
            }
        )

        profile_df = build_profile_dataset(source_df)
        row = profile_df.iloc[0]

        self.assertEqual(row["employee_name"], "J Moore")
        self.assertEqual(row["technician_key"], "jmoore")
        self.assertEqual(row["skill_count"], 3)
        self.assertTrue(row["primary_skill_domain"] in {"Applications", "Network & Security", "Service Management"})


if __name__ == "__main__":
    unittest.main()
