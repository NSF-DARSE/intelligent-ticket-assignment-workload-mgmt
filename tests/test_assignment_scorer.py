from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from assignment_scorer import (  # noqa: E402
    canonicalize_technician_key,
    compute_skill_alignment,
    infer_required_skills,
    infer_ticket_domains,
    normalize_inverse,
)


class AssignmentScorerUnitTests(unittest.TestCase):
    def test_canonicalize_technician_key_applies_alias_and_cleans(self) -> None:
        self.assertEqual(canonicalize_technician_key(" AJOHSON "), "ajohnson")
        self.assertTrue(pd.isna(canonicalize_technician_key("")))
        self.assertTrue(pd.isna(canonicalize_technician_key(None)))

    def test_normalize_inverse_handles_constant_series(self) -> None:
        series = pd.Series([5.0, 5.0, 5.0])
        normalized = normalize_inverse(series)
        self.assertListEqual(normalized.round(4).tolist(), [1.0, 1.0, 1.0])

    def test_infer_required_skills_captures_vpn_and_access_work(self) -> None:
        ticket = pd.Series(
            {
                "title": "VPN access not working",
                "description": "User cannot connect after MFA reset",
                "ticket_text": "vpn remote access user cannot sign in",
                "issue_type": "SW:Support",
                "ticket_text_has_vpn": True,
                "ticket_text_has_access_issue": True,
            }
        )

        skills = infer_required_skills(ticket)

        self.assertIn("VPN Client Software Support", skills)
        self.assertIn("Remote Access Tools", skills)
        self.assertIn("Access Management", skills)
        self.assertIn("MFA Enrollment Support", skills)
        self.assertIn("Incident Management", skills)

    def test_infer_ticket_domains_prefers_application_and_network_signals(self) -> None:
        ticket = pd.Series(
            {
                "title": "Browser VPN issue",
                "description": "Remote user cannot access the web portal",
                "ticket_text": "vpn browser remote access problem",
                "issue_type": "SW:Support",
                "issue_type_group": "Support",
            }
        )

        domains = infer_ticket_domains(ticket)

        self.assertIn("Applications", domains)
        self.assertIn("Network & Security", domains)

    def test_compute_skill_alignment_rewards_matching_skills_and_domain(self) -> None:
        ticket = pd.Series(
            {
                "title": "Printer queue failure",
                "description": "Office printer stopped scanning",
                "ticket_text": "printer scan queue issue",
                "issue_type": "IT:Hardware",
                "ticket_text_has_printer": True,
            }
        )
        skill_lookup = {
            "jmoore": {
                "skills": {
                    "Printer Repair & Maintenance",
                    "Printer Installation (Driver/Queue)",
                    "Desktop/Laptop Support",
                },
                "primary_skill_domain": "Endpoint & Hardware",
                "employee_name": "J Moore",
                "role": "Support Specialist",
            }
        }

        result = compute_skill_alignment(ticket, "jmoore", skill_lookup)

        self.assertGreaterEqual(result["matched_skill_count"], 2)
        self.assertGreater(result["skill_alignment_score"], 0.7)
        self.assertEqual(result["employee_name"], "J Moore")


if __name__ == "__main__":
    unittest.main()
