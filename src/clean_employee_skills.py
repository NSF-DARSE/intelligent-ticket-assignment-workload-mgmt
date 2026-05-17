from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_ROOT / "Skillsdataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "Feature_Engineered"
PROFILE_OUTPUT_PATH = OUTPUT_DIR / "employee_skills_profile.csv"
NORMALIZED_OUTPUT_PATH = OUTPUT_DIR / "employee_skills_normalized.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "employee_skills_summary.json"
TECHNICIAN_KEY_ALIASES = {
    "ajohson": "ajohnson",
}


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_technician_key(employee_name: str) -> str:
    cleaned = normalize_whitespace(employee_name).replace(".", "")
    parts = cleaned.split(" ")
    if len(parts) < 2:
        return re.sub(r"[^a-z0-9]+", "", cleaned.lower())
    first_initial = re.sub(r"[^a-z0-9]+", "", parts[0].lower())[:1]
    last_name = re.sub(r"[^a-z0-9]+", "", parts[-1].lower())
    key = f"{first_initial}{last_name}"
    return TECHNICIAN_KEY_ALIASES.get(key, key)


def split_skills(skills_value: str) -> list[str]:
    skills = [normalize_whitespace(skill) for skill in str(skills_value or "").split(";")]
    return [skill for skill in skills if skill]


def infer_primary_skill_domain(skills: list[str]) -> str:
    joined = " | ".join(skills).lower()
    domain_rules = [
        ("Service Management", ["incident", "problem", "service", "change", "knowledge", "itsm", "cab", "release"]),
        ("Customer Support", ["customer", "communication", "intake", "onboarding", "documentation"]),
        ("Applications", ["browser", "outlook", "teams", "web app", "odbc", "application", "quickbooks"]),
        ("AI & Automation", ["ai", "machine learning", "automation", "scripting", "powershell"]),
        ("Cloud & Infrastructure", ["cloud", "server", "virtualization", "disaster recovery", "azure", "datto bcdr"]),
        ("Network & Security", ["network", "firewall", "vpn", "security", "defender", "siem", "lan/wan"]),
        ("Endpoint & Hardware", ["desktop", "laptop", "printer", "hardware", "windows os", "device", "voip"]),
        ("Business & Reporting", ["excel", "powerpoint", "forms", "analysis", "reporting"]),
    ]
    for domain, terms in domain_rules:
        if any(term in joined for term in terms):
            return domain
    return "General Support"


def build_profile_dataset(source_df: pd.DataFrame) -> pd.DataFrame:
    profile_df = source_df.copy()
    profile_df.columns = ["employee_name", "role", "skills"]
    profile_df["employee_name"] = profile_df["employee_name"].map(normalize_whitespace)
    profile_df["role"] = profile_df["role"].map(normalize_whitespace)
    profile_df["skills"] = profile_df["skills"].map(normalize_whitespace)
    profile_df["technician_key"] = profile_df["employee_name"].map(build_technician_key)
    profile_df["skill_count"] = profile_df["skills"].map(lambda value: len(split_skills(value)))
    profile_df["primary_skill_domain"] = profile_df["skills"].map(lambda value: infer_primary_skill_domain(split_skills(value)))
    profile_df["email"] = ""
    profile_df["contact_number"] = ""
    profile_df["department"] = ""
    profile_df["location"] = ""
    return profile_df[
        [
            "employee_name",
            "technician_key",
            "role",
            "primary_skill_domain",
            "skill_count",
            "email",
            "contact_number",
            "department",
            "location",
            "skills",
        ]
    ]


def build_normalized_dataset(profile_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, record in profile_df.iterrows():
        skill_list = split_skills(record["skills"])
        for index, skill in enumerate(skill_list, start=1):
            rows.append(
                {
                    "employee_name": record["employee_name"],
                    "technician_key": record["technician_key"],
                    "role": record["role"],
                    "primary_skill_domain": record["primary_skill_domain"],
                    "skill_name": skill,
                    "skill_rank": index,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(SOURCE_PATH)
    profile_df = build_profile_dataset(source_df)
    normalized_df = build_normalized_dataset(profile_df)

    profile_df.to_csv(PROFILE_OUTPUT_PATH, index=False)
    normalized_df.to_csv(NORMALIZED_OUTPUT_PATH, index=False)

    summary = {
        "source_file": str(SOURCE_PATH),
        "profile_output_file": str(PROFILE_OUTPUT_PATH),
        "normalized_output_file": str(NORMALIZED_OUTPUT_PATH),
        "employee_count": int(len(profile_df)),
        "skill_row_count": int(len(normalized_df)),
        "unique_roles": int(profile_df["role"].nunique()),
        "unique_skill_domains": int(profile_df["primary_skill_domain"].nunique()),
    }
    SUMMARY_OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Employee skills cleaning completed.")
    print(f"Profiles: {len(profile_df)} rows -> {PROFILE_OUTPUT_PATH}")
    print(f"Normalized skills: {len(normalized_df)} rows -> {NORMALIZED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
