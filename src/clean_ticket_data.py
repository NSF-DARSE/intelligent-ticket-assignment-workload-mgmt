from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/Raw_Data/autotask_raw_data.csv")
CLEAN_DATA_DIR = Path("data/Cleaned_Data")
CLEAN_DATA_PATH = CLEAN_DATA_DIR / "autotask_cleaned_data.csv"
SUMMARY_PATH = CLEAN_DATA_DIR / "autotask_cleaning_summary.json"

DATETIME_COLUMNS = [
    "created_at",
    "completed_at",
    "due_at",
    "first_response_at",
]

TEXT_COLUMNS = [
    "ticket_id",
    "title",
    "description",
    "account",
    "location",
    "status",
    "priority",
    "source",
    "primary_resource",
    "role",
    "queue",
    "issue_type",
    "sub_issue_type",
    "work_type",
    "contract_name",
    "sla",
    "created_by",
    "resolution",
    "completed_by",
]

MISSING_TOKENS = {
    "",
    "unknown",
    "not available",
    "uncategorized",
    "unassigned",
    "nan",
    "none",
    "null",
}


def parse_datetime_column(series: pd.Series) -> pd.Series:
    """Parse mixed timestamp formats into UTC-aware datetimes."""
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def normalize_text_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = " ".join(text.split())

    if not text:
        return pd.NA

    return text


def normalize_missing_category(value: object) -> object:
    normalized = normalize_text_value(value)
    if pd.isna(normalized):
        return pd.NA

    if str(normalized).lower() in MISSING_TOKENS:
        return pd.NA

    return normalized


def normalize_priority(value: object) -> object:
    normalized = normalize_missing_category(value)
    if pd.isna(normalized):
        return pd.NA

    mapping = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "critical": "Critical",
    }
    return mapping.get(str(normalized).lower(), normalized)


def normalize_status(value: object) -> object:
    normalized = normalize_missing_category(value)
    if pd.isna(normalized):
        return pd.NA

    return str(normalized).title()


def safe_hours_delta(end_series: pd.Series, start_series: pd.Series) -> pd.Series:
    hours = (end_series - start_series).dt.total_seconds() / 3600
    invalid_mask = (end_series.isna()) | (start_series.isna()) | (hours < 0)
    return hours.mask(invalid_mask)


def clean_ticket_data(input_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(input_path)

    df.columns = [column.strip().lower() for column in df.columns]

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(normalize_text_value)

    categorical_missing_columns = [
        "location",
        "source",
        "primary_resource",
        "role",
        "queue",
        "issue_type",
        "sub_issue_type",
        "work_type",
        "contract_name",
        "sla",
        "completed_by",
        "resolution",
    ]
    for column in categorical_missing_columns:
        if column in df.columns:
            df[column] = df[column].map(normalize_missing_category)

    if "priority" in df.columns:
        df["priority"] = df["priority"].map(normalize_priority)

    if "status" in df.columns:
        df["status"] = df["status"].map(normalize_status)

    for column in DATETIME_COLUMNS:
        if column in df.columns:
            # Normalize all timestamps to UTC so downstream feature logic can
            # safely compare dates from mixed source formats.
            df[column] = parse_datetime_column(df[column])

    if "estimated_hours" in df.columns:
        df["estimated_hours"] = pd.to_numeric(df["estimated_hours"], errors="coerce")
        df["estimated_hours_clean"] = df["estimated_hours"].mask(df["estimated_hours"] <= 0)

    if "priority_numeric" in df.columns:
        df["priority_numeric"] = pd.to_numeric(df["priority_numeric"], errors="coerce").astype("Int64")

    if "is_legacy" in df.columns:
        df["is_legacy"] = pd.to_numeric(df["is_legacy"], errors="coerce").fillna(0).astype("Int64").astype(bool)

    df["is_completed"] = df["completed_at"].notna() if "completed_at" in df.columns else False

    if {"completed_at", "created_at"}.issubset(df.columns):
        df["resolution_hours"] = safe_hours_delta(df["completed_at"], df["created_at"]).round(2)
        df["resolution_days"] = (df["resolution_hours"] / 24).round(2)

    if {"first_response_at", "created_at"}.issubset(df.columns):
        df["first_response_minutes"] = (
            safe_hours_delta(df["first_response_at"], df["created_at"]) * 60
        ).round(1)

    if {"due_at", "created_at"}.issubset(df.columns):
        df["hours_until_due"] = safe_hours_delta(df["due_at"], df["created_at"]).round(2)

    if {"due_at", "completed_at"}.issubset(df.columns):
        df["hours_past_due"] = safe_hours_delta(df["completed_at"], df["due_at"]).round(2)

    if "title" in df.columns:
        df["title_word_count"] = df["title"].fillna("").str.split().str.len().astype("Int64")

    if "description" in df.columns:
        df["description_word_count"] = df["description"].fillna("").str.split().str.len().astype("Int64")

    if {"title", "description"}.issubset(df.columns):
        df["ticket_text"] = (
            df["title"].fillna("") + " " + df["description"].fillna("")
        ).str.strip().replace("", pd.NA)

    summary = {
        "input_path": str(input_path),
        "output_rows": int(len(df)),
        "output_columns": int(len(df.columns)),
        "completed_ticket_count": int(df["is_completed"].sum()) if "is_completed" in df.columns else 0,
        "missing_completed_at_count": int(df["completed_at"].isna().sum()) if "completed_at" in df.columns else 0,
        "priority_distribution": (
            df["priority"].fillna("Missing").value_counts().to_dict() if "priority" in df.columns else {}
        ),
        "status_distribution_top10": (
            df["status"].fillna("Missing").value_counts().head(10).to_dict() if "status" in df.columns else {}
        ),
        "null_percent_by_column": {
            column: round(float(percent), 2)
            for column, percent in (df.isna().mean() * 100).sort_values(ascending=False).items()
        },
    }

    return df, summary


def main() -> None:
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df, summary = clean_ticket_data()

    cleaned_df.to_csv(CLEAN_DATA_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Cleaned data saved to: {CLEAN_DATA_PATH}")
    print(f"Cleaning summary saved to: {SUMMARY_PATH}")
    print(f"Rows: {summary['output_rows']}")
    print(f"Columns: {summary['output_columns']}")
    print(f"Completed tickets: {summary['completed_ticket_count']}")


if __name__ == "__main__":
    main()
