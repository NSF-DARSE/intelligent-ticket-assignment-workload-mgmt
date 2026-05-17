from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_DATA_PATH = Path("data/Cleaned_Data/autotask_cleaned_data.csv")
FEATURE_DATA_DIR = Path("data/Feature_Engineered")
FEATURE_DATA_PATH = FEATURE_DATA_DIR / "autotask_feature_engineered.csv"
TRAINING_DATA_PATH = FEATURE_DATA_DIR / "autotask_training_dataset.csv"
OPEN_TICKETS_PATH = FEATURE_DATA_DIR / "autotask_open_tickets_dataset.csv"
TECHNICIAN_PROFILE_PATH = FEATURE_DATA_DIR / "technician_profiles.csv"
SUMMARY_PATH = FEATURE_DATA_DIR / "feature_engineering_summary.json"

DATETIME_COLUMNS = ["created_at", "completed_at", "due_at", "first_response_at"]
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
ACTIVE_STATUSES = {
    "New",
    "Scheduled",
    "Escalate",
    "Waiting Customer",
    "In Progress",
    "Waiting Dispatch",
    "Waiting Approval",
    "Waiting Materials",
    "Waiting Vendor",
}
SERVICE_REQUEST_PATTERNS = [
    r"\brequest\b",
    r"\bsetup\b",
    r"\bnew user\b",
    r"\bnew employee\b",
    r"\baccount creation\b",
    r"\buser account creation\b",
    r"\binstall\b",
    r"\blicense\b",
]
MAINTENANCE_PATTERNS = [
    r"\bpatch\b",
    r"\bupdate\b",
    r"\bmaintenance\b",
    r"\bupgrade\b",
    r"\bwindows update\b",
]
SLA_RULES = {
    "Low": {"initial_response_hours": 4.0, "status_update_hours": 48.0, "sla_weight": 1},
    "Medium": {"initial_response_hours": 2.0, "status_update_hours": 8.0, "sla_weight": 2},
    "High": {"initial_response_hours": 0.25, "status_update_hours": 4.0, "sla_weight": 4},
    "Service Request": {"initial_response_hours": 24.0, "status_update_hours": 24.0, "sla_weight": 1},
    "Maintenance": {"initial_response_hours": 24.0, "status_update_hours": 24.0, "sla_weight": 1},
}


def parse_datetime_column(series: pd.Series) -> pd.Series:
    """Parse mixed timestamp formats into UTC-aware datetimes."""
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def load_cleaned_data(input_path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    for column in DATETIME_COLUMNS:
        if column in df.columns:
            df[column] = parse_datetime_column(df[column])

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    created = df["created_at"]

    df["created_year"] = created.dt.year.astype("Int64")
    df["created_month"] = created.dt.month.astype("Int64")
    df["created_day"] = created.dt.day.astype("Int64")
    df["created_hour"] = created.dt.hour.astype("Int64")
    df["created_day_of_week"] = created.dt.dayofweek.astype("Int64")
    df["created_week_of_year"] = created.dt.isocalendar().week.astype("Int64")
    df["created_is_weekend"] = created.dt.dayofweek.isin([5, 6])
    df["created_is_business_hours"] = created.dt.hour.between(8, 17, inclusive="left") & ~df[
        "created_is_weekend"
    ]
    df["created_part_of_day"] = pd.cut(
        created.dt.hour,
        bins=[-1, 5, 11, 17, 21, 24],
        labels=["Overnight", "Morning", "Afternoon", "Evening", "Late Night"],
    ).astype("object")

    return df


def add_priority_features(df: pd.DataFrame) -> pd.DataFrame:
    for priority in PRIORITY_LEVELS:
        feature_name = f"is_priority_{priority.lower()}"
        df[feature_name] = df["priority"].eq(priority)

    df["priority_weight"] = df["priority"].map(
        {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    ).astype("Int64")

    return df


def add_status_features(df: pd.DataFrame) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(tz="UTC")

    df["is_active_ticket"] = df["status"].isin(ACTIVE_STATUSES)
    df["is_waiting_state"] = df["status"].astype("string").str.startswith("Waiting", na=False)
    df["is_unassigned"] = df["primary_resource"].isna()
    df["is_overdue_open"] = df["is_active_ticket"] & df["due_at"].notna() & (df["due_at"] < now_utc)
    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    ticket_text = df["ticket_text"].fillna("")

    df["ticket_text_char_count"] = ticket_text.str.len().astype("Int64")
    df["ticket_text_has_server"] = ticket_text.str.contains(r"\bserver\b", case=False, regex=True)
    df["ticket_text_has_backup"] = ticket_text.str.contains(r"\bbackup\b", case=False, regex=True)
    df["ticket_text_has_vpn"] = ticket_text.str.contains(r"\bvpn\b", case=False, regex=True)
    df["ticket_text_has_email"] = ticket_text.str.contains(r"\bemail\b|\boutlook\b", case=False, regex=True)
    df["ticket_text_has_printer"] = ticket_text.str.contains(r"\bprinter\b", case=False, regex=True)
    df["ticket_text_has_access_issue"] = ticket_text.str.contains(
        r"\baccess\b|\bpermission\b|\bdenied\b", case=False, regex=True
    )
    df["ticket_text_has_urgent_language"] = ticket_text.str.contains(
        r"\bcritical\b|\burgent\b|\basap\b|\bimmediately\b", case=False, regex=True
    )

    return df


def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    df["issue_type_group"] = df["issue_type"].astype("string").str.split(":").str[0]
    df["queue_group"] = df["queue"].astype("string").str.split(":").str[0]
    df["role_group"] = df["role"].astype("string").str.split(":").str[0]

    return df


def add_sla_features(df: pd.DataFrame) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(tz="UTC")
    text = df["ticket_text"].fillna("").str.lower()
    queue = df["queue"].fillna("").str.lower()
    issue_type = df["issue_type"].fillna("").str.lower()

    is_maintenance = text.str.contains("|".join(MAINTENANCE_PATTERNS), regex=True)
    is_service_request = (
        queue.str.contains(r"change requests", regex=True)
        | issue_type.str.contains(r"change", regex=True)
        | text.str.contains("|".join(SERVICE_REQUEST_PATTERNS), regex=True)
    ) & ~is_maintenance

    sla_priority_class = np.select(
        [
            is_maintenance,
            is_service_request,
            df["priority"].eq("Critical") | df["priority"].eq("High"),
            df["priority"].eq("Medium"),
            df["priority"].eq("Low"),
        ],
        [
            "Maintenance",
            "Service Request",
            "High",
            "Medium",
            "Low",
        ],
        default="Medium",
    )

    df["sla_priority_class"] = pd.Series(sla_priority_class, index=df.index, dtype="string")
    df["sla_coverage_type"] = np.where(df["created_is_business_hours"], "Standard", "Expanded")
    df["sla_initial_response_hours"] = df["sla_priority_class"].map(
        {key: value["initial_response_hours"] for key, value in SLA_RULES.items()}
    )
    df["sla_status_update_hours"] = df["sla_priority_class"].map(
        {key: value["status_update_hours"] for key, value in SLA_RULES.items()}
    )
    df["sla_weight"] = df["sla_priority_class"].map({key: value["sla_weight"] for key, value in SLA_RULES.items()})
    df["is_sla_high_urgency"] = df["sla_priority_class"].eq("High")
    df["is_service_request"] = df["sla_priority_class"].eq("Service Request")
    df["is_maintenance"] = df["sla_priority_class"].eq("Maintenance")
    df["first_response_sla_met"] = (
        df["first_response_minutes"].notna()
        & df["sla_initial_response_hours"].notna()
        & (df["first_response_minutes"] <= df["sla_initial_response_hours"] * 60)
    )

    age_hours = (now_utc - df["created_at"]).dt.total_seconds() / 3600
    df["ticket_age_hours"] = age_hours.round(2)
    df["sla_age_ratio"] = (age_hours / df["sla_initial_response_hours"]).round(2)
    df["sla_breach_risk"] = df["is_active_ticket"] & (df["sla_age_ratio"] >= 1.0)
    df["sla_breach_severity"] = np.select(
        [
            df["sla_age_ratio"] >= 4,
            df["sla_age_ratio"] >= 2,
            df["sla_age_ratio"] >= 1,
        ],
        ["Critical", "High", "Medium"],
        default="Low",
    )

    return df


def add_resolution_target_features(df: pd.DataFrame) -> pd.DataFrame:
    df["resolution_hours_log"] = np.log1p(df["resolution_hours"])
    df["met_estimate_flag"] = (
        df["resolution_hours"].notna()
        & df["estimated_hours_clean"].notna()
        & (df["resolution_hours"] <= df["estimated_hours_clean"] * 1.2)
    )
    df["estimate_error_hours"] = df["resolution_hours"] - df["estimated_hours_clean"]

    return df


def build_technician_profiles(df: pd.DataFrame) -> pd.DataFrame:
    technician_df = df[df["completed_by"].notna()].copy()

    if technician_df.empty:
        return pd.DataFrame()

    grouped = technician_df.groupby("completed_by", dropna=False)
    profiles = grouped.agg(
        tickets_completed=("ticket_id", "count"),
        avg_resolution_hours=("resolution_hours", "mean"),
        median_resolution_hours=("resolution_hours", "median"),
        avg_estimated_hours=("estimated_hours_clean", "mean"),
        avg_first_response_minutes=("first_response_minutes", "mean"),
        critical_tickets_completed=("is_priority_critical", "sum"),
        high_tickets_completed=("is_priority_high", "sum"),
        distinct_issue_types=("issue_type", "nunique"),
        distinct_accounts=("account", "nunique"),
    ).reset_index()

    profiles["avg_resolution_hours"] = profiles["avg_resolution_hours"].round(2)
    profiles["median_resolution_hours"] = profiles["median_resolution_hours"].round(2)
    profiles["avg_estimated_hours"] = profiles["avg_estimated_hours"].round(2)
    profiles["avg_first_response_minutes"] = profiles["avg_first_response_minutes"].round(2)

    return profiles.sort_values(["tickets_completed", "avg_resolution_hours"], ascending=[False, True])


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    feature_df = df.copy()

    for column in DATETIME_COLUMNS:
        if column in feature_df.columns:
            feature_df[column] = parse_datetime_column(feature_df[column])

    feature_df = add_time_features(feature_df)
    feature_df = add_priority_features(feature_df)
    feature_df = add_status_features(feature_df)
    feature_df = add_text_features(feature_df)
    feature_df = add_category_features(feature_df)
    feature_df = add_sla_features(feature_df)
    feature_df = add_resolution_target_features(feature_df)

    training_df = feature_df[feature_df["resolution_hours"].notna()].copy()
    open_tickets_df = feature_df[feature_df["resolution_hours"].isna()].copy()
    technician_profiles_df = build_technician_profiles(feature_df)

    summary = {
        "input_rows": int(len(df)),
        "feature_rows": int(len(feature_df)),
        "feature_columns": int(len(feature_df.columns)),
        "training_rows": int(len(training_df)),
        "open_ticket_rows": int(len(open_tickets_df)),
        "technician_profile_rows": int(len(technician_profiles_df)),
        "active_ticket_count": int(feature_df["is_active_ticket"].sum()),
        "unassigned_ticket_count": int(feature_df["is_unassigned"].sum()),
        "overdue_open_ticket_count": int(feature_df["is_overdue_open"].sum()),
        "sla_breach_risk_count": int(feature_df["sla_breach_risk"].sum()),
        "business_hours_ticket_pct": round(float(feature_df["created_is_business_hours"].mean() * 100), 2),
        "weekend_ticket_pct": round(float(feature_df["created_is_weekend"].mean() * 100), 2),
        "sla_priority_distribution": feature_df["sla_priority_class"].value_counts().to_dict(),
        "top_issue_types": feature_df["issue_type"].fillna("Missing").value_counts().head(10).to_dict(),
        "top_technicians_by_completed_tickets": (
            technician_profiles_df[["completed_by", "tickets_completed"]].head(10).to_dict(orient="records")
            if not technician_profiles_df.empty
            else []
        ),
    }

    return feature_df, training_df, open_tickets_df, technician_profiles_df, summary


def main() -> None:
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df = load_cleaned_data()
    feature_df, training_df, open_tickets_df, technician_profiles_df, summary = engineer_features(cleaned_df)

    feature_df.to_csv(FEATURE_DATA_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    training_df.to_csv(TRAINING_DATA_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    open_tickets_df.to_csv(OPEN_TICKETS_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")
    technician_profiles_df.to_csv(TECHNICIAN_PROFILE_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Feature-engineered data saved to: {FEATURE_DATA_PATH}")
    print(f"Training dataset saved to: {TRAINING_DATA_PATH}")
    print(f"Open tickets dataset saved to: {OPEN_TICKETS_PATH}")
    print(f"Technician profiles saved to: {TECHNICIAN_PROFILE_PATH}")
    print(f"Feature summary saved to: {SUMMARY_PATH}")
    print(f"Feature rows: {summary['feature_rows']}")
    print(f"Training rows: {summary['training_rows']}")
    print(f"Open ticket rows: {summary['open_ticket_rows']}")


if __name__ == "__main__":
    main()
