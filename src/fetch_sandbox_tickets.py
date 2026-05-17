from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from autotask_api_client import AutotaskApiClient, AutotaskConfig
from load_outputs_to_postgres import get_db_url


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "Raw_Data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

SANDBOX_RAW_PATH = RAW_DATA_DIR / "autotask_sandbox_raw_data.csv"
CANONICAL_RAW_PATH = RAW_DATA_DIR / "autotask_raw_data.csv"
SUMMARY_PATH = RAW_DATA_DIR / "autotask_sandbox_fetch_summary.json"

EXPECTED_COLUMNS = [
    "ticket_id",
    "title",
    "description",
    "account",
    "location",
    "status",
    "priority",
    "source",
    "estimated_hours",
    "primary_resource",
    "role",
    "queue",
    "issue_type",
    "sub_issue_type",
    "work_type",
    "contract_name",
    "sla",
    "created_at",
    "created_by",
    "completed_at",
    "due_at",
    "first_response_at",
    "resolution",
    "completed_by",
    "is_legacy",
    "priority_numeric",
]

PRIORITY_LABEL_MAP = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Critical",
}

OPEN_STATUS_LABELS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch tickets from the Autotask sandbox API and normalize them into the project raw schema."
    )
    parser.add_argument("--days-back", type=int, default=365, help="Fetch tickets created in the last N days.")
    parser.add_argument(
        "--all-tickets",
        action="store_true",
        help="Fetch all tickets from the sandbox without applying the created-date filter.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on records fetched.")
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Keep only active/open tickets after normalization.",
    )
    parser.add_argument(
        "--skip-reference-data",
        action="store_true",
        help="Skip auxiliary lookups for company/resource/queue names.",
    )
    parser.add_argument(
        "--replace-main-raw",
        action="store_true",
        help="Also overwrite data/Raw_Data/autotask_raw_data.csv for the existing Phase 2 pipeline.",
    )
    parser.add_argument(
        "--load-postgres",
        action="store_true",
        help="Load the normalized sandbox tickets into the autotask_raw PostgreSQL table.",
    )
    return parser.parse_args()


def iso_timestamp_days_back(days_back: int) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return cutoff.replace(microsecond=0).isoformat()


def pick_first(record: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        value = record.get(field)
        if value not in (None, ""):
            return value
    return None


def stringify(value: Any) -> str | None:
    if value in (None, "", "None"):
        return None
    return str(value)


def maybe_lookup(value: Any, mapping: dict[Any, str] | None = None) -> str | None:
    if value in (None, "", "None"):
        return None
    if mapping and value in mapping:
        return mapping[value]
    if mapping and str(value).isdigit():
        int_value = int(value)
        if int_value in mapping:
            return mapping[int_value]
    return stringify(value)


def normalize_priority_label(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and int(value) in PRIORITY_LABEL_MAP:
        return PRIORITY_LABEL_MAP[int(value)]
    text = str(value).strip()
    if text.isdigit():
        return PRIORITY_LABEL_MAP.get(int(text), text)
    return text


def map_picklist_value(value: Any, picklist_map: dict[str, str] | None = None) -> str | None:
    if value in (None, ""):
        return None

    text = str(value)
    if picklist_map and text in picklist_map:
        return picklist_map[text]
    return text


def normalize_priority_numeric(priority_value: Any, priority_label: str | None) -> int | None:
    if priority_value in (None, "") and not priority_label:
        return None
    if isinstance(priority_value, (int, float)):
        return int(priority_value)
    text = str(priority_value).strip() if priority_value not in (None, "") else ""
    if text.isdigit():
        return int(text)
    inverse_map = {label: numeric for numeric, label in PRIORITY_LABEL_MAP.items()}
    return inverse_map.get(priority_label or "")


def normalize_ticket_record(ticket: dict[str, Any], refs: dict[str, dict[Any, str]]) -> dict[str, Any]:
    picklists = refs.get("picklists", {})

    priority_raw = pick_first(ticket, ["priority", "priorityLabel", "priorityName", "priorityID"])
    priority = normalize_priority_label(map_picklist_value(priority_raw, picklists.get("priority")))

    account_value = pick_first(ticket, ["accountName", "companyName", "companyID", "accountID"])
    resource_value = pick_first(ticket, ["primaryResourceName", "assignedResourceName", "assignedResourceID"])
    creator_value = pick_first(ticket, ["createdByName", "creatorResourceName", "creatorResourceID"])
    completed_by_value = pick_first(ticket, ["completedByName", "completedByResourceName", "completedByResourceID"])
    queue_value = pick_first(ticket, ["queueName", "queueID"])
    contract_value = pick_first(ticket, ["contractName", "contractID"])
    sla_value = pick_first(ticket, ["serviceLevelAgreementName", "serviceLevelAgreementID", "slaName", "slaID"])

    record = {
        "ticket_id": stringify(pick_first(ticket, ["ticketNumber", "id"])),
        "title": stringify(pick_first(ticket, ["title"])),
        "description": stringify(pick_first(ticket, ["description"])),
        "account": maybe_lookup(account_value, refs.get("companies")),
        "location": stringify(pick_first(ticket, ["locationName", "companyLocationName", "location"])),
        "status": stringify(
            map_picklist_value(pick_first(ticket, ["statusLabel", "statusName", "status"]), picklists.get("status"))
        ),
        "priority": priority,
        "source": stringify(
            map_picklist_value(pick_first(ticket, ["sourceLabel", "sourceName", "source"]), picklists.get("source"))
        ),
        "estimated_hours": pick_first(ticket, ["estimatedHours", "estimatedHoursToComplete", "hoursToBeScheduled"]),
        "primary_resource": maybe_lookup(resource_value, refs.get("resources")),
        "role": stringify(
            map_picklist_value(
                pick_first(ticket, ["roleName", "assignedResourceRoleName", "role", "assignedResourceRoleID"]),
                picklists.get("assignedResourceRoleID"),
            )
        ),
        "queue": maybe_lookup(queue_value, refs.get("queues")),
        "issue_type": stringify(
            map_picklist_value(pick_first(ticket, ["issueTypeName", "issueType"]), picklists.get("issueType"))
        ),
        "sub_issue_type": stringify(
            map_picklist_value(
                pick_first(ticket, ["subIssueTypeName", "subIssueType"]),
                picklists.get("subIssueType"),
            )
        ),
        "work_type": stringify(
            map_picklist_value(pick_first(ticket, ["workTypeName", "workType"]), picklists.get("ticketType"))
        ),
        "contract_name": maybe_lookup(contract_value, refs.get("contracts")),
        "sla": maybe_lookup(sla_value, refs.get("service_levels")),
        "created_at": stringify(pick_first(ticket, ["createDate", "createdDate", "createdAt"])),
        "created_by": maybe_lookup(creator_value, refs.get("resources")),
        "completed_at": stringify(pick_first(ticket, ["completedDate", "completedDateTime", "completedAt"])),
        "due_at": stringify(pick_first(ticket, ["dueDateTime", "dueDate", "dueAt"])),
        "first_response_at": stringify(
            pick_first(ticket, ["firstResponseDateTime", "firstResponseDate", "firstResponseAt"])
        ),
        "resolution": stringify(pick_first(ticket, ["resolution"])),
        "completed_by": maybe_lookup(completed_by_value, refs.get("resources")),
        "is_legacy": False,
        "priority_numeric": normalize_priority_numeric(priority_raw, priority),
    }

    return record


def normalize_tickets(tickets: list[dict[str, Any]], refs: dict[str, dict[Any, str]]) -> pd.DataFrame:
    records = [normalize_ticket_record(ticket, refs) for ticket in tickets]
    df = pd.DataFrame(records)

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[EXPECTED_COLUMNS].copy()
    df["estimated_hours"] = pd.to_numeric(df["estimated_hours"], errors="coerce")
    df["priority_numeric"] = pd.to_numeric(df["priority_numeric"], errors="coerce").astype("Int64")
    return df


def filter_open_tickets(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].isin(OPEN_STATUS_LABELS)].copy()


def build_filters(days_back: int) -> list[dict[str, Any]]:
    return [{"field": "createDate", "op": "gte", "value": iso_timestamp_days_back(days_back)}]


def save_summary(df: pd.DataFrame, args: argparse.Namespace, postgres_loaded: bool) -> None:
    created_at = pd.to_datetime(df["created_at"], errors="coerce") if "created_at" in df.columns else pd.Series([], dtype="datetime64[ns]")
    summary = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows_fetched": int(len(df)),
        "open_only": bool(args.open_only),
        "days_back": int(args.days_back),
        "max_records": args.max_records,
        "replace_main_raw": bool(args.replace_main_raw),
        "postgres_loaded": postgres_loaded,
        "created_at_min": created_at.min().isoformat() if not created_at.empty and created_at.notna().any() else None,
        "created_at_max": created_at.max().isoformat() if not created_at.empty and created_at.notna().any() else None,
        "sandbox_output": str(SANDBOX_RAW_PATH),
        "canonical_output": str(CANONICAL_RAW_PATH) if args.replace_main_raw else None,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_csv_with_fallback(df: pd.DataFrame, target_path: Path, allow_fallback: bool) -> Path:
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        if not allow_fallback:
            raise

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = target_path.with_name(f"{target_path.stem}_{timestamp}{target_path.suffix}")
        df.to_csv(fallback_path, index=False)
        return fallback_path


def load_to_postgres(df: pd.DataFrame) -> None:
    engine = create_engine(get_db_url())
    df.to_sql("autotask_raw", engine, if_exists="replace", index=False)

    with engine.begin() as conn:
        conn.execute(text("SELECT COUNT(*) FROM autotask_raw"))


def main() -> None:
    args = parse_args()
    config = AutotaskConfig.from_env(PROJECT_ROOT / ".env")
    client = AutotaskApiClient(config)

    refs = {} if args.skip_reference_data else client.fetch_reference_maps()
    refs["picklists"] = client.fetch_picklist_maps("Tickets")
    filters = [{"field": "id", "op": "gte", "value": 0}] if args.all_tickets else build_filters(args.days_back)
    tickets = client.query_entity("Tickets", filters=filters, max_records=args.max_records)
    df = normalize_tickets(tickets, refs)

    if args.open_only:
        df = filter_open_tickets(df)

    sandbox_output_path = write_csv_with_fallback(df, SANDBOX_RAW_PATH, allow_fallback=True)

    if args.replace_main_raw:
        write_csv_with_fallback(df, CANONICAL_RAW_PATH, allow_fallback=False)

    postgres_loaded = False
    if args.load_postgres:
        load_to_postgres(df)
        postgres_loaded = True

    save_summary(df, args, postgres_loaded)

    print("Autotask sandbox fetch completed.")
    print(f"Rows fetched: {len(df)}")
    print(f"Sandbox raw file: {sandbox_output_path}")
    if args.replace_main_raw:
        print(f"Canonical raw file updated: {CANONICAL_RAW_PATH}")
    if postgres_loaded:
        print("PostgreSQL table autotask_raw refreshed.")


if __name__ == "__main__":
    main()
