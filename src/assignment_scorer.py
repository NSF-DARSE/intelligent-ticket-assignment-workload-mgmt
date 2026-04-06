from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DATA_PATH = Path("data/Feature_Engineered/autotask_feature_engineered.csv")
COMPLEXITY_PATH = Path("data/Complexity/autotask_complexity_scored.csv")
RECOMMENDATION_DIR = Path("data/Recommendations")
WORKLOAD_PATH = RECOMMENDATION_DIR / "technician_workload_snapshot.csv"
RECOMMENDATIONS_PATH = RECOMMENDATION_DIR / "assignment_recommendations.csv"
SUMMARY_PATH = RECOMMENDATION_DIR / "recommendation_summary.json"

TECHNICIAN_WEIGHTS = {
    "issue_type_skill": 0.25,
    "queue_group_skill": 0.10,
    "account_familiarity": 0.10,
    "workload_hours": 0.15,
    "workload_count": 0.10,
    "priority_balance": 0.10,
    "resolution_efficiency": 0.10,
    "sla_pressure": 0.10,
    "sla_urgency_fit": 0.10,
    "complexity_fit": 0.10,
}


def load_feature_data() -> pd.DataFrame:
    return pd.read_csv(FEATURE_DATA_PATH)


def load_complexity_data() -> pd.DataFrame:
    return pd.read_csv(COMPLEXITY_PATH) if COMPLEXITY_PATH.exists() else pd.DataFrame()


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator in (0, np.nan) or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def normalize_inverse(series: pd.Series) -> pd.Series:
    if series.empty:
        return series

    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return pd.Series([1.0] * len(series), index=series.index)

    return 1 - ((series - minimum) / (maximum - minimum))


def build_workload_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    active = df[df["is_active_ticket"]].copy()
    active = active[active["primary_resource"].notna()].copy()

    if active.empty:
        return pd.DataFrame(
            columns=[
                "technician",
                "open_ticket_count",
                "open_estimated_hours",
                "critical_open_count",
                "high_open_count",
                "medium_open_count",
                "low_open_count",
                "high_sla_open_count",
                "service_request_open_count",
                "maintenance_open_count",
            ]
        )

    workload = (
        active.groupby("primary_resource")
        .agg(
            open_ticket_count=("ticket_id", "count"),
            open_estimated_hours=("estimated_hours_clean", lambda s: s.fillna(2.0).sum()),
            critical_open_count=("is_priority_critical", "sum"),
            high_open_count=("is_priority_high", "sum"),
            medium_open_count=("is_priority_medium", "sum"),
            low_open_count=("is_priority_low", "sum"),
            high_sla_open_count=("is_sla_high_urgency", "sum"),
            service_request_open_count=("is_service_request", "sum"),
            maintenance_open_count=("is_maintenance", "sum"),
        )
        .reset_index()
        .rename(columns={"primary_resource": "technician"})
    )

    workload["open_estimated_hours"] = workload["open_estimated_hours"].round(2)
    workload["workload_hours_score"] = normalize_inverse(workload["open_estimated_hours"]).round(4)
    workload["workload_count_score"] = normalize_inverse(workload["open_ticket_count"]).round(4)

    return workload


def build_technician_history(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["resolution_hours"].notna() & df["completed_by"].notna()].copy()
    if completed.empty:
        return pd.DataFrame()

    history = (
        completed.groupby("completed_by")
        .agg(
            completed_ticket_count=("ticket_id", "count"),
            avg_resolution_hours=("resolution_hours", "mean"),
        )
        .reset_index()
        .rename(columns={"completed_by": "technician"})
    )

    history["avg_resolution_hours"] = history["avg_resolution_hours"].round(2)
    history["resolution_efficiency_score"] = normalize_inverse(history["avg_resolution_hours"]).round(4)

    return history


def get_technician_pool(df: pd.DataFrame) -> list[str]:
    completed_techs = set(df["completed_by"].dropna().astype(str).unique())
    return sorted(completed_techs)


def compute_skill_score(ticket: pd.Series, technician: str, completed: pd.DataFrame) -> dict:
    technician_history = completed[completed["completed_by"] == technician]
    total_completed = len(technician_history)

    issue_type_matches = technician_history["issue_type"].eq(ticket["issue_type"]).sum()
    queue_group_matches = technician_history["queue_group"].eq(ticket["queue_group"]).sum()
    account_matches = technician_history["account"].eq(ticket["account"]).sum()

    same_issue_tickets = technician_history[technician_history["issue_type"] == ticket["issue_type"]]
    overall_resolution = technician_history["resolution_hours"].mean()
    issue_resolution = same_issue_tickets["resolution_hours"].mean() if not same_issue_tickets.empty else overall_resolution

    return {
        "issue_type_skill": round(safe_ratio(issue_type_matches, total_completed), 4),
        "queue_group_skill": round(safe_ratio(queue_group_matches, total_completed), 4),
        "account_familiarity": round(safe_ratio(account_matches, total_completed), 4),
        "issue_type_match_count": int(issue_type_matches),
        "queue_group_match_count": int(queue_group_matches),
        "account_match_count": int(account_matches),
        "issue_type_avg_resolution_hours": round(float(issue_resolution), 2) if pd.notna(issue_resolution) else np.nan,
    }


def priority_balance_score(ticket: pd.Series, workload_row: pd.Series) -> float:
    if ticket["sla_priority_class"] == "High" or ticket["priority"] == "Critical":
        pressure = workload_row.get("critical_open_count", 0)
    elif ticket["priority"] == "High":
        pressure = workload_row.get("critical_open_count", 0) + workload_row.get("high_open_count", 0)
    else:
        pressure = workload_row.get("open_ticket_count", 0)

    max_pressure = max(
        workload_row.get("open_ticket_count", 0),
        workload_row.get("critical_open_count", 0) + workload_row.get("high_open_count", 0),
        1,
    )

    return round(1 - min(float(pressure) / float(max_pressure), 1.0), 4)


def sla_pressure_score(ticket: pd.Series, workload_row: pd.Series) -> float:
    if ticket["sla_priority_class"] == "High":
        pressure = workload_row.get("high_sla_open_count", 0)
    elif ticket["sla_priority_class"] == "Service Request":
        pressure = workload_row.get("service_request_open_count", 0)
    elif ticket["sla_priority_class"] == "Maintenance":
        pressure = workload_row.get("maintenance_open_count", 0)
    else:
        pressure = workload_row.get("open_ticket_count", 0)

    open_count = max(float(workload_row.get("open_ticket_count", 0)), 1.0)
    return round(1 - min(float(pressure) / open_count, 1.0), 4)


def sla_urgency_fit_score(ticket: pd.Series, skill: dict, tech_workload: dict) -> float:
    if ticket["sla_priority_class"] == "High":
        skill_component = min(skill["issue_type_skill"] * 2, 1.0)
        workload_component = tech_workload["workload_hours_score"]
        return round((skill_component * 0.6) + (workload_component * 0.4), 4)
    if ticket["sla_priority_class"] in {"Service Request", "Maintenance"}:
        return round((skill["queue_group_skill"] * 0.5) + (tech_workload["workload_count_score"] * 0.5), 4)
    return round((skill["issue_type_skill"] * 0.5) + (tech_workload["workload_hours_score"] * 0.5), 4)


def complexity_fit_score(ticket: pd.Series, tech_workload: dict) -> float:
    complexity_score = float(ticket.get("complexity_score", 3.0) or 3.0)
    if complexity_score >= 3.6:
        return round(max(0.0, tech_workload["workload_hours_score"]), 4)
    if complexity_score <= 2.4:
        return round(min(1.0, 0.5 + (tech_workload["workload_count_score"] * 0.5)), 4)
    return round(min(1.0, 0.4 + (tech_workload["workload_hours_score"] * 0.6)), 4)


def recommend_assignments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    complexity_df = load_complexity_data()
    if not complexity_df.empty:
        df = df.merge(
            complexity_df[["ticket_id", "complexity_score", "complexity_class", "complexity_reason"]],
            on="ticket_id",
            how="left",
        )

    technician_pool = get_technician_pool(df)
    completed = df[df["resolution_hours"].notna() & df["completed_by"].notna()].copy()
    open_tickets = df[df["is_active_ticket"]].copy()

    workload = build_workload_snapshot(df)
    history = build_technician_history(df)

    if not technician_pool or open_tickets.empty:
        return workload, pd.DataFrame(), {"open_ticket_count": int(len(open_tickets)), "recommendation_rows": 0}

    workload_lookup = workload.set_index("technician").to_dict(orient="index") if not workload.empty else {}
    history_lookup = history.set_index("technician").to_dict(orient="index") if not history.empty else {}

    recommendation_rows = []

    for _, ticket in open_tickets.iterrows():
        scored_rows = []

        for technician in technician_pool:
            tech_history = history_lookup.get(
                technician,
                {"resolution_efficiency_score": 0.5, "avg_resolution_hours": np.nan, "completed_ticket_count": 0},
            )
            tech_workload = workload_lookup.get(
                technician,
                {
                    "open_ticket_count": 0,
                    "open_estimated_hours": 0.0,
                    "critical_open_count": 0,
                    "high_open_count": 0,
                    "medium_open_count": 0,
                    "low_open_count": 0,
                    "high_sla_open_count": 0,
                    "service_request_open_count": 0,
                    "maintenance_open_count": 0,
                    "workload_hours_score": 1.0,
                    "workload_count_score": 1.0,
                },
            )

            skill = compute_skill_score(ticket, technician, completed)
            balance_score = priority_balance_score(ticket, pd.Series(tech_workload))
            sla_pressure = sla_pressure_score(ticket, pd.Series(tech_workload))
            sla_urgency_fit = sla_urgency_fit_score(ticket, skill, tech_workload)
            complexity_fit = complexity_fit_score(ticket, tech_workload)

            total_score = (
                TECHNICIAN_WEIGHTS["issue_type_skill"] * skill["issue_type_skill"]
                + TECHNICIAN_WEIGHTS["queue_group_skill"] * skill["queue_group_skill"]
                + TECHNICIAN_WEIGHTS["account_familiarity"] * skill["account_familiarity"]
                + TECHNICIAN_WEIGHTS["workload_hours"] * tech_workload["workload_hours_score"]
                + TECHNICIAN_WEIGHTS["workload_count"] * tech_workload["workload_count_score"]
                + TECHNICIAN_WEIGHTS["priority_balance"] * balance_score
                + TECHNICIAN_WEIGHTS["resolution_efficiency"] * tech_history["resolution_efficiency_score"]
                + TECHNICIAN_WEIGHTS["sla_pressure"] * sla_pressure
                + TECHNICIAN_WEIGHTS["sla_urgency_fit"] * sla_urgency_fit
                + TECHNICIAN_WEIGHTS["complexity_fit"] * complexity_fit
            )

            rationale = []
            rationale.append(f"SLA class {ticket['sla_priority_class']}")
            if pd.notna(ticket.get("complexity_class")):
                rationale.append(f"complexity {ticket['complexity_class']}")
            if skill["issue_type_match_count"] > 0:
                rationale.append(f"{skill['issue_type_match_count']} similar issue-type tickets")
            if skill["account_match_count"] > 0:
                rationale.append(f"familiar with account {ticket['account']}")
            if tech_workload["open_ticket_count"] == 0:
                rationale.append("currently no active tickets assigned")
            else:
                rationale.append(f"{int(tech_workload['open_ticket_count'])} active tickets")
            rationale.append(f"{tech_workload['open_estimated_hours']:.1f} open estimated hours")

            scored_rows.append(
                {
                    "ticket_id": ticket["ticket_id"],
                    "ticket_title": ticket["title"],
                    "ticket_priority": ticket["priority"],
                    "ticket_sla_priority_class": ticket["sla_priority_class"],
                    "ticket_sla_breach_risk": ticket["sla_breach_risk"],
                    "ticket_issue_type": ticket["issue_type"],
                    "ticket_queue": ticket["queue"],
                    "ticket_complexity_score": ticket.get("complexity_score"),
                    "ticket_complexity_class": ticket.get("complexity_class"),
                    "ticket_complexity_reason": ticket.get("complexity_reason"),
                    "current_assignee": ticket.get("primary_resource"),
                    "recommended_technician": technician,
                    "recommendation_score": round(float(total_score), 4),
                    "issue_type_skill_score": skill["issue_type_skill"],
                    "queue_group_skill_score": skill["queue_group_skill"],
                    "account_familiarity_score": skill["account_familiarity"],
                    "workload_hours_score": tech_workload["workload_hours_score"],
                    "workload_count_score": tech_workload["workload_count_score"],
                    "priority_balance_score": balance_score,
                    "sla_pressure_score": sla_pressure,
                    "sla_urgency_fit_score": sla_urgency_fit,
                    "complexity_fit_score": complexity_fit,
                    "resolution_efficiency_score": tech_history["resolution_efficiency_score"],
                    "open_ticket_count": int(tech_workload["open_ticket_count"]),
                    "open_estimated_hours": float(tech_workload["open_estimated_hours"]),
                    "historical_completed_tickets": int(tech_history["completed_ticket_count"]),
                    "avg_resolution_hours": tech_history["avg_resolution_hours"],
                    "rationale": "; ".join(rationale),
                }
            )

        ranked = sorted(scored_rows, key=lambda row: row["recommendation_score"], reverse=True)
        for rank, row in enumerate(ranked[:3], start=1):
            row["recommendation_rank"] = rank
            recommendation_rows.append(row)

    recommendations = pd.DataFrame(recommendation_rows)

    summary = {
        "open_ticket_count": int(len(open_tickets)),
        "technician_pool_size": int(len(technician_pool)),
        "recommendation_rows": int(len(recommendations)),
        "tickets_with_recommendations": int(recommendations["ticket_id"].nunique()) if not recommendations.empty else 0,
        "sla_priority_distribution_open": open_tickets["sla_priority_class"].value_counts().to_dict(),
        "complexity_distribution_open": open_tickets["complexity_class"].value_counts().to_dict()
        if "complexity_class" in open_tickets.columns
        else {},
    }

    return workload, recommendations, summary


def main() -> None:
    RECOMMENDATION_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_data()
    workload, recommendations, summary = recommend_assignments(df)

    workload.to_csv(WORKLOAD_PATH, index=False)
    recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Workload snapshot saved to: {WORKLOAD_PATH}")
    print(f"Recommendations saved to: {RECOMMENDATIONS_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Open tickets scored: {summary['open_ticket_count']}")
    print(f"Recommendation rows: {summary['recommendation_rows']}")


if __name__ == "__main__":
    main()
    
