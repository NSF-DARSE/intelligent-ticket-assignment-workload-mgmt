from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DATA_PATH = Path("data/Feature_Engineered/autotask_feature_engineered.csv")
COMPLEXITY_PATH = Path("data/Complexity/autotask_complexity_scored.csv")
NLP_MATCHES_PATH = Path("data/NLP/ticket_similarity_matches.csv")
RECOMMENDATION_DIR = Path("data/Recommendations")
WORKLOAD_PATH = RECOMMENDATION_DIR / "technician_workload_snapshot.csv"
RECOMMENDATIONS_PATH = RECOMMENDATION_DIR / "assignment_recommendations.csv"
SUMMARY_PATH = RECOMMENDATION_DIR / "recommendation_summary.json"

TECHNICIAN_WEIGHTS = {
    "issue_type_skill": 0.25,
    "bm25_text_expertise": 0.15,
    "queue_group_skill": 0.10,
    "account_familiarity": 0.05,
    "workload_hours": 0.12,
    "workload_count": 0.12,
    "priority_balance": 0.10,
    "resolution_efficiency": 0.05,
    "sla_pressure": 0.10,
    "sla_urgency_fit": 0.10,
    "complexity_fit": 0.10,
}

SOFT_TICKET_CAP = 20
CAPACITY_HOUR_CAP = 40.0
NEW_TECH_COMPLETED_THRESHOLD = 15
NEW_TECH_TOP1_CAP = 5


def default_workload_record() -> dict:
    return {
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
    }


def load_feature_data() -> pd.DataFrame:
    return pd.read_csv(FEATURE_DATA_PATH)


def load_complexity_data() -> pd.DataFrame:
    return pd.read_csv(COMPLEXITY_PATH) if COMPLEXITY_PATH.exists() else pd.DataFrame()


def load_nlp_matches() -> pd.DataFrame:
    return pd.read_csv(NLP_MATCHES_PATH) if NLP_MATCHES_PATH.exists() else pd.DataFrame()


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


def recompute_workload_scores(workload_lookup: dict[str, dict]) -> None:
    if not workload_lookup:
        return

    counts = pd.Series({tech: values.get("open_ticket_count", 0) for tech, values in workload_lookup.items()}, dtype=float)
    hours = pd.Series({tech: values.get("open_estimated_hours", 0.0) for tech, values in workload_lookup.items()}, dtype=float)

    count_scores = normalize_inverse(counts)
    hour_scores = normalize_inverse(hours)

    for technician in workload_lookup:
        workload_lookup[technician]["workload_count_score"] = round(float(count_scores.loc[technician]), 4)
        workload_lookup[technician]["workload_hours_score"] = round(float(hour_scores.loc[technician]), 4)


def projected_effort_hours(ticket: pd.Series) -> float:
    for field in ["predicted_resolution_hours_final", "estimated_resolution_hours_nlp", "estimated_hours_clean", "estimated_hours"]:
        value = ticket.get(field)
        if pd.notna(value):
            value = float(value)
            if value > 0:
                if field.startswith("predicted_resolution_hours") or field == "estimated_resolution_hours_nlp":
                    return min(value, 24.0)
                return value
    return 2.0


def apply_projected_assignment(workload_lookup: dict[str, dict], technician: str, ticket: pd.Series) -> None:
    tech_workload = workload_lookup.setdefault(technician, default_workload_record())
    tech_workload["open_ticket_count"] = float(tech_workload.get("open_ticket_count", 0)) + 1
    tech_workload["open_estimated_hours"] = float(tech_workload.get("open_estimated_hours", 0.0)) + projected_effort_hours(ticket)

    priority = ticket.get("priority")
    if priority == "Critical":
        tech_workload["critical_open_count"] = float(tech_workload.get("critical_open_count", 0)) + 1
    elif priority == "High":
        tech_workload["high_open_count"] = float(tech_workload.get("high_open_count", 0)) + 1
    elif priority == "Medium":
        tech_workload["medium_open_count"] = float(tech_workload.get("medium_open_count", 0)) + 1
    elif priority == "Low":
        tech_workload["low_open_count"] = float(tech_workload.get("low_open_count", 0)) + 1

    if ticket.get("sla_priority_class") == "High":
        tech_workload["high_sla_open_count"] = float(tech_workload.get("high_sla_open_count", 0)) + 1
    if bool(ticket.get("is_service_request")):
        tech_workload["service_request_open_count"] = float(tech_workload.get("service_request_open_count", 0)) + 1
    if bool(ticket.get("is_maintenance")):
        tech_workload["maintenance_open_count"] = float(tech_workload.get("maintenance_open_count", 0)) + 1

    recompute_workload_scores(workload_lookup)


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
    active_techs = set(df["primary_resource"].dropna().astype(str).unique())
    return sorted(completed_techs | active_techs)


def build_text_expertise_lookup(matches_df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    if matches_df.empty or "matched_completed_by" not in matches_df.columns:
        return {}

    score_column = "hybrid_text_score" if "hybrid_text_score" in matches_df.columns else "similarity_score"
    matches = matches_df.copy()
    matches[score_column] = pd.to_numeric(matches[score_column], errors="coerce").fillna(0.0)
    matches = matches[matches["matched_completed_by"].notna()].copy()

    if matches.empty:
        return {}

    grouped = (
        matches.groupby(["open_ticket_id", "matched_completed_by"])
        .agg(
            text_match_count=("matched_ticket_id", "count"),
            bm25_text_expertise_score=(score_column, "mean"),
            best_text_match_score=(score_column, "max"),
        )
        .reset_index()
    )

    if "match_rank" in matches.columns:
        best_rank = (
            matches.groupby(["open_ticket_id", "matched_completed_by"])["match_rank"]
            .min()
            .reset_index(name="best_text_match_rank")
        )
        grouped = grouped.merge(best_rank, on=["open_ticket_id", "matched_completed_by"], how="left")
    else:
        grouped["best_text_match_rank"] = np.nan

    lookup: dict[tuple[str, str], dict] = {}
    for _, row in grouped.iterrows():
        lookup[(str(row["open_ticket_id"]), str(row["matched_completed_by"]))] = {
            "bm25_text_expertise_score": round(float(row["bm25_text_expertise_score"]), 4),
            "best_text_match_score": round(float(row["best_text_match_score"]), 4),
            "text_match_count": int(row["text_match_count"]),
            "best_text_match_rank": int(row["best_text_match_rank"])
            if pd.notna(row["best_text_match_rank"])
            else np.nan,
        }

    return lookup


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


def new_technician_bonus(tech_history: dict, tech_workload: dict, ticket: pd.Series) -> float:
    completed_count = int(tech_history.get("completed_ticket_count", 0))
    if completed_count >= NEW_TECH_COMPLETED_THRESHOLD:
        return 0.0

    if not is_low_risk_ticket(ticket):
        return 0.0

    base_bonus = 0.10
    workload_component = 0.10 * float(tech_workload.get("workload_count_score", 1.0))
    hour_component = 0.05 * float(tech_workload.get("workload_hours_score", 1.0))
    return round(max(0.0, base_bonus + workload_component + hour_component), 4)


def is_new_technician(tech_history: dict) -> bool:
    return int(tech_history.get("completed_ticket_count", 0)) < NEW_TECH_COMPLETED_THRESHOLD


def is_high_risk_ticket(ticket: pd.Series) -> bool:
    complexity_score = float(ticket.get("complexity_score", 3.0) or 3.0)
    complexity_class = str(ticket.get("complexity_class", "") or "")
    return bool(
        ticket.get("priority") in {"Critical", "High"}
        or ticket.get("sla_priority_class") == "High"
        or complexity_class == "High"
        or complexity_score >= 3.6
    )


def is_low_risk_ticket(ticket: pd.Series) -> bool:
    complexity_score = float(ticket.get("complexity_score", 3.0) or 3.0)
    complexity_class = str(ticket.get("complexity_class", "") or "")
    return bool(
        ticket.get("priority") in {"Low", "Medium"}
        and ticket.get("sla_priority_class") != "High"
        and complexity_class != "High"
        and complexity_score < 3.6
    )


def new_technician_penalty(
    tech_history: dict,
    tech_workload: dict,
    ticket: pd.Series,
    assigned_top1_count: int,
) -> float:
    if not is_new_technician(tech_history):
        return 0.0

    penalty = 0.0
    if is_high_risk_ticket(ticket):
        penalty += 0.30
    if assigned_top1_count >= NEW_TECH_TOP1_CAP:
        penalty += 0.15
    if float(tech_workload.get("open_ticket_count", 0)) >= SOFT_TICKET_CAP:
        penalty += 0.05

    return round(penalty, 4)


def high_risk_new_technician_block(tech_history: dict, ticket: pd.Series) -> bool:
    return is_new_technician(tech_history) and is_high_risk_ticket(ticket)


def new_technician_top1_cap_block(tech_history: dict, assigned_top1_count: int) -> bool:
    return is_new_technician(tech_history) and assigned_top1_count >= NEW_TECH_TOP1_CAP


def experienced_low_risk_penalty(tech_history: dict, tech_workload: dict, ticket: pd.Series) -> float:
    if is_new_technician(tech_history) or not is_low_risk_ticket(ticket):
        return 0.0

    open_ticket_count = float(tech_workload.get("open_ticket_count", 0))
    open_estimated_hours = float(tech_workload.get("open_estimated_hours", 0.0))
    count_penalty = 0.0 if open_ticket_count < 20 else min((open_ticket_count - 20) * 0.004, 0.10)
    hour_penalty = 0.0 if open_estimated_hours < 40 else min(((open_estimated_hours - 40) / 10.0) * 0.008, 0.06)
    return round(count_penalty + hour_penalty + 0.05, 4)


def exploration_capacity_bonus(
    tech_history: dict,
    tech_workload: dict,
    ticket: pd.Series,
    assigned_top1_count: int,
) -> float:
    if not is_new_technician(tech_history) or not is_low_risk_ticket(ticket):
        return 0.0

    if assigned_top1_count >= NEW_TECH_TOP1_CAP:
        return 0.0

    remaining_capacity_ratio = max(0.0, (NEW_TECH_TOP1_CAP - assigned_top1_count) / NEW_TECH_TOP1_CAP)
    workload_component = 0.06 * float(tech_workload.get("workload_count_score", 1.0))
    return round((0.08 + workload_component) * remaining_capacity_ratio, 4)


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


def capacity_penalty(tech_workload: dict) -> float:
    open_ticket_count = float(tech_workload.get("open_ticket_count", 0))
    open_estimated_hours = float(tech_workload.get("open_estimated_hours", 0.0))

    ticket_overage = max(0.0, open_ticket_count - SOFT_TICKET_CAP)
    hour_overage = max(0.0, open_estimated_hours - CAPACITY_HOUR_CAP)

    ticket_penalty = min(ticket_overage * 0.005, 0.15)
    hour_penalty = min((hour_overage / 10.0) * 0.01, 0.08)

    return round(ticket_penalty + hour_penalty, 4)


def distribution_penalty(tech_workload: dict, mean_open_count: float, mean_open_hours: float) -> float:
    open_ticket_count = float(tech_workload.get("open_ticket_count", 0))
    open_estimated_hours = float(tech_workload.get("open_estimated_hours", 0.0))

    count_over = max(0.0, open_ticket_count - mean_open_count)
    hours_over = max(0.0, open_estimated_hours - mean_open_hours)

    count_penalty = min(count_over * 0.01, 0.18)
    hour_penalty = min((hours_over / 8.0) * 0.01, 0.12)
    return round(count_penalty + hour_penalty, 4)


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
    text_expertise_lookup = build_text_expertise_lookup(load_nlp_matches())

    workload = build_workload_snapshot(df)
    history = build_technician_history(df)

    if not technician_pool or open_tickets.empty:
        return workload, pd.DataFrame(), {"open_ticket_count": int(len(open_tickets)), "recommendation_rows": 0}

    workload_lookup = workload.set_index("technician").to_dict(orient="index") if not workload.empty else {}
    history_lookup = history.set_index("technician").to_dict(orient="index") if not history.empty else {}
    for technician in technician_pool:
        workload_lookup.setdefault(technician, default_workload_record())
    recompute_workload_scores(workload_lookup)

    recommendation_rows = []
    new_tech_top1_counts: dict[str, int] = {}
    open_tickets = open_tickets.sort_values(
        by=["sla_weight", "priority_weight", "complexity_score"],
        ascending=[False, False, False],
        na_position="last",
    )

    for _, ticket in open_tickets.iterrows():
        scored_rows = []

        for technician in technician_pool:
            tech_history = history_lookup.get(
                technician,
                {"resolution_efficiency_score": 0.5, "avg_resolution_hours": np.nan, "completed_ticket_count": 0},
            )
            tech_workload = workload_lookup.get(technician, default_workload_record())

            skill = compute_skill_score(ticket, technician, completed)
            text_expertise = text_expertise_lookup.get(
                (str(ticket["ticket_id"]), technician),
                {
                    "bm25_text_expertise_score": 0.0,
                    "best_text_match_score": 0.0,
                    "text_match_count": 0,
                    "best_text_match_rank": np.nan,
                },
            )
            balance_score = priority_balance_score(ticket, pd.Series(tech_workload))
            sla_pressure = sla_pressure_score(ticket, pd.Series(tech_workload))
            sla_urgency_fit = sla_urgency_fit_score(ticket, skill, tech_workload)
            complexity_fit = complexity_fit_score(ticket, tech_workload)
            overload_penalty = capacity_penalty(tech_workload)
            current_mean_count = np.mean([float(row.get("open_ticket_count", 0)) for row in workload_lookup.values()])
            current_mean_hours = np.mean([float(row.get("open_estimated_hours", 0.0)) for row in workload_lookup.values()])
            fairness_penalty = distribution_penalty(tech_workload, current_mean_count, current_mean_hours)
            exploration_bonus = new_technician_bonus(tech_history, tech_workload, ticket)
            low_risk_senior_penalty = experienced_low_risk_penalty(tech_history, tech_workload, ticket)
            low_risk_new_tech_bonus = exploration_capacity_bonus(
                tech_history,
                tech_workload,
                ticket,
                new_tech_top1_counts.get(technician, 0),
            )
            onboarding_penalty = new_technician_penalty(
                tech_history,
                tech_workload,
                ticket,
                new_tech_top1_counts.get(technician, 0),
            )

            total_score = (
                TECHNICIAN_WEIGHTS["issue_type_skill"] * skill["issue_type_skill"]
                + TECHNICIAN_WEIGHTS["bm25_text_expertise"] * text_expertise["bm25_text_expertise_score"]
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
            total_score = max(
                0.0,
                total_score
                - overload_penalty
                - fairness_penalty
                - onboarding_penalty
                - low_risk_senior_penalty
                + exploration_bonus
                + low_risk_new_tech_bonus,
            )
            if high_risk_new_technician_block(tech_history, ticket):
                total_score = total_score * 0.05
            if new_technician_top1_cap_block(tech_history, new_tech_top1_counts.get(technician, 0)):
                total_score = total_score * 0.10

            rationale = []
            rationale.append(f"SLA class {ticket['sla_priority_class']}")
            if pd.notna(ticket.get("complexity_class")):
                rationale.append(f"complexity {ticket['complexity_class']}")
            if skill["issue_type_match_count"] > 0:
                rationale.append(f"{skill['issue_type_match_count']} similar issue-type tickets")
            if text_expertise["text_match_count"] > 0:
                rationale.append(
                    f"{text_expertise['text_match_count']} BM25/TF-IDF text matches"
                )
            if skill["account_match_count"] > 0:
                rationale.append(f"familiar with account {ticket['account']}")
            if tech_workload["open_ticket_count"] == 0:
                rationale.append("currently no active tickets assigned")
            else:
                rationale.append(f"{int(tech_workload['open_ticket_count'])} active tickets")
            rationale.append(f"{tech_workload['open_estimated_hours']:.1f} open estimated hours")
            if overload_penalty > 0:
                rationale.append(f"capacity penalty {overload_penalty:.2f} after workload cap")
            if fairness_penalty > 0:
                rationale.append(f"fair distribution penalty {fairness_penalty:.2f}")
            if exploration_bonus > 0:
                rationale.append("new technician exploration bonus applied")
            if low_risk_senior_penalty > 0:
                rationale.append("low-risk work redistributed away from senior technicians")
            if low_risk_new_tech_bonus > 0:
                rationale.append("low-risk onboarding bonus applied")
            if onboarding_penalty > 0 and is_high_risk_ticket(ticket):
                rationale.append("new technician penalty applied for high-risk ticket")
            elif onboarding_penalty > 0:
                rationale.append("new technician onboarding cap penalty applied")
            if high_risk_new_technician_block(tech_history, ticket):
                rationale.append("high-risk ticket kept with experienced technicians")
            if new_technician_top1_cap_block(tech_history, new_tech_top1_counts.get(technician, 0)):
                rationale.append("new technician top-1 onboarding cap reached")

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
                    "bm25_text_expertise_score": text_expertise["bm25_text_expertise_score"],
                    "best_text_match_score": text_expertise["best_text_match_score"],
                    "text_match_count": text_expertise["text_match_count"],
                    "best_text_match_rank": text_expertise["best_text_match_rank"],
                    "queue_group_skill_score": skill["queue_group_skill"],
                    "account_familiarity_score": skill["account_familiarity"],
                    "workload_hours_score": tech_workload["workload_hours_score"],
                    "workload_count_score": tech_workload["workload_count_score"],
                    "priority_balance_score": balance_score,
                    "sla_pressure_score": sla_pressure,
                    "sla_urgency_fit_score": sla_urgency_fit,
                    "complexity_fit_score": complexity_fit,
                    "capacity_penalty_score": overload_penalty,
                    "distribution_penalty_score": fairness_penalty,
                    "new_technician_bonus_score": exploration_bonus,
                    "low_risk_senior_penalty_score": low_risk_senior_penalty,
                    "low_risk_new_tech_bonus_score": low_risk_new_tech_bonus,
                    "new_technician_penalty_score": onboarding_penalty,
                    "resolution_efficiency_score": tech_history["resolution_efficiency_score"],
                    "open_ticket_count": int(tech_workload["open_ticket_count"]),
                    "open_estimated_hours": float(tech_workload["open_estimated_hours"]),
                    "historical_completed_tickets": int(tech_history["completed_ticket_count"]),
                    "avg_resolution_hours": tech_history["avg_resolution_hours"],
                    "rationale": "; ".join(rationale),
                }
            )

        ranked = sorted(scored_rows, key=lambda row: row["recommendation_score"], reverse=True)
        if ranked:
            top_choice = ranked[0]["recommended_technician"]
            apply_projected_assignment(workload_lookup, top_choice, ticket)
            top_history_count = int(
                history_lookup.get(top_choice, {"completed_ticket_count": 0}).get("completed_ticket_count", 0)
            )
            if top_history_count < NEW_TECH_COMPLETED_THRESHOLD:
                new_tech_top1_counts[top_choice] = new_tech_top1_counts.get(top_choice, 0) + 1
        for rank, row in enumerate(ranked[:3], start=1):
            row["recommendation_rank"] = rank
            recommendation_rows.append(row)

    recommendations = pd.DataFrame(recommendation_rows)

    summary = {
        "open_ticket_count": int(len(open_tickets)),
        "technician_pool_size": int(len(technician_pool)),
        "recommendation_rows": int(len(recommendations)),
        "tickets_with_recommendations": int(recommendations["ticket_id"].nunique()) if not recommendations.empty else 0,
        "uses_bm25_text_expertise": True,
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
    
