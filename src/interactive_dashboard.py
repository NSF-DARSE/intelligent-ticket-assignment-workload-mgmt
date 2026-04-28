from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

from load_outputs_to_postgres import get_db_url


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "data" / "Feature_Engineered" / "autotask_feature_engineered.csv"
COMPLEXITY_PATH = BASE_DIR / "data" / "Complexity" / "autotask_complexity_scored.csv"
RECOMMENDATIONS_PATH = BASE_DIR / "data" / "Recommendations" / "assignment_recommendations.csv"
TIME_PATH = BASE_DIR / "data" / "Time_Estimation" / "time_estimation_open_ticket_predictions.csv"
NLP_PATH = BASE_DIR / "data" / "NLP" / "ticket_similarity_summary.csv"
EMPLOYEE_SKILLS_PROFILE_PATH = BASE_DIR / "data" / "Feature_Engineered" / "employee_skills_profile.csv"

BRAND_COLORS = {
    "cream": "#f5f5f2",
    "ink": "#17324d",
    "blue": "#1f679e",
    "blue_dark": "#174e79",
    "orange": "#ff9d35",
    "orange_dark": "#f1871d",
    "slate": "#5d7082",
    "rose": "#d55a4d",
    "white": "#ffffff",
}


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(255, 157, 53, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(31, 103, 158, 0.16), transparent 24%),
                linear-gradient(180deg, {BRAND_COLORS["cream"]} 0%, #ffffff 18%, #ffffff 100%);
        }}
        .block-container {{
            max-width: 1450px;
            padding-top: 0.6rem;
            padding-bottom: 2.2rem;
        }}
        .hero-panel {{
            background:
                linear-gradient(135deg, {BRAND_COLORS["orange"]} 0%, {BRAND_COLORS["orange_dark"]} 46%, {BRAND_COLORS["blue"]} 100%);
            color: white;
            border-radius: 28px;
            padding: 1.7rem 1.8rem 1.45rem 1.8rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 20px 46px rgba(34, 49, 63, 0.20);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
        }}
        .hero-panel::before {{
            content: "";
            position: absolute;
            inset: auto -8% -38% auto;
            width: 280px;
            height: 280px;
            background: radial-gradient(circle, rgba(255,255,255,0.22), transparent 68%);
        }}
        .hero-panel h1 {{
            margin: 0 0 0.35rem 0;
            font-size: 2.45rem;
            line-height: 1.08;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-weight: 800;
        }}
        .hero-panel p {{
            margin: 0;
            opacity: 0.93;
            font-size: 1rem;
            max-width: 860px;
        }}
        .portal-strip {{
            display: flex;
            gap: 0.85rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }}
        .portal-badge {{
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.18);
            color: white;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 600;
            backdrop-filter: blur(3px);
        }}
        div[data-testid="stMetric"] {{
            background: {BRAND_COLORS["white"]};
            border: 1px solid rgba(97, 113, 125, 0.16);
            border-radius: 18px;
            padding: 0.75rem 0.9rem;
            box-shadow: 0 10px 26px rgba(34, 49, 63, 0.06);
        }}
        div[data-testid="stMetricLabel"] {{
            color: {BRAND_COLORS["slate"]};
        }}
        div[data-testid="stPopover"] > button {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(31, 103, 158, 0.18);
            background: {BRAND_COLORS["white"]};
            color: {BRAND_COLORS["ink"]};
            font-weight: 600;
            min-height: 3.4rem;
            box-shadow: 0 12px 24px rgba(34, 49, 63, 0.06);
            font-size: 1rem;
        }}
        div[data-testid="stPopover"] > button:hover {{
            border-color: rgba(255, 157, 53, 0.52);
            box-shadow: 0 16px 28px rgba(34, 49, 63, 0.10);
        }}
        div[data-testid="stTabs"] button[role="tab"] {{
            border-radius: 999px;
            padding: 0.45rem 1rem;
            margin-right: 0.45rem;
        }}
        div[data-testid="stTabs"] button[aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(255, 157, 53, 0.18), rgba(31, 103, 158, 0.16));
            color: {BRAND_COLORS["ink"]};
            border: 1px solid rgba(31, 103, 158, 0.18);
        }}
        div[data-testid="stDataFrame"] {{
            background: {BRAND_COLORS["white"]};
            border-radius: 18px;
            border: 1px solid rgba(97, 113, 125, 0.14);
            box-shadow: 0 12px 26px rgba(34, 49, 63, 0.05);
            padding: 0.15rem;
        }}
        div[data-testid="stPlotlyChart"] {{
            background: {BRAND_COLORS["white"]};
            border-radius: 22px;
            border: 1px solid rgba(97, 113, 125, 0.12);
            box-shadow: 0 14px 32px rgba(34, 49, 63, 0.05);
            padding: 0.45rem 0.4rem 0.2rem 0.4rem;
        }}
        div[data-testid="stInfo"] {{
            background: rgba(229, 239, 248, 0.80);
            border: 1px solid rgba(31, 103, 158, 0.14);
            border-radius: 18px;
        }}
        .employee-card {{
            background: white;
            border: 1px solid rgba(31, 103, 158, 0.12);
            border-radius: 22px;
            padding: 1rem 0.85rem 0.8rem 0.85rem;
            text-align: center;
            box-shadow: 0 10px 24px rgba(34, 49, 63, 0.05);
            margin-bottom: 0.45rem;
        }}
        .employee-card svg {{
            width: 64px;
            height: 64px;
            margin-bottom: 0.45rem;
        }}
        .employee-card-name {{
            color: #17324d;
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }}
        .employee-card-meta {{
            color: #5d7082;
            font-size: 0.84rem;
            line-height: 1.45;
        }}
        h2, h3 {{
            color: {BRAND_COLORS["ink"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_round(series: pd.Series, digits: int = 2) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(digits)


@st.cache_data(ttl=30)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = load_table_with_fallback("autotask_feature_engineered", FEATURE_PATH)
    complexity_df = load_table_with_fallback("autotask_complexity_scored", COMPLEXITY_PATH)
    recommendations_df = load_table_with_fallback("autotask_assignment_recommendations", RECOMMENDATIONS_PATH)
    time_df = load_table_with_fallback("autotask_time_estimation_open_ticket_predictions", TIME_PATH)
    nlp_df = load_table_with_fallback("autotask_ticket_similarity_summary", NLP_PATH)
    employee_skills_df = load_table_with_fallback("autotask_employee_skills_profile", EMPLOYEE_SKILLS_PROFILE_PATH)

    feature_df = feature_df.merge(
        complexity_df[["ticket_id", "complexity_score", "complexity_class", "complexity_reason"]],
        on="ticket_id",
        how="left",
    )
    feature_df = feature_df.merge(
        time_df[["ticket_id", "predicted_resolution_hours_final"]],
        on="ticket_id",
        how="left",
    )
    feature_df = feature_df.merge(
        nlp_df[["ticket_id", "top_similarity_score", "estimated_resolution_hours_nlp"]],
        on="ticket_id",
        how="left",
    )

    return feature_df, complexity_df, recommendations_df, time_df, nlp_df, employee_skills_df


def load_table_with_fallback(table_name: str, csv_path: Path) -> pd.DataFrame:
    try:
        engine = create_engine(get_db_url())
        return pd.read_sql_table(table_name, engine)
    except Exception:
        return pd.read_csv(csv_path)


def build_employee_name_lookup(employee_skills_df: pd.DataFrame) -> dict[str, str]:
    if employee_skills_df.empty:
        return {}
    required = {"technician_key", "employee_name"}
    if not required.issubset(employee_skills_df.columns):
        return {}
    lookup_df = employee_skills_df[list(required)].copy().dropna()
    lookup_df["technician_key"] = lookup_df["technician_key"].astype(str).str.strip().str.lower()
    lookup_df["employee_name"] = lookup_df["employee_name"].astype(str).str.strip()
    lookup_df = lookup_df.drop_duplicates("technician_key")
    return dict(zip(lookup_df["technician_key"], lookup_df["employee_name"]))


def format_employee_name(value: object, name_lookup: dict[str, str]) -> str:
    if pd.isna(value):
        return "Unassigned"
    key = str(value).strip().lower()
    if key in {"", "nan", "none"}:
        return "Unassigned"
    return name_lookup.get(key, str(value))


def apply_display_names(
    feature_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    employee_skills_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    name_lookup = build_employee_name_lookup(employee_skills_df)
    feature_df = feature_df.copy()
    recommendations_df = recommendations_df.copy()

    feature_df["primary_resource_display"] = feature_df["primary_resource"].map(
        lambda value: format_employee_name(value, name_lookup)
    )
    feature_df["completed_by_display"] = feature_df["completed_by"].map(
        lambda value: format_employee_name(value, name_lookup)
    )

    if "recommended_employee_name" not in recommendations_df.columns:
        recommendations_df["recommended_employee_name"] = recommendations_df["recommended_technician"].map(
            lambda value: format_employee_name(value, name_lookup)
        )
    else:
        recommendations_df["recommended_employee_name"] = recommendations_df["recommended_employee_name"].fillna(
            recommendations_df["recommended_technician"].map(lambda value: format_employee_name(value, name_lookup))
        )

    if "current_assignee" in recommendations_df.columns:
        recommendations_df["current_assignee_display"] = recommendations_df["current_assignee"].map(
            lambda value: format_employee_name(value, name_lookup)
        )

    return feature_df, recommendations_df


def build_employee_summary(
    feature_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    employee_skills_df: pd.DataFrame,
) -> pd.DataFrame:
    completed = feature_df[feature_df["resolution_hours"].notna() & feature_df["completed_by"].notna()].copy()
    active = feature_df[feature_df["is_active_ticket"] & feature_df["primary_resource"].notna()].copy()
    top_rec = recommendations_df[recommendations_df["recommendation_rank"] == 1].copy()

    completed_summary = (
        completed.groupby(["completed_by", "completed_by_display"])
        .agg(
            completed_tickets=("ticket_id", "count"),
            avg_complexity_score=("complexity_score", "mean"),
            unique_issue_types=("issue_type", "nunique"),
        )
        .reset_index()
        .rename(columns={"completed_by": "technician", "completed_by_display": "employee_name"})
    )

    active_summary = (
        active.groupby(["primary_resource", "primary_resource_display"])
        .agg(
            open_tickets=("ticket_id", "count"),
            high_complexity_open=("complexity_class", lambda s: int((s == "High").sum())),
            critical_open=("priority", lambda s: int((s == "Critical").sum())),
        )
        .reset_index()
        .rename(columns={"primary_resource": "technician", "primary_resource_display": "employee_name"})
    )

    recommendation_summary = (
        top_rec.groupby(["recommended_technician", "recommended_employee_name"])
        .agg(
            top_recommendations=("ticket_id", "count"),
            avg_recommendation_score=("recommendation_score", "mean"),
        )
        .reset_index()
        .rename(columns={"recommended_technician": "technician", "recommended_employee_name": "employee_name"})
    )

    summary = completed_summary.merge(active_summary, on=["technician", "employee_name"], how="outer")
    summary = summary.merge(recommendation_summary, on=["technician", "employee_name"], how="outer")

    if not employee_skills_df.empty:
        profile_columns = [col for col in ["technician_key", "employee_name", "role", "primary_skill_domain", "skill_count"] if col in employee_skills_df.columns]
        if {"technician_key", "employee_name"}.issubset(profile_columns):
            profiles = employee_skills_df[profile_columns].drop_duplicates("technician_key").rename(
                columns={"technician_key": "technician"}
            )
            summary = profiles.merge(summary, on=["technician", "employee_name"], how="left")

    summary = summary.fillna(0)

    for col in ["avg_complexity_score", "avg_recommendation_score"]:
        if col in summary.columns:
            summary[col] = summary[col].round(2)

    return summary.sort_values(["completed_tickets", "open_tickets"], ascending=[False, False])


def build_attention_table(active_df: pd.DataFrame) -> pd.DataFrame:
    attention = active_df.copy()
    severity_weights = {
        "Low": 1.0,
        "Medium": 2.0,
        "High": 3.0,
        "Critical": 4.0,
    }
    severity_score = attention["sla_breach_severity"].map(severity_weights).fillna(0.0)
    attention["attention_score"] = (
        severity_score * 2.0
        + attention["complexity_score"].fillna(0) * 1.3
        + attention["is_unassigned"].fillna(False).astype(int) * 1.5
        + attention["sla_breach_risk"].fillna(False).astype(int) * 1.2
    )
    attention = attention.sort_values("attention_score", ascending=False)
    return attention[
        [
            "ticket_id",
            "title",
            "primary_resource_display",
            "priority",
            "sla_priority_class",
            "complexity_class",
            "attention_score",
        ]
    ].rename(columns={"primary_resource_display": "primary_resource"}).head(12)


def build_ticket_recommendation_board(recommendations_df: pd.DataFrame) -> pd.DataFrame:
    if recommendations_df.empty:
        return pd.DataFrame(
            columns=[
                "ticket_id",
                "ticket_title",
                "ticket_status",
                "ticket_type",
                "ticket_priority",
                "current_assignee",
                "top_1_technician",
                "top_1_score",
                "top_2_technician",
                "top_2_score",
                "top_3_technician",
                "top_3_score",
            ]
        )

    rows = []
    for ticket_id, group in recommendations_df.sort_values(
        ["ticket_id", "recommendation_rank"]
    ).groupby("ticket_id", sort=False):
        group = group.sort_values("recommendation_rank")
        row = {
            "ticket_id": ticket_id,
            "ticket_title": group["ticket_title"].iloc[0],
            "ticket_status": group["ticket_status"].iloc[0] if "ticket_status" in group.columns else "Open",
            "ticket_type": group["ticket_issue_type"].iloc[0],
            "ticket_priority": group["ticket_priority"].iloc[0],
            "current_assignee": group["current_assignee_display"].iloc[0]
            if "current_assignee_display" in group.columns
            else (group["current_assignee"].iloc[0] if "current_assignee" in group.columns else None),
        }

        for rank in (1, 2, 3):
            ranked = group[group["recommendation_rank"] == rank]
            if ranked.empty:
                row[f"top_{rank}_technician"] = None
                row[f"top_{rank}_score"] = None
            else:
                row[f"top_{rank}_technician"] = ranked["recommended_employee_name"].iloc[0]
                row[f"top_{rank}_score"] = ranked["recommendation_score"].iloc[0]

        rows.append(row)

    board = pd.DataFrame(rows)
    for column in ["top_1_score", "top_2_score", "top_3_score"]:
        if column in board.columns:
            board[column] = safe_round(board[column], 4)
    if "current_assignee" in board.columns:
        board["current_assignee"] = board["current_assignee"].fillna("Unassigned")
    return board


def dataframe_download(label: str, df: pd.DataFrame, file_name: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def build_ticket_breakdown(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    issue_type_counts = (
        df["issue_type"]
        .fillna("Missing")
        .value_counts()
        .reset_index()
    )
    issue_type_counts.columns = ["Issue Type", "Tickets"]

    category_source = "issue_type_group" if "issue_type_group" in df.columns else "queue_group"
    category_counts = (
        df[category_source]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .reset_index()
    )
    category_counts.columns = ["Category", "Tickets"]
    return issue_type_counts, category_counts


def build_popup_technician_summary(df: pd.DataFrame) -> pd.DataFrame:
    working_column = "primary_resource_display" if "primary_resource_display" in df.columns else "primary_resource"
    solved_column = "completed_by_display" if "completed_by_display" in df.columns else "completed_by"

    solved_summary = (
        df[df["resolution_hours"].notna() & df["completed_by"].notna()]
        .groupby(solved_column)
        .agg(tickets_solved=("ticket_id", "count"))
        .reset_index()
        .rename(columns={solved_column: "Technician"})
    )

    working_summary = (
        df[df["is_active_ticket"] == True]
        .groupby(working_column)
        .agg(tickets_working=("ticket_id", "count"))
        .reset_index()
        .rename(columns={working_column: "Technician"})
    )

    technician_summary = solved_summary.merge(working_summary, on="Technician", how="outer").fillna(0)
    if technician_summary.empty:
        return pd.DataFrame(columns=["Technician", "Tickets Solved", "Tickets Working"])

    technician_summary["Tickets Solved"] = technician_summary["tickets_solved"].astype(int)
    technician_summary["Tickets Working"] = technician_summary["tickets_working"].astype(int)
    technician_summary["Technician"] = technician_summary["Technician"].replace({"Unassigned": pd.NA}).fillna("Unassigned")
    technician_summary = technician_summary[["Technician", "Tickets Solved", "Tickets Working"]]
    return technician_summary.sort_values(["Tickets Solved", "Tickets Working", "Technician"], ascending=[False, False, True])


def summary_breakdown_button(
    label: str,
    count_text: str,
    df: pd.DataFrame,
    key_prefix: str,
    delta_text: str | None = None,
) -> None:
    button_label = f"{label} {count_text}"

    with st.popover(button_label, use_container_width=True):
        st.metric(label, count_text, delta=delta_text)
        st.caption("Click summary buttons to inspect issue types and ticket categories for that ticket set.")
        if df.empty:
            st.info("No tickets are available for this view.")
            return

        issue_type_counts, category_counts = build_ticket_breakdown(df)
        technician_summary = build_popup_technician_summary(df)
        pop_col1, pop_col2 = st.columns(2)
        with pop_col1:
            st.markdown("**Ticket Types**")
            st.dataframe(issue_type_counts, use_container_width=True, hide_index=True)
            dataframe_download(
                f"Download {label} Ticket Types",
                issue_type_counts,
                f"{key_prefix}_ticket_types.csv",
            )
        with pop_col2:
            st.markdown("**Ticket Categories**")
            st.dataframe(category_counts, use_container_width=True, hide_index=True)
            dataframe_download(
                f"Download {label} Ticket Categories",
                category_counts,
                f"{key_prefix}_ticket_categories.csv",
            )

        st.markdown("**Technician Summary**")
        st.dataframe(technician_summary, use_container_width=True, hide_index=True)
        dataframe_download(
            f"Download {label} Technician Summary",
            technician_summary,
            f"{key_prefix}_technician_summary.csv",
        )


def build_recommendation_summary_table(recommendations_df: pd.DataFrame) -> pd.DataFrame:
    if recommendations_df.empty:
        return pd.DataFrame(columns=["Technician Name", "Recommended Tickets"])

    summary = (
        recommendations_df[recommendations_df["recommendation_rank"] == 1]
        .groupby("recommended_employee_name")
        .agg(recommended_tickets=("ticket_id", "nunique"))
        .reset_index()
        .rename(
            columns={
                "recommended_employee_name": "Technician Name",
                "recommended_tickets": "Recommended Tickets",
            }
        )
        .sort_values(["Recommended Tickets", "Technician Name"], ascending=[False, True])
    )
    return summary


def main() -> None:
    st.set_page_config(
        page_title="Autotask AI Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_theme()

    feature_df, complexity_df, recommendations_df, time_df, nlp_df, employee_skills_df = load_data()
    feature_df, recommendations_df = apply_display_names(feature_df, recommendations_df, employee_skills_df)
    if "ticket_status" not in recommendations_df.columns:
        status_map = feature_df[["ticket_id", "status"]].drop_duplicates()
        recommendations_df = recommendations_df.merge(status_map, on="ticket_id", how="left")
        recommendations_df = recommendations_df.rename(columns={"status": "ticket_status"})

    st.markdown(
        """
        <div class="hero-panel">
          <h1>AUTO TASK AI TICKET RECOMMADATION SYSTEM</h1>
          <p>Smart technician recommendation, workload balancing, and ticket intelligence in one unified operations portal.</p>
          <div class="portal-strip">
            <span class="portal-badge">Live Ticket Intelligence</span>
            <span class="portal-badge">Skills-Aware Recommendations</span>
            <span class="portal-badge">Workload Balanced Assignment</span>
            <span class="portal-badge">SLA & Complexity Tracking</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_filtered = feature_df[feature_df["is_active_ticket"] == True].copy()

    employee_summary = build_employee_summary(feature_df, recommendations_df, employee_skills_df)
    top_recommendations = recommendations_df[recommendations_df["recommendation_rank"] == 1].copy()
    filtered_top_recommendations = top_recommendations[
        top_recommendations["ticket_id"].isin(active_filtered["ticket_id"])
    ].copy()
    filtered_recommendation_rows = recommendations_df[
        recommendations_df["ticket_id"].isin(active_filtered["ticket_id"])
    ].copy()
    ticket_recommendation_board = build_ticket_recommendation_board(filtered_recommendation_rows)
    attention_table = build_attention_table(active_filtered)

    insight_col1, insight_col2, insight_col3 = st.columns(3)
    with insight_col1:
        busiest = active_filtered["primary_resource_display"].fillna("Unassigned").value_counts()
        busiest_text = busiest.index[0] if not busiest.empty else "N/A"
        st.info(f"Most open tickets assigned to: `{busiest_text}`")
    with insight_col2:
        highest_sla = active_filtered["sla_priority_class"].fillna("Missing").value_counts()
        highest_sla_text = highest_sla.index[0] if not highest_sla.empty else "N/A"
        st.info(f"Most common filtered SLA class: `{highest_sla_text}`")
    with insight_col3:
        highest_complexity = active_filtered["complexity_class"].fillna("Missing").value_counts()
        highest_complexity_text = highest_complexity.index[0] if not highest_complexity.empty else "N/A"
        st.info(f"Most common filtered complexity: `{highest_complexity_text}`")

    open_ticket_count = int(feature_df["is_active_ticket"].sum())
    assigned_open_ticket_count = int(active_filtered["primary_resource"].notna().sum())
    unassigned_open_ticket_count = int(active_filtered["primary_resource"].isna().sum())
    tickets_with_recommendations = int(ticket_recommendation_board["ticket_id"].nunique())
    completed_df = feature_df[feature_df["resolution_hours"].notna()].copy()
    assigned_open_df = active_filtered[active_filtered["primary_resource"].notna()].copy()
    unassigned_open_df = active_filtered[active_filtered["primary_resource"].isna()].copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        summary_breakdown_button("Total Tickets", f"{len(feature_df):,}", feature_df, "total_tickets")
    with c2:
        summary_breakdown_button("Completed Tickets", f"{len(completed_df):,}", completed_df, "completed_tickets")
    with c3:
        summary_breakdown_button(
            "Open Tickets",
            f"{open_ticket_count:,}",
            active_filtered,
            "open_tickets",
        )
    with c4:
        summary_breakdown_button("Assigned Open", f"{assigned_open_ticket_count:,}", assigned_open_df, "assigned_open_tickets")
    with c5:
        summary_breakdown_button("Unassigned Open", f"{unassigned_open_ticket_count:,}", unassigned_open_df, "unassigned_open_tickets")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Employees", "Recommendations", "Ticket Assignment Board"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            sla_counts = active_filtered["sla_priority_class"].fillna("Missing").value_counts().reset_index()
            sla_counts.columns = ["SLA Class", "Tickets"]
            fig = px.bar(
                sla_counts,
                x="SLA Class",
                y="Tickets",
                color="SLA Class",
                title="Active Tickets by SLA Class",
                color_discrete_sequence=[
                    BRAND_COLORS["blue"],
                    BRAND_COLORS["orange"],
                    BRAND_COLORS["rose"],
                    BRAND_COLORS["blue_dark"],
                    BRAND_COLORS["slate"],
                ],
            )
            fig.update_traces(text=sla_counts["Tickets"], textposition="inside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            complexity_counts = active_filtered["complexity_class"].fillna("Missing").value_counts().reset_index()
            complexity_counts.columns = ["Complexity", "Tickets"]
            fig = px.pie(
                complexity_counts,
                names="Complexity",
                values="Tickets",
                title="Active Ticket Complexity Mix",
                hole=0.45,
                color_discrete_sequence=[BRAND_COLORS["blue"], BRAND_COLORS["orange"], BRAND_COLORS["rose"]],
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            priority_mix = (
                active_filtered.groupby(["primary_resource_display", "priority"]).size().reset_index(name="Tickets")
            )
            fig = px.bar(
                priority_mix,
                x="primary_resource_display",
                y="Tickets",
                color="priority",
                barmode="stack",
                title="Active Priority Mix by Technician",
            )
            fig.update_traces(texttemplate="%{y}", textposition="inside")
            fig.update_layout(xaxis_title="", yaxis_title="Tickets")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            issue_counts = active_filtered["issue_type"].fillna("Missing").value_counts().head(10).reset_index()
            issue_counts.columns = ["Issue Type", "Tickets"]
            fig = px.bar(issue_counts, x="Issue Type", y="Tickets", title="Top Active Issue Types", color="Tickets")
            fig.update_traces(text=issue_counts["Tickets"], textposition="inside")
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            technician_load = (
                active_filtered["primary_resource_display"]
                .fillna("Unassigned")
                .value_counts()
                .reset_index()
                .head(10)
            )
            technician_load.columns = ["Technician", "Open Tickets"]
            fig = px.bar(
                technician_load,
                x="Technician",
                y="Open Tickets",
                color="Open Tickets",
                title="Top Technicians by Open Tickets",
                color_discrete_sequence=[BRAND_COLORS["blue"]],
            )
            fig.update_traces(text=technician_load["Open Tickets"], textposition="inside")
            fig.update_layout(xaxis_title="", yaxis_title="Open Tickets")
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            account_load = (
                active_filtered["account"]
                .fillna("Missing")
                .value_counts()
                .reset_index()
                .head(8)
            )
            account_load.columns = ["Account", "Open Tickets"]
            fig = px.bar(
                account_load,
                x="Open Tickets",
                y="Account",
                orientation="h",
                color="Open Tickets",
                title="Top Accounts with Open Tickets",
                color_discrete_sequence=[BRAND_COLORS["orange"]],
            )
            fig.update_traces(text=account_load["Open Tickets"], textposition="inside")
            fig.update_layout(xaxis_title="Open Tickets", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tickets Needing the Most Attention")
        attention_display = attention_table.copy()
        for column in ["attention_score"]:
            attention_display[column] = safe_round(attention_display[column])
        dataframe_download("Download Attention Table", attention_display, "attention_tickets.csv")
        st.dataframe(attention_display, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Employee Profiles")
        st.caption("Pick an employee to open their profile, ticket summary, completed tickets, and active tickets.")

        employee_profiles = employee_summary.copy()
        employee_profiles["profile_label"] = employee_profiles["employee_name"].astype(str)
        employee_names = employee_profiles["profile_label"].tolist()

        if "selected_employee_profile" not in st.session_state:
            st.session_state["selected_employee_profile"] = employee_names[0] if employee_names else None

        if employee_names:
            st.markdown("### Team Members")
            summary_action_col1, summary_action_col2 = st.columns([3.2, 1.1], vertical_alignment="center")
            with summary_action_col2:
                dataframe_download("Download Technician Summary", employee_summary, "technician_summary.csv")

            profile_columns = st.columns(4)
            for idx, employee in enumerate(employee_names):
                employee_row = employee_profiles[employee_profiles["profile_label"] == employee].iloc[0]
                with profile_columns[idx % 4]:
                    st.markdown(
                        f"""
                        <div class="employee-card">
                          <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                            <circle cx="50" cy="50" r="43" stroke="#3b3b3b" stroke-width="4"/>
                            <circle cx="50" cy="37" r="15" stroke="#3b3b3b" stroke-width="4"/>
                            <path d="M27 74C31 60 41 54 50 54C59 54 69 60 73 74" stroke="#3b3b3b" stroke-width="4" stroke-linecap="round"/>
                          </svg>
                          <div class="employee-card-name">{employee}</div>
                          <div class="employee-card-meta">
                            Completed: {int(employee_row['completed_tickets'])}<br/>
                            Open: {int(employee_row['open_tickets'])}<br/>
                            Top recs: {int(employee_row['top_recommendations'])}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"View {employee}",
                        key=f"profile_{employee}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_employee_profile"] = employee

            selected_employee = st.session_state["selected_employee_profile"]
            selected_profile = employee_profiles[employee_profiles["profile_label"] == selected_employee].iloc[0]

            st.markdown(f"## {selected_employee}")
            profile_metric_1, profile_metric_2, profile_metric_3 = st.columns(3)
            profile_metric_1.metric("Completed Tickets", f"{int(selected_profile['completed_tickets']):,}")
            profile_metric_2.metric("Open Tickets", f"{int(selected_profile['open_tickets']):,}")
            profile_metric_3.metric("Top Recommendations", f"{int(selected_profile['top_recommendations']):,}")

            profile_left, profile_right = st.columns([1.4, 1.1])
            with profile_left:
                st.markdown("### Employee Profile")
                st.write(
                    {
                        "Employee Name": selected_employee,
                        "Role": selected_profile["role"] if "role" in selected_profile else "",
                        "Primary Skill Domain": selected_profile["primary_skill_domain"] if "primary_skill_domain" in selected_profile else "",
                        "Email": "",
                        "Contact Number": "",
                        "Department": "",
                        "Location": "",
                    }
                )

            selected_technician = selected_profile["technician"]
            employee_active = active_filtered[active_filtered["primary_resource"] == selected_technician].copy()
            employee_completed = feature_df[
                (feature_df["resolution_hours"].notna()) & (feature_df["completed_by"] == selected_technician)
            ].copy()
            employee_recommendations = filtered_recommendation_rows[
                filtered_recommendation_rows["recommended_technician"] == selected_technician
            ].copy()

            with profile_right:
                top_issue_types = (
                    employee_completed["issue_type"].fillna("Missing").value_counts().head(8).reset_index()
                )
                top_issue_types.columns = ["Issue Type", "Tickets"]
                if not top_issue_types.empty:
                    fig = px.bar(
                        top_issue_types,
                        x="Issue Type",
                        y="Tickets",
                        color="Tickets",
                        title="Top Ticket Types Worked",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No completed ticket history is available for this employee yet.")

            st.markdown("### Ticket Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Tickets Worked", f"{len(employee_completed):,}")
            summary_col2.metric("Tickets In Progress", f"{len(employee_active):,}")
            summary_col3.metric("Overall Recommended Tickets", f"{employee_recommendations['ticket_id'].nunique():,}")

            completed_cols = [
                "ticket_id",
                "title",
                "priority",
                "issue_type",
                "sla_priority_class",
                "complexity_class",
                "account",
            ]
            active_cols = [
                "ticket_id",
                "title",
                "priority",
                "issue_type",
                "sla_priority_class",
                "complexity_class",
                "account",
            ]

            st.markdown("### Tickets Worked")
            available_completed_cols = [col for col in completed_cols if col in employee_completed.columns]
            if available_completed_cols and not employee_completed.empty:
                completed_display = employee_completed[available_completed_cols].copy()
                dataframe_download(
                    "Download Completed Tickets",
                    completed_display,
                    f"{selected_employee}_completed_tickets.csv",
                )
                st.dataframe(completed_display, use_container_width=True, hide_index=True)
            else:
                st.info("No completed tickets found for this employee.")

            st.markdown("### Tickets Currently Working On")
            available_active_cols = [col for col in active_cols if col in employee_active.columns]
            if available_active_cols and not employee_active.empty:
                active_display = employee_active[available_active_cols].copy()
                dataframe_download(
                    "Download Active Tickets",
                    active_display,
                    f"{selected_employee}_active_tickets.csv",
                )
                st.dataframe(active_display, use_container_width=True, hide_index=True)
            else:
                st.info("This employee has no active tickets in the current filtered view.")

            st.markdown("### Tickets Recommended To This Employee")
            recommendation_cols = [
                "ticket_id",
                "ticket_title",
                "ticket_priority",
                "ticket_sla_priority_class",
                "ticket_complexity_class",
                "recommendation_rank",
                "recommendation_score",
                "matched_skill_count",
                "rationale",
            ]
            available_recommendation_cols = [col for col in recommendation_cols if col in employee_recommendations.columns]
            if available_recommendation_cols and not employee_recommendations.empty:
                recommendation_display = employee_recommendations[available_recommendation_cols].copy()
                if "recommendation_score" in recommendation_display.columns:
                    recommendation_display["recommendation_score"] = safe_round(recommendation_display["recommendation_score"], 4)
                dataframe_download(
                    "Download Recommended Tickets",
                    recommendation_display,
                    f"{selected_employee}_recommended_tickets.csv",
                )
                st.dataframe(recommendation_display, use_container_width=True, hide_index=True)
            else:
                st.info("This employee does not currently have recommended tickets in the current view.")
        else:
            st.info("No employee profiles are available in the current dataset.")

    with tab3:
        st.subheader("Recommendation Summary")
        recommendation_board_view = ticket_recommendation_board[
            [
                "ticket_id",
                "ticket_type",
                "ticket_priority",
                "current_assignee",
                "top_1_technician",
                "top_2_technician",
                "top_3_technician",
            ]
        ].copy() if not ticket_recommendation_board.empty else pd.DataFrame(
            columns=[
                "ticket_id",
                "ticket_type",
                "ticket_priority",
                "current_assignee",
                "top_1_technician",
                "top_2_technician",
                "top_3_technician",
            ]
        )
        st.markdown("### Ticket-Level Recommendations")
        dataframe_download(
            "Download Ticket Recommendation Table",
            recommendation_board_view,
            "ticket_level_recommendations.csv",
        )
        st.dataframe(
            recommendation_board_view,
            use_container_width=True,
            hide_index=True,
        )

        technician_recommendation_summary = build_recommendation_summary_table(filtered_recommendation_rows)
        st.markdown("### Technician Recommendation Summary")
        dataframe_download(
            "Download Technician Recommendation Summary",
            technician_recommendation_summary,
            "technician_recommendation_summary.csv",
        )
        st.dataframe(
            technician_recommendation_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Recommendation Detail Rows")
        dataframe_download(
            "Download Filtered Recommendation Details",
            filtered_top_recommendations,
            "filtered_recommendation_details.csv",
        )
        st.dataframe(
            filtered_top_recommendations[
                [
                    "ticket_id",
                    "ticket_title",
                    "ticket_priority",
                    "ticket_sla_priority_class",
                    "ticket_complexity_class",
                    "recommended_employee_name",
                    "recommendation_score",
                    "skill_experience_score",
                    "skill_alignment_score",
                    "matched_skill_count",
                    "rationale",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        rec_counts = filtered_top_recommendations["recommended_employee_name"].value_counts().reset_index()
        rec_counts.columns = ["Technician", "Top Recommendations"]
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                rec_counts,
                x="Technician",
                y="Top Recommendations",
                color="Top Recommendations",
                title="Top Recommendation Counts",
            )
            fig.update_traces(text=rec_counts["Top Recommendations"], textposition="inside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            avg_score = (
                filtered_top_recommendations.groupby("recommended_employee_name")["recommendation_score"]
                .mean()
                .reset_index()
            )
            avg_score["recommendation_score"] = safe_round(avg_score["recommendation_score"])
            fig = px.bar(
                avg_score,
                x="recommended_employee_name",
                y="recommendation_score",
                color="recommendation_score",
                title="Average Top Recommendation Score",
            )
            fig.update_traces(text=avg_score["recommendation_score"], textposition="inside")
            fig.update_layout(xaxis_title="", yaxis_title="Average Score")
            st.plotly_chart(fig, use_container_width=True)

        score_columns = [
            "issue_type_skill_score",
            "skill_experience_score",
            "skill_alignment_score",
            "skill_coverage_score",
            "skill_domain_match_score",
            "bm25_text_expertise_score",
            "embedding_text_expertise_score",
            "queue_group_skill_score",
            "account_familiarity_score",
            "priority_balance_score",
            "sla_pressure_score",
            "sla_urgency_fit_score",
            "complexity_fit_score",
            "resolution_efficiency_score",
        ]
        available_scores = [col for col in score_columns if col in filtered_top_recommendations.columns]
        if available_scores:
            averages = filtered_top_recommendations[available_scores].mean().sort_values(ascending=True)
            fig = go.Figure(
                go.Bar(
                    x=averages.values.round(3),
                    y=averages.index,
                    orientation="h",
                    marker_color=BRAND_COLORS["blue"],
                    text=averages.values.round(3),
                    textposition="inside",
                )
            )
            fig.update_layout(
                title="Average Contribution of Recommendation Components",
                xaxis_title="Average Score",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Open Ticket Recommendation Board")
        st.caption("One row per open ticket showing the current assigned technician and the top 3 recommended technicians from the PostgreSQL-backed recommendation output.")
        board_kpi1, board_kpi2, board_kpi3, board_kpi4 = st.columns(4)
        board_kpi1.metric("Open Tickets in View", f"{len(active_filtered):,}")
        board_kpi2.metric("Currently Assigned", f"{assigned_open_ticket_count:,}")
        board_kpi3.metric("Currently Unassigned", f"{unassigned_open_ticket_count:,}")
        board_kpi4.metric("Tickets With Top-3", f"{tickets_with_recommendations:,}")

        if not active_filtered.empty:
            assignee_summary = (
                active_filtered["primary_resource_display"]
                .fillna("Unassigned")
                .value_counts()
                .reset_index()
            )
            assignee_summary.columns = ["Current Assignee", "Open Tickets"]
            fig = px.bar(
                assignee_summary,
                x="Current Assignee",
                y="Open Tickets",
                color="Open Tickets",
                title="Open Tickets by Current Assigned Technician",
            )
            fig.update_traces(text=assignee_summary["Open Tickets"], textposition="inside")
            fig.update_layout(xaxis_title="", yaxis_title="Open Tickets")
            st.plotly_chart(fig, use_container_width=True)

        if not filtered_recommendation_rows.empty:
            st.subheader("Recommended Technician Workload Distribution")
            rec_workload = (
                filtered_recommendation_rows.groupby(["recommended_employee_name", "recommendation_rank"])
                .agg(
                    recommended_ticket_count=("ticket_id", "nunique"),
                    avg_recommendation_score=("recommendation_score", "mean"),
                )
                .reset_index()
            )
            rec_workload["recommendation_rank"] = "Rank " + rec_workload["recommendation_rank"].astype(str)
            rec_workload["avg_recommendation_score"] = safe_round(rec_workload["avg_recommendation_score"], 4)

            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                fig = px.bar(
                    rec_workload,
                    x="recommended_employee_name",
                    y="recommended_ticket_count",
                    color="recommendation_rank",
                    barmode="stack",
                    title="Top-3 Recommendation Load by Technician",
                )
                fig.update_traces(texttemplate="%{y}", textposition="inside")
                fig.update_layout(xaxis_title="Recommended Technician", yaxis_title="Recommended Open Tickets")
                st.plotly_chart(fig, use_container_width=True)

            with rec_col2:
                top1_load = (
                    filtered_recommendation_rows[filtered_recommendation_rows["recommendation_rank"] == 1]
                    .groupby("recommended_employee_name")
                    .agg(
                        top1_recommended_tickets=("ticket_id", "nunique"),
                        avg_top1_score=("recommendation_score", "mean"),
                    )
                    .reset_index()
                    .sort_values("top1_recommended_tickets", ascending=False)
                )
                top1_load["avg_top1_score"] = safe_round(top1_load["avg_top1_score"], 4)
                fig = px.bar(
                    top1_load,
                    x="recommended_employee_name",
                    y="top1_recommended_tickets",
                    color="avg_top1_score",
                    title="Rank-1 Recommendation Load by Technician",
                )
                fig.update_traces(text=top1_load["top1_recommended_tickets"], textposition="inside")
                fig.update_layout(xaxis_title="Recommended Technician", yaxis_title="Rank-1 Open Tickets")
                st.plotly_chart(fig, use_container_width=True)

        dataframe_download(
            "Download Ticket Recommendation Board",
            ticket_recommendation_board,
            "ticket_recommendation_board.csv",
        )
        board_display = ticket_recommendation_board.sort_values(
            ["ticket_priority", "ticket_id"],
            ascending=[True, True],
        )
        st.dataframe(board_display, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
