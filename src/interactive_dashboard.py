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

BRAND_COLORS = {
    "sand": "#f4efe6",
    "ink": "#22313f",
    "teal": "#2c7a7b",
    "amber": "#c89b3c",
    "coral": "#d26a4d",
    "slate": "#61717d",
    "rose": "#c94f4f",
}


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {BRAND_COLORS["sand"]} 0%, #ffffff 22%, #ffffff 100%);
        }}
        .block-container {{
            max-width: 1450px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }}
        .hero-panel {{
            background: linear-gradient(135deg, {BRAND_COLORS["teal"]}, {BRAND_COLORS["amber"]});
            color: white;
            border-radius: 22px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 14px 36px rgba(34, 49, 63, 0.18);
        }}
        .hero-panel h1 {{
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
        }}
        .hero-panel p {{
            margin: 0;
            opacity: 0.96;
            font-size: 1rem;
        }}
        div[data-testid="stMetric"] {{
            background: white;
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
            border: 1px solid rgba(44, 122, 123, 0.28);
            background: linear-gradient(135deg, rgba(44, 122, 123, 0.12), rgba(200, 155, 60, 0.18));
            color: {BRAND_COLORS["ink"]};
            font-weight: 600;
            min-height: 2.9rem;
            box-shadow: 0 10px 22px rgba(34, 49, 63, 0.08);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_round(series: pd.Series, digits: int = 2) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(digits)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = load_table_with_fallback("autotask_feature_engineered", FEATURE_PATH)
    complexity_df = load_table_with_fallback("autotask_complexity_scored", COMPLEXITY_PATH)
    recommendations_df = load_table_with_fallback("autotask_assignment_recommendations", RECOMMENDATIONS_PATH)
    time_df = load_table_with_fallback("autotask_time_estimation_open_ticket_predictions", TIME_PATH)
    nlp_df = load_table_with_fallback("autotask_ticket_similarity_summary", NLP_PATH)

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

    return feature_df, complexity_df, recommendations_df, time_df, nlp_df


def load_table_with_fallback(table_name: str, csv_path: Path) -> pd.DataFrame:
    try:
        engine = create_engine(get_db_url())
        return pd.read_sql_table(table_name, engine)
    except Exception:
        return pd.read_csv(csv_path)


def filter_active_tickets(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    active = feature_df[feature_df["is_active_ticket"] == True].copy()

    technicians = sorted([x for x in active["primary_resource"].dropna().unique().tolist()])
    priorities = sorted([x for x in active["priority"].dropna().unique().tolist()])
    sla_classes = sorted([x for x in active["sla_priority_class"].dropna().unique().tolist()])
    complexity_classes = sorted([x for x in active["complexity_class"].dropna().unique().tolist()])
    issue_types = sorted([x for x in active["issue_type"].dropna().unique().tolist()])
    filter_summary = []

    with st.popover("Filter Tickets", use_container_width=True):
        st.caption("Refine the live ticket view without changing saved outputs.")
        selected_technicians = st.multiselect("Technician", technicians)
        selected_priorities = st.multiselect("Priority", priorities)
        selected_complexity = st.multiselect("Complexity", complexity_classes)
        selected_issue_types = st.multiselect("Issue Type", issue_types)
        selected_sla = st.multiselect("SLA Class", sla_classes)

    if selected_technicians:
        active = active[active["primary_resource"].isin(selected_technicians)]
        filter_summary.append(f"{len(selected_technicians)} technician")
    if selected_priorities:
        active = active[active["priority"].isin(selected_priorities)]
        filter_summary.append(f"{len(selected_priorities)} priority")
    if selected_complexity:
        active = active[active["complexity_class"].isin(selected_complexity)]
        filter_summary.append(f"{len(selected_complexity)} complexity")
    if selected_issue_types:
        active = active[active["issue_type"].isin(selected_issue_types)]
        filter_summary.append(f"{len(selected_issue_types)} issue type")
    if selected_sla:
        active = active[active["sla_priority_class"].isin(selected_sla)]
        filter_summary.append(f"{len(selected_sla)} SLA class")

    return active, filter_summary


def build_employee_summary(feature_df: pd.DataFrame, recommendations_df: pd.DataFrame) -> pd.DataFrame:
    completed = feature_df[feature_df["resolution_hours"].notna() & feature_df["completed_by"].notna()].copy()
    active = feature_df[feature_df["is_active_ticket"] & feature_df["primary_resource"].notna()].copy()
    top_rec = recommendations_df[recommendations_df["recommendation_rank"] == 1].copy()

    completed_summary = (
        completed.groupby("completed_by")
        .agg(
            completed_tickets=("ticket_id", "count"),
            avg_complexity_score=("complexity_score", "mean"),
            unique_issue_types=("issue_type", "nunique"),
        )
        .reset_index()
        .rename(columns={"completed_by": "technician"})
    )

    active_summary = (
        active.groupby("primary_resource")
        .agg(
            open_tickets=("ticket_id", "count"),
            high_complexity_open=("complexity_class", lambda s: int((s == "High").sum())),
            critical_open=("priority", lambda s: int((s == "Critical").sum())),
        )
        .reset_index()
        .rename(columns={"primary_resource": "technician"})
    )

    recommendation_summary = (
        top_rec.groupby("recommended_technician")
        .agg(
            top_recommendations=("ticket_id", "count"),
            avg_recommendation_score=("recommendation_score", "mean"),
        )
        .reset_index()
        .rename(columns={"recommended_technician": "technician"})
    )

    summary = completed_summary.merge(active_summary, on="technician", how="outer")
    summary = summary.merge(recommendation_summary, on="technician", how="outer").fillna(0)

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
            "primary_resource",
            "priority",
            "sla_priority_class",
            "complexity_class",
            "attention_score",
        ]
    ].head(12)


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
            "current_assignee": group["current_assignee"].iloc[0] if "current_assignee" in group.columns else None,
        }

        for rank in (1, 2, 3):
            ranked = group[group["recommendation_rank"] == rank]
            if ranked.empty:
                row[f"top_{rank}_technician"] = None
                row[f"top_{rank}_score"] = None
            else:
                row[f"top_{rank}_technician"] = ranked["recommended_technician"].iloc[0]
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


def main() -> None:
    st.set_page_config(
        page_title="Autotask AI Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_theme()

    feature_df, complexity_df, recommendations_df, time_df, nlp_df = load_data()
    if "ticket_status" not in recommendations_df.columns:
        status_map = feature_df[["ticket_id", "status"]].drop_duplicates()
        recommendations_df = recommendations_df.merge(status_map, on="ticket_id", how="left")
        recommendations_df = recommendations_df.rename(columns={"status": "ticket_status"})

    hero_col, filter_col = st.columns([6.8, 1.4], vertical_alignment="top")
    with hero_col:
        st.markdown(
            """
            <div class="hero-panel">
              <h1>Autotask AI Dashboard</h1>
              <p>Interactive Phase 2 view for PostgreSQL-backed open tickets, technician recommendations, SLA pressure, complexity, and predicted effort.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with filter_col:
        st.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)
        active_filtered, filter_summary = filter_active_tickets(feature_df)

    st.caption("Primary source: PostgreSQL analytics tables. CSV files are used only as a fallback if the database is unavailable.")
    if filter_summary:
        st.caption("Active filters: " + ", ".join(filter_summary))

    employee_summary = build_employee_summary(feature_df, recommendations_df)
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
        busiest = active_filtered["primary_resource"].fillna("Unassigned").value_counts()
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
    filtered_open_ticket_count = int(len(active_filtered))
    assigned_open_ticket_count = int(active_filtered["primary_resource"].notna().sum())
    unassigned_open_ticket_count = int(active_filtered["primary_resource"].isna().sum())
    tickets_with_recommendations = int(ticket_recommendation_board["ticket_id"].nunique())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Tickets", f"{len(feature_df):,}")
    c2.metric("Completed Tickets", f"{int(feature_df['resolution_hours'].notna().sum()):,}")
    c3.metric("Open Tickets", f"{open_ticket_count:,}", delta=f"{filtered_open_ticket_count:,} after filters")
    c4.metric(
        "Assigned Open",
        f"{assigned_open_ticket_count:,}",
    )
    c5.metric(
        "Unassigned Open",
        f"{unassigned_open_ticket_count:,}",
    )
    c6.metric(
        "Open With Top-3",
        f"{tickets_with_recommendations:,}",
    )

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
                    BRAND_COLORS["teal"],
                    BRAND_COLORS["amber"],
                    BRAND_COLORS["coral"],
                    BRAND_COLORS["rose"],
                    BRAND_COLORS["slate"],
                ],
            )
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
                color_discrete_sequence=[BRAND_COLORS["teal"], BRAND_COLORS["amber"], BRAND_COLORS["coral"]],
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            priority_mix = (
                active_filtered.groupby(["primary_resource", "priority"]).size().reset_index(name="Tickets")
            )
            fig = px.bar(
                priority_mix,
                x="primary_resource",
                y="Tickets",
                color="priority",
                barmode="stack",
                title="Active Priority Mix by Technician",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Tickets")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            issue_counts = active_filtered["issue_type"].fillna("Missing").value_counts().head(10).reset_index()
            issue_counts.columns = ["Issue Type", "Tickets"]
            fig = px.bar(issue_counts, x="Issue Type", y="Tickets", title="Top Active Issue Types", color="Tickets")
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            workload = (
                active_filtered.groupby("primary_resource")
                .agg(
                    active_tickets=("ticket_id", "count"),
                    high_complexity_tickets=("complexity_class", lambda s: int((s == "High").sum())),
                )
                .reset_index()
                .sort_values("active_tickets", ascending=False)
            )
            fig = px.scatter(
                workload,
                x="active_tickets",
                y="high_complexity_tickets",
                size="active_tickets",
                color="primary_resource",
                title="Technician Workload: Ticket Count vs High-Complexity Tickets",
            )
            fig.update_layout(xaxis_title="Open Tickets", yaxis_title="High-Complexity Tickets")
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            breach_counts = (
                active_filtered["sla_breach_risk"].map({True: "Breach Risk", False: "Within SLA"}).value_counts().reset_index()
            )
            breach_counts.columns = ["Status", "Tickets"]
            fig = px.bar(
                breach_counts,
                x="Status",
                y="Tickets",
                color="Status",
                title="Current SLA Breach Risk",
                color_discrete_sequence=[BRAND_COLORS["rose"], BRAND_COLORS["teal"]],
            )
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
        employee_profiles["profile_label"] = employee_profiles["technician"].astype(str)
        employee_names = employee_profiles["profile_label"].tolist()

        if "selected_employee_profile" not in st.session_state:
            st.session_state["selected_employee_profile"] = employee_names[0] if employee_names else None

        if employee_names:
            picker_col, summary_col = st.columns([2.1, 1.2], vertical_alignment="center")
            with picker_col:
                selected_employee = st.selectbox(
                    "Employee",
                    options=employee_names,
                    index=employee_names.index(st.session_state["selected_employee_profile"])
                    if st.session_state["selected_employee_profile"] in employee_names
                    else 0,
                    label_visibility="collapsed",
                )
                st.session_state["selected_employee_profile"] = selected_employee
            with summary_col:
                dataframe_download("Download Technician Summary", employee_summary, "technician_summary.csv")

            st.markdown("### Team Members")
            profile_columns = st.columns(3)
            for idx, employee in enumerate(employee_names):
                employee_row = employee_profiles[employee_profiles["profile_label"] == employee].iloc[0]
                with profile_columns[idx % 3]:
                    if st.button(
                        f"{employee}",
                        key=f"profile_{employee}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_employee_profile"] = employee
                    st.caption(
                        f"Completed: {int(employee_row['completed_tickets'])} | "
                        f"Open: {int(employee_row['open_tickets'])} | "
                        f"Top recs: {int(employee_row['top_recommendations'])}"
                    )

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
                        "Email": "",
                        "Contact Number": "",
                        "Department": "",
                        "Location": "",
                    }
                )

            employee_active = active_filtered[active_filtered["primary_resource"] == selected_employee].copy()
            employee_completed = feature_df[
                (feature_df["resolution_hours"].notna()) & (feature_df["completed_by"] == selected_employee)
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
            summary_col1, summary_col2 = st.columns(2)
            summary_col1.metric("Tickets Worked", f"{len(employee_completed):,}")
            summary_col2.metric("Tickets In Progress", f"{len(employee_active):,}")

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
        else:
            st.info("No employee profiles are available in the current dataset.")

    with tab3:
        st.subheader("Recommendation Summary")
        dataframe_download(
            "Download Filtered Recommendations",
            filtered_top_recommendations,
            "filtered_recommendations.csv",
        )
        st.dataframe(
            filtered_top_recommendations[
                [
                    "ticket_id",
                    "ticket_title",
                    "ticket_priority",
                    "ticket_sla_priority_class",
                    "ticket_complexity_class",
                    "recommended_technician",
                    "recommendation_score",
                    "rationale",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        rec_counts = filtered_top_recommendations["recommended_technician"].value_counts().reset_index()
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
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            avg_score = (
                filtered_top_recommendations.groupby("recommended_technician")["recommendation_score"]
                .mean()
                .reset_index()
            )
            avg_score["recommendation_score"] = safe_round(avg_score["recommendation_score"])
            fig = px.bar(
                avg_score,
                x="recommended_technician",
                y="recommendation_score",
                color="recommendation_score",
                title="Average Top Recommendation Score",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Average Score")
            st.plotly_chart(fig, use_container_width=True)

        score_columns = [
            "issue_type_skill_score",
            "bm25_text_expertise_score",
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
                    marker_color=BRAND_COLORS["teal"],
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
                active_filtered["primary_resource"]
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
            fig.update_layout(xaxis_title="", yaxis_title="Open Tickets")
            st.plotly_chart(fig, use_container_width=True)

        if not filtered_recommendation_rows.empty:
            st.subheader("Recommended Technician Workload Distribution")
            rec_workload = (
                filtered_recommendation_rows.groupby(["recommended_technician", "recommendation_rank"])
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
                    x="recommended_technician",
                    y="recommended_ticket_count",
                    color="recommendation_rank",
                    barmode="stack",
                    title="Top-3 Recommendation Load by Technician",
                )
                fig.update_layout(xaxis_title="Recommended Technician", yaxis_title="Recommended Open Tickets")
                st.plotly_chart(fig, use_container_width=True)

            with rec_col2:
                top1_load = (
                    filtered_recommendation_rows[filtered_recommendation_rows["recommendation_rank"] == 1]
                    .groupby("recommended_technician")
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
                    x="recommended_technician",
                    y="top1_recommended_tickets",
                    color="avg_top1_score",
                    title="Rank-1 Recommendation Load by Technician",
                )
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
