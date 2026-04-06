from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURE_DATA_PATH = Path("data/Feature_Engineered/autotask_feature_engineered.csv")
RECOMMENDATIONS_PATH = Path("data/Recommendations/assignment_recommendations.csv")
COMPLEXITY_PATH = Path("data/Complexity/autotask_complexity_scored.csv")
DASHBOARD_DIR = Path("reports/dashboard")
DASHBOARD_PATH = DASHBOARD_DIR / "employee_ticket_dashboard.html"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = pd.read_csv(FEATURE_DATA_PATH)
    recommendation_df = pd.read_csv(RECOMMENDATIONS_PATH)
    if COMPLEXITY_PATH.exists():
        complexity_df = pd.read_csv(COMPLEXITY_PATH)[
            ["ticket_id", "complexity_score", "complexity_class", "complexity_reason"]
        ]
        feature_df = feature_df.merge(complexity_df, on="ticket_id", how="left")
    return feature_df, recommendation_df


def encode_plot_to_base64(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_bar_chart(data: pd.DataFrame, x: str, y: str, title: str, color: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=data, x=x, y=y, ax=ax, color=color)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    return encode_plot_to_base64(fig)


def build_stacked_priority_chart(priority_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    priority_df.set_index("technician")[["Low", "Medium", "High", "Critical"]].plot(
        kind="bar", stacked=True, ax=ax, colormap="viridis"
    )
    ax.set_title("Current Open Ticket Priority Mix by Technician", fontsize=13, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Tickets")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Priority", bbox_to_anchor=(1.02, 1), loc="upper left")
    return encode_plot_to_base64(fig)


def summarize_employee_stats(feature_df: pd.DataFrame, recommendation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = feature_df[feature_df["resolution_hours"].notna() & feature_df["completed_by"].notna()].copy()
    active = feature_df[feature_df["is_active_ticket"] & feature_df["primary_resource"].notna()].copy()

    completed_summary = (
        completed.groupby("completed_by")
        .agg(
            completed_tickets=("ticket_id", "count"),
            avg_resolution_hours=("resolution_hours", "mean"),
            median_resolution_hours=("resolution_hours", "median"),
            avg_first_response_minutes=("first_response_minutes", "mean"),
            critical_completed=("is_priority_critical", "sum"),
            high_completed=("is_priority_high", "sum"),
            unique_issue_types=("issue_type", "nunique"),
            avg_complexity_score=("complexity_score", "mean"),
        )
        .reset_index()
        .rename(columns={"completed_by": "technician"})
    )

    active_summary = (
        active.groupby("primary_resource")
        .agg(
            open_tickets=("ticket_id", "count"),
            open_estimated_hours=("estimated_hours_clean", lambda s: s.fillna(2.0).sum()),
            current_critical=("is_priority_critical", "sum"),
            current_high=("is_priority_high", "sum"),
            current_medium=("is_priority_medium", "sum"),
            current_low=("is_priority_low", "sum"),
            current_high_complexity=(
                "complexity_class",
                lambda s: int((s == "High").sum()),
            ),
        )
        .reset_index()
        .rename(columns={"primary_resource": "technician"})
    )

    recommendation_top = recommendation_df[recommendation_df["recommendation_rank"] == 1].copy()
    recommendation_summary = (
        recommendation_top.groupby("recommended_technician")
        .agg(
            top_recommendations=("ticket_id", "count"),
            avg_recommendation_score=("recommendation_score", "mean"),
        )
        .reset_index()
        .rename(columns={"recommended_technician": "technician"})
    )

    employee_summary = completed_summary.merge(active_summary, on="technician", how="outer")
    employee_summary = employee_summary.merge(recommendation_summary, on="technician", how="outer")
    employee_summary = employee_summary.fillna(0)

    numeric_columns = [
        "avg_resolution_hours",
        "median_resolution_hours",
        "avg_first_response_minutes",
        "open_estimated_hours",
        "avg_recommendation_score",
        "avg_complexity_score",
    ]
    for column in numeric_columns:
        if column in employee_summary.columns:
            employee_summary[column] = employee_summary[column].round(2)

    employee_summary = employee_summary.sort_values(
        ["completed_tickets", "open_tickets", "top_recommendations"], ascending=[False, False, False]
    )

    priority_mix = active_summary.copy()
    for col in ["current_low", "current_medium", "current_high", "current_critical"]:
        if col not in priority_mix.columns:
            priority_mix[col] = 0
    priority_mix = priority_mix.rename(
        columns={
            "current_low": "Low",
            "current_medium": "Medium",
            "current_high": "High",
            "current_critical": "Critical",
        }
    )

    return employee_summary, priority_mix


def build_employee_detail_sections(feature_df: pd.DataFrame, employee_summary: pd.DataFrame) -> str:
    completed = feature_df[feature_df["resolution_hours"].notna() & feature_df["completed_by"].notna()].copy()
    active = feature_df[feature_df["is_active_ticket"] & feature_df["primary_resource"].notna()].copy()

    sections = []
    for _, employee in employee_summary.iterrows():
        technician = employee["technician"]
        employee_completed = completed[completed["completed_by"] == technician]
        employee_active = active[active["primary_resource"] == technician]

        top_issue_types = (
            employee_completed["issue_type"]
            .fillna("Missing")
            .value_counts()
            .head(5)
            .rename_axis("issue_type")
            .reset_index(name="tickets")
        )
        active_tickets = employee_active[
            ["ticket_id", "title", "priority", "sla_priority_class", "complexity_class"]
        ].head(5)

        top_issue_html = top_issue_types.to_html(index=False, classes="mini-table")
        active_tickets_html = (
            active_tickets.to_html(index=False, classes="mini-table") if not active_tickets.empty else "<p>No active tickets assigned.</p>"
        )

        sections.append(
            f"""
            <section class="employee-card">
              <h3>{technician}</h3>
              <div class="employee-metrics">
                <div><strong>Completed Tickets</strong><span>{int(employee['completed_tickets'])}</span></div>
                <div><strong>Open Tickets</strong><span>{int(employee['open_tickets'])}</span></div>
                <div><strong>Avg Resolution Hours</strong><span>{employee['avg_resolution_hours']}</span></div>
                <div><strong>Top Recommendations</strong><span>{int(employee['top_recommendations'])}</span></div>
                <div><strong>Avg Complexity</strong><span>{employee['avg_complexity_score']}</span></div>
                <div><strong>High Complexity Open</strong><span>{int(employee['current_high_complexity'])}</span></div>
              </div>
              <div class="employee-panels">
                <div>
                  <h4>Top Issue Types Solved</h4>
                  {top_issue_html}
                </div>
                <div>
                  <h4>Current Active Tickets</h4>
                  {active_tickets_html}
                </div>
              </div>
            </section>
            """
        )

    return "\n".join(sections)


def generate_dashboard() -> Path:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    feature_df, recommendation_df = load_data()
    employee_summary, priority_mix = summarize_employee_stats(feature_df, recommendation_df)

    completed_chart = build_bar_chart(
        employee_summary, "technician", "completed_tickets", "Completed Tickets by Technician", "#2a9d8f"
    )
    open_chart = build_bar_chart(
        employee_summary, "technician", "open_tickets", "Current Open Tickets by Technician", "#e76f51"
    )
    resolution_chart = build_bar_chart(
        employee_summary, "technician", "avg_resolution_hours", "Average Resolution Hours by Technician", "#457b9d"
    )
    priority_chart = build_stacked_priority_chart(priority_mix)

    overall_cards = {
        "Total Tickets": int(len(feature_df)),
        "Completed Tickets": int(feature_df["resolution_hours"].notna().sum()),
        "Active Tickets": int(feature_df["is_active_ticket"].sum()),
        "Technicians": int(employee_summary["technician"].nunique()),
        "High Complexity Open": int(
            ((feature_df["is_active_ticket"]) & (feature_df["complexity_class"] == "High")).sum()
        ),
    }

    cards_html = "\n".join(
        f'<div class="stat-card"><span class="label">{label}</span><span class="value">{value}</span></div>'
        for label, value in overall_cards.items()
    )

    summary_table_html = employee_summary.to_html(index=False, classes="summary-table")
    detail_sections_html = build_employee_detail_sections(feature_df, employee_summary)

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Employee Ticket Dashboard</title>
      <style>
        body {{
          font-family: Georgia, 'Times New Roman', serif;
          margin: 0;
          background: linear-gradient(180deg, #f3f1eb 0%, #ffffff 100%);
          color: #1f2933;
        }}
        .container {{
          max-width: 1320px;
          margin: 0 auto;
          padding: 32px 24px 56px;
        }}
        h1 {{
          margin-bottom: 8px;
          font-size: 36px;
        }}
        .subtitle {{
          color: #52606d;
          margin-bottom: 28px;
          font-size: 17px;
        }}
        .stats-grid {{
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 16px;
          margin-bottom: 28px;
        }}
        .stat-card {{
          background: #fffaf2;
          border: 1px solid #eadfce;
          border-radius: 18px;
          padding: 18px 20px;
          box-shadow: 0 12px 32px rgba(63, 55, 45, 0.08);
        }}
        .stat-card .label {{
          display: block;
          color: #7b8794;
          font-size: 13px;
          margin-bottom: 8px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }}
        .stat-card .value {{
          font-size: 30px;
          font-weight: bold;
        }}
        .chart-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 18px;
          margin-bottom: 28px;
        }}
        .panel {{
          background: white;
          border-radius: 20px;
          padding: 18px;
          border: 1px solid #e5e7eb;
          box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
        }}
        .panel h2 {{
          margin-top: 0;
          font-size: 20px;
        }}
        img {{
          width: 100%;
          border-radius: 12px;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
        }}
        .summary-table, .mini-table {{
          font-size: 14px;
        }}
        th, td {{
          border-bottom: 1px solid #e5e7eb;
          padding: 10px 8px;
          text-align: left;
          vertical-align: top;
        }}
        th {{
          background: #f8fafc;
        }}
        .employee-grid {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 18px;
          margin-top: 28px;
        }}
        .employee-card {{
          background: white;
          border-radius: 20px;
          border: 1px solid #e5e7eb;
          padding: 18px;
          box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
        }}
        .employee-card h3 {{
          margin-top: 0;
          font-size: 22px;
        }}
        .employee-metrics {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
          margin-bottom: 18px;
        }}
        .employee-metrics div {{
          background: #f8fafc;
          border-radius: 14px;
          padding: 12px;
        }}
        .employee-metrics strong {{
          display: block;
          color: #52606d;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 6px;
        }}
        .employee-metrics span {{
          font-size: 22px;
          font-weight: bold;
        }}
        .employee-panels {{
          display: grid;
          grid-template-columns: 1fr;
          gap: 14px;
        }}
        @media (max-width: 980px) {{
          .stats-grid, .chart-grid, .employee-grid {{
            grid-template-columns: 1fr;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Employee Ticket Dashboard</h1>
        <p class="subtitle">Current project analytics view built from cleaned tickets, feature-engineered data, SLA-aware scoring, and recommendation outputs.</p>

        <section class="stats-grid">
          {cards_html}
        </section>

        <section class="chart-grid">
          <div class="panel">
            <h2>Completed Tickets</h2>
            <img src="data:image/png;base64,{completed_chart}" alt="Completed tickets chart" />
          </div>
          <div class="panel">
            <h2>Open Tickets</h2>
            <img src="data:image/png;base64,{open_chart}" alt="Open tickets chart" />
          </div>
          <div class="panel">
            <h2>Average Resolution Hours</h2>
            <img src="data:image/png;base64,{resolution_chart}" alt="Resolution chart" />
          </div>
          <div class="panel">
            <h2>Priority Mix</h2>
            <img src="data:image/png;base64,{priority_chart}" alt="Priority mix chart" />
          </div>
        </section>

        <section class="panel">
          <h2>Technician Summary Table</h2>
          {summary_table_html}
        </section>

        <section class="employee-grid">
          {detail_sections_html}
        </section>
      </div>
    </body>
    </html>
    """

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    return DASHBOARD_PATH


def main() -> None:
    path = generate_dashboard()
    print(f"Dashboard saved to: {path}")


if __name__ == "__main__":
    main()
