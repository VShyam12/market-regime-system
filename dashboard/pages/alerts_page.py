"""Alerts page for regime change events and market notifications."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.components import render_header, render_alert_card, get_regime_colour


def load_saved_alerts(output_dir: Path) -> list:
    """Load saved alerts from inference/alerts.json if it exists."""
    alerts_path = Path(output_dir) / "alerts.json"
    if not alerts_path.exists():
        return []

    try:
        with open(alerts_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _generate_alert_payload(start_date: str, end_date: str) -> list:
    """Cached helper to avoid repeated alert generation in session."""
    from inference.alerts import AlertGenerator
    from inference.predict import MarketRegimePredictor

    predictor = MarketRegimePredictor()
    predictions = predictor.predict(start_date=start_date, end_date=end_date)
    generator = AlertGenerator()
    alerts = generator.generate_all_alerts(predictions)
    return [alert.to_dict() for alert in alerts]


def generate_fresh_alerts(predictor, start_date: str, end_date: str) -> list:
    """Generate fresh alerts from the live predictor for the chosen date range."""
    try:
        predictions = predictor.predict(start_date=start_date, end_date=end_date)

        from inference.alerts import AlertGenerator

        generator = AlertGenerator()
        alerts = generator.generate_all_alerts(predictions)
        return [alert.to_dict() for alert in alerts]
    except Exception as exc:
        st.error(f"Failed to generate alerts: {exc}")
        return []


def render_alert_summary(alerts: list) -> None:
    """Render a four-card alert summary for HIGH, MEDIUM, LOW, and TOTAL counts."""
    high_count = sum(1 for alert in alerts if alert.get("priority") == "HIGH")
    medium_count = sum(1 for alert in alerts if alert.get("priority") == "MEDIUM")
    low_count = sum(1 for alert in alerts if alert.get("priority") == "LOW")
    total_count = len(alerts)

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0 20px 0;">
            <div style="background: #0F172A; border: 1px solid #7F1D1D; border-radius: 14px; padding: 16px 18px;">
                <div style="font-size: 2rem; font-weight: 700; color: #DC2626; line-height: 1;">{high_count}</div>
                <div style="color: #E5E7EB; font-size: 0.9rem; margin-top: 6px;">HIGH</div>
            </div>
            <div style="background: #0F172A; border: 1px solid #78350F; border-radius: 14px; padding: 16px 18px;">
                <div style="font-size: 2rem; font-weight: 700; color: #D97706; line-height: 1;">{medium_count}</div>
                <div style="color: #E5E7EB; font-size: 0.9rem; margin-top: 6px;">MEDIUM</div>
            </div>
            <div style="background: #0F172A; border: 1px solid #14532D; border-radius: 14px; padding: 16px 18px;">
                <div style="font-size: 2rem; font-weight: 700; color: #16A34A; line-height: 1;">{low_count}</div>
                <div style="color: #E5E7EB; font-size: 0.9rem; margin-top: 6px;">LOW</div>
            </div>
            <div style="background: #0F172A; border: 1px solid #1D4ED8; border-radius: 14px; padding: 16px 18px;">
                <div style="font-size: 2rem; font-weight: 700; color: #3B82F6; line-height: 1;">{total_count}</div>
                <div style="color: #E5E7EB; font-size: 0.9rem; margin-top: 6px;">TOTAL</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_alert_timeline(alerts: list):
    """Create a timeline of alerts by priority using Plotly."""
    import plotly.graph_objects as go

    if not alerts:
        return None

    df = pd.DataFrame(alerts).copy()
    if df.empty or "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return None

    priority_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    colors = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"}

    df["priority_level"] = df["priority"].map(priority_order).fillna(0).astype(int)
    df["priority_label"] = df["priority"].fillna("LOW")
    df["title"] = df["title"].fillna("")

    fig = go.Figure()
    for priority in ["LOW", "MEDIUM", "HIGH"]:
        subset = df[df["priority_label"] == priority]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=[priority] * len(subset),
                mode="markers",
                marker=dict(symbol="diamond", size=12, color=colors[priority]),
                name=priority,
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Title: %{customdata}<extra></extra>",
                customdata=subset["title"],
            )
        )

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(
            title="Priority",
            categoryorder="array",
            categoryarray=["LOW", "MEDIUM", "HIGH"],
            gridcolor="rgba(148,163,184,0.15)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    return fig


def _sort_alerts(alerts: list, sort_choice: str) -> list:
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    sorted_alerts = list(alerts)
    
    if sort_choice == "Most recent first":
        sorted_alerts.sort(key=lambda a: pd.to_datetime(a.get("date", "1970-01-01")), reverse=True)
    elif sort_choice == "Oldest first":
        sorted_alerts.sort(key=lambda a: pd.to_datetime(a.get("date", "1970-01-01")))
    elif sort_choice == "Priority (High first)":
        sorted_alerts.sort(
            key=lambda a: (
                priority_rank.get(a.get("priority", "LOW"), 99),
                pd.to_datetime(a.get("date", "1970-01-01")),
            ),
            reverse=False,
        )
    return sorted_alerts


def render(predictor):
    """Render the Alerts page."""
    render_header("🔔 Alerts", "Regime change events and market notifications")

    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "alerts_loaded" not in st.session_state:
        st.session_state.alerts_loaded = False

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.Timestamp("2024-01-01").date(), min_value=pd.to_datetime("2000-01-01"))
    with col2:
        end_date = st.date_input("End date", value=pd.Timestamp("2024-12-31").date())

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        load_clicked = st.button("📂 Load Saved", use_container_width=True)
    with btn_col2:
        generate_clicked = st.button("⚡ Generate New", type="primary", use_container_width=True)

    if load_clicked:
        st.session_state.alerts = load_saved_alerts(Path("inference"))
        st.session_state.alerts_loaded = True
        st.success(f"Loaded {len(st.session_state.alerts)} saved alerts.")

    if generate_clicked:
        st.session_state.alerts = generate_fresh_alerts(
            predictor,
            str(start_date),
            str(end_date),
        )
        st.session_state.alerts_loaded = True
        st.success(f"Generated {len(st.session_state.alerts)} fresh alerts.")

    if not st.session_state.alerts_loaded and not st.session_state.alerts:
        st.session_state.alerts = load_saved_alerts(Path("inference"))
        if st.session_state.alerts:
            st.session_state.alerts_loaded = True

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

    if not st.session_state.alerts:
        st.info("No alerts available. Load saved alerts or generate new alerts for the selected date range.")
        return

    priority_options = ["HIGH", "MEDIUM", "LOW"]
    selected_priorities = st.multiselect(
        "Priority filter",
        options=priority_options,
        default=priority_options,
    )

    sort_choice = st.selectbox(
        "Sort alerts",
        ["Most recent first", "Oldest first", "Priority (High first)"],
        index=0,
    )

    filtered_alerts = [a for a in st.session_state.alerts if a.get("priority") in selected_priorities]
    filtered_alerts = _sort_alerts(filtered_alerts, sort_choice)

    if not filtered_alerts:
        st.info("No alerts match the selected filters.")
        return

    render_alert_summary(filtered_alerts)

    timeline_fig = plot_alert_timeline(filtered_alerts)
    if timeline_fig is not None:
        st.plotly_chart(
            timeline_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("### Alert Feed", unsafe_allow_html=True)
    for alert in filtered_alerts:
        render_alert_card(alert)

    export_df = pd.DataFrame(filtered_alerts)

    with st.expander("Export alerts", expanded=False):
        st.dataframe(export_df, use_container_width=True, hide_index=True)
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="alerts_export.csv",
            mime="text/csv",
        )
