"""Regime Timeline page — full interactive implementation."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.components import (
    render_header, render_stat_card, get_regime_colour
)


def get_regime_predictions(predictor, start_date, end_date):
    try:
        with st.spinner("Running inference pipeline..."):
            predictions = predictor.predict(
                start_date=start_date,
                end_date=end_date
            )
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def plot_regime_timeline(predictions: pd.DataFrame,
                         spy_df: pd.DataFrame = None):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    regime_colours = {
        "Growth": "rgba(22,163,74,0.15)",
        "Transition": "rgba(217,119,6,0.15)",
        "Panic": "rgba(220,38,38,0.15)"
    }
    border_colours = {
        "Growth": "rgba(22,163,74,0.4)",
        "Transition": "rgba(217,119,6,0.4)",
        "Panic": "rgba(220,38,38,0.4)"
    }

    if len(predictions) > 0:
        current_regime = predictions['regime'].iloc[0]
        start = predictions['date'].iloc[0]

        for i in range(1, len(predictions)):
            row_regime = predictions['regime'].iloc[i]
            if row_regime != current_regime or \
                    i == len(predictions) - 1:
                end = predictions['date'].iloc[i]
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=regime_colours.get(
                        current_regime,
                        "rgba(100,116,139,0.15)"),
                    line=dict(
                        color=border_colours.get(
                            current_regime,
                            "rgba(100,116,139,0.4)"),
                        width=0.5),
                    row=1, col=1
                )
                current_regime = row_regime
                start = predictions['date'].iloc[i]

    if spy_df is not None and len(spy_df) > 0:
        merged = predictions.merge(
            spy_df[['Close']],
            left_on='date',
            right_index=True,
            how='left'
        )
        fig.add_trace(
            go.Scatter(
                x=merged['date'],
                y=merged['Close'],
                mode='lines',
                name='SPY Price',
                line=dict(color='#93C5FD', width=1.5),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'SPY: $%{y:.2f}<extra></extra>'
                )
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=predictions['date'],
            y=predictions['confidence'],
            mode='lines',
            name='Confidence',
            line=dict(color='#8B5CF6', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(139,92,246,0.1)',
            hovertemplate=(
                '<b>%{x}</b><br>'
                'Confidence: %{y:.1%}<extra></extra>'
            )
        ),
        row=2, col=1
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.4)',
        font=dict(
            family='Inter, sans-serif',
            color='#94A3B8',
            size=12
        ),
        legend=dict(
            bgcolor='rgba(15,23,42,0.8)',
            bordercolor='rgba(59,130,246,0.2)',
            borderwidth=1,
            font=dict(size=11)
        ),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=10, b=0),
        height=500,
        xaxis=dict(
            gridcolor='rgba(59,130,246,0.08)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(59,130,246,0.08)',
            showgrid=True,
            zeroline=False,
            title='Price ($)'
        ),
        xaxis2=dict(
            gridcolor='rgba(59,130,246,0.08)',
            showgrid=True,
            zeroline=False
        ),
        yaxis2=dict(
            gridcolor='rgba(59,130,246,0.08)',
            showgrid=True,
            zeroline=False,
            tickformat='.0%',
            title='Confidence'
        )
    )

    return fig


def plot_regime_distribution(predictions: pd.DataFrame):
    counts = predictions['regime'].value_counts()
    colours = {
        'Growth': '#16A34A',
        'Transition': '#D97706',
        'Panic': '#DC2626'
    }
    bar_colours = [
        colours.get(r, '#64748B') for r in counts.index
    ]

    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker=dict(
            color=bar_colours,
            opacity=0.7,
            line=dict(color=bar_colours, width=1.5)
        ),
        text=[
            f'{v/len(predictions):.1%}'
            for v in counts.values
        ],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=12),
        hovertemplate=(
            '<b>%{x}</b><br>Days: %{y}<extra></extra>'
        )
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.4)',
        font=dict(
            family='Inter, sans-serif',
            color='#94A3B8',
            size=12
        ),
        height=220,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(
            gridcolor='rgba(59,130,246,0.08)'
        ),
        yaxis=dict(
            gridcolor='rgba(59,130,246,0.08)',
            title='Days'
        )
    )
    return fig


def plot_transition_timeline(transitions: pd.DataFrame):
    if transitions is None or len(transitions) == 0:
        return None

    colours = {
        'Growth': '#16A34A',
        'Transition': '#D97706',
        'Panic': '#DC2626'
    }

    fig = go.Figure()

    for _, row in transitions.iterrows():
        from_colour = colours.get(
            row['from_regime'], '#64748B')
        to_colour = colours.get(
            row['to_regime'], '#64748B')

        fig.add_trace(go.Scatter(
            x=[row['date']],
            y=[0.5],
            mode='markers+text',
            marker=dict(
                size=14,
                color=to_colour,
                line=dict(color=from_colour, width=2),
                symbol='diamond'
            ),
            text=[
                f"{row['from_regime']}→{row['to_regime']}"
            ],
            textposition='top center',
            textfont=dict(size=10, color='#94A3B8'),
            hovertemplate=(
                f"<b>{row['date']}</b><br>"
                f"{row['from_regime']} → "
                f"{row['to_regime']}<br>"
                f"Previous: {row['duration_days']} days"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.4)',
        font=dict(
            family='Inter, sans-serif',
            color='#94A3B8',
            size=11
        ),
        height=120,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            gridcolor='rgba(59,130,246,0.08)'
        ),
        yaxis=dict(visible=False, range=[0, 1])
    )
    return fig


def render_regime_card(regime, confidence, since_date,
                       vix_level, p_growth,
                       p_transition, p_panic):
    colour = get_regime_colour(regime)
    icons = {
        "Growth": "🌱",
        "Transition": "⚡",
        "Panic": "🚨"
    }
    icon = icons.get(regime, "📊")

    try:
        vix_display = (
            f"{float(vix_level):.2f}"
            if vix_level is not None and not (
                isinstance(vix_level, float) and
                np.isnan(float(vix_level)))
            else "N/A"
        )
    except Exception:
        vix_display = "N/A"

    st.markdown(f"""
    <div style="background: rgba(30,41,59,0.6);
                border: 1px solid {colour}44;
                border-left: 4px solid {colour};
                border-radius: 16px; padding: 24px;
                margin-bottom: 1.5rem;">
        <div style="display: flex;
                    justify-content: space-between;
                    align-items: center;">
            <div>
                <p style="color: #64748B; font-size: 11px;
                          text-transform: uppercase;
                          letter-spacing: 0.1em;
                          margin: 0 0 8px 0;
                          font-family: Inter, sans-serif;">
                    Current Regime
                </p>
                <div style="display: flex;
                            align-items: center;
                            gap: 10px; margin-bottom: 6px;">
                    <span style="font-size: 28px;">{icon}</span>
                    <h2 style="color: {colour};
                               font-size: 36px;
                               font-weight: 800; margin: 0;
                               font-family: Inter, sans-serif;
                               letter-spacing: -0.03em;">
                        {regime}
                    </h2>
                </div>
                <p style="color: #94A3B8; font-size: 13px;
                          margin: 0;
                          font-family: Inter, sans-serif;">
                    Since {since_date} · VIX: {vix_display}
                </p>
            </div>
            <div style="text-align: right;">
                <p style="color: #64748B; font-size: 11px;
                          text-transform: uppercase;
                          letter-spacing: 0.1em;
                          margin: 0 0 4px 0;
                          font-family: Inter, sans-serif;">
                    Confidence
                </p>
                <p style="color: #F8FAFC; font-size: 40px;
                          font-weight: 700; margin: 0;
                          font-family: JetBrains Mono,
                          monospace;
                          letter-spacing: -0.02em;">
                    {confidence:.1%}
                </p>
            </div>
        </div>
        <div style="display: flex; gap: 12px;
                    margin-top: 16px; padding-top: 16px;
                    border-top: 1px solid {colour}22;">
            <div style="flex: 1;
                        background: rgba(22,163,74,0.1);
                        border: 1px solid rgba(22,163,74,0.2);
                        border-radius: 8px; padding: 10px;
                        text-align: center;">
                <p style="color: #16A34A; font-size: 18px;
                          font-weight: 700; margin: 0;
                          font-family: JetBrains Mono,
                          monospace;">
                    {p_growth:.1%}
                </p>
                <p style="color: #64748B; font-size: 11px;
                          margin: 2px 0 0 0;
                          font-family: Inter, sans-serif;">
                    Growth
                </p>
            </div>
            <div style="flex: 1;
                        background: rgba(217,119,6,0.1);
                        border: 1px solid rgba(217,119,6,0.2);
                        border-radius: 8px; padding: 10px;
                        text-align: center;">
                <p style="color: #D97706; font-size: 18px;
                          font-weight: 700; margin: 0;
                          font-family: JetBrains Mono,
                          monospace;">
                    {p_transition:.1%}
                </p>
                <p style="color: #64748B; font-size: 11px;
                          margin: 2px 0 0 0;
                          font-family: Inter, sans-serif;">
                    Transition
                </p>
            </div>
            <div style="flex: 1;
                        background: rgba(220,38,38,0.1);
                        border: 1px solid rgba(220,38,38,0.2);
                        border-radius: 8px; padding: 10px;
                        text-align: center;">
                <p style="color: #DC2626; font-size: 18px;
                          font-weight: 700; margin: 0;
                          font-family: JetBrains Mono,
                          monospace;">
                    {p_panic:.1%}
                </p>
                <p style="color: #64748B; font-size: 11px;
                          margin: 2px 0 0 0;
                          font-family: Inter, sans-serif;">
                    Panic
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render(predictor):
    render_header(
        "🌱 Regime Timeline",
        "Market state detection across 2000–2024 · "
        "LSTM → BAM → Markov pipeline"
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input(
            "Start date",
            value=pd.to_datetime("2024-01-01"),
            min_value=pd.to_datetime("2000-01-01"),
            max_value=pd.to_datetime("2024-12-31")
        )
    with col2:
        end_date = st.date_input(
            "End date",
            value=pd.to_datetime("2024-12-31"),
            min_value=pd.to_datetime("2000-01-01"),
            max_value=pd.to_datetime("2024-12-31")
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button(
            "⚡ Run Analysis",
            type="primary"
        )

    if predictor is None:
        st.error(
            "Model not loaded. Please check your setup.")
        return

    cache_key = f"{start_date}_{end_date}"
    if run_btn or \
            "predictions" not in st.session_state or \
            st.session_state.get("pred_key") != cache_key:

        predictions = get_regime_predictions(
            predictor,
            str(start_date),
            str(end_date)
        )

        if predictions is not None and \
                len(predictions) > 0:
            st.session_state.predictions = predictions
            st.session_state.pred_key = cache_key
        else:
            st.warning(
                "No predictions available for "
                "selected range.")
            return

    if "predictions" not in st.session_state:
        st.info(
            "Select a date range and click "
            "Run Analysis.")
        return

    predictions = st.session_state.predictions

    if len(predictions) == 0:
        st.warning("No data available.")
        return

    current_row = predictions.iloc[-1]
    regime = str(current_row.get('regime', 'Unknown'))
    confidence = float(
        current_row.get('confidence', 0.0))
    p_growth = float(current_row.get('p_growth', 0.0))
    p_transition = float(
        current_row.get('p_transition', 0.0))
    p_panic = float(current_row.get('p_panic', 0.0))
    vix_level = current_row.get('vix_level', 0.0)
    since_date = str(
        current_row.get('date', ''))[:10]

    render_regime_card(
        regime=regime,
        confidence=confidence,
        since_date=since_date,
        vix_level=vix_level,
        p_growth=p_growth,
        p_transition=p_transition,
        p_panic=p_panic
    )

    st.markdown("""
    <p style="color: #475569; font-size: 11px;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              margin: 0 0 12px 0;
              font-family: Inter, sans-serif;
              font-weight: 600;">
        Regime Timeline Chart
    </p>
    """, unsafe_allow_html=True)

    try:
        raw_dir = Path("data/raw")
        spy_df = None
        spy_path = raw_dir / "SPY.parquet"
        if spy_path.exists():
            spy_df = pd.read_parquet(spy_path)
            spy_df.index = pd.to_datetime(spy_df.index)
            mask = (
                (spy_df.index >= str(start_date)) &
                (spy_df.index <= str(end_date))
            )
            spy_df = spy_df[mask]
    except Exception:
        spy_df = None

    predictions['date'] = pd.to_datetime(
        predictions['date'])
    fig = plot_regime_timeline(predictions, spy_df)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={'displayModeBar': False}
    )

    st.markdown("""
    <div style="display: flex; gap: 16px;
                margin: 0 0 1.5rem 0; flex-wrap: wrap;">
        <div style="display: flex; align-items: center;
                    gap: 6px;">
            <div style="width: 12px; height: 12px;
                        border-radius: 2px;
                        background: rgba(22,163,74,0.4);">
            </div>
            <span style="color: #64748B; font-size: 12px;
                         font-family: Inter, sans-serif;">
                Growth
            </span>
        </div>
        <div style="display: flex; align-items: center;
                    gap: 6px;">
            <div style="width: 12px; height: 12px;
                        border-radius: 2px;
                        background: rgba(217,119,6,0.4);">
            </div>
            <span style="color: #64748B; font-size: 12px;
                         font-family: Inter, sans-serif;">
                Transition
            </span>
        </div>
        <div style="display: flex; align-items: center;
                    gap: 6px;">
            <div style="width: 12px; height: 12px;
                        border-radius: 2px;
                        background: rgba(220,38,38,0.4);">
            </div>
            <span style="color: #64748B; font-size: 12px;
                         font-family: Inter, sans-serif;">
                Panic
            </span>
        </div>
        <div style="display: flex; align-items: center;
                    gap: 6px;">
            <div style="width: 2px; height: 12px;
                        background: #8B5CF6;"></div>
            <span style="color: #64748B; font-size: 12px;
                         font-family: Inter, sans-serif;">
                Confidence
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    total = len(predictions)
    growth_days = len(
        predictions[predictions['regime'] == 'Growth'])
    trans_days = len(
        predictions[predictions['regime'] == 'Transition'])
    panic_days = len(
        predictions[predictions['regime'] == 'Panic'])

    with col1:
        render_stat_card(
            label="Growth",
            value=f"{growth_days} days",
            subtitle=f"{growth_days/total:.1%} of period",
            colour="#16A34A",
            icon="🌱"
        )
    with col2:
        render_stat_card(
            label="Transition",
            value=f"{trans_days} days",
            subtitle=f"{trans_days/total:.1%} of period",
            colour="#D97706",
            icon="⚡"
        )
    with col3:
        render_stat_card(
            label="Panic",
            value=f"{panic_days} days",
            subtitle=f"{panic_days/total:.1%} of period",
            colour="#DC2626",
            icon="🚨"
        )

    with st.expander(
            "📊 Regime Distribution", expanded=False):
        fig_dist = plot_regime_distribution(predictions)
        st.plotly_chart(
            fig_dist,
            use_container_width=True,
            config={'displayModeBar': False}
        )

    with st.expander(
            "🔄 Regime Transitions", expanded=False):
        try:
            from inference.predict import \
                get_regime_transitions
            transitions = get_regime_transitions(
                predictions)
            if transitions is not None and \
                    len(transitions) > 0:
                fig_trans = plot_transition_timeline(
                    transitions)
                if fig_trans:
                    st.plotly_chart(
                        fig_trans,
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )
                st.dataframe(
                    transitions,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "No regime transitions in "
                    "selected period.")
        except Exception as e:
            st.warning(
                f"Could not load transitions: {e}")

    with st.expander(
            "🔍 Raw Predictions Data", expanded=False):
        display_cols = [
            'date', 'regime', 'confidence',
            'p_growth', 'p_transition', 'p_panic'
        ]
        available = [
            c for c in display_cols
            if c in predictions.columns
        ]
        st.dataframe(
            predictions[available].tail(50),
            use_container_width=True,
            hide_index=True
        )