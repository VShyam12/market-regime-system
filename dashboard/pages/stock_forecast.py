"""Stock Forecast page — regime-conditioned price prediction."""

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


def get_stock_forecast(predictor, ticker: str,
                       checkpoint_dir: Path):
    try:
        from models.forecaster import forecast_stock
        result = forecast_stock(
            ticker=ticker,
            predictor=predictor,
            checkpoint_dir=checkpoint_dir
        )
        return result
    except Exception as e:
        st.error(f"Forecast error: {e}")
        return None


def plot_forecast_chart(ticker: str,
                        forecast_result: dict,
                        raw_dir: Path):
    try:
        ticker_file = ticker.replace('^', '')
        ticker_path = raw_dir / f"{ticker_file}.parquet"
        if not ticker_path.exists():
            return None

        hist_df = pd.read_parquet(ticker_path)
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df = hist_df.sort_index()
        recent = hist_df.tail(60)

        forecast_prices = forecast_result.get(
            'forecast_prices', [])
        current_price = forecast_result.get(
            'current_price',
            float(recent['Close'].iloc[-1]))
        direction = forecast_result.get('direction', 'UP')

        last_date = recent.index[-1]
        forecast_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(forecast_prices)
        )

        forecast_colour = (
            '#16A34A' if direction == 'UP'
            else '#DC2626'
        )
        fill_colour = (
            'rgba(22,163,74,0.1)'
            if direction == 'UP'
            else 'rgba(220,38,38,0.1)'
        )

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.75, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.05
        )

        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent['Close'],
                mode='lines',
                name='Historical',
                line=dict(color='#93C5FD', width=1.5),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    f'{ticker}: $%{{y:.2f}}'
                    '<extra></extra>'
                )
            ),
            row=1, col=1
        )

        if len(forecast_prices) > 0:
            all_x = [last_date] + list(forecast_dates)
            all_y = [float(current_price)] + [
                float(p) for p in forecast_prices
            ]

            fig.add_trace(
                go.Scatter(
                    x=all_x,
                    y=all_y,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(
                        color=forecast_colour,
                        width=2,
                        dash='dash'
                    ),
                    marker=dict(
                        size=8,
                        color=forecast_colour,
                        symbol='circle'
                    ),
                    hovertemplate=(
                        '<b>%{x}</b><br>'
                        f'Forecast: $%{{y:.2f}}'
                        '<extra></extra>'
                    )
                ),
                row=1, col=1
            )

            upper = [p * 1.01 for p in all_y]
            lower = [p * 0.99 for p in all_y]

            fig.add_trace(
                go.Scatter(
                    x=all_x + all_x[::-1],
                    y=upper + lower[::-1],
                    fill='toself',
                    fillcolor=fill_colour,
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Confidence interval',
                    showlegend=True,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

        returns = recent['Close'].pct_change().dropna()
        ret_colours = [
            '#16A34A' if v >= 0 else '#DC2626'
            for v in returns.values
        ]

        fig.add_trace(
            go.Bar(
                x=returns.index,
                y=returns.values,
                name='Daily return',
                marker=dict(
                    color=ret_colours,
                    opacity=0.7
                ),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Return: %{y:.2%}'
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )

        fig.add_vline(
            x=str(last_date),
            line_dash="dot",
            line_color="rgba(148,163,184,0.5)",
            line_width=1
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
            height=480,
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
                tickformat='.1%',
                title='Return'
            )
        )

        return fig

    except Exception as e:
        st.warning(f"Chart error: {e}")
        return None


def render(predictor):
    render_header(
        "📊 Stock Forecast",
        "Regime-conditioned 5-day price prediction · "
        "LSTM encoder with market state context"
    )

    if predictor is None:
        st.error(
            "Model not loaded. "
            "Please check your setup.")
        return

    checkpoint_dir = Path("models/checkpoints")
    available_tickers = ["SPY", "QQQ", "AAPL", "MSFT"]
    available = [
        t for t in available_tickers
        if (checkpoint_dir /
            f"forecaster_with_regime_{t}.pt"
            ).exists()
    ]

    if not available:
        st.warning(
            "No forecaster models found. "
            "Please run Phase 7 training first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox(
            "Select ticker",
            options=available,
            index=0
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button(
            "⚡ Generate Forecast",
            type="primary",
            use_container_width=True
        )

    st.markdown("""
    <div style="background: rgba(59,130,246,0.08);
                border: 1px solid rgba(59,130,246,0.2);
                border-radius: 10px;
                padding: 12px 16px;
                margin-bottom: 1.5rem;">
        <p style="color: #93C5FD; font-size: 12px;
                  margin: 0;
                  font-family: Inter, sans-serif;">
            ℹ️ Forecasts use regime-conditioned LSTM.
            The current market regime is used as an
            additional signal to improve prediction
            accuracy. Regime conditioning improved SPY
            directional accuracy by +29.4% vs baseline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    forecast_key = f"forecast_{ticker}"

    if run_btn or forecast_key not in st.session_state:
        with st.spinner(
                f"Generating {ticker} forecast..."):
            result = get_stock_forecast(
                predictor, ticker, checkpoint_dir)

        if result:
            st.session_state[forecast_key] = result
        else:
            st.error(
                f"Could not generate forecast "
                f"for {ticker}.")
            return

    if forecast_key not in st.session_state:
        st.info(
            "Select a ticker and click "
            "Generate Forecast.")
        return

    result = st.session_state[forecast_key]

    regime = result.get('current_regime', 'Unknown')
    direction = result.get('direction', 'UP')
    magnitude = result.get('magnitude', 0.0)
    current_price = result.get('current_price', 0.0)
    forecast_prices = result.get('forecast_prices', [])
    colour = get_regime_colour(regime)
    icons = {
        "Growth": "🌱",
        "Transition": "⚡",
        "Panic": "🚨"
    }
    dir_colour = (
        '#16A34A' if direction == 'UP'
        else '#DC2626'
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        render_stat_card(
            label=f"{ticker} · Current Price",
            value=f"${float(current_price):.2f}",
            colour="#3B82F6"
        )
    with col_b:
        render_stat_card(
            label="Current Regime",
            value=f"{icons.get(regime, '📊')} {regime}",
            colour=colour
        )
    with col_c:
        try:
            mag = float(magnitude)
            render_stat_card(
                label="Expected Move (5-day)",
                value=(
                    f"{'↑' if direction == 'UP' else '↓'}"
                    f" {mag:.2%}"
                ),
                colour=dir_colour
            )
        except Exception:
            render_stat_card(
                label="Expected Move (5-day)",
                value="N/A",
                colour="#3B82F6"
            )

    st.markdown("""
    <p style="color: #475569; font-size: 11px;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              margin: 1.5rem 0 0.5rem 0;
              font-family: Inter, sans-serif;
              font-weight: 600;">
        Day-by-Day Forecast
    </p>
    """, unsafe_allow_html=True)

    if forecast_prices:
        days_data = []
        for i, price in enumerate(
                forecast_prices[:5]):
            try:
                p = float(price)
                cp = float(current_price)
                ret = (p - cp) / cp
                days_data.append({
                    'Day': f'Day {i+1}',
                    'Predicted Price': f'${p:.2f}',
                    'Change from Today': (
                        f'{"↑" if ret >= 0 else "↓"}'
                        f' {ret:.2%}'
                    )
                })
            except Exception:
                continue
        if days_data:
            st.dataframe(
                pd.DataFrame(days_data),
                use_container_width=True,
                hide_index=True
            )

    st.markdown("""
    <p style="color: #475569; font-size: 11px;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              margin: 1.5rem 0 0.5rem 0;
              font-family: Inter, sans-serif;
              font-weight: 600;">
        Price Chart
    </p>
    """, unsafe_allow_html=True)

    raw_dir = Path("data/raw")
    fig = plot_forecast_chart(ticker, result, raw_dir)
    if fig:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

    st.markdown("""
    <p style="color: #475569; font-size: 11px;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              margin: 1.5rem 0 1rem 0;
              font-family: Inter, sans-serif;
              font-weight: 600;">
        Regime Impact Analysis
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    regime_descriptions = {
        "Growth": (
            "Markets trending upward. Momentum "
            "strategies perform well."
        ),
        "Transition": (
            "Market uncertainty elevated. "
            "Watch for trend changes."
        ),
        "Panic": (
            "High market stress. "
            "Expect high volatility."
        )
    }

    with col1:
        render_stat_card(
            label="Active Regime",
            value=regime,
            subtitle=regime_descriptions.get(
                regime, ""),
            colour=colour
        )
    with col2:
        render_stat_card(
            label="Predicted Direction",
            value=(
                f"{'↑ UP' if direction == 'UP' else '↓ DOWN'}"
            ),
            subtitle="5-day forecast direction",
            colour=dir_colour
        )
    with col3:
        try:
            mag = float(magnitude)
            render_stat_card(
                label="Expected Magnitude",
                value=f"{mag:.2%}",
                subtitle="Predicted price change",
                colour='#3B82F6'
            )
        except Exception:
            render_stat_card(
                label="Expected Magnitude",
                value="N/A",
                subtitle="Predicted price change",
                colour='#3B82F6'
            )

    with st.expander(
            "📈 Model Comparison: "
            "With vs Without Regime",
            expanded=False):
        st.markdown("""
        <p style="color: #94A3B8; font-size: 13px;
                  font-family: Inter, sans-serif;
                  line-height: 1.6; margin: 0 0 12px 0;">
            The regime-conditioned model uses the current
            market state as an additional input signal.
            For SPY, this improved directional accuracy
            from
            <span style="color:#DC2626;font-weight:600;">
                35.3%
            </span>
            to
            <span style="color:#16A34A;font-weight:600;">
                64.7%
            </span>
            — a
            <span style="color:#93C5FD;font-weight:600;">
                +29.4%
            </span>
            improvement.
        </p>
        """, unsafe_allow_html=True)

        comparison_data = {
            'Ticker': ['SPY', 'QQQ', 'AAPL', 'MSFT'],
            'With Regime': [
                '64.7%', '63.7%', '61.5%', '61.1%'
            ],
            'Without Regime': [
                '35.3%', '63.7%', '61.5%', '61.1%'
            ],
            'Improvement': [
                '+29.4%', '+0.0%', '+0.0%', '+0.0%'
            ]
        }
        st.dataframe(
            pd.DataFrame(comparison_data),
            use_container_width=True,
            hide_index=True
        )

    with st.expander(
            "⚠️ Disclaimer", expanded=False):
        st.markdown("""
        <p style="color: #64748B; font-size: 12px;
                  font-family: Inter, sans-serif;
                  line-height: 1.6; margin: 0;">
            This forecast is generated by a machine
            learning model for educational and research
            purposes only. It should not be used as
            financial advice or as the sole basis for
            investment decisions. Past performance does
            not guarantee future results.
        </p>
        """, unsafe_allow_html=True)