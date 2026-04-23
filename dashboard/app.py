"""Main Streamlit dashboard for Market Regime Detection System."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Market Regime Detection System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    [data-testid="stSidebarNav"] { display: none; }

    .stApp {
        background: linear-gradient(135deg, #020817 0%, #0a1628 50%, #020817 100%);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(59, 130, 246, 0.15);
        backdrop-filter: blur(20px);
    }

    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid transparent;
        color: #64748B;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        margin-bottom: 4px;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
        color: #93C5FD;
        transform: translateX(4px);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
        border-bottom: 1px solid rgba(59, 130, 246, 0.1);
    }

    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 10px;
        padding: 4px;
        border: 1px solid rgba(59, 130, 246, 0.15);
    }

    .stTabs [data-baseweb="tab"] {
        color: #64748B;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
    }

    .stSelectbox select, .stTextInput input {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }

    ::-webkit-scrollbar {
        width: 4px;
        height: 4px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(59, 130, 246, 0.3);
        border-radius: 2px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(59, 130, 246, 0.5);
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        background: rgba(30, 41, 59, 0.6);
    }

    .regime-growth {
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.15), rgba(22, 163, 74, 0.05));
        border: 1px solid rgba(22, 163, 74, 0.3);
        border-radius: 16px;
        padding: 20px;
    }

    .regime-transition {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.15), rgba(217, 119, 6, 0.05));
        border: 1px solid rgba(217, 119, 6, 0.3);
        border-radius: 16px;
        padding: 20px;
    }

    .regime-panic {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(220, 38, 38, 0.05));
        border: 1px solid rgba(220, 38, 38, 0.3);
        border-radius: 16px;
        padding: 20px;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #16A34A;
        animation: pulse 2s infinite;
        margin-right: 6px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(22, 163, 74, 0); }
        100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
    }

    .nav-active button {
        background: rgba(59, 130, 246, 0.15) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
        color: #93C5FD !important;
    }

    .gradient-text {
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stDataFrame {
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
        overflow: hidden;
    }

    .stAlert {
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initialising market intelligence engine...")
def load_predictor():
    try:
        from inference.predict import MarketRegimePredictor
        predictor = MarketRegimePredictor()
        return predictor
    except Exception as e:
        return None


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem 0.5rem 1rem 0.5rem;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <div style="width: 36px; height: 36px;
                            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
                            border-radius: 10px; display: flex;
                            align-items: center; justify-content: center;
                            font-size: 18px;">📈</div>
                <div>
                    <h2 style="color: #F8FAFC; font-size: 18px;
                               font-weight: 700; margin: 0;
                               font-family: Inter, sans-serif;">RegimeAI</h2>
                    <p style="color: #475569; font-size: 11px;
                              margin: 0; font-family: Inter, sans-serif;">
                        Market Intelligence System
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="height: 1px;
                    background: linear-gradient(90deg, transparent,
                    rgba(59,130,246,0.3), transparent);
                    margin: 0 0 1rem 0;"></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style="color: #334155; font-size: 10px;
                  text-transform: uppercase; letter-spacing: 0.1em;
                  margin: 0 0 8px 8px; font-family: Inter, sans-serif;
                  font-weight: 600;">Navigation</p>
        """, unsafe_allow_html=True)

        pages = [
            ("🌱  Regime Timeline", "Regime Timeline",
             "Detect market states"),
            ("📊  Stock Forecast", "Stock Forecast",
             "5-day price prediction"),
            ("🔔  Alerts", "Alerts",
             "Regime change events"),
            ("🧠  Model Intelligence", "Model Intelligence",
             "System internals"),
        ]

        for label, page_key, desc in pages:
            is_active = st.session_state.get(
                "current_page") == page_key
            container_style = """
                background: rgba(59,130,246,0.12);
                border: 1px solid rgba(59,130,246,0.3);
                border-radius: 10px; margin-bottom: 4px;
            """ if is_active else "margin-bottom: 4px;"

            st.markdown(
                f'<div style="{container_style}">',
                unsafe_allow_html=True)
            if st.button(label, key=f"nav_{page_key}",
                        use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="height: 1px;
                    background: linear-gradient(90deg, transparent,
                    rgba(59,130,246,0.3), transparent);
                    margin: 1rem 0;"></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style="color: #334155; font-size: 10px;
                  text-transform: uppercase; letter-spacing: 0.1em;
                  margin: 0 0 12px 8px; font-family: Inter, sans-serif;
                  font-weight: 600;">System Status</p>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="padding: 0 8px;">
            <div style="display: flex; align-items: center;
                        margin-bottom: 8px;">
                <span class="pulse-dot"></span>
                <span style="color: #94A3B8; font-size: 12px;
                             font-family: Inter, sans-serif;">
                    Model Active
                </span>
            </div>
            <div style="background: rgba(15,23,42,0.6);
                        border: 1px solid rgba(59,130,246,0.1);
                        border-radius: 8px; padding: 10px 12px;
                        margin-bottom: 8px;">
                <p style="color: #475569; font-size: 10px;
                          text-transform: uppercase;
                          letter-spacing: 0.05em; margin: 0 0 4px 0;
                          font-family: Inter, sans-serif;">
                    Data Coverage
                </p>
                <p style="color: #93C5FD; font-size: 12px;
                          font-family: JetBrains Mono, monospace;
                          font-weight: 500; margin: 0;">
                    2000 → 2024
                </p>
            </div>
            <div style="background: rgba(15,23,42,0.6);
                        border: 1px solid rgba(59,130,246,0.1);
                        border-radius: 8px; padding: 10px 12px;">
                <p style="color: #475569; font-size: 10px;
                          text-transform: uppercase;
                          letter-spacing: 0.05em; margin: 0 0 4px 0;
                          font-family: Inter, sans-serif;">
                    Architecture
                </p>
                <p style="color: #93C5FD; font-size: 12px;
                          font-family: JetBrains Mono, monospace;
                          font-weight: 500; margin: 0;">
                    LSTM → BAM → Markov
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: 2rem; padding: 0 8px;">
            <p style="color: #1E293B; font-size: 11px;
                      text-align: center; margin: 0;
                      font-family: Inter, sans-serif;">
                v1.0 · Built with PyTorch
            </p>
        </div>
        """, unsafe_allow_html=True)


def main():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Regime Timeline"

    render_sidebar()
    predictor = load_predictor()

    try:
        current = st.session_state.current_page

        if current == "Regime Timeline":
            from dashboard.pages.regime_timeline import render
            render(predictor)
        elif current == "Stock Forecast":
            from dashboard.pages.stock_forecast import render
            render(predictor)
        elif current == "Alerts":
            from dashboard.pages.alerts_page import render
            render(predictor)
        elif current == "Model Intelligence":
            from dashboard.pages.model_intelligence import render
            render(predictor)

    except Exception as e:
        st.error(f"Page error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()