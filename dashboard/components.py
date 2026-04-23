"""Shared UI components for Market Regime Detection dashboard."""

import streamlit as st


def get_regime_colour(regime: str) -> str:
    colours = {
        "Growth": "#16A34A",
        "Transition": "#D97706",
        "Panic": "#DC2626"
    }
    return colours.get(regime, "#64748B")


def get_regime_gradient(regime: str) -> str:
    gradients = {
        "Growth": "linear-gradient(135deg, rgba(22,163,74,0.2), rgba(22,163,74,0.05))",
        "Transition": "linear-gradient(135deg, rgba(217,119,6,0.2), rgba(217,119,6,0.05))",
        "Panic": "linear-gradient(135deg, rgba(220,38,38,0.2), rgba(220,38,38,0.05))"
    }
    return gradients.get(regime, "rgba(30,41,59,0.4)")


def render_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #F8FAFC; font-size: 26px;
                   font-weight: 700; margin: 0;
                   font-family: Inter, sans-serif;
                   letter-spacing: -0.02em;">{title}</h1>
        {f'<p style="color: #64748B; font-size: 13px; margin: 6px 0 0 0; font-family: Inter, sans-serif;">{subtitle}</p>' if subtitle else ''}
        <div style="height: 2px; width: 40px;
                    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
                    border-radius: 2px; margin-top: 10px;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_regime_badge(regime: str,
                        confidence: float = None) -> str:
    colour = get_regime_colour(regime)
    conf_text = f" · {confidence:.1%}" if confidence else ""
    icons = {"Growth": "↑", "Transition": "→", "Panic": "↓"}
    icon = icons.get(regime, "·")
    return f"""
    <span style="background: {colour}22;
                 color: {colour};
                 border: 1px solid {colour}44;
                 padding: 4px 12px; border-radius: 20px;
                 font-size: 12px; font-weight: 600;
                 font-family: Inter, sans-serif;
                 letter-spacing: 0.02em;">
        {icon} {regime}{conf_text}
    </span>
    """


def render_current_regime_card(regime: str,
                                confidence: float,
                                since_date: str,
                                duration_days: int,
                                vix_level: float,
                                p_growth: float,
                                p_transition: float,
                                p_panic: float) -> None:
    colour = get_regime_colour(regime)
    gradient = get_regime_gradient(regime)
    icons = {"Growth": "🌱", "Transition": "⚡", "Panic": "🚨"}
    icon = icons.get(regime, "📊")

    st.markdown(f"""
    <div style="background: {gradient};
                border: 1px solid {colour}44;
                border-radius: 16px; padding: 24px;
                margin-bottom: 1.5rem;
                position: relative; overflow: hidden;">
        <div style="position: absolute; top: -20px; right: -20px;
                    width: 100px; height: 100px;
                    background: {colour}11;
                    border-radius: 50%;"></div>
        <div style="position: absolute; top: 10px; right: 10px;
                    width: 60px; height: 60px;
                    background: {colour}08;
                    border-radius: 50%;"></div>

        <div style="display: flex; justify-content: space-between;
                    align-items: flex-start; position: relative;">
            <div>
                <p style="color: #64748B; font-size: 11px;
                          text-transform: uppercase;
                          letter-spacing: 0.1em; margin: 0 0 8px 0;
                          font-family: Inter, sans-serif;
                          font-weight: 600;">Current Regime</p>
                <div style="display: flex; align-items: center;
                            gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 32px;">{icon}</span>
                    <h2 style="color: {colour}; font-size: 36px;
                               font-weight: 800; margin: 0;
                               font-family: Inter, sans-serif;
                               letter-spacing: -0.03em;">{regime}</h2>
                </div>
                <p style="color: #94A3B8; font-size: 13px; margin: 0;
                          font-family: Inter, sans-serif;">
                    Since {since_date} · {duration_days} days
                </p>
            </div>

            <div style="text-align: right;">
                <p style="color: #64748B; font-size: 11px;
                          text-transform: uppercase;
                          letter-spacing: 0.1em; margin: 0 0 4px 0;
                          font-family: Inter, sans-serif;">Confidence</p>
                <p style="color: #F8FAFC; font-size: 40px;
                          font-weight: 700; margin: 0;
                          font-family: JetBrains Mono, monospace;
                          letter-spacing: -0.02em;">{confidence:.1%}</p>
                <p style="color: #64748B; font-size: 12px; margin: 4px 0 0 0;
                          font-family: Inter, sans-serif;">VIX: {vix_level:.2f}</p>
            </div>
        </div>

        <div style="margin-top: 20px; padding-top: 16px;
                    border-top: 1px solid {colour}22;">
            <p style="color: #475569; font-size: 11px;
                      text-transform: uppercase; letter-spacing: 0.08em;
                      margin: 0 0 10px 0; font-family: Inter, sans-serif;
                      font-weight: 600;">Probability Distribution</p>
            <div style="display: flex; gap: 12px;">
                <div style="flex: 1; background: rgba(22,163,74,0.1);
                            border: 1px solid rgba(22,163,74,0.2);
                            border-radius: 8px; padding: 10px;
                            text-align: center;">
                    <p style="color: #16A34A; font-size: 18px;
                              font-weight: 700; margin: 0;
                              font-family: JetBrains Mono, monospace;">
                        {p_growth:.1%}
                    </p>
                    <p style="color: #64748B; font-size: 11px;
                              margin: 2px 0 0 0;
                              font-family: Inter, sans-serif;">Growth</p>
                </div>
                <div style="flex: 1; background: rgba(217,119,6,0.1);
                            border: 1px solid rgba(217,119,6,0.2);
                            border-radius: 8px; padding: 10px;
                            text-align: center;">
                    <p style="color: #D97706; font-size: 18px;
                              font-weight: 700; margin: 0;
                              font-family: JetBrains Mono, monospace;">
                        {p_transition:.1%}
                    </p>
                    <p style="color: #64748B; font-size: 11px;
                              margin: 2px 0 0 0;
                              font-family: Inter, sans-serif;">Transition</p>
                </div>
                <div style="flex: 1; background: rgba(220,38,38,0.1);
                            border: 1px solid rgba(220,38,38,0.2);
                            border-radius: 8px; padding: 10px;
                            text-align: center;">
                    <p style="color: #DC2626; font-size: 18px;
                              font-weight: 700; margin: 0;
                              font-family: JetBrains Mono, monospace;">
                        {p_panic:.1%}
                    </p>
                    <p style="color: #64748B; font-size: 11px;
                              margin: 2px 0 0 0;
                              font-family: Inter, sans-serif;">Panic</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_card(label: str, value: str,
                     subtitle: str = "",
                     colour: str = "#3B82F6",
                     icon: str = "") -> None:
    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.6);
                border: 1px solid rgba(59,130,246,0.12);
                border-radius: 12px; padding: 16px 20px;
                border-left: 3px solid {colour};
                transition: all 0.2s ease;
                margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between;
                    align-items: flex-start;">
            <div>
                <p style="color: #475569; font-size: 11px;
                          text-transform: uppercase;
                          letter-spacing: 0.08em; margin: 0 0 6px 0;
                          font-family: Inter, sans-serif;
                          font-weight: 600;">{label}</p>
                <p style="color: #F8FAFC; font-size: 24px;
                          font-weight: 700; margin: 0;
                          font-family: JetBrains Mono, monospace;
                          letter-spacing: -0.02em;">{value}</p>
                {f'<p style="color: #64748B; font-size: 12px; margin: 4px 0 0 0; font-family: Inter, sans-serif;">{subtitle}</p>' if subtitle else ''}
            </div>
            {f'<span style="font-size: 24px; opacity: 0.6;">{icon}</span>' if icon else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_alert_card(alert: dict) -> None:
    priority_colours = {
        "HIGH": "#DC2626",
        "MEDIUM": "#D97706",
        "LOW": "#16A34A"
    }
    priority_bg = {
        "HIGH": "rgba(220,38,38,0.08)",
        "MEDIUM": "rgba(217,119,6,0.08)",
        "LOW": "rgba(22,163,74,0.08)"
    }
    priority = alert.get("priority", "LOW")
    colour = priority_colours.get(priority, "#64748B")
    bg = priority_bg.get(priority, "rgba(30,41,59,0.4)")

    st.markdown(f"""
    <div style="background: {bg};
                border: 1px solid {colour}33;
                border-left: 3px solid {colour};
                border-radius: 12px; padding: 16px 20px;
                margin-bottom: 10px;
                transition: all 0.2s ease;">
        <div style="display: flex; justify-content: space-between;
                    align-items: center; margin-bottom: 8px;">
            <span style="background: {colour}22; color: {colour};
                         border: 1px solid {colour}44;
                         padding: 2px 10px; border-radius: 20px;
                         font-size: 10px; font-weight: 700;
                         font-family: Inter, sans-serif;
                         letter-spacing: 0.08em;
                         text-transform: uppercase;">
                {priority}
            </span>
            <span style="color: #475569; font-size: 12px;
                         font-family: JetBrains Mono, monospace;">
                {alert.get('date', '')}
            </span>
        </div>
        <p style="color: #F8FAFC; font-size: 14px; font-weight: 600;
                  margin: 0 0 6px 0; font-family: Inter, sans-serif;">
            {alert.get('title', '')}
        </p>
        <p style="color: #94A3B8; font-size: 13px; margin: 0;
                  font-family: Inter, sans-serif; line-height: 1.5;">
            {alert.get('message', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str,
                       delta: str = None,
                       colour: str = "#3B82F6") -> None:
    delta_html = ""
    if delta:
        delta_colour = "#16A34A" if "+" in delta else "#DC2626"
        arrow = "↑" if "+" in delta else "↓"
        delta_html = f"""
        <p style="color:{delta_colour}; font-size:12px;
                  margin:4px 0 0 0; font-family: Inter, sans-serif;
                  font-weight: 500;">{arrow} {delta}</p>
        """
    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.6);
                border: 1px solid rgba(59,130,246,0.1);
                border-top: 2px solid {colour};
                border-radius: 12px; padding: 16px 20px;
                margin-bottom: 8px;">
        <p style="color:#475569; font-size:11px;
                  text-transform:uppercase; letter-spacing:0.08em;
                  margin:0 0 6px 0; font-family: Inter, sans-serif;
                  font-weight: 600;">{label}</p>
        <p style="color:#F8FAFC; font-size:26px; font-weight:700;
                  margin:0; font-family: JetBrains Mono, monospace;
                  letter-spacing: -0.02em;">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def get_regime_colour(regime: str) -> str:
    colours = {
        "Growth": "#16A34A",
        "Transition": "#D97706",
        "Panic": "#DC2626"
    }
    return colours.get(regime, "#64748B")


def render_sidebar_status(predictor_loaded: bool) -> None:
    status = "🟢 Active" if predictor_loaded else "🔴 Offline"
    st.sidebar.markdown(
        f'<p style="color:#94A3B8;font-size:12px;">{status}</p>',
        unsafe_allow_html=True)