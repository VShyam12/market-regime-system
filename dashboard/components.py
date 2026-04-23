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
        "Growth": (
            "linear-gradient(135deg, "
            "rgba(22,163,74,0.2), "
            "rgba(22,163,74,0.05))"
        ),
        "Transition": (
            "linear-gradient(135deg, "
            "rgba(217,119,6,0.2), "
            "rgba(217,119,6,0.05))"
        ),
        "Panic": (
            "linear-gradient(135deg, "
            "rgba(220,38,38,0.2), "
            "rgba(220,38,38,0.05))"
        )
    }
    return gradients.get(
        regime, "rgba(30,41,59,0.4)")


def render_header(title: str,
                  subtitle: str = "") -> None:
    sub_html = (
        f'<p style="color:#64748B;font-size:13px;'
        f'margin:6px 0 0 0;'
        f'font-family:Inter,sans-serif;">'
        f'{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin-bottom:2rem;">'
        f'<h1 style="color:#F8FAFC;font-size:26px;'
        f'font-weight:700;margin:0;'
        f'font-family:Inter,sans-serif;'
        f'letter-spacing:-0.02em;">{title}</h1>'
        f'{sub_html}'
        f'<div style="height:2px;width:40px;'
        f'background:linear-gradient('
        f'90deg,#3B82F6,#8B5CF6);'
        f'border-radius:2px;'
        f'margin-top:10px;"></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_regime_badge(regime: str,
                        confidence: float = None
                        ) -> str:
    colour = get_regime_colour(regime)
    conf_text = (
        f" · {confidence:.1%}" if confidence else "")
    icons = {
        "Growth": "↑",
        "Transition": "→",
        "Panic": "↓"
    }
    icon = icons.get(regime, "·")
    return (
        f'<span style="background:{colour}22;'
        f'color:{colour};'
        f'border:1px solid {colour}44;'
        f'padding:4px 12px;border-radius:20px;'
        f'font-size:12px;font-weight:600;'
        f'font-family:Inter,sans-serif;'
        f'letter-spacing:0.02em;">'
        f'{icon} {regime}{conf_text}'
        f'</span>'
    )


def render_stat_card(label: str,
                     value: str,
                     subtitle: str = "",
                     colour: str = "#3B82F6",
                     icon: str = "") -> None:
    icon_part = (
        f'<span style="font-size:20px;'
        f'opacity:0.7;">{icon}</span>'
        if icon else ""
    )
    sub_part = (
        f'<p style="color:#64748B;'
        f'font-size:12px;margin:4px 0 0 0;'
        f'font-family:Inter,sans-serif;'
        f'line-height:1.4;">{subtitle}</p>'
        if subtitle else ""
    )
    html = (
        f'<div style="background:rgba(15,23,42,0.6);'
        f'border:1px solid rgba(59,130,246,0.12);'
        f'border-radius:12px;padding:16px 20px;'
        f'border-left:3px solid {colour};'
        f'margin-bottom:8px;">'
        f'<div style="display:flex;'
        f'justify-content:space-between;'
        f'align-items:flex-start;">'
        f'<div style="flex:1;">'
        f'<p style="color:#475569;font-size:11px;'
        f'text-transform:uppercase;'
        f'letter-spacing:0.08em;margin:0 0 6px 0;'
        f'font-family:Inter,sans-serif;'
        f'font-weight:600;">{label}</p>'
        f'<p style="color:#F8FAFC;font-size:22px;'
        f'font-weight:700;margin:0;'
        f'font-family:JetBrains Mono,monospace;'
        f'letter-spacing:-0.02em;">{value}</p>'
        f'{sub_part}'
        f'</div>'
        f'{icon_part}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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
    colour = priority_colours.get(
        priority, "#64748B")
    bg = priority_bg.get(
        priority, "rgba(30,41,59,0.4)")

    html = (
        f'<div style="background:{bg};'
        f'border:1px solid {colour}33;'
        f'border-left:3px solid {colour};'
        f'border-radius:12px;'
        f'padding:16px 20px;margin-bottom:10px;">'
        f'<div style="display:flex;'
        f'justify-content:space-between;'
        f'align-items:center;margin-bottom:8px;">'
        f'<span style="background:{colour}22;'
        f'color:{colour};'
        f'border:1px solid {colour}44;'
        f'padding:2px 10px;border-radius:20px;'
        f'font-size:10px;font-weight:700;'
        f'font-family:Inter,sans-serif;'
        f'letter-spacing:0.08em;'
        f'text-transform:uppercase;">'
        f'{priority}</span>'
        f'<span style="color:#475569;font-size:12px;'
        f'font-family:JetBrains Mono,monospace;">'
        f'{alert.get("date", "")}</span>'
        f'</div>'
        f'<p style="color:#F8FAFC;font-size:14px;'
        f'font-weight:600;margin:0 0 6px 0;'
        f'font-family:Inter,sans-serif;">'
        f'{alert.get("title", "")}</p>'
        f'<p style="color:#94A3B8;font-size:13px;'
        f'margin:0;font-family:Inter,sans-serif;'
        f'line-height:1.5;">'
        f'{alert.get("message", "")}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str,
                       value: str,
                       delta: str = None,
                       colour: str = "#3B82F6"
                       ) -> None:
    delta_html = ""
    if delta:
        delta_colour = (
            "#16A34A" if "+" in delta
            else "#DC2626"
        )
        arrow = "↑" if "+" in delta else "↓"
        delta_html = (
            f'<p style="color:{delta_colour};'
            f'font-size:12px;margin:4px 0 0 0;'
            f'font-family:Inter,sans-serif;'
            f'font-weight:500;">'
            f'{arrow} {delta}</p>'
        )
    html = (
        f'<div style="background:rgba(15,23,42,0.6);'
        f'border:1px solid rgba(59,130,246,0.1);'
        f'border-top:2px solid {colour};'
        f'border-radius:12px;padding:16px 20px;'
        f'margin-bottom:8px;">'
        f'<p style="color:#475569;font-size:11px;'
        f'text-transform:uppercase;'
        f'letter-spacing:0.08em;margin:0 0 6px 0;'
        f'font-family:Inter,sans-serif;'
        f'font-weight:600;">{label}</p>'
        f'<p style="color:#F8FAFC;font-size:26px;'
        f'font-weight:700;margin:0;'
        f'font-family:JetBrains Mono,monospace;'
        f'letter-spacing:-0.02em;">{value}</p>'
        f'{delta_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_status(
        predictor_loaded: bool) -> None:
    status = (
        "🟢 Active" if predictor_loaded
        else "🔴 Offline"
    )
    st.sidebar.markdown(
        f'<p style="color:#94A3B8;font-size:12px;">'
        f'{status}</p>',
        unsafe_allow_html=True
    )