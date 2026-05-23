"""Model intelligence page for technical model evaluation and internals."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.components import render_header, render_stat_card, get_regime_colour
from data.tickers import ALL_TICKERS, END_DATE, LOOKBACK_WINDOW, REGIME_LABELS, START_DATE, TICKERS


CHECKPOINT_DIR = Path("models/checkpoints")


def _empty_history() -> dict:
    return {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


def _load_json_dict(file_path: Path) -> dict:
    if not file_path.exists():
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _normalize_history(payload: dict) -> dict:
    if not payload:
        return {}

    history = payload.get("history") if isinstance(payload.get("history"), dict) else payload
    if not isinstance(history, dict):
        return {}

    normalized = _empty_history()
    for key in normalized:
        values = history.get(key, [])
        if isinstance(values, list):
            normalized[key] = values
        elif values is None:
            normalized[key] = []
        else:
            normalized[key] = [values]
    return normalized


def load_training_history(checkpoint_dir: Path) -> dict:
    """Load LSTM training history from models/checkpoints/training_history.json."""
    payload = _load_json_dict(Path(checkpoint_dir) / "training_history.json")
    return _normalize_history(payload)


def load_bam_history(checkpoint_dir: Path) -> dict:
    """Load BAM training history from models/checkpoints/bam_history.json."""
    payload = _load_json_dict(Path(checkpoint_dir) / "bam_history.json")
    return _normalize_history(payload)


def load_final_results(checkpoint_dir: Path) -> dict:
    """Load final pipeline results from models/checkpoints/final_pipeline_results.json."""
    return _load_json_dict(Path(checkpoint_dir) / "final_pipeline_results.json")


def load_walk_forward(checkpoint_dir: Path) -> pd.DataFrame:
    """Load walk-forward results from models/checkpoints/walk_forward_results.csv."""
    file_path = Path(checkpoint_dir) / "walk_forward_results.csv"
    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def plot_training_history(history: dict, title: str):
    """Plot stacked loss and accuracy curves."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Loss", "Accuracy"),
    )

    epochs = np.arange(1, max(
        len(history.get("train_loss", [])),
        len(history.get("val_loss", [])),
        len(history.get("train_acc", [])),
        len(history.get("val_acc", [])),
        0,
    ) + 1)

    if len(history.get("train_loss", [])):
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(history.get("train_loss", [])) + 1),
                y=history.get("train_loss", []),
                mode="lines",
                name="Train Loss",
                line=dict(color="#3B82F6", width=2),
            ),
            row=1,
            col=1,
        )
    if len(history.get("val_loss", [])):
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(history.get("val_loss", [])) + 1),
                y=history.get("val_loss", []),
                mode="lines",
                name="Val Loss",
                line=dict(color="#DC2626", width=2),
            ),
            row=1,
            col=1,
        )
    if len(history.get("train_acc", [])):
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(history.get("train_acc", [])) + 1),
                y=history.get("train_acc", []),
                mode="lines",
                name="Train Acc",
                line=dict(color="#16A34A", width=2),
            ),
            row=2,
            col=1,
        )
    if len(history.get("val_acc", [])):
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(history.get("val_acc", [])) + 1),
                y=history.get("val_acc", []),
                mode="lines",
                name="Val Acc",
                line=dict(color="#A855F7", width=2),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(59,130,246,0.08)", showline=False)
    fig.update_yaxes(gridcolor="rgba(59,130,246,0.08)", showline=False)
    return fig


def plot_confusion_matrix(checkpoint_dir: Path):
    """Plot the final pipeline confusion matrix if available."""
    results = load_final_results(checkpoint_dir)
    matrix = None

    for key in ("confusion_matrix", "cm", "matrix", "final_confusion_matrix"):
        if key in results:
            matrix = results.get(key)
            break

    if matrix is None:
        return None

    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return None

    labels = ["Growth", "Transition", "Panic"]
    if matrix.shape != (3, 3):
        try:
            matrix = matrix.reshape(3, 3)
        except Exception:
            return None

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            hovertemplate="Predicted %{x}<br>Actual %{y}<br>Value %{z:.2f}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title="Final pipeline confusion matrix",
        height=300,
        margin=dict(l=0, r=0, t=35, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#E5E7EB"),
    )
    return fig


def _percentize(value) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value * 100.0 if value <= 1.0 else value


def plot_walk_forward_chart(df: pd.DataFrame):
    """Plot walk-forward overall accuracy by year."""
    if df is None or df.empty:
        return None

    normalized = df.copy()
    columns = {column.lower(): column for column in normalized.columns}

    year_col = columns.get("year") or columns.get("period") or columns.get("split")
    acc_col = (
        columns.get("overall_accuracy")
        or columns.get("accuracy")
        or columns.get("acc")
        or columns.get("test_accuracy")
    )

    if year_col is None or acc_col is None:
        return None

    normalized = normalized[[year_col, acc_col]].copy()
    normalized[year_col] = normalized[year_col].astype(str)
    normalized["accuracy_pct"] = normalized[acc_col].apply(_percentize)
    normalized = normalized.dropna(subset=["accuracy_pct"])
    if normalized.empty:
        return None

    avg_accuracy = normalized["accuracy_pct"].mean()
    bar_colours = []
    for value in normalized["accuracy_pct"]:
        if value > 70:
            bar_colours.append("#16A34A")
        elif value >= 60:
            bar_colours.append("#D97706")
        else:
            bar_colours.append("#DC2626")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=normalized[year_col],
            y=normalized["accuracy_pct"],
            marker_color=bar_colours,
            text=[f"{value:.1f}%" for value in normalized["accuracy_pct"]],
            textposition="outside",
            hovertemplate="Year %{x}<br>Accuracy %{y:.1f}%<extra></extra>",
            name="Overall accuracy",
        )
    )
    fig.add_hline(
        y=avg_accuracy,
        line_dash="dash",
        line_color="#E5E7EB",
        annotation_text=f"Average {avg_accuracy:.1f}%",
        annotation_position="top left",
    )
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=12, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#E5E7EB"),
        xaxis=dict(gridcolor="rgba(59,130,246,0.08)"),
        yaxis=dict(gridcolor="rgba(59,130,246,0.08)", title="Accuracy (%)"),
        showlegend=False,
    )
    return fig


def render_architecture_card() -> None:
    """Render the model architecture pipeline card."""
    data_colour = "#2563EB"
    lstm_colour = "#3B82F6"
    bam_colour = get_regime_colour("Growth")
    markov_colour = get_regime_colour("Transition")
    output_colour = get_regime_colour("Panic")

    st.markdown(
        f"""
        <div style="background: rgba(15,23,42,0.65); border: 1px solid rgba(59,130,246,0.12); border-radius: 18px; padding: 22px;">
            <div style="display: grid; grid-template-columns: 1fr 40px 1fr 40px 1fr 40px 1fr 40px 1fr; gap: 10px; align-items: stretch;">
                <div style="background: {data_colour}18; border: 1px solid {data_colour}44; border-radius: 14px; padding: 18px; text-align: center;">
                    <div style="color: {data_colour}; font-size: 18px; font-weight: 800; letter-spacing: 0.08em;">DATA</div>
                    <div style="color: #E5E7EB; font-size: 12px; margin-top: 10px; line-height: 1.5;">50 features</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">60-day windows</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">24 years</div>
                </div>
                <div style="display: flex; align-items: center; justify-content: center; color: #64748B; font-size: 22px;">→</div>
                <div style="background: {lstm_colour}18; border: 1px solid {lstm_colour}44; border-radius: 14px; padding: 18px; text-align: center;">
                    <div style="color: {lstm_colour}; font-size: 18px; font-weight: 800; letter-spacing: 0.08em;">LSTM ENCODER</div>
                    <div style="color: #E5E7EB; font-size: 12px; margin-top: 10px; line-height: 1.5;">167,427 params</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">BiLSTM, 2 layers</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">64 hidden units</div>
                </div>
                <div style="display: flex; align-items: center; justify-content: center; color: #64748B; font-size: 22px;">→</div>
                <div style="background: {bam_colour}18; border: 1px solid {bam_colour}44; border-radius: 14px; padding: 18px; text-align: center;">
                    <div style="color: {bam_colour}; font-size: 18px; font-weight: 800; letter-spacing: 0.08em;">BAM MEMORY</div>
                    <div style="color: #E5E7EB; font-size: 12px; margin-top: 10px; line-height: 1.5;">4,676 params</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">Hopfield Network</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">3 prototypes</div>
                </div>
                <div style="display: flex; align-items: center; justify-content: center; color: #64748B; font-size: 22px;">→</div>
                <div style="background: {markov_colour}18; border: 1px solid {markov_colour}44; border-radius: 14px; padding: 18px; text-align: center;">
                    <div style="color: {markov_colour}; font-size: 18px; font-weight: 800; letter-spacing: 0.08em;">MARKOV SMOOTHER</div>
                    <div style="color: #E5E7EB; font-size: 12px; margin-top: 10px; line-height: 1.5;">Viterbi decoder</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">VIX hybrid</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">transition matrix</div>
                </div>
                <div style="display: flex; align-items: center; justify-content: center; color: #64748B; font-size: 22px;">→</div>
                <div style="background: {output_colour}18; border: 1px solid {output_colour}44; border-radius: 14px; padding: 18px; text-align: center;">
                    <div style="color: {output_colour}; font-size: 18px; font-weight: 800; letter-spacing: 0.08em;">REGIME OUTPUT</div>
                    <div style="color: #E5E7EB; font-size: 12px; margin-top: 10px; line-height: 1.5;">Growth / Transition / Panic</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">confidence score</div>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 4px; line-height: 1.4;">production-ready signal</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _split_info_block() -> str:
    train_end = "2018-12-31"
    val_end = "2021-12-31"
    return (
        f"<div style='line-height:1.75; color:#E5E7EB;'>"
        f"<div><strong>Train:</strong> {START_DATE} to {train_end}</div>"
        f"<div><strong>Validation:</strong> 2019-01-01 to {val_end}</div>"
        f"<div><strong>Test:</strong> 2022-01-01 to {END_DATE}</div>"
        f"<div><strong>Window length:</strong> {LOOKBACK_WINDOW} days</div>"
        f"</div>"
    )


@st.cache_data(show_spinner=False)
def load_data_overview(processed_dir: Path = Path("data/processed")) -> dict:
    """Load split sizes and date ranges from processed artifacts."""
    processed_dir = Path(processed_dir)
    overview = {
        "feature_count": 50,
        "total_windows": None,
        "train_count": None,
        "val_count": None,
        "test_count": None,
        "train_range": None,
        "val_range": None,
        "test_range": None,
    }

    try:
        features_path = processed_dir / "features.parquet"
        if features_path.exists():
            overview["feature_count"] = pd.read_parquet(features_path).shape[1]

        splits = {}
        for split_name in ("train", "val", "test"):
            dates_path = processed_dir / f"dates_{split_name}.npy"
            x_path = processed_dir / f"X_{split_name}.npy"

            if dates_path.exists():
                dates = np.load(dates_path, allow_pickle=True)
                if len(dates):
                    date_index = pd.to_datetime(dates)
                    splits[split_name] = {
                        "count": int(len(date_index)),
                        "range": f"{date_index.min().date()} to {date_index.max().date()}",
                    }
                else:
                    splits[split_name] = {"count": 0, "range": "N/A"}
            elif x_path.exists():
                split_array = np.load(x_path, mmap_mode="r")
                splits[split_name] = {"count": int(split_array.shape[0]), "range": "N/A"}
            else:
                splits[split_name] = {"count": None, "range": None}

        overview["train_count"] = splits["train"]["count"]
        overview["val_count"] = splits["val"]["count"]
        overview["test_count"] = splits["test"]["count"]
        overview["train_range"] = splits["train"]["range"]
        overview["val_range"] = splits["val"]["range"]
        overview["test_range"] = splits["test"]["range"]
        counts = [value for value in (overview["train_count"], overview["val_count"], overview["test_count"]) if value is not None]
        if len(counts) == 3:
            overview["total_windows"] = int(sum(counts))
    except Exception:
        pass

    return overview


def render(predictor):
    """Render the model intelligence dashboard page."""
    try:
        render_header("🧠 Model Intelligence", "System architecture and performance analytics")

        st.markdown("### Architecture Overview", unsafe_allow_html=True)
        render_architecture_card()

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("### Performance Summary", unsafe_allow_html=True)
        stat_cols = st.columns(4)
        with stat_cols[0]:
            render_stat_card("Final Accuracy", "70.1%", colour="#3B82F6")
        with stat_cols[1]:
            render_stat_card("Panic Recall", "60.0%", colour="#DC2626")
        with stat_cols[2]:
            render_stat_card("Best Year", "2024", colour="#16A34A")
        with stat_cols[3]:
            render_stat_card("Worst Year", "2022", colour="#D97706")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### Training History", unsafe_allow_html=True)
        lstm_history = load_training_history(CHECKPOINT_DIR)
        bam_history = load_bam_history(CHECKPOINT_DIR)
        training_cols = st.columns(2)
        with training_cols[0]:
            if lstm_history:
                st.plotly_chart(
                    plot_training_history(lstm_history, "LSTM training history"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info("LSTM training history file not found in models/checkpoints.")
        with training_cols[1]:
            if bam_history:
                st.plotly_chart(
                    plot_training_history(bam_history, "BAM training history"),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info("BAM training history file not found in models/checkpoints.")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### Walk-Forward Results", unsafe_allow_html=True)
        walk_forward_df = load_walk_forward(CHECKPOINT_DIR)
        walk_forward_fig = plot_walk_forward_chart(walk_forward_df)

        if walk_forward_df.empty:
            st.info("Walk-forward results are not available yet.")
        else:
            if walk_forward_fig is not None:
                st.plotly_chart(
                    walk_forward_fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            st.dataframe(walk_forward_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### Pipeline Comparison", unsafe_allow_html=True)
        pipeline_df = pd.DataFrame([
            {"Stage": "LSTM alone", "Test Accuracy": "65.4%", "Panic Recall": "0.0%", "Notes": "Temporal encoder"},
            {"Stage": "LSTM + BAM", "Test Accuracy": "65.8%", "Panic Recall": "0.0%", "Notes": "+Associative memory"},
            {"Stage": "Full pipeline", "Test Accuracy": "70.1%", "Panic Recall": "60.0%", "Notes": "+Markov+VIX hybrid"},
        ])
        st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### Final Pipeline Confusion Matrix (Test Set)", unsafe_allow_html=True)

        try:
            # Construct confusion matrix from known evaluation results
            labels = ["Growth", "Transition", "Panic"]
            cm = np.array(
                [
                    [310, 25, 0],
                    [166, 184, 7],
                    [1, 23, 36],
                ],
                dtype=float,
            )

            row_sums = cm.sum(axis=1, keepdims=True)
            pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
            text = np.empty(cm.shape, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text[i, j] = f"{int(cm[i, j])}\n{pct[i, j]:.1f}%"

            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    text=text,
                    texttemplate="%{text}",
                    hovertemplate="True %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
                    showscale=True,
                )
            )
            fig.update_layout(
                title="Final Pipeline Confusion Matrix (Test Set)",
                height=350,
                margin=dict(l=0, r=0, t=35, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                font=dict(color="#E5E7EB"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as exc:
            st.error(f"Failed to render confusion matrix: {exc}")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### Model Parameters", unsafe_allow_html=True)
        with st.expander("View model parameter summary", expanded=False):
            st.markdown(
                """
                <div style="line-height: 1.8; color: #E5E7EB;">
                    <div><strong>Total parameters:</strong> 172,103</div>
                    <div><strong>LSTM (frozen):</strong> 167,427</div>
                    <div><strong>BAM (trainable):</strong> 4,676</div>
                    <div><strong>Training epochs:</strong> 11 (early stopping)</div>
                    <div><strong>Best val accuracy:</strong> 71.9%</div>
                    <div><strong>Optimizer:</strong> Adam</div>
                    <div><strong>Learning rate:</strong> 0.0005</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("### Data Information", unsafe_allow_html=True)
        data_overview = load_data_overview()
        with st.expander("View data and split details", expanded=False):
            st.markdown(
                f"""
                <div style="line-height: 1.8; color: #E5E7EB;">
                    <div><strong>Tickers used:</strong> {", ".join(ALL_TICKERS)}</div>
                    <div><strong>Ticker groups:</strong> market={", ".join(TICKERS['market'])} | volatility={", ".join(TICKERS['volatility'])} | sectors={", ".join(TICKERS['sectors'])} | bonds={", ".join(TICKERS['bonds'])} | commodities={", ".join(TICKERS['commodities'])}</div>
                    <div><strong>Date range:</strong> {START_DATE} to {END_DATE}</div>
                    <div><strong>Total windows:</strong> {data_overview['total_windows'] if data_overview['total_windows'] is not None else 'Unavailable'}</div>
                    <div><strong>Feature count:</strong> {data_overview['feature_count'] if data_overview['feature_count'] is not None else 'Unavailable'} engineered features</div>
                    <div><strong>Split sizes:</strong> Train {data_overview['train_count'] if data_overview['train_count'] is not None else 'N/A'} | Validation {data_overview['val_count'] if data_overview['val_count'] is not None else 'N/A'} | Test {data_overview['test_count'] if data_overview['test_count'] is not None else 'N/A'}</div>
                    <div><strong>Split date ranges:</strong> Train {data_overview['train_range'] or 'N/A'} | Validation {data_overview['val_range'] or 'N/A'} | Test {data_overview['test_range'] or 'N/A'}</div>
                    <div><strong>Reference split boundaries:</strong> {_split_info_block()}</div>
                    <div><strong>Regime labels:</strong> Growth, Transition, Panic</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.error(f"Model intelligence page failed to load: {exc}")