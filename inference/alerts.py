"""Alert generation for market regime predictions.

This module creates human-readable alerts from regime prediction sequences.
It supports transition alerts, panic lifecycle alerts, and extended regime
alerts, plus JSON persistence helpers and a runnable entry point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

from inference.predict import MarketRegimePredictor, get_regime_transitions


@dataclass
class RegimeAlert:
    """Container for a single regime alert."""

    alert_id: str
    alert_type: str
    priority: str
    date: str
    title: str
    message: str
    from_regime: str | None
    to_regime: str | None
    confidence: float | None
    vix_level: float | None
    duration_days: int | None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return all alert fields as a dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Return a compact alert summary string."""
        return f"[{self.priority}] {self.date} - {self.title}: {self.message}"


class AlertGenerator:
    """Generate alerts from regime prediction outputs."""

    def __init__(
        self,
        panic_entry_threshold: int = 1,
        extended_growth_days: int = 180,
        extended_panic_days: int = 30,
    ):
        self.panic_entry_threshold = int(panic_entry_threshold)
        self.extended_growth_days = int(extended_growth_days)
        self.extended_panic_days = int(extended_panic_days)

    @staticmethod
    def _prepare_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "regime", "regime_id", "confidence", "vix_level"}
        missing = sorted(required - set(predictions_df.columns))
        if missing:
            raise ValueError(f"predictions_df missing required columns: {missing}")

        df = predictions_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _make_alert_id(alert_date: pd.Timestamp, alert_type: str) -> str:
        return f"alert_{alert_date.strftime('%Y%m%d')}_{alert_type}"

    def generate_transition_alerts(
        self,
        transitions_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
    ) -> list[RegimeAlert]:
        """Generate alerts for each regime transition."""
        try:
            if transitions_df is None or transitions_df.empty:
                return []

            preds = self._prepare_predictions(predictions_df)
            preds_by_date = preds.set_index("date")

            alerts: list[RegimeAlert] = []
            tx = transitions_df.copy()
            tx["date"] = pd.to_datetime(tx["date"])

            for _, row in tx.sort_values("date").iterrows():
                transition_date = pd.to_datetime(row["date"])
                from_regime = str(row.get("from_regime", "Unknown"))
                to_regime = str(row.get("to_regime", "Unknown"))
                duration = int(row.get("duration_days", 0))

                conf = None
                vix = None
                if transition_date in preds_by_date.index:
                    pred_row = preds_by_date.loc[transition_date]
                    if isinstance(pred_row, pd.DataFrame):
                        pred_row = pred_row.iloc[-1]
                    conf = float(pred_row.get("confidence", float("nan")))
                    vix = float(pred_row.get("vix_level", float("nan")))
                    if pd.isna(conf):
                        conf = None
                    if pd.isna(vix):
                        vix = None

                alert_type = "transition"
                priority = "LOW"
                title = "Regime transition detected"
                message = f"Market shifted from {from_regime} to {to_regime}."

                if from_regime == "Growth" and to_regime == "Transition":
                    priority = "MEDIUM"
                    title = "Market entering transition phase"
                    message = (
                        f"Growth regime ended after {duration} days. "
                        "Market showing signs of uncertainty."
                    )
                elif from_regime == "Transition" and to_regime == "Growth":
                    priority = "LOW"
                    title = "Growth regime confirmed"
                    message = (
                        f"Market returned to Growth after {duration} days "
                        "of transition."
                    )
                elif from_regime == "Transition" and to_regime == "Panic":
                    alert_type = "panic_entry"
                    priority = "HIGH"
                    title = "PANIC REGIME DETECTED"
                    message = (
                        "Market entered Panic regime. VIX elevated. "
                        f"Previous transition lasted {duration} days."
                    )
                elif from_regime == "Panic" and to_regime == "Transition":
                    alert_type = "recovery"
                    priority = "MEDIUM"
                    title = "Panic regime easing"
                    message = (
                        "Market showing signs of stabilisation. "
                        f"Panic lasted {duration} days."
                    )
                elif from_regime == "Growth" and to_regime == "Panic":
                    alert_type = "panic_entry"
                    priority = "HIGH"
                    title = "SUDDEN MARKET SHOCK"
                    message = (
                        "Direct Growth to Panic transition detected. "
                        "Rare event - high uncertainty."
                    )

                alerts.append(
                    RegimeAlert(
                        alert_id=self._make_alert_id(transition_date, alert_type),
                        alert_type=alert_type,
                        priority=priority,
                        date=transition_date.date().isoformat(),
                        title=title,
                        message=message,
                        from_regime=from_regime,
                        to_regime=to_regime,
                        confidence=conf,
                        vix_level=vix,
                        duration_days=duration,
                        metadata={"source": "transition"},
                    )
                )

            return alerts
        except Exception as exc:
            raise RuntimeError(f"generate_transition_alerts failed: {exc}") from exc

    def generate_panic_alerts(self, predictions_df: pd.DataFrame) -> list[RegimeAlert]:
        """Generate panic entry/exit alerts for each detected Panic period."""
        try:
            preds = self._prepare_predictions(predictions_df)
            if preds.empty:
                return []

            alerts: list[RegimeAlert] = []
            regime_ids = preds["regime_id"].astype(int).to_numpy()
            dates = preds["date"].to_numpy()

            i = 0
            while i < len(preds):
                if regime_ids[i] != 2:
                    i += 1
                    continue

                start = i
                while i + 1 < len(preds) and regime_ids[i + 1] == 2:
                    i += 1
                end = i
                duration = end - start + 1

                start_row = preds.iloc[start]
                end_row = preds.iloc[end]

                if duration >= self.panic_entry_threshold:
                    entry_date = pd.to_datetime(start_row["date"])
                    entry_conf = float(start_row["confidence"])
                    entry_vix = (
                        float(start_row["vix_level"]) if pd.notna(start_row["vix_level"]) else None
                    )
                    vix_text = f"{entry_vix:.1f}" if entry_vix is not None else "N/A"

                    alerts.append(
                        RegimeAlert(
                            alert_id=self._make_alert_id(entry_date, "panic_entry"),
                            alert_type="panic_entry",
                            priority="HIGH",
                            date=entry_date.date().isoformat(),
                            title="Panic period began",
                            message=(
                                f"Panic regime started on {entry_date.date().isoformat()}. "
                                f"VIX={vix_text}. Confidence={entry_conf:.1%}"
                            ),
                            from_regime=(
                                str(preds.iloc[start - 1]["regime"]) if start > 0 else None
                            ),
                            to_regime="Panic",
                            confidence=entry_conf,
                            vix_level=entry_vix,
                            duration_days=duration,
                            metadata={"panic_start": entry_date.date().isoformat()},
                        )
                    )

                if end + 1 < len(preds):
                    exit_idx = end + 1
                    exit_row = preds.iloc[exit_idx]
                    exit_date = pd.to_datetime(exit_row["date"])
                    exit_conf = float(exit_row["confidence"])
                    exit_vix = (
                        float(exit_row["vix_level"]) if pd.notna(exit_row["vix_level"]) else None
                    )

                    alerts.append(
                        RegimeAlert(
                            alert_id=self._make_alert_id(exit_date, "panic_exit"),
                            alert_type="panic_exit",
                            priority="MEDIUM",
                            date=exit_date.date().isoformat(),
                            title="Panic period ended",
                            message=(
                                f"Panic regime ended on {exit_date.date().isoformat()} after {duration} days. "
                                "Market recovering."
                            ),
                            from_regime="Panic",
                            to_regime=str(exit_row["regime"]),
                            confidence=exit_conf,
                            vix_level=exit_vix,
                            duration_days=duration,
                            metadata={"panic_end": exit_date.date().isoformat()},
                        )
                    )

                i += 1

            return alerts
        except Exception as exc:
            raise RuntimeError(f"generate_panic_alerts failed: {exc}") from exc

    def generate_extended_regime_alerts(self, predictions_df: pd.DataFrame) -> list[RegimeAlert]:
        """Generate alerts for unusually long Growth/Panic regime runs."""
        try:
            preds = self._prepare_predictions(predictions_df)
            if preds.empty:
                return []

            alerts: list[RegimeAlert] = []
            regime_ids = preds["regime_id"].astype(int).to_numpy()

            run_start = 0
            for i in range(1, len(preds) + 1):
                run_break = i == len(preds) or regime_ids[i] != regime_ids[run_start]
                if not run_break:
                    continue

                run_end = i - 1
                run_days = run_end - run_start + 1
                rid = int(regime_ids[run_start])
                start_date = pd.to_datetime(preds.iloc[run_start]["date"])
                end_date = pd.to_datetime(preds.iloc[run_end]["date"])
                end_row = preds.iloc[run_end]

                if rid == 0 and run_days > self.extended_growth_days:
                    alerts.append(
                        RegimeAlert(
                            alert_id=self._make_alert_id(end_date, "extended_regime_growth"),
                            alert_type="extended_regime",
                            priority="LOW",
                            date=end_date.date().isoformat(),
                            title="Extended Growth regime",
                            message=(
                                f"Growth regime has lasted {run_days} days. "
                                "Historically elevated risk of correction."
                            ),
                            from_regime=None,
                            to_regime="Growth",
                            confidence=float(end_row["confidence"]),
                            vix_level=(
                                float(end_row["vix_level"]) if pd.notna(end_row["vix_level"]) else None
                            ),
                            duration_days=run_days,
                            metadata={
                                "run_start": start_date.date().isoformat(),
                                "run_end": end_date.date().isoformat(),
                            },
                        )
                    )

                if rid == 2 and run_days > self.extended_panic_days:
                    alerts.append(
                        RegimeAlert(
                            alert_id=self._make_alert_id(end_date, "extended_regime_panic"),
                            alert_type="extended_regime",
                            priority="HIGH",
                            date=end_date.date().isoformat(),
                            title="Extended Panic regime",
                            message=(
                                f"Panic regime persisting for {run_days} days. "
                                "Monitor for regime shift signals."
                            ),
                            from_regime=None,
                            to_regime="Panic",
                            confidence=float(end_row["confidence"]),
                            vix_level=(
                                float(end_row["vix_level"]) if pd.notna(end_row["vix_level"]) else None
                            ),
                            duration_days=run_days,
                            metadata={
                                "run_start": start_date.date().isoformat(),
                                "run_end": end_date.date().isoformat(),
                            },
                        )
                    )

                run_start = i

            return alerts
        except Exception as exc:
            raise RuntimeError(f"generate_extended_regime_alerts failed: {exc}") from exc

    def generate_all_alerts(self, predictions_df: pd.DataFrame) -> list[RegimeAlert]:
        """Generate transition, panic, and extended regime alerts in one list."""
        try:
            transitions_df = get_regime_transitions(predictions_df)

            transition_alerts = self.generate_transition_alerts(transitions_df, predictions_df)
            panic_alerts = self.generate_panic_alerts(predictions_df)
            extended_alerts = self.generate_extended_regime_alerts(predictions_df)

            all_alerts = transition_alerts + panic_alerts + extended_alerts

            deduped: dict[str, RegimeAlert] = {}
            for alert in all_alerts:
                if alert.alert_id not in deduped:
                    deduped[alert.alert_id] = alert

            sorted_alerts = sorted(
                deduped.values(),
                key=lambda a: pd.Timestamp(a.date),
                reverse=True,
            )
            return sorted_alerts
        except Exception as exc:
            raise RuntimeError(f"generate_all_alerts failed: {exc}") from exc


def save_alerts(alerts: list[RegimeAlert], save_path: Path) -> None:
    """Save alerts as JSON."""
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = [alert.to_dict() for alert in alerts]
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

        print(f"Saved {len(alerts)} alerts to {save_path}")
    except Exception as exc:
        raise RuntimeError(f"save_alerts failed: {exc}") from exc


def load_alerts(load_path: Path) -> list[RegimeAlert]:
    """Load alerts from JSON and reconstruct dataclass objects."""
    try:
        load_path = Path(load_path)
        with open(load_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)

        alerts = [RegimeAlert(**item) for item in payload]
        return alerts
    except Exception as exc:
        raise RuntimeError(f"load_alerts failed: {exc}") from exc


def print_alerts(alerts: list[RegimeAlert], max_alerts: int = 10) -> None:
    """Print a readable summary of the most recent alerts."""
    try:
        total = len(alerts)
        shown = min(max_alerts, total)

        print(f"Recent Alerts (showing {shown} of {total} total)")
        print("=" * 60)

        for alert in alerts[:shown]:
            print(f"[{alert.priority}] {alert.date} - {alert.title}")
            print(f"  {alert.message}")

            if alert.from_regime is not None:
                duration_text = (
                    f" ({alert.duration_days} days)" if alert.duration_days is not None else ""
                )
                print(f"  Previous regime: {alert.from_regime}{duration_text}")

            print("")
    except Exception as exc:
        raise RuntimeError(f"print_alerts failed: {exc}") from exc


def run_alert_generation(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    output_dir: Path = Path("inference"),
) -> list[RegimeAlert]:
    """Run end-to-end alert generation and save output JSON."""
    try:
        predictor = MarketRegimePredictor()
        predictions = predictor.predict(start_date=start_date, end_date=end_date)

        generator = AlertGenerator()
        alerts = generator.generate_all_alerts(predictions)

        print_alerts(alerts)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_alerts(alerts, output_dir / "alerts.json")

        return alerts
    except Exception as exc:
        raise RuntimeError(f"run_alert_generation failed: {exc}") from exc


if __name__ == "__main__":
    try:
        generated_alerts = run_alert_generation()

        high_count = sum(1 for alert in generated_alerts if alert.priority == "HIGH")
        medium_count = sum(1 for alert in generated_alerts if alert.priority == "MEDIUM")
        low_count = sum(1 for alert in generated_alerts if alert.priority == "LOW")

        print(f"HIGH priority alerts: {high_count}")
        print(f"MEDIUM priority alerts: {medium_count}")
        print(f"LOW priority alerts: {low_count}")
    except Exception as exc:
        print(f"Alert generation failed: {exc}", file=sys.stderr)
        raise
