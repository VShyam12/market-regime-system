"""Central ticker and regime configuration for the Market Regime Detection System.

This module defines ticker groups, display names, regime-specific indicator sets,
and core date/label/window constants used across data collection, feature
engineering, training, and inference workflows.
"""

TICKERS: dict[str, list[str]] = {
    "market": ["SPY", "QQQ", "IWM"],
    "volatility": ["^VIX"],
    "sectors": ["XLK", "XLF", "XLV", "XLU", "XLE", "XLI"],
    "bonds": ["TLT", "IEF", "HYG"],
    "commodities": ["GLD", "USO"],
}

# Flatten all ticker groups while preserving first-seen order.
ALL_TICKERS: list[str] = list(dict.fromkeys(ticker for group in TICKERS.values() for ticker in group))

TICKER_NAMES: dict[str, str] = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF",
    "^VIX": "CBOE Volatility Index",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "GLD": "SPDR Gold Shares",
    "USO": "United States Oil Fund, LP",
}

REGIME_INDICATORS: dict[str, list[str]] = {
    "primary": ["SPY", "^VIX", "TLT"],
    "secondary": ["QQQ", "IWM", "HYG", "GLD"],
}

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
REGIME_LABELS: dict[int, str] = {0: "Growth", 1: "Transition", 2: "Panic"}
LOOKBACK_WINDOW = 60
FORECAST_HORIZON = 30


def get_tickers_by_category(category: str) -> list[str]:
    """Return the ticker list for a given category.

    Args:
        category: One of the keys defined in TICKERS.

    Returns:
        A copy of the ticker list for the category.

    Raises:
        ValueError: If the category is not defined.
    """
    if category not in TICKERS:
        available = ", ".join(sorted(TICKERS))
        raise ValueError(f"Unknown category '{category}'. Available categories: {available}")

    return TICKERS[category].copy()


if __name__ == "__main__":
    print(f"Total number of tickers: {len(ALL_TICKERS)}")

    print("\nTickers by category:")
    for category_name, tickers in TICKERS.items():
        print(f"- {category_name}: {tickers}")

    print(f"\nPrimary indicators: {REGIME_INDICATORS['primary']}")
    print(f"Secondary indicators: {REGIME_INDICATORS['secondary']}")
