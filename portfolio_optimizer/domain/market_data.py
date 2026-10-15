from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class MarketDataset:
    """Contenitore per i dati di mercato normalizzati."""

    prices: pd.DataFrame
    log_returns: pd.DataFrame
    dividends: pd.DataFrame
    splits: pd.DataFrame
    risk_free_rate_annual: pd.Series
    risk_free_rate_daily: pd.Series
    reference_currency: str = "USD"
    ticker_currencies: dict[str, str] = field(default_factory=dict)
    currency_metadata: dict[str, dict] = field(default_factory=dict)
    provenance_manifest: dict = field(default_factory=dict)
    history_impact: dict = field(default_factory=dict)
