from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame, annualization: int = 252) -> pd.DataFrame:
    """Matrice di covarianza campionaria classica."""
    return returns.cov() * annualization


def ledoit_wolf_covariance(
    returns: pd.DataFrame, annualization: int = 252
) -> pd.DataFrame:
    """Stimatore di covarianza con contrazione (shrinkage) di Ledoit-Wolf."""
    clean = returns.dropna()
    if len(clean) < 2:
        raise ValueError(
            f"LedoitWolf requires at least 2 observations; got {len(clean)}. "
            "One or more tickers may have no price data."
        )
    model = LedoitWolf().fit(clean.values)
    cov = pd.DataFrame(
        model.covariance_, index=returns.columns, columns=returns.columns
    )
    return cov * annualization


def ewma_covariance(
    returns: pd.DataFrame,
    lam: float = 0.94,
    annualization: int = 252,
) -> pd.DataFrame:
    """Matrice di covarianza a medie mobili esponenziali (EWMA).

    Assegna pesi che decadono esponenzialmente: il giorno piu' recente prende
    peso (1-λ), ogni giorno precedente viene scontato di λ. λ=0.94 e' lo
    standard giornaliero RiskMetrics (J.P. Morgan, RiskMetrics Technical
    Document, 4a ed., 1996).

    Nota: questa implementazione sottrae la media (demeaning) usando la
    media pesata EWMA prima di calcolare il prodotto esterno. La specifica
    RiskMetrics originale assume media zero (nessun demeaning). Il demeaning
    qui riduce il bias quando i titoli hanno un drift diverso da zero, ma si
    allontana dalla formulazione RiskMetrics stretta a media zero.
    """
    clean = returns.dropna()
    T = len(clean)
    if T < 2:
        return sample_covariance(returns, annualization)

    raw_weights = np.array([(1 - lam) * lam**i for i in range(T - 1, -1, -1)])
    weights = raw_weights / raw_weights.sum()

    ewma_mean = (weights[:, np.newaxis] * clean.values).sum(axis=0)
    demeaned = clean.values - ewma_mean
    cov = (weights[:, np.newaxis] * demeaned).T @ demeaned
    return pd.DataFrame(
        cov * annualization, index=returns.columns, columns=returns.columns
    )
