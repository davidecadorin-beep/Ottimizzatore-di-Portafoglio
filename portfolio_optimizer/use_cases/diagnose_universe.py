"""Caratterizzazione dell'universo investibile prima di scegliere un modello.

Il sistema non puo' raccomandare un criterio senza prima descrivere cosa sta
guardando. Le grandezze calcolate qui sono esattamente quelle su cui si basa
l'evidenza empirica: quanti titoli ci sono rispetto alla lunghezza della
finestra di stima, quanto i rendimenti si allontanano dalla normalita', quanto
sono correlati tra loro, e se la volatilita' attuale sta sopra o sotto la sua
media storica.

Qui non si giudica niente, si descrive solo. La regola che decide in base
a questa diagnosi sta in ``model_selector`` e si porta dietro l'evidenza
empirica che giustifica ogni ramo.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from portfolio_optimizer.risk.metrics import normality_tests


# Finestra recente usata per giudicare il regime di volatilita' attuale. Un
# trimestre va bene: e' lungo a sufficienza da non farsi influenzare da una
# singola settimana ma corto abbastanza da restare "attuale".
REGIME_WINDOW = 63


@dataclass(frozen=True, slots=True)
class UniverseDiagnosis:
    """Cosa osserva il sistema sui titoli che gli sono stati dati."""

    n_assets: int
    n_days: int
    assets_per_observation: float
    median_excess_kurtosis: float
    max_excess_kurtosis: float
    share_rejecting_normality: float
    mean_correlation: float
    max_correlation: float
    volatility_recent: float
    volatility_historical: float
    volatility_regime: str
    condition_number: float

    @property
    def has_fat_tails(self) -> bool:
        """Se le code sono abbastanza pesanti da contare per un criterio di coda.

        Una curtosi in eccesso sopra 1 sul titolo mediano e' ben oltre quello
        che produce una normale, ed e' la condizione per cui un criterio
        costruito sulla coda (e non sulla varianza) ha davvero qualcosa da
        sfruttare.
        """
        return self.median_excess_kurtosis > 1.0

    @property
    def estimation_is_stretched(self) -> bool:
        """Se il campione e' corto rispetto al numero di parametri da stimare.

        La matrice di covarianza di n titoli ha n(n+1)/2 parametri. Sopra
        circa un titolo ogni venti osservazioni la covarianza campionaria
        diventa mal condizionata e a quel punto la contrazione (shrinkage)
        non e' piu opzionale, va fatta.
        """
        return self.assets_per_observation > 0.05

    @property
    def is_highly_correlated(self) -> bool:
        return self.mean_correlation > 0.60

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "has_fat_tails": self.has_fat_tails,
                "estimation_is_stretched": self.estimation_is_stretched,
                "is_highly_correlated": self.is_highly_correlated,
            }
        )
        return payload

    def describe(self) -> list[str]:
        """La diagnosi scritta in frasi, da mostrare vicino alla raccomandazione."""
        lines = [
            f"{self.n_assets} titoli su {self.n_days} osservazioni "
            f"({self.assets_per_observation:.3f} titoli per osservazione)",
            f"curtosi in eccesso mediana {self.median_excess_kurtosis:.2f}, "
            f"massima {self.max_excess_kurtosis:.2f}",
            f"normalita' rifiutata sul {self.share_rejecting_normality:.0%} dei titoli",
            f"correlazione media {self.mean_correlation:.2f}, "
            f"massima {self.max_correlation:.2f}",
            f"volatilita' recente {self.volatility_recent:.1%} contro "
            f"{self.volatility_historical:.1%} storica: regime {self.volatility_regime}",
        ]
        if self.estimation_is_stretched:
            lines.append(
                "campione corto rispetto al numero di parametri da stimare: "
                "la covarianza campionaria e' mal condizionata"
            )
        return lines


def _volatility_regime(recent: float, historical: float) -> str:
    if not np.isfinite(recent) or not np.isfinite(historical) or historical <= 0:
        return "indeterminato"
    ratio = recent / historical
    if ratio > 1.30:
        return "elevato"
    if ratio < 0.75:
        return "quieto"
    return "ordinario"


def diagnose_universe(
    log_returns: pd.DataFrame, periods: int = 252, regime_window: int = REGIME_WINDOW
) -> UniverseDiagnosis:
    """Descrive l'universo partendo dalla sua storia di rendimenti."""
    clean = log_returns.dropna(how="all").dropna(axis=1, how="all")
    if clean.shape[1] < 2:
        raise ValueError("At least two assets are required to diagnose a universe.")
    clean = clean.dropna()
    n_days, n_assets = clean.shape
    if n_days < 60:
        raise ValueError(
            f"At least 60 common observations are required; got {n_days}."
        )

    shape = normality_tests(clean)
    kurtosis = shape["excess_kurtosis"]
    rejecting = float((~shape["is_normal_jb"]).mean())

    correlation = clean.corr().to_numpy()
    off_diagonal = correlation[~np.eye(n_assets, dtype=bool)]

    scale = np.sqrt(periods)
    equal_weight = clean.mean(axis=1)
    window = min(regime_window, n_days)
    volatility_recent = float(equal_weight.iloc[-window:].std(ddof=1) * scale)
    volatility_historical = float(equal_weight.std(ddof=1) * scale)

    covariance = clean.cov().to_numpy()
    eigenvalues = np.linalg.eigvalsh(covariance)
    smallest = float(eigenvalues.min())
    condition = (
        float(eigenvalues.max() / smallest) if smallest > 1e-18 else float("inf")
    )

    return UniverseDiagnosis(
        n_assets=int(n_assets),
        n_days=int(n_days),
        assets_per_observation=float(n_assets / n_days),
        median_excess_kurtosis=float(kurtosis.median()),
        max_excess_kurtosis=float(kurtosis.max()),
        share_rejecting_normality=rejecting,
        mean_correlation=float(off_diagonal.mean()),
        max_correlation=float(off_diagonal.max()),
        volatility_recent=volatility_recent,
        volatility_historical=volatility_historical,
        volatility_regime=_volatility_regime(volatility_recent, volatility_historical),
        condition_number=condition,
    )
