"""Criteri di ottimizzazione come componenti intercambiabili.

Ogni criterio e' esposto con la stessa firma — in ingresso una finestra di
rendimenti log, in uscita i pesi — cosi' chi chiama puo' far variare una
dimensione alla volta. Gli stimatori dei rendimenti attesi e della matrice
di covarianza sono dimensioni separate, combinate qui in una strategia.

La correzione di Ito applicata prima dell'ottimizzazione media-varianza
segue la stessa convenzione dell'applicazione web: il criterio e'
definito sui rendimenti aritmetici mentre la stima viene fatta sui
rendimenti log.

Perche' questo modulo sta in ``backtest`` e non in ``experiments``. E'
stato scritto per la griglia sperimentale, ma l'applicazione web ha
bisogno esattamente degli stessi criteri quando confronta i modelli fuori
campione, e i contratti di import vietano a qualsiasi layer di importare
``experiments``. Duplicare i criteri nel layer applicativo e' l'altra
opzione, ed e' la peggiore: due definizioni dello stesso criterio
divergono in silenzio, e il confronto su cui si basa la tesi smetterebbe
di essere un confronto fra gli stessi oggetti. Il modulo quindi vive dove
entrambi i chiamanti possono raggiungerlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from portfolio_optimizer.backtest.walk_forward import StrategyOutput
from portfolio_optimizer.models.black_litterman import implied_equilibrium_returns
from portfolio_optimizer.models.covariance import (
    ewma_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
)
from portfolio_optimizer.models.equal_weight import equal_weight_portfolio
from portfolio_optimizer.models.markowitz import (
    maximum_sharpe_portfolio,
    minimum_cvar_portfolio,
    minimum_variance_portfolio,
)
from portfolio_optimizer.models.returns_estimator import bayes_stein_expected_returns
from portfolio_optimizer.models.risk_parity import equal_risk_contribution


# ---------------------------------------------------------------------------
# Stimatori
# ---------------------------------------------------------------------------

CovarianceEstimator = Callable[[pd.DataFrame], pd.DataFrame]
ReturnsEstimator = Callable[[pd.DataFrame], pd.Series]


def historical_expected_returns(
    returns: pd.DataFrame, annualization: int = 252
) -> pd.Series:
    """Media campionaria annualizzata: la base la cui instabilita' motiva tutto il resto."""
    return returns.mean() * annualization


COVARIANCE_ESTIMATORS: dict[str, CovarianceEstimator] = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "ewma": ewma_covariance,
}

RETURNS_ESTIMATORS: dict[str, ReturnsEstimator] = {
    "historical": historical_expected_returns,
    "bayes_stein": bayes_stein_expected_returns,
}


def ito_adjusted(expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """E[R_simple] ~= E[r_log] + variance / 2, applicata titolo per titolo."""
    adjustment = pd.Series(
        0.5 * np.diag(covariance.to_numpy()), index=covariance.index
    )
    return expected_returns.add(adjustment, fill_value=0.0)


# ---------------------------------------------------------------------------
# Specifica della strategia
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Una cella della griglia sperimentale, o una colonna del confronto."""

    criterion: str
    returns_method: str = "historical"
    cov_method: str = "ledoit_wolf"
    max_weight: float = 1.0
    risk_free_rate: float = 0.0
    cvar_alpha: float = 0.95

    @property
    def uses_expected_returns(self) -> bool:
        """Se il criterio ha bisogno di una stima dei rendimenti attesi.

        I criteri che non ne hanno bisogno sono interessanti proprio
        perche' evitano l'input piu' rumoroso del problema.
        """
        return self.criterion in {"max_sharpe"}

    @property
    def label(self) -> str:
        if self.uses_expected_returns:
            return f"{self.criterion}|{self.returns_method}|{self.cov_method}"
        if self.criterion in {"min_variance", "risk_parity", "black_litterman"}:
            return f"{self.criterion}|{self.cov_method}"
        return self.criterion


CRITERIA_WITHOUT_COVARIANCE = {"equal_weight", "min_cvar"}

# I criteri che la griglia empirica esegue di default. Black-Litterman e'
# assente apposta: senza view esplicite restituisce il portafoglio di
# equilibrio, quindi una cella della griglia aggiungerebbe una colonna che
# duplica il prior invece di un criterio davvero sotto test. Resta
# costruibile — la pagina di confronto lo mostra — ma aggiungerlo qui
# cambierebbe in silenzio la dimensione dell'esperimento.
ALL_CRITERIA = (
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "risk_parity",
    "min_cvar",
)

# Tutto quello che ``build_strategy`` accetta.
SUPPORTED_CRITERIA = ALL_CRITERIA + ("black_litterman",)


def _black_litterman_weights(
    window: pd.DataFrame,
    covariance: pd.DataFrame,
    spec: StrategySpec,
) -> pd.Series:
    """Portafoglio di equilibrio, senza nessuna view espressa.

    L'applicazione lascia che l'utente dichiari le view; una run
    walk-forward non ha nessuno che possa dichiararle, e inventarle
    renderebbe la colonna non testabile. Il criterio viene quindi
    valutato nell'unica forma in cui e' riproducibile: si riottimizzano
    al contrario i pesi del prior in rendimenti di equilibrio, poi si
    ottimizza su quelli. Senza capitalizzazioni di mercato disponibili il
    prior e' il portafoglio equipesato, quindi il risultato e' il
    portafoglio equipesato soggetto al tetto di concentrazione. E' una
    proprieta' del metodo senza view, non un difetto
    dell'implementazione, e la pagina di confronto lo dichiara invece di
    presentare due colonne quasi identiche come se fossero evidenza
    indipendente.

    I passaggi riproducono ``_build_bl_weights`` nel caso d'uso
    dell'ottimizzazione per il ramo senza view, cosi' la colonna mostrata
    nel confronto e' lo stesso oggetto che produce l'applicazione e non
    una sua seconda implementazione.

    Due operazioni sono assenti apposta, e la loro assenza e' il punto.
    Il rendimento di equilibrio Pi = delta * Sigma * w e' un PREMIO per
    il rischio, gia' in termini aritmetici: qui la correzione di Ito non
    ha niente da convertire e aggiungerebbe un termine proporzionale alla
    varianza di ogni titolo, inclinando il portafoglio verso i titoli
    piu' volatili; e il tasso privo di rischio e' gia' stato sottratto,
    quindi sottrarlo di nuovo dentro il problema tangente lo toglie due
    volte. Con entrambe assenti l'ottimizzazione restituisce esattamente
    i pesi del prior, che e' l'identita' di reverse-optimization su cui
    il modello e' costruito.
    """
    n = covariance.shape[0]
    prior = pd.Series(np.ones(n) / n, index=covariance.index)
    bl_mu = implied_equilibrium_returns(covariance, prior)
    return maximum_sharpe_portfolio(
        bl_mu,
        covariance,
        risk_free_rate=0.0,
        max_weight=spec.max_weight,
    )


def build_strategy(spec: StrategySpec) -> Callable[[pd.DataFrame], StrategyOutput]:
    """Trasforma una specifica in una funzione chiamabile dal runner walk-forward."""
    if spec.criterion not in SUPPORTED_CRITERIA:
        raise ValueError(
            f"Unknown criterion '{spec.criterion}'; expected one of {SUPPORTED_CRITERIA}."
        )
    if spec.cov_method not in COVARIANCE_ESTIMATORS:
        raise ValueError(f"Unknown covariance estimator '{spec.cov_method}'.")
    if spec.returns_method not in RETURNS_ESTIMATORS:
        raise ValueError(f"Unknown returns estimator '{spec.returns_method}'.")

    estimate_cov = COVARIANCE_ESTIMATORS[spec.cov_method]
    estimate_mu = RETURNS_ESTIMATORS[spec.returns_method]

    def strategy(train: pd.DataFrame) -> StrategyOutput:
        window = train.dropna(axis=1, how="all").dropna()
        if window.shape[1] == 0 or len(window) < 2:
            raise ValueError("Estimation window is too short or empty.")

        covariance = estimate_cov(window)
        expected = estimate_mu(window)
        mu_mvo = ito_adjusted(expected, covariance)

        if spec.criterion == "equal_weight":
            weights = equal_weight_portfolio(window)
        elif spec.criterion == "min_variance":
            weights = minimum_variance_portfolio(covariance, max_weight=spec.max_weight)
        elif spec.criterion == "max_sharpe":
            weights = maximum_sharpe_portfolio(
                mu_mvo,
                covariance,
                risk_free_rate=spec.risk_free_rate,
                max_weight=spec.max_weight,
            )
        elif spec.criterion == "risk_parity":
            weights = equal_risk_contribution(covariance, max_weight=spec.max_weight)
        elif spec.criterion == "black_litterman":
            weights = _black_litterman_weights(window, covariance, spec)
        else:  # min_cvar
            weights = minimum_cvar_portfolio(
                window, alpha=spec.cvar_alpha, max_weight=spec.max_weight
            )

        weights = weights.reindex(window.columns).fillna(0.0)
        total = float(weights.sum())
        if total <= 0:
            raise RuntimeError(f"Criterion '{spec.criterion}' returned null weights.")
        weights = weights / total

        return StrategyOutput(
            weights=weights,
            expected_returns=mu_mvo,
            covariance=covariance,
        )

    return strategy
