from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def feasible_cap(n_assets: int, max_weight: float | None) -> None:
    """Rifiuta un tetto di concentrazione che nessun portafoglio puo' soddisfare.

    Con ``n`` titoli e un tetto ``c``, i pesi possono sommare a uno solo se
    ``n * c >= 1``. Sotto quella soglia l'insieme dei vincoli e' vuoto, e un
    ottimizzatore a cui si da' un insieme ammissibile vuoto non fallisce in
    modo rumoroso: SLSQP torna un risultato non convergente che chi chiama
    legge come una soluzione.

    La conseguenza era che ogni criterio restituiva in silenzio il
    portafoglio equipesato — il criterio sbagliato E una violazione proprio
    del tetto che era stato chiesto. Si arriva li' anche con i default
    dell'applicazione: i criteri a minima varianza e minimo CVaR portano un
    tetto suggerito di 0.30, che richiede quattro titoli, e l'applicazione ne
    accetta due.
    """
    if max_weight is None or max_weight >= 1.0:
        return
    if n_assets * max_weight < 1.0 - 1e-12:
        minimum = int(np.ceil(1.0 / max_weight))
        raise ValueError(
            f"Il tetto del {max_weight:.0%} per titolo e' irraggiungibile con "
            f"{n_assets} titoli: i pesi non potrebbero sommare a uno. Servono "
            f"almeno {minimum} titoli, oppure un tetto di almeno "
            f"{1.0 / n_assets:.0%}."
        )


def _risk_contributions(weights: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return weights * (sigma @ weights)


def equal_risk_contribution(cov: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    """Equal Risk Contribution (ERC), nel rispetto del tetto di concentrazione.

    Senza vincoli, il problema e' la forma convessa a log-barriera di Spinu
    (2013):

        minimizza  w'Sigma w - (1/n) * sum(log w_i)   soggetto a  w_i > 0

    la cui condizione di primo ordine ``2 Sigma w = 1/(n w)`` rende uguale
    ogni ``w_i (Sigma w)_i`` — questa e' la proprieta' ERC. E' strettamente
    convesso, quindi la soluzione e' unica, e normalizzare non rompe la
    proprieta'.

    Quando il tetto e' vincolante i due obiettivi vanno in conflitto, e la
    vecchia implementazione risolveva il conflitto non risolvendo nessuno
    dei due: risolveva il problema senza vincoli, troncava i pesi al tetto
    e poi rinormalizzava — il che divide per un numero minore di uno e
    rimanda i pesi troncati esattamente sopra il tetto. Un tetto al 30 per
    cento restituiva un peso massimo del 40.8 per cento, una violazione di
    un terzo, e nemmeno i contributi al rischio erano piu' uguali.

    Il tetto e' un vincolo, non una preferenza, quindi viene imposto
    direttamente sul problema: sotto di esso si minimizza direttamente la
    dispersione dei contributi al rischio. Il risultato coincide con la
    soluzione di Spinu quando il tetto non e' vincolante, ed e' il
    portafoglio piu' vicino alla parita' di rischio raggiungibile quando lo
    e'. L'uguaglianza esatta dei contributi al rischio e' irraggiungibile
    sotto un tetto vincolante, ed e' una proprieta' del vincolo, non del
    risolutore.
    """
    n = cov.shape[0]
    feasible_cap(n, max_weight)
    sigma = cov.to_numpy()
    x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma @ w) - float(np.log(w).sum()) / n

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * sigma @ w - 1.0 / (n * w)

    res = minimize(
        objective,
        x0=x0,
        jac=gradient,
        bounds=[(1e-8, None)] * n,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9},
    )

    w = res.x if (res.x is not None and not np.any(np.isnan(res.x))) else x0
    w = np.maximum(w, 0.0)
    total = w.sum()
    w = w / total if total > 0 else np.ones(n) / n

    cap = 1.0 if max_weight is None else float(max_weight)
    if cap >= 1.0 or w.max() <= cap + 1e-9:
        return pd.Series(w, index=cov.index, name="risk_parity_weights")

    # Il tetto vincola: si minimizza la dispersione dei contributi al rischio
    # sotto il vincolo, anziche' troncare una soluzione che lo ignora.
    def dispersion(x: np.ndarray) -> float:
        contributions = _risk_contributions(x, sigma)
        return float(np.sum((contributions - contributions.mean()) ** 2))

    constrained = minimize(
        dispersion,
        x0=np.clip(w, 0.0, cap) / np.clip(w, 0.0, cap).sum(),
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-14},
    )
    if constrained.success and np.isfinite(constrained.fun):
        solution = np.clip(constrained.x, 0.0, cap)
        total = solution.sum()
        if total > 0:
            solution = solution / total
            # La proiezione finale non puo' riportare nulla sopra il tetto:
            # se lo facesse, il vincolo tornerebbe a essere una preferenza.
            if solution.max() <= cap + 1e-6:
                return pd.Series(
                    solution, index=cov.index, name="risk_parity_weights"
                )

    logger.warning(
        "Parita' di rischio col tetto %.2f non convergente; uso il portafoglio "
        "ammissibile piu' vicino all'equipesato.",
        cap,
    )
    fallback = np.minimum(np.ones(n) / n, cap)
    return pd.Series(
        fallback / fallback.sum(), index=cov.index, name="risk_parity_weights"
    )
