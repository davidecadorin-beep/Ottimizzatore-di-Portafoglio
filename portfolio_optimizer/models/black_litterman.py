from __future__ import annotations

import numpy as np
import pandas as pd


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float | None = None,
) -> pd.Series:
    """Pi = delta * Sigma * w.

    Se risk_aversion e' None, deduce delta da un premio azionario globale
    ipotizzato del 5%, seguendo He & Litterman (1999):
        delta = equity_premium / sigma_m^2
    troncato a [0.5, 10].
    """
    port_var = float(market_weights.values @ cov.values @ market_weights.values)
    if risk_aversion is None:
        _EQUITY_PREMIUM = 0.05
        risk_aversion = float(np.clip(_EQUITY_PREMIUM / max(port_var, 1e-8), 0.5, 10.0))
    pi = risk_aversion * cov.values @ market_weights.values
    return pd.Series(pi, index=cov.index, name="implied_returns")



_CONDITION_LIMIT = 1.0 / np.finfo(float).eps


def _inverse(matrix: np.ndarray, what: str) -> np.ndarray:
    """Inverte, o dice precisamente cosa non si poteva invertire e perche'.

    ``np.linalg.inv`` solleva un nudo ``LinAlgError: Singular matrix`` che non
    nomina ne' la matrice ne' la causa. Chi chiama lo registra nei log e
    ripiega sul portafoglio di equilibrio, quindi all'utente si dice che
    Black-Litterman e' fallito senza dirgli che due dei suoi titoli sono
    indistinguibili l'uno dall'altro -- che e' la parte su cui puo' davvero
    agire.
    """
    array = np.asarray(matrix, dtype=float)
    condition = np.linalg.cond(array)
    if not np.isfinite(condition) or condition > _CONDITION_LIMIT:
        raise np.linalg.LinAlgError(
            f"{what} non e' invertibile (numero di condizionamento {condition:.2e}). "
            "Di norma significa che due titoli hanno rendimenti quasi identici, "
            "oppure che le osservazioni sono meno dei titoli. Uno stimatore con "
            "shrinkage (Ledoit-Wolf) risolve entrambi i casi."
        )
    return np.linalg.inv(array)


def black_litterman_posterior(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    p_matrix: np.ndarray,
    q_views: np.ndarray,
    omega: np.ndarray | None = None,
    tau: float = 0.05,
    risk_aversion: float | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Calcola i rendimenti attesi a posteriori e la covarianza a posteriori di BL.

    Returns
    -------
    posterior_mu : pd.Series
    posterior_cov : pd.DataFrame
        Sigma + M, dove M = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1}.
        Passarla come covarianza all'ottimizzatore per tenere conto
        dell'incertezza delle view.
    """
    sigma = cov.values
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion).values
    p = np.asarray(p_matrix, dtype=float)
    q = np.asarray(q_views, dtype=float).reshape(-1, 1)
    if omega is None:
        omega = np.diag(np.diag(p @ (tau * sigma) @ p.T))

    # tau*Sigma e Omega vanno invertite entrambe. Una covarianza campionaria
    # su titoli collineari -- lo stesso titolo su due mercati, un ETF e il suo
    # gemello, piu' titoli che osservazioni -- e' singolare, e su di essa
    # np.linalg.inv solleva LinAlgError senza dire quale matrice sia il
    # problema. Verificato: con covarianza campionaria e due serie identiche il
    # determinante e' esattamente nullo; con Ledoit-Wolf lo shrinkage la rende
    # definita positiva e il calcolo procede. Anche Omega puo' esserlo, se una
    # view viene dichiarata a incertezza nulla.
    tau_sigma_inv = _inverse(tau * sigma, "la matrice di covarianza")
    omega_inv = _inverse(omega, "la matrice di incertezza delle view (Omega)")
    M_inv = tau_sigma_inv + p.T @ omega_inv @ p
    M = _inverse(M_inv, "la precisione a posteriori")
    m2 = tau_sigma_inv @ pi.reshape(-1, 1) + p.T @ omega_inv @ q
    posterior_mu = (M @ m2).ravel()

    posterior_cov_array = sigma + M
    return (
        pd.Series(posterior_mu, index=cov.index, name="bl_posterior_returns"),
        pd.DataFrame(posterior_cov_array, index=cov.index, columns=cov.columns),
    )
