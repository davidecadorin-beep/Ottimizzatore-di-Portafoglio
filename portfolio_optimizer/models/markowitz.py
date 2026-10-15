from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from portfolio_optimizer.models.risk_parity import feasible_cap

_N_TRIALS_SHARPE = 6  # multi-start serve solo perche' lo Sharpe non e' convesso
_N_TRIALS_CONVEX = 1  # i problemi quadratici sulla varianza sono convessi — basta un solo punto di partenza


def _portfolio_performance(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray
) -> tuple[float, float]:
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(np.clip(np.dot(weights.T, np.dot(cov, weights)), 0, None)))
    return ret, vol


def _constraints(n_assets: int) -> list[dict]:
    return [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones(n_assets),
        }
    ]


def _bounds(
    n_assets: int, max_weight: float | None = None
) -> list[tuple[float, float]]:
    upper = 1.0 if max_weight is None else max_weight
    return [(0.0, upper)] * n_assets


def _starting_points(
    n: int, n_trials: int, seed: int | np.random.Generator | None = 42
) -> list[np.ndarray]:
    """Pesi equi + estrazioni Dirichlet riproducibili, per il multi-start."""
    points = [np.ones(n) / n]
    if n_trials > 1:
        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        for _ in range(n_trials - 1):
            points.append(rng.dirichlet(np.ones(n)))
    return points


def _best_minimize(
    objective,
    jac,
    n: int,
    bounds,
    constraints,
    n_trials: int = _N_TRIALS_SHARPE,
    seed: int | np.random.Generator | None = 42,
) -> np.ndarray:
    """Prova piu' punti di partenza; torna il miglior risultato ammissibile."""
    best_x, best_val = None, float("inf")
    for x0 in _starting_points(n, n_trials, seed=seed):
        res = minimize(
            objective,
            x0=x0,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 2000, "ftol": 1e-9},
        )
        # La convergenza va verificata. Senza il controllo su ``success`` un
        # risultato non convergente veniva letto come l'ottimo, e con un
        # vincolo infattibile ogni criterio restituiva l'equipesato: il
        # criterio sbagliato e, insieme, la violazione del tetto richiesto.
        if res.success and res.fun < best_val and not np.any(np.isnan(res.x)):
            best_val = res.fun
            best_x = res.x

    upper = bounds[0][1] if bounds else 1.0
    if best_x is None:
        raise RuntimeError(
            "Ottimizzazione non convergente su nessuno dei punti di partenza. "
            "Puo' dipendere da un vincolo troppo stretto o da una matrice di "
            "covarianza mal condizionata."
        )

    best_x = np.clip(best_x, 0.0, upper)
    total = best_x.sum()
    if total <= 0:
        raise RuntimeError("Ottimizzazione degenere: pesi tutti nulli.")
    best_x = best_x / total
    # La rinormalizzazione dopo il troncamento puo' riportare un peso sopra il
    # tetto — e' l'errore che rendeva il vincolo una preferenza nella parita'
    # di rischio. Qui il tetto e' gia' nei ``bounds`` del risolutore, quindi
    # uno sforamento residuo segnala una soluzione non ammissibile e non un
    # arrotondamento.
    if upper < 1.0 and best_x.max() > upper + 1e-6:
        raise RuntimeError(
            f"Soluzione non ammissibile: peso massimo {best_x.max():.4f} contro "
            f"un tetto di {upper:.4f}."
        )
    return best_x


def minimum_variance_portfolio(
    cov: pd.DataFrame,
    max_weight: float | None = None,
    seed: int | np.random.Generator | None = 42,
) -> pd.Series:
    n = cov.shape[0]
    feasible_cap(n, max_weight)
    sigma = cov.values

    def objective(w: np.ndarray) -> float:
        return float(np.dot(w.T, np.dot(sigma, w)))

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * sigma @ w

    bounds = _bounds(n, max_weight)
    weights = _best_minimize(
        objective,
        jac,
        n,
        bounds,
        _constraints(n),
        n_trials=_N_TRIALS_CONVEX,
        seed=seed,
    )
    return pd.Series(weights, index=cov.index, name="min_variance_weights")


def maximum_sharpe_portfolio(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float | None = None,
    seed: int | np.random.Generator | None = 42,
) -> pd.Series:
    n = len(expected_returns)
    feasible_cap(n, max_weight)
    mu = expected_returns.values.astype(np.float64)
    sigma = cov.values.astype(np.float64)

    def objective(w: np.ndarray) -> float:
        ret, vol = _portfolio_performance(w, mu, sigma)
        if vol < 1e-10:
            return 1e9
        return -((ret - risk_free_rate) / vol)

    def jac(w: np.ndarray) -> np.ndarray:
        ret, vol = _portfolio_performance(w, mu, sigma)
        if vol < 1e-10:
            return np.zeros(n)
        # ∇[-( μ'w - rf)/vol] = -μ/vol + (μ'w - rf) * Σw / vol³
        sigma_w = sigma @ w
        return -mu / vol + (ret - risk_free_rate) * sigma_w / (vol**3)

    bounds = _bounds(n, max_weight)
    weights = _best_minimize(
        objective,
        jac,
        n,
        bounds,
        _constraints(n),
        n_trials=_N_TRIALS_SHARPE,
        seed=seed,
    )
    return pd.Series(weights, index=expected_returns.index, name="max_sharpe_weights")


def efficient_frontier(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 30,
    max_weight: float | None = None,
    seed: int | np.random.Generator | None = 42,
) -> pd.DataFrame:
    n = len(expected_returns)
    feasible_cap(n, max_weight)
    mu = expected_returns.values.astype(np.float64)
    sigma = cov.values.astype(np.float64)
    bounds = _bounds(n, max_weight)

    # Risolve il rendimento minimo e massimo raggiungibile sotto i vincoli, in modo esatto
    res_min = linprog(c=mu, A_eq=[np.ones(n)], b_eq=[1.0], bounds=bounds, method="highs")
    min_ret = float(res_min.fun) if res_min.success else float(mu.min())

    res_max = linprog(c=-mu, A_eq=[np.ones(n)], b_eq=[1.0], bounds=bounds, method="highs")
    max_ret = float(-res_max.fun) if res_max.success else float(mu.max())

    if max_ret < min_ret:
        max_ret = min_ret

    targets = np.linspace(min_ret, max_ret, n_points)
    rows: list[dict] = []

    def obj(w: np.ndarray) -> float:
        return float(np.dot(w.T, np.dot(sigma, w)))

    def obj_jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * sigma @ w

    for target in targets:
        constraints = _constraints(n) + [
            {
                "type": "eq",
                "fun": lambda w, t=target: float(np.dot(w, mu) - t),
                "jac": lambda w, m=mu: m,
            }
        ]
        best_x, best_val = None, float("inf")
        for x0 in _starting_points(n, _N_TRIALS_CONVEX, seed=seed):
            res = minimize(
                obj,
                x0=x0,
                jac=obj_jac,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            if res.success and not np.any(np.isnan(res.x)) and res.fun < best_val:
                best_val, best_x = res.fun, res.x
        if best_x is None:
            warnings.warn(
                f"Frontiera efficiente: target return {target:.6f} non raggiungibile o non convergente.",
                UserWarning,
                stacklevel=2,
            )
            continue
        ret, vol = _portfolio_performance(best_x, mu, sigma)
        rows.append(
            {
                "target_return": target,
                "return": ret,
                "volatility": vol,
                "weights": best_x,
            }
        )
    return pd.DataFrame(rows)



def minimum_cvar_portfolio(
    scenarios: pd.DataFrame,
    alpha: float = 0.95,
    max_weight: float = 1.0,
) -> pd.Series:
    """Minimizza il CVaR sugli scenari di rendimento usando la programmazione lineare."""
    import cvxpy as cp

    # Il CVaR e' definito sui rendimenti semplici; convertiamo i log-rendimenti in semplici
    scenarios = np.expm1(scenarios)

    n_assets = scenarios.shape[1]
    feasible_cap(n_assets, max_weight)
    n_obs = scenarios.shape[0]
    w = cp.Variable(n_assets)
    z = cp.Variable(n_obs)
    var = cp.Variable()
    losses = -(scenarios.values @ w)
    cvar = var + (1.0 / ((1 - alpha) * n_obs)) * cp.sum(z)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        z >= 0,
        z >= losses - var,
    ]
    problem = cp.Problem(cp.Minimize(cvar), constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    tol = 1e-4
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"L'ottimizzazione del CVaR non ha trovato una soluzione ottima (stato: '{problem.status}')."
        )
    if w.value is None:
        raise RuntimeError("L'ottimizzazione del CVaR non ha restituito nessun peso.")

    w_arr = np.asarray(w.value, dtype=float).ravel()
    if not np.all(np.isfinite(w_arr)):
        raise RuntimeError("L'ottimizzazione del CVaR ha restituito pesi non finiti.")
    if abs(float(np.sum(w_arr)) - 1.0) > tol:
        raise RuntimeError(
            f"I pesi dell'ottimizzazione CVaR non sommano a 1 (somma={np.sum(w_arr):.6f}, tolleranza={tol})."
        )
    if np.any(w_arr < -tol):
        raise RuntimeError(
            f"I pesi dell'ottimizzazione CVaR violano il vincolo di non negativita' (min={np.min(w_arr):.6f}, tolleranza={tol})."
        )
    if np.any(w_arr > max_weight + tol):
        raise RuntimeError(
            f"I pesi dell'ottimizzazione CVaR violano il vincolo max_weight (max={np.max(w_arr):.6f}, max_weight={max_weight}, tolleranza={tol})."
        )

    w_arr = np.clip(w_arr, 0.0, max_weight)
    total_w = float(np.sum(w_arr))
    if total_w > 0:
        w_arr = w_arr / total_w
    return pd.Series(w_arr, index=scenarios.columns, name="min_cvar_weights")
