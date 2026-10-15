"""Valutazione delle previsioni di volatilita' che i modelli emettono su se' stessi.

Qui vivono due domande distinte.

Primo, se la volatilita' che un ottimizzatore dichiara per il proprio
portafoglio e' non distorta. In genere non lo e': scegliendo la
combinazione di titoli la cui covarianza stimata e' piu' bassa,
l'ottimizzatore seleziona anche gli errori di stima che puntano verso il
basso, quindi la volatilita' che annuncia e' sistematicamente sotto
quella che poi realizza. Il bias cresce col numero di titoli rispetto
alla lunghezza della finestra di stima. Misurarlo sui propri dati e' un
risultato di per se', e ha una conseguenza diretta per l'applicazione:
una volatilita' con un bias noto andrebbe corretta, o mostrata con un
intervallo, non presentata come una stima puntuale.

Secondo, quale stimatore di covarianza prevede meglio. La volatilita'
non si osserva, quindi il confronto usa la volatilita' realizzata del
periodo di detenzione come proxy e una funzione di perdita che resta
consistente anche con un proxy rumoroso — QLIKE e l'errore quadratico
sulla varianza vanno bene entrambe (Patton, 2011). Le differenze nella
perdita media vengono poi testate con Diebold e Mariano (1995), con la
correzione per piccoli campioni di Harvey, Leybourne e Newbold (1997).

Riferimenti
----------
Mincer, J. e Zarnowitz, V. (1969), in Economic Forecasts and Expectations.
Diebold, F. X. e Mariano, R. S. (1995), JBES 13(3).
Harvey, D., Leybourne, S. e Newbold, P. (1997), IJF 13(2).
Patton, A. J. (2011), Journal of Econometrics 160(1).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Funzioni di perdita
# ---------------------------------------------------------------------------


def mse_variance_loss(forecast_vol: np.ndarray, realized_vol: np.ndarray) -> np.ndarray:
    """Errore quadratico sulla varianza."""
    return (realized_vol**2 - forecast_vol**2) ** 2


def qlike_loss(forecast_vol: np.ndarray, realized_vol: np.ndarray) -> np.ndarray:
    """Perdita QLIKE: log(h) + proxy / h, sulle varianze.

    Penalizza la sottostima piu' pesantemente della sovrastima, che e'
    l'asimmetria che un risk manager affronta davvero, e resta
    consistente anche quando il proxy della varianza vera e' rumoroso.
    """
    h = np.maximum(forecast_vol**2, 1e-12)
    proxy = realized_vol**2
    return np.log(h) + proxy / h


LOSS_FUNCTIONS = {"qlike": qlike_loss, "mse": mse_variance_loss}


# ---------------------------------------------------------------------------
# Strumenti HAC
# ---------------------------------------------------------------------------


def _bartlett_variance(series: np.ndarray, bandwidth: int | None = None) -> float:
    """Varianza di lungo periodo di una serie scalare, kernel di Bartlett."""
    n = series.size
    centred = series - series.mean()
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(bandwidth, n - 1))
    value = float(centred @ centred) / n
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        value += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / n
    return max(value, 1e-18)


# ---------------------------------------------------------------------------
# Bias della volatilita' ex-ante
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VolatilityBiasResult:
    """Se la volatilita' dichiarata da un modello corrisponde a quella realizzata."""

    group: str
    n_obs: int
    mean_forecast: float
    mean_realized: float
    mean_error: float
    ratio_realized_to_forecast: float
    t_statistic: float
    p_value: float
    mz_intercept: float
    mz_slope: float
    mz_slope_t_vs_one: float
    mz_r_squared: float
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def _mincer_zarnowitz(
    forecast: np.ndarray, realized: np.ndarray
) -> tuple[float, float, float, float]:
    """Regredisce il realizzato sulla previsione; una previsione non distorta da' (0, 1).

    Gli errori standard sono corretti HAC: quando le finestre di
    ribilanciamento si sovrappongono i residui sono serialmente
    correlati e l'errore OLS semplice sottostima l'incertezza.
    """
    n = forecast.size
    design = np.column_stack([np.ones(n), forecast])
    coefficients, *_ = np.linalg.lstsq(design, realized, rcond=None)
    residuals = realized - design @ coefficients

    total = float(((realized - realized.mean()) ** 2).sum())
    r_squared = float(1.0 - (residuals**2).sum() / total) if total > 0 else np.nan

    xtx_inv = np.linalg.pinv(design.T @ design)
    scores = design * residuals[:, None]
    meat = np.zeros((2, 2))
    bandwidth = max(0, min(int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), n - 1))
    meat += scores.T @ scores
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inv @ meat @ xtx_inv
    slope_se = float(np.sqrt(max(covariance[1, 1], 1e-24)))
    slope_t = float((coefficients[1] - 1.0) / slope_se)
    return float(coefficients[0]), float(coefficients[1]), slope_t, r_squared


def volatility_bias(
    frame: pd.DataFrame,
    group_by: str | list[str] | None = None,
    label: str = "tutte",
) -> pd.DataFrame:
    """Confronta volatilita' dichiarata e realizzata, in totale o per gruppo.

    ``frame`` e' l'export di un ForecastRegistry. Le righe senza sia una
    previsione che un valore realizzato vengono scartate.

    ``group_by`` accetta piu' colonne insieme, e passarne piu' di una conta
    davvero: una scomposizione a una sola via per criterio di ottimizzazione
    mescola insieme gli stimatori di covarianza con cui quel criterio e'
    stato eseguito, e attribuisce all'ottimizzatore un bias che potrebbe
    appartenere a uno stimatore solo. Su questo progetto quella confusione
    ha prodotto una lettura sbagliata dei risultati, quindi la
    scomposizione a due vie e' quella da riportare.
    """
    required = {"expected_volatility_annual", "realized_volatility_annual"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns in the forecast frame: {sorted(missing)}")

    groups: list[tuple[str, pd.DataFrame]]
    if group_by is None:
        groups = [(label, frame)]
    elif isinstance(group_by, str):
        groups = [(str(key), part) for key, part in frame.groupby(group_by)]
    else:
        missing_cols = [c for c in group_by if c not in frame.columns]
        if missing_cols:
            raise KeyError(f"Missing grouping columns: {missing_cols}")
        groups = [
            (" | ".join(str(k) for k in key), part)
            for key, part in frame.groupby(list(group_by))
        ]

    rows = []
    for name, part in groups:
        clean = part[list(required)].replace([np.inf, -np.inf], np.nan).dropna()
        clean = clean[(clean["expected_volatility_annual"] > 0)]
        if len(clean) < 10:
            continue
        forecast = clean["expected_volatility_annual"].to_numpy()
        realized = clean["realized_volatility_annual"].to_numpy()
        error = realized - forecast

        variance = _bartlett_variance(error)
        t_stat = float(error.mean() / np.sqrt(variance / error.size))
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
        intercept, slope, slope_t, r_squared = _mincer_zarnowitz(forecast, realized)

        if p_value >= 0.05:
            verdict = "nessuna distorsione rilevabile"
        elif error.mean() > 0:
            verdict = "volatilita dichiarata inferiore alla realizzata (sottostima)"
        else:
            verdict = "volatilita dichiarata superiore alla realizzata (sovrastima)"

        rows.append(
            VolatilityBiasResult(
                group=name,
                n_obs=len(clean),
                mean_forecast=float(forecast.mean()),
                mean_realized=float(realized.mean()),
                mean_error=float(error.mean()),
                ratio_realized_to_forecast=float(realized.mean() / forecast.mean()),
                t_statistic=t_stat,
                p_value=p_value,
                mz_intercept=intercept,
                mz_slope=slope,
                mz_slope_t_vs_one=slope_t,
                mz_r_squared=r_squared,
                verdict=verdict,
            ).as_dict()
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ForecastComparisonResult:
    """Accuratezza comparata di due modelli previsivi sotto una stessa funzione di perdita."""

    model_1: str
    model_2: str
    loss: str
    n_obs: int
    mean_loss_1: float
    mean_loss_2: float
    loss_difference: float
    dm_statistic: float
    p_value: float
    better: str

    def as_dict(self) -> dict:
        return asdict(self)


def diebold_mariano(
    loss_1: np.ndarray,
    loss_2: np.ndarray,
    horizon: int = 1,
    small_sample_correction: bool = True,
) -> tuple[float, float]:
    """Test H0: i due modelli previsivi hanno la stessa perdita attesa.

    Una statistica negativa favorisce il primo modello. La correzione di
    Harvey-Leybourne-Newbold conta qui perche' il numero di date di
    ribilanciamento in un esercizio walk-forward e' piccolo.
    """
    difference = np.asarray(loss_1, dtype=float) - np.asarray(loss_2, dtype=float)
    difference = difference[np.isfinite(difference)]
    n = difference.size
    if n < 10:
        raise ValueError(f"At least 10 paired observations are required; got {n}.")

    variance = _bartlett_variance(difference, bandwidth=max(horizon - 1, 0) or None)
    statistic = float(difference.mean() / np.sqrt(variance / n))

    if small_sample_correction:
        factor = np.sqrt(
            (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        )
        statistic *= float(factor)
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1)))
    else:
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))
    return statistic, p_value


_CONTEXT_COLS: tuple[str, ...] = (
    "scenario",
    "model",
    "returns_method",
    "horizon_days",
    "step_days",
)


def compare_estimators(
    frame: pd.DataFrame,
    group_by: str = "cov_method",
    loss: str = "qlike",
    pair_on: str = "issue_date",
    context_cols: tuple[str, ...] = _CONTEXT_COLS,
) -> pd.DataFrame:
    """Confronto a coppie dell'accuratezza delle previsioni di volatilita', per stimatore.

    Fa girare un test di Diebold-Mariano per ogni (gruppo di contesto,
    coppia di stimatori). Ogni gruppo di contesto e' una combinazione
    unica di ``context_cols`` (es. scenario, modello, returns_method,
    horizon_days, step_days). Dentro ogni gruppo le previsioni sono
    accoppiate su ``pair_on`` (tipicamente ``issue_date``), cosi' il
    confronto e' omogeneo: due stimatori vengono sempre giudicati sulle
    stesse date di ribilanciamento. Mescolare le configurazioni come se
    fossero un'unica serie temporale gonfierebbe n e tratterebbe la
    variazione cross-sectional come indipendenza temporale — e' esattamente
    l'errore che questo raggruppamento evita.

    Le colonne di contesto assenti da ``frame`` vengono ignorate in silenzio.
    """
    if loss not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss '{loss}'; expected one of {sorted(LOSS_FUNCTIONS)}.")
    loss_fn = LOSS_FUNCTIONS[loss]

    working = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["expected_volatility_annual", "realized_volatility_annual"]
    )
    working = working[working["expected_volatility_annual"] > 0]

    if pair_on not in working.columns:
        raise KeyError(f"Pairing column '{pair_on}' not found in the frame.")

    actual_context = [c for c in context_cols if c in working.columns]

    # Evita un'aggregazione silenziosa: dentro ogni cella (contesto + pair_on)
    # ci deve essere al massimo un'osservazione per stimatore.
    dup_keys = actual_context + [pair_on, group_by]
    duplicates = working[working.duplicated(subset=dup_keys, keep=False)]
    if not duplicates.empty:
        example_dup = duplicates[dup_keys].iloc[0].to_dict()
        raise ValueError(
            f"Duplicate observations found for pairing keys {dup_keys}: "
            f"e.g. {example_dup}. Pairing requires strictly unique keys per estimator."
        )

    working = working.assign(
        _loss=loss_fn(
            working["expected_volatility_annual"].to_numpy(),
            working["realized_volatility_annual"].to_numpy(),
        )
    )

    rows = []

    if actual_context:
        groups: list = list(working.groupby(actual_context, sort=False))
    else:
        groups = [(None, working)]

    for group_key, group_df in groups:
        if group_key is None:
            ctx: dict = {}
        elif isinstance(group_key, tuple):
            ctx = dict(zip(actual_context, group_key))
        else:
            ctx = {actual_context[0]: group_key}

        pivot = group_df.pivot(index=pair_on, columns=group_by, values="_loss")
        estimators = [c for c in pivot.columns if pivot[c].notna().sum() >= 10]

        for i, first in enumerate(estimators):
            for second in estimators[i + 1 :]:
                pair = pivot[[first, second]].dropna()
                if len(pair) < 10:
                    continue
                l1 = pair[first].to_numpy()
                l2 = pair[second].to_numpy()
                try:
                    statistic, p_value = diebold_mariano(l1, l2)
                except ValueError:
                    continue
                if p_value >= 0.05:
                    better = "differenza non significativa"
                else:
                    better = first if l1.mean() < l2.mean() else second
                row = ForecastComparisonResult(
                    model_1=str(first),
                    model_2=str(second),
                    loss=loss,
                    n_obs=len(pair),
                    mean_loss_1=float(l1.mean()),
                    mean_loss_2=float(l2.mean()),
                    loss_difference=float(l1.mean() - l2.mean()),
                    dm_statistic=statistic,
                    p_value=p_value,
                    better=better,
                ).as_dict()
                row.update(ctx)
                rows.append(row)

    return pd.DataFrame(rows)
