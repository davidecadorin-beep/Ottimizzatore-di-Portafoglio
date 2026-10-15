"""Inferenza statistica sulle differenze di performance aggiustata per il rischio.

Confrontare due strategie guardando solo i loro indici di Sharpe non basta:
va dimostrato che la differenza e' distinguibile da zero. Questo modulo
implementa i due test standard per farlo.

- Jobson e Korkie (1981) con la correzione di Memmel (2003): formula chiusa,
  assume rendimenti iid normali. Veloce, va bene come primo screening.
- Ledoit e Wolf (2008): errore standard HAC via il metodo delta sul vettore
  dei momenti, robusto ad autocorrelazione e code pesanti, con la
  possibilita' di un p-value da bootstrap a blocchi circolari studentizzato.
  E' il test da riportare quando i rendimenti sono giornalieri e non
  normali, che e' il caso di solito.

Tutti i test lavorano su indici di Sharpe NON annualizzati; l'annualizzazione
moltiplica entrambi i rapporti per la stessa costante e viene riportata a
parte per leggibilita'.

Riferimenti
----------
Jobson, J. D. e Korkie, B. M. (1981), Journal of Finance 36(4).
Memmel, C. (2003), Finance Letters 1.
Ledoit, O. e Wolf, M. (2008), Journal of Empirical Finance 15(5).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from portfolio_optimizer.risk.metrics import is_degenerate_dispersion


@dataclass(slots=True)
class SharpeTestResult:
    """Risultato di un test sulla differenza fra due indici di Sharpe."""

    test: str
    n_obs: int
    sharpe_1: float
    sharpe_2: float
    sharpe_1_annual: float
    sharpe_2_annual: float
    difference_annual: float
    std_error: float
    statistic: float
    p_value: float
    significant_5pct: bool

    def as_dict(self) -> dict:
        return asdict(self)


MIN_OBSERVATIONS = 30


def _align(
    returns_1: pd.Series, returns_2: pd.Series, rf_daily: float
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.concat([returns_1, returns_2], axis=1).dropna()
    if len(frame) < MIN_OBSERVATIONS:
        raise ValueError(
            f"At least {MIN_OBSERVATIONS} overlapping observations are required; "
            f"got {len(frame)}."
        )
    x = frame.iloc[:, 0].to_numpy(dtype=float) - rf_daily
    y = frame.iloc[:, 1].to_numpy(dtype=float) - rf_daily
    return x, y


def non_annualised_sharpe(excess: np.ndarray) -> float:
    """Indice di Sharpe NON annualizzato di un array di rendimenti in eccesso.

    Chiamato apposta NON ``sharpe_ratio``. Quel nome appartiene a
    ``risk.metrics``, dove e' il rapporto annualizzato calcolato da una
    serie di rendimenti log e un tasso privo di rischio annuale. I due
    differiscono per un fattore sqrt(252) e per cosa si aspettano in
    ingresso: questo riceve eccessi gia' netti del tasso privo di rischio,
    quello se li netta da solo.

    Entrambi sono corretti dove vengono usati -- i test di
    Jobson-Korkie-Memmel e Ledoit-Wolf sono definiti sul rapporto non
    annualizzato -- ma avere due definizioni della stessa grandezza con lo
    stesso nome e' la forma di quasi tutti i difetti trovati nell'audit. Il
    nome e' quello che li tiene separati.
    """
    sd = float(np.asarray(excess, dtype=float).std(ddof=1))
    if is_degenerate_dispersion(excess, sd):
        raise ValueError(
            "Zero standard deviation; Sharpe ratio is undefined. "
            "The guard is relative to the scale of the series: an exact "
            "comparison with zero lets a constant series through with a "
            "ratio of order 1e16."
        )
    return float(np.asarray(excess, dtype=float).mean() / sd)


_DEGENERATE_TOLERANCE = 1e-12


def _degenerate_safe(variance: float, difference: float) -> tuple[float, float, float]:
    """Errore standard, statistica e p-value, tollerando una varianza degenere.

    Due strategie perfettamente correlate e con lo stesso indice di Sharpe
    — una serie confrontata con se' stessa, o con un suo multiplo positivo
    — implicano una varianza stimata esattamente zero. Non e' un errore
    numerico, e' la risposta corretta a una domanda senza contenuto: la
    differenza e' zero e non si puo' distinguere da zero.
    """
    if variance > _DEGENERATE_TOLERANCE:
        std_error = float(np.sqrt(variance))
        statistic = float(difference / std_error)
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))
        return std_error, statistic, p_value
    if abs(difference) <= _DEGENERATE_TOLERANCE:
        return 0.0, 0.0, 1.0
    raise ValueError(
        "Non-positive variance estimate with a non-zero Sharpe difference; "
        "the sample is too short or the series are degenerate."
    )


# ---------------------------------------------------------------------------
# Jobson-Korkie / Memmel
# ---------------------------------------------------------------------------


def jobson_korkie_memmel(
    returns_1: pd.Series,
    returns_2: pd.Series,
    rf_daily: float = 0.0,
    periods: int = 252,
) -> SharpeTestResult:
    """Test H0: SR_1 == SR_2 sotto normalita' iid.

    La varianza della differenza di Sharpe segue Memmel (2003), che
    corregge un errore algebrico nella derivazione originale di
    Jobson-Korkie.
    """
    x, y = _align(returns_1, returns_2, rf_daily)
    n = len(x)
    sr_x, sr_y = non_annualised_sharpe(x), non_annualised_sharpe(y)
    rho = float(np.corrcoef(x, y)[0, 1])

    variance = (
        2.0
        - 2.0 * rho
        + 0.5 * (sr_x**2 + sr_y**2 - 2.0 * sr_x * sr_y * rho**2)
    ) / n
    std_error, statistic, p_value = _degenerate_safe(variance, sr_x - sr_y)
    scale = np.sqrt(periods)

    return SharpeTestResult(
        test="jobson_korkie_memmel",
        n_obs=n,
        sharpe_1=sr_x,
        sharpe_2=sr_y,
        sharpe_1_annual=sr_x * scale,
        sharpe_2_annual=sr_y * scale,
        difference_annual=(sr_x - sr_y) * scale,
        std_error=std_error,
        statistic=float(statistic),
        p_value=p_value,
        significant_5pct=p_value < 0.05,
    )


# ---------------------------------------------------------------------------
# Ledoit-Wolf: errore standard HAC e bootstrap a blocchi studentizzato
# ---------------------------------------------------------------------------


def _sharpe_difference(moments: np.ndarray) -> float:
    """Delta = SR_1 - SR_2 in funzione di v = (mu_1, mu_2, gamma_1, gamma_2)."""
    mu_1, mu_2, gamma_1, gamma_2 = moments
    var_1 = gamma_1 - mu_1**2
    var_2 = gamma_2 - mu_2**2
    if var_1 <= 0 or var_2 <= 0:
        raise ValueError("Non-positive implied variance in Sharpe difference.")
    return float(mu_1 / np.sqrt(var_1) - mu_2 / np.sqrt(var_2))


def _gradient(moments: np.ndarray) -> np.ndarray:
    """Gradiente della differenza di Sharpe rispetto al vettore dei momenti."""
    mu_1, mu_2, gamma_1, gamma_2 = moments
    var_1 = gamma_1 - mu_1**2
    var_2 = gamma_2 - mu_2**2
    return np.array(
        [
            gamma_1 / var_1**1.5,
            -gamma_2 / var_2**1.5,
            -0.5 * mu_1 / var_1**1.5,
            0.5 * mu_2 / var_2**1.5,
        ]
    )


def _bartlett_hac(series: np.ndarray, bandwidth: int | None = None) -> np.ndarray:
    """Stima HAC (Newey-West / Bartlett) della matrice di covarianza di lungo periodo."""
    n, k = series.shape
    centred = series - series.mean(axis=0)
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(bandwidth, n - 1))

    psi = centred.T @ centred / n
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = centred[lag:].T @ centred[:-lag] / n
        psi += weight * (gamma + gamma.T)
    return psi


def _moment_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack([x, y, x**2, y**2])


def _hac_variance(x: np.ndarray, y: np.ndarray, bandwidth: int | None) -> float:
    """Varianza della differenza di Sharpe col metodo delta, corretta HAC."""
    matrix = _moment_matrix(x, y)
    moments = matrix.mean(axis=0)
    grad = _gradient(moments)
    psi = _bartlett_hac(matrix, bandwidth)
    return float(grad @ psi @ grad) / len(x)


def ledoit_wolf_sharpe_test(
    returns_1: pd.Series,
    returns_2: pd.Series,
    rf_daily: float = 0.0,
    periods: int = 252,
    bandwidth: int | None = None,
    bootstrap: int = 0,
    block_size: int = 21,
    random_state: int = 42,
) -> SharpeTestResult:
    """Test H0: SR_1 == SR_2, robusto ad autocorrelazione e code pesanti.

    L'errore standard si ottiene col metodo delta applicato al vettore dei
    momenti (mu_1, mu_2, E[r_1^2], E[r_2^2]), la cui covarianza di lungo
    periodo e' stimata con un kernel di Bartlett. Con ``bootstrap`` maggiore
    di zero il p-value viene da un bootstrap a blocchi circolari
    studentizzato, che e' la procedura consigliata da Ledoit e Wolf per i
    dati giornalieri; altrimenti si usa l'approssimazione normale.
    """
    x, y = _align(returns_1, returns_2, rf_daily)
    n = len(x)

    moments = _moment_matrix(x, y).mean(axis=0)
    observed = _sharpe_difference(moments)
    std_error, statistic, p_value = _degenerate_safe(
        _hac_variance(x, y, bandwidth), observed
    )
    test_name = "ledoit_wolf_hac"

    if bootstrap > 0 and std_error > 0:
        rng = np.random.default_rng(random_state)
        n_blocks = int(np.ceil(n / block_size))
        studentized = np.empty(bootstrap)
        kept = 0
        for _ in range(bootstrap):
            starts = rng.integers(0, n, size=n_blocks)
            index = np.concatenate(
                [(np.arange(s, s + block_size) % n) for s in starts]
            )[:n]
            xb, yb = x[index], y[index]
            try:
                mb = _moment_matrix(xb, yb).mean(axis=0)
                diff_b = _sharpe_difference(mb)
                var_b = _hac_variance(xb, yb, bandwidth)
            except ValueError:
                continue
            if var_b <= _DEGENERATE_TOLERANCE:
                continue
            studentized[kept] = (diff_b - observed) / np.sqrt(var_b)
            kept += 1
        if kept < bootstrap // 2:
            raise ValueError("Bootstrap failed on too many resamples.")
        count = int(np.sum(np.abs(studentized[:kept]) >= abs(statistic)))
        p_value = float((1 + count) / (kept + 1))
        test_name = "ledoit_wolf_bootstrap"

    scale = np.sqrt(periods)
    sr_x, sr_y = non_annualised_sharpe(x), non_annualised_sharpe(y)
    return SharpeTestResult(
        test=test_name,
        n_obs=n,
        sharpe_1=sr_x,
        sharpe_2=sr_y,
        sharpe_1_annual=sr_x * scale,
        sharpe_2_annual=sr_y * scale,
        difference_annual=observed * scale,
        std_error=std_error,
        statistic=float(statistic),
        p_value=p_value,
        significant_5pct=p_value < 0.05,
    )


# ---------------------------------------------------------------------------
# Confronto a coppie contro una strategia di riferimento
# ---------------------------------------------------------------------------


def compare_against_benchmark(
    strategies: dict[str, pd.Series],
    benchmark_name: str,
    rf_daily: float = 0.0,
    periods: int = 252,
    robust: bool = True,
    bootstrap: int = 0,
) -> pd.DataFrame:
    """Testa ogni strategia contro un riferimento (di solito il portafoglio 1/N).

    Restituisce una riga per strategia, ordinata per differenza di Sharpe
    annualizzata.
    """
    if benchmark_name not in strategies:
        raise KeyError(f"Benchmark '{benchmark_name}' not among the strategies.")

    reference = strategies[benchmark_name]
    rows = []
    for name, series in strategies.items():
        if name == benchmark_name:
            continue
        if robust:
            result = ledoit_wolf_sharpe_test(
                series, reference, rf_daily, periods, bootstrap=bootstrap
            )
        else:
            result = jobson_korkie_memmel(series, reference, rf_daily, periods)
        row = result.as_dict()
        row["strategy"] = name
        row["benchmark"] = benchmark_name
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("strategy")
    return frame.sort_values("difference_annual", ascending=False)


# ---------------------------------------------------------------------------
# Romano-Wolf: lo stesso confronto, corretto per averlo ripetuto tante volte
# ---------------------------------------------------------------------------


def romano_wolf_stepdown(
    strategies: dict[str, pd.Series],
    benchmark_name: str,
    rf_daily: float = 0.0,
    periods: int = 252,
    bootstrap: int = 1000,
    block_size: int = 21,
    bandwidth: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Quali strategie battono il benchmark, tenendo conto della ricerca stessa.

    ``compare_against_benchmark`` risponde a una domanda alla volta: QUESTA
    strategia e' migliore del riferimento? Scegliere la migliore fra tante
    risposte e' una domanda diversa, e il p-value del vincitore non e' il
    p-value della ricerca che l'ha trovato. Una griglia di ventotto
    configurazioni ne testa ventisette al cinque per cento ciascuna e poi
    tiene la migliore: sotto l'ipotesi che nessuna sia davvero migliore, la
    probabilita' di trovarne almeno una "significativa" e' stata misurata
    sulle serie fuori campione di questo stesso progetto fra il 25 e il 29
    per cento, secondo lo scenario -- cinque volte il livello nominale. Non
    e' il 75 per cento che darebbe l'indipendenza, perche' le configurazioni
    sono molto correlate fra loro (correlazione media delle differenze fra
    0.36 e 0.44), ed e' proprio per questo che Bonferroni qui sarebbe il
    rimedio sbagliato: dividere alpha per ventisette assume un'indipendenza
    che i dati contraddicono, e butterebbe via vantaggi reali.

    La procedura stepdown di Romano e Wolf (2005) usa la correlazione
    invece di far finta che non ci sia. Si estrae un solo bootstrap a
    blocchi circolari e le STESSE date ricampionate vengono usate per ogni
    strategia, cosi' la distribuzione congiunta delle statistiche viene
    riprodotta insieme alle marginali. Il valore critico e' quindi il
    quantile della statistica MASSIMA tra le strategie ancora sotto test, e
    la procedura scende un passo alla volta: una volta che i vincitori piu'
    chiari vengono rigettati esci dalla famiglia, e quelle restanti vengono
    giudicate contro il massimo su un insieme piu' piccolo, il che
    recupera la potenza che una correzione in un solo passo butterebbe via.

    Il test e' A UNA CODA, perche' la domanda della raccomandazione e' a
    una coda: la regola propone una configurazione sopra il benchmark solo
    quando e' migliore, mai quando e' semplicemente diversa.

    Restituisce una riga per strategia con la differenza osservata, il
    p-value nominale che avrebbe avuto da sola, il p-value corretto per
    famiglia, e se viene rigettata al cinque per cento dopo la correzione.
    """
    if benchmark_name not in strategies:
        raise KeyError(f"Benchmark '{benchmark_name}' not among the strategies.")
    names = [n for n in strategies if n != benchmark_name]
    if not names:
        raise ValueError("Serve almeno una strategia oltre al riferimento.")

    frame = pd.concat(
        [strategies[benchmark_name].rename("__benchmark__")]
        + [strategies[n].rename(n) for n in names],
        axis=1,
    ).dropna()
    if len(frame) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Servono almeno {MIN_OBSERVATIONS} osservazioni comuni; ce ne sono "
            f"{len(frame)}."
        )

    reference = frame["__benchmark__"].to_numpy(dtype=float) - rf_daily
    excess = {n: frame[n].to_numpy(dtype=float) - rf_daily for n in names}
    n_obs = len(reference)

    observed = np.empty(len(names))
    std_errors = np.empty(len(names))
    for j, name in enumerate(names):
        moments = _moment_matrix(excess[name], reference).mean(axis=0)
        observed[j] = _sharpe_difference(moments)
        variance = _hac_variance(excess[name], reference, bandwidth)
        std_errors[j] = np.sqrt(variance) if variance > _DEGENERATE_TOLERANCE else np.nan

    statistic = observed / std_errors

    # Un solo insieme di date ricampionate per tutte le strategie: e' cio' che
    # conserva la correlazione fra loro, ed e' l'intera ragione per cui questa
    # correzione e' meno severa di Bonferroni.
    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n_obs / block_size))
    draws = np.full((bootstrap, len(names)), np.nan)
    for b in range(bootstrap):
        starts = rng.integers(0, n_obs, size=n_blocks)
        index = np.concatenate([np.arange(s, s + block_size) % n_obs for s in starts])
        index = index[:n_obs]
        reference_b = reference[index]
        for j, name in enumerate(names):
            series_b = excess[name][index]
            try:
                moments_b = _moment_matrix(series_b, reference_b).mean(axis=0)
                difference_b = _sharpe_difference(moments_b)
                variance_b = _hac_variance(series_b, reference_b, bandwidth)
            except (ValueError, ZeroDivisionError):
                continue
            if variance_b <= _DEGENERATE_TOLERANCE:
                continue
            draws[b, j] = (difference_b - observed[j]) / np.sqrt(variance_b)

    usable = ~np.isnan(draws).all(axis=1)
    draws = draws[usable]
    if len(draws) < bootstrap // 2:
        raise ValueError("Il bootstrap e' fallito su troppi ricampionamenti.")
    draws = np.nan_to_num(draws, nan=-np.inf)

    order = np.argsort(-np.nan_to_num(statistic, nan=-np.inf))
    adjusted = np.ones(len(names))
    remaining = list(order)
    previous = 0.0
    while remaining:
        maxima = draws[:, remaining].max(axis=1)
        j = remaining[0]
        if np.isnan(statistic[j]):
            adjusted[j] = 1.0
            remaining.pop(0)
            continue
        count = int(np.sum(maxima >= statistic[j]))
        p_value = float((1 + count) / (len(draws) + 1))
        # Monotonia lungo l'ordinamento: un p-value corretto non puo' scendere
        # scendendo in classifica.
        p_value = max(p_value, previous)
        adjusted[j] = p_value
        previous = p_value
        remaining.pop(0)

    single = np.array(
        [
            float((1 + int(np.sum(draws[:, j] >= statistic[j]))) / (len(draws) + 1))
            if not np.isnan(statistic[j])
            else 1.0
            for j in range(len(names))
        ]
    )

    scale = np.sqrt(periods)
    table = pd.DataFrame(
        {
            "strategy": names,
            "benchmark": benchmark_name,
            "difference_annual": observed * scale,
            "statistic": statistic,
            "p_value_single": single,
            "p_value_stepdown": adjusted,
            "significant_5pct": adjusted < 0.05,
            "n_obs": n_obs,
            "n_tested": len(names),
        }
    ).set_index("strategy")
    return table.sort_values("difference_annual", ascending=False)
