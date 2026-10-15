from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew


def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def normality_tests(returns: pd.DataFrame) -> pd.DataFrame:
    """Forma della distribuzione dei rendimenti per titolo e test formali di normalita'.

    Convenzione sul quarto momento: la colonna e' la curtosi IN ECCESSO
    (``fisher=True``), zero per una distribuzione normale, positiva per una
    leptocurtica. La letteratura finanziaria riporta quasi sempre la forma
    in eccesso (Cont, 2001), ed e' la forma che questo progetto usa
    dappertutto, inclusa la metrica a livello di portafoglio in
    ``optimize_portfolio``. La colonna si chiama ``excess_kurtosis`` e non
    ``kurtosis`` apposta, cosi' la convenzione non si puo' leggere male solo
    guardando l'output: una tabella di "kurtosis" intorno a 3 e una di
    "excess kurtosis" intorno a 0 descrivono lo stesso identico dato.
    """
    from scipy.stats import jarque_bera, kurtosis, shapiro, skew

    rows = []
    for col in returns.columns:
        series = returns[col].dropna()
        _, jb_p = jarque_bera(series)
        _, sh_p = shapiro(
            series if len(series) <= 5000 else series.sample(5000, random_state=42)
        )
        rows.append(
            {
                "asset": col,
                "skewness": float(skew(series)),
                "excess_kurtosis": float(kurtosis(series, fisher=True)),
                "jarque_bera_p": jb_p,
                "shapiro_p": sh_p,
                "is_normal_jb": jb_p > 0.05,
            }
        )
    return pd.DataFrame(rows).set_index("asset")


def var_historical(returns: pd.Series, alpha: float = 0.95) -> float:
    return float(-np.percentile(returns.dropna(), (1 - alpha) * 100))


def cvar_historical(returns: pd.Series, alpha: float = 0.95) -> float:
    threshold = np.percentile(returns.dropna(), (1 - alpha) * 100)
    tail = returns[returns <= threshold]
    return float(-tail.mean())


def equity_curve(
    returns: pd.Series | np.ndarray,
    initial_capital: float = 1.0,
    is_log_returns: bool = True,
) -> pd.Series | np.ndarray:
    """Calcola la traiettoria di ricchezza/equity partendo da initial_capital."""
    if isinstance(returns, pd.Series):
        clean = returns.dropna()
        if is_log_returns:
            return float(initial_capital) * np.exp(clean.cumsum())
        return float(initial_capital) * (1.0 + clean).cumprod()
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return np.array([])
    if is_log_returns:
        return float(initial_capital) * np.exp(np.cumsum(arr))
    return float(initial_capital) * np.cumprod(1.0 + arr)


def drawdown_series(
    cumulative_returns: pd.Series | np.ndarray,
    initial_capital: float = 1.0,
) -> pd.Series | np.ndarray:
    """Calcola la serie di drawdown rispetto al massimo corrente, incluso il capitale iniziale.

    Il drawdown e' non-positivo: 0.0 sul picco, negativo durante i ribassi.
    """
    if isinstance(cumulative_returns, pd.Series):
        running_max = np.maximum(float(initial_capital), cumulative_returns.cummax())
        return cumulative_returns / running_max - 1.0
    arr = np.asarray(cumulative_returns, dtype=float)
    if arr.size == 0:
        return np.array([])
    running_max = np.maximum(float(initial_capital), np.maximum.accumulate(arr))
    return arr / running_max - 1.0


def max_drawdown(
    cumulative_returns: pd.Series | np.ndarray,
    initial_capital: float = 1.0,
) -> float:
    """Massimo calo picco-valle, espresso come un float non-positivo.

    Il massimo corrente include il capitale iniziale (default 1.0), cosi'
    una perdita immediata alla prima osservazione viene misurata
    correttamente come drawdown invece di risultare 0.0.
    """
    dd = drawdown_series(cumulative_returns, initial_capital=initial_capital)
    if len(dd) == 0:
        return 0.0
    return float(np.min(dd))



def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
) -> float:
    """Indice di Sortino con la semi-deviazione verso il basso corretta: sqrt(E[min(r-rf,0)^2] * periods)."""
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess = returns - rf_daily
    downside_sq = np.minimum(excess, 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean() * periods))
    if downside_dev < 1e-10 or np.isnan(downside_dev):
        return np.nan
    return float(excess.mean() * periods / downside_dev)


def is_degenerate_dispersion(values: np.ndarray | pd.Series, dispersion: float) -> bool:
    """Se una deviazione standard e' zero per tutti gli scopi che contano.

    ``sd == 0`` e' la guardia ovvia e non funziona. L'algoritmo a due
    passate che numpy e pandas usano lascia un residuo su una serie
    costante: una serie di 0.0001 ripetuto ha una deviazione standard di
    circa 2.7e-20, non zero, quindi la guardia passa e il rapporto viene
    fuori 5.9e16 invece di essere rifiutato. Entrambe le definizioni
    dell'indice di Sharpe in questo package avevano questo difetto, e il
    modulo di inferenza poi riportava la differenza come significativa
    all'uno per cento.

    Il test e' quindi relativo alla scala dei dati, ed e' proprio questo
    che lo rende indipendente dall'unita' in cui sono espressi i
    rendimenti.
    """
    array = np.asarray(values, dtype=float)
    if not np.isfinite(dispersion):
        return True
    scale = float(np.mean(np.abs(array))) if array.size else 0.0
    return dispersion <= 1e-12 * max(scale, np.finfo(float).tiny)


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
) -> float:
    vol = returns.std(ddof=1)
    if is_degenerate_dispersion(returns, float(vol)):
        return np.nan
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    excess = returns.mean() - rf_daily
    return float((excess / vol) * np.sqrt(periods))


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    cum = np.exp(returns.cumsum())
    mdd = abs(max_drawdown(cum))
    if mdd == 0:
        return np.nan
    n = max(len(returns), 1)
    cagr = float(cum.iloc[-1] ** (periods / n) - 1)
    return cagr / mdd


def omega_ratio(
    returns: pd.Series,
    threshold: float | None = None,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """Ω(L) = Σ(r_i - L | r_i > L) / Σ(L - r_i | r_i ≤ L).

    threshold : MAR giornaliero. Se None, viene derivato da risk_free_rate
        con la formula composta (1+rf)^(1/252)-1. Il default (sia None che
        rf=0) e' la convenzione originale di Keating-Shadwick (2002). Per
        valutare un portafoglio passa risk_free_rate=rf_annual.
    """
    if threshold is None:
        threshold = (1.0 + risk_free_rate) ** (1.0 / periods) - 1.0
    gains = (returns - threshold)[returns > threshold].sum()
    losses = (threshold - returns)[returns <= threshold].sum()
    if losses == 0:
        return np.nan
    return float(gains / losses)


def summary_metrics(
    portfolio_log_returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods: int = 252,
) -> dict[str, float]:
    """Il pannello di misure che l'applicazione riporta per un portafoglio.

    Una sola definizione, chiamata da tutti i punti in cui il pannello
    viene mostrato. Il caso d'uso dell'ottimizzazione e il confronto fra
    modelli costruivano questo dizionario separatamente, ed e' cosi' che
    una tabella di valori in campione e una fuori campione potevano finire
    per voler dire cose diverse sotto la stessa intestazione di colonna.
    Un'unica funzione le rende confrontabili per costruzione.

    Convenzioni, tutte verificate in ``tests/test_conventions.py``:

    - l'input e' una serie di rendimenti LOG GIORNALIERI del portafoglio;
    - il rendimento annualizzato e' geometrico (CAGR), non la media
      aritmetica;
    - il tasso privo di rischio viene composto in forma giornaliera, non
      diviso per 252;
    - VaR e CVaR sono grandezze di perdita POSITIVE al livello del 95 per
      cento, quindi un valore piu' piccolo e' meglio;
    - il massimo drawdown e' NEGATIVO, quindi un valore piu' grande e'
      meglio;
    - il quarto momento e' la curtosi IN ECCESSO, zero per una
      distribuzione normale.

    Le convenzioni di segno contano, e non e' solo contabilita': una
    tabella di confronto deve decidere quale colonna vince, e puo' farlo
    solo partendo dalla convenzione di segno della misura. Sono scritte
    qui perche' questo e' l'unico posto in cui vengono stabilite.
    """
    series = portfolio_log_returns.dropna()
    n = max(len(series), 1)
    cumulative = float(np.exp(series.sum()))
    return {
        "return_annual": float(cumulative ** (periods / n) - 1),
        "volatility_annual": float(series.std(ddof=1) * np.sqrt(periods)),
        "sharpe_ratio": sharpe_ratio(series, risk_free_rate=risk_free_annual, periods=periods),
        "sortino_ratio": sortino_ratio(series, risk_free_rate=risk_free_annual, periods=periods),
        "calmar_ratio": calmar_ratio(series, periods=periods),
        "omega_ratio": omega_ratio(series, risk_free_rate=risk_free_annual, periods=periods),
        "var_95": var_historical(series, alpha=0.95),
        "cvar_95": cvar_historical(series, alpha=0.95),
        "max_drawdown": max_drawdown(np.exp(series.cumsum())),
        "skewness": float(_skew(series)),
        "kurtosis": float(_kurtosis(series, fisher=True)),
    }


# Direzione di preferenza per ogni misura di ``summary_metrics``. Una
# tabella di confronto non puo' segnare una colonna migliore e una
# peggiore senza questo, e leggere la direzione dall'etichetta e'
# esattamente l'errore che questa costante serve a evitare: "Max
# Drawdown" e' negativo, quindi piu' alto e' meglio, mentre "VaR 95%" e'
# una perdita positiva, quindi piu' basso e' meglio.

def sharpe_is_orderable(returns: pd.Series, risk_free_annual: float = 0.0,
                        periods: int = 252) -> bool:
    """Se l'indice di Sharpe di questa serie si puo' usare per classificarla.

    Il rapporto smette di essere monotono nella desiderabilita' appena il
    rendimento in eccesso diventa negativo: una strategia piu' volatile
    divide un numeratore negativo per un denominatore piu' grande e viene
    fuori meglio di una piu' calma che ha perso meno. Verificato su due
    serie di tre anni, una che perde il 10 per cento e l'altra il 30: gli
    indici di Sharpe sono -0.670 e -0.354, quindi la strategia che ha
    perso tre volte tanto mostra una differenza di +0.316 e qualsiasi
    regola che legga "differenza > 0" la dichiara vincitrice.

    Una sola definizione, condivisa da tutto quello che fa classifiche: le
    tabelle per sottoperiodo (G17) e la pagina di confronto (G30) ce
    l'avevano separata, che e' esattamente la duplicazione che questo
    audit continua a trovare.
    """
    series = returns.dropna()
    if series.empty:
        return False
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / periods) - 1.0
    return bool(float(series.mean()) > rf_daily)


HIGHER_IS_BETTER: dict[str, bool] = {
    "return_annual": True,
    "volatility_annual": False,
    "sharpe_ratio": True,
    "sortino_ratio": True,
    "calmar_ratio": True,
    "omega_ratio": True,
    "var_95": False,
    "cvar_95": False,
    "max_drawdown": True,
}
