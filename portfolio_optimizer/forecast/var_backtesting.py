"""Backtesting delle previsioni di Value at Risk.

Una previsione di VaR al livello alpha dichiara che le perdite la
supereranno in una frazione (1 - alpha) dei giorni. Due cose possono
andare storte, e si testano separatamente.

- Frequenza sbagliata. Le violazioni capitano troppo spesso o troppo
  raramente. E' il test di copertura incondizionata di Kupiec (1995).
- Timing sbagliato. Le violazioni capitano al tasso medio giusto ma si
  raggruppano a raffica, il che vuol dire che il modello non reagisce ai
  cambi di regime. E' il test di indipendenza di Christoffersen (1998).
  E' il fallimento piu' serio: un modello che fallisce qui e' cieco
  esattamente quando conta.

Il test congiunto di copertura condizionata e' la somma delle due
statistiche.

Il semaforo del Comitato di Basilea da' una lettura prudenziale del
numero di violazioni su 250 giorni di contrattazione al livello del 99
per cento, ed e' un modo compatto di presentare il risultato.

Riferimenti
----------
Kupiec, P. (1995), Journal of Derivatives 3(2).
Christoffersen, P. F. (1998), International Economic Review 39(4).
Basel Committee on Banking Supervision (1996), Supervisory Framework for the
Use of Backtesting in Conjunction with the Internal Models Approach.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(slots=True)
class CoverageTestResult:
    """Risultato dei test di copertura su una serie di violazioni di VaR."""

    confidence_level: float
    n_obs: int
    n_violations: int
    observed_rate: float
    expected_rate: float
    lr_unconditional: float
    p_unconditional: float
    lr_independence: float
    p_independence: float
    lr_conditional: float
    p_conditional: float
    basel_zone: str
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def _as_hits(hits: pd.Series | np.ndarray | list) -> np.ndarray:
    array = np.asarray(pd.Series(hits).dropna().to_numpy(), dtype=float)
    if array.size == 0:
        raise ValueError("Empty violation series.")
    unique = set(np.unique(array).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError("The violation series must contain only 0 and 1.")
    return array.astype(int)


def violations_from_returns(
    realized_returns: pd.Series, var_forecast: pd.Series | float
) -> pd.Series:
    """Serie binaria di violazioni da rendimenti realizzati e una previsione di VaR.

    ``var_forecast`` e' espresso come perdita positiva, quindi una
    violazione si ha quando il rendimento realizzato scende sotto il suo
    negativo.
    """
    realized = pd.Series(realized_returns).dropna()
    if np.isscalar(var_forecast):
        threshold = pd.Series(float(var_forecast), index=realized.index)
    else:
        threshold = pd.Series(var_forecast).reindex(realized.index).ffill()
    return (realized < -threshold).astype(int)


# ---------------------------------------------------------------------------
# Kupiec: copertura incondizionata
# ---------------------------------------------------------------------------


def kupiec_pof(hits: pd.Series | np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    """Test proportion-of-failures. Restituisce (statistica LR, p-value).

    H0: la probabilita' di violazione e' uguale a 1 - confidence_level.
    """
    array = _as_hits(hits)
    n = array.size
    x = int(array.sum())
    p = 1.0 - confidence_level

    if x == 0:
        lr = -2.0 * (n * np.log(1.0 - p))
    elif x == n:
        lr = -2.0 * (n * np.log(p))
    else:
        pi = x / n
        log_null = (n - x) * np.log(1.0 - p) + x * np.log(p)
        log_alt = (n - x) * np.log(1.0 - pi) + x * np.log(pi)
        lr = -2.0 * (log_null - log_alt)

    lr = float(max(lr, 0.0))
    return lr, float(1.0 - stats.chi2.cdf(lr, df=1))


# ---------------------------------------------------------------------------
# Christoffersen: indipendenza e copertura condizionata
# ---------------------------------------------------------------------------


def christoffersen_independence(hits: pd.Series | np.ndarray) -> tuple[float, float]:
    """Test di Markov per l'indipendenza fra violazioni consecutive.

    H0: la probabilita' di una violazione non dipende dal fatto che il
    giorno precedente fosse esso stesso una violazione.
    """
    array = _as_hits(hits)
    if array.size < 2:
        return 0.0, 1.0

    previous, current = array[:-1], array[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    # Casi degeneri: non si osserva nessuna transizione di uno dei due tipi.
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0, 1.0

    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi <= 0.0 or pi >= 1.0:
        return 0.0, 1.0

    def _safe_log(value: float) -> float:
        return np.log(value) if value > 0 else 0.0

    log_null = (n00 + n10) * _safe_log(1.0 - pi) + (n01 + n11) * _safe_log(pi)
    log_alt = (
        n00 * _safe_log(1.0 - pi_01)
        + n01 * _safe_log(pi_01)
        + n10 * _safe_log(1.0 - pi_11)
        + n11 * _safe_log(pi_11)
    )
    lr = float(max(-2.0 * (log_null - log_alt), 0.0))
    return lr, float(1.0 - stats.chi2.cdf(lr, df=1))


def christoffersen_conditional_coverage(
    hits: pd.Series | np.ndarray, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Test congiunto di frequenza corretta e indipendenza."""
    lr_uc, _ = kupiec_pof(hits, confidence_level)
    lr_ind, _ = christoffersen_independence(hits)
    lr_cc = lr_uc + lr_ind
    return float(lr_cc), float(1.0 - stats.chi2.cdf(lr_cc, df=2))


# ---------------------------------------------------------------------------
# Semaforo di Basilea
# ---------------------------------------------------------------------------


def basel_traffic_light(
    n_violations: int, n_obs: int = 250, confidence_level: float = 0.99
) -> str:
    """Zona prudenziale di un conteggio di violazioni, scalata alla dimensione del campione.

    Le soglie pubblicate si riferiscono a 250 giorni di contrattazione al
    livello del 99 per cento; sono riprodotte qui dalla distribuzione
    binomiale cosi' la lettura resta valida per campioni di lunghezza
    diversa.
    """
    p = 1.0 - confidence_level
    cumulative = stats.binom.cdf(n_violations, n_obs, p)
    if cumulative < 0.95:
        return "verde"
    if cumulative < 0.9999:
        return "giallo"
    return "rosso"


# ---------------------------------------------------------------------------
# Valutazione completa
# ---------------------------------------------------------------------------


def evaluate_var_forecasts(
    hits: pd.Series | np.ndarray, confidence_level: float = 0.95
) -> CoverageTestResult:
    """Fa girare tutta la batteria di test su una serie di violazioni."""
    array = _as_hits(hits)
    n = array.size
    x = int(array.sum())

    lr_uc, p_uc = kupiec_pof(array, confidence_level)
    lr_ind, p_ind = christoffersen_independence(array)
    lr_cc, p_cc = christoffersen_conditional_coverage(array, confidence_level)

    if p_uc >= 0.05 and p_ind >= 0.05:
        verdict = "copertura corretta e violazioni indipendenti"
    elif p_uc < 0.05 and p_ind >= 0.05:
        verdict = "frequenza delle violazioni non conforme al livello nominale"
    elif p_uc >= 0.05 and p_ind < 0.05:
        verdict = "violazioni raggruppate nel tempo: il modello non reagisce ai cambi di regime"
    else:
        verdict = "frequenza non conforme e violazioni raggruppate"

    return CoverageTestResult(
        confidence_level=confidence_level,
        n_obs=n,
        n_violations=x,
        observed_rate=x / n,
        expected_rate=1.0 - confidence_level,
        lr_unconditional=lr_uc,
        p_unconditional=p_uc,
        lr_independence=lr_ind,
        p_independence=p_ind,
        lr_conditional=lr_cc,
        p_conditional=p_cc,
        basel_zone=basel_traffic_light(x, n, confidence_level),
        verdict=verdict,
    )


def evaluate_registry(
    registry, levels: tuple[int, ...] = (95, 99), min_obs: int = 250
) -> pd.DataFrame:
    """Fa girare i test di copertura su ogni configurazione nel registro.

    Una riga per configurazione e livello di confidenza. I test non
    vengono mai eseguiti su un campione aggregato: piu' configurazioni
    valutate sugli stessi giorni di calendario condividono le loro
    violazioni, quindi aggregarle moltiplicherebbe la dimensione
    apparente del campione senza aggiungere informazione e manderebbe
    ogni p-value a zero.

    Usa ``summarise_coverage`` per ridurre queste righe a una per
    modello.
    """
    rows = []
    for config_id in registry.config_ids:
        for level in levels:
            hits = registry.hit_series(level, config_id=config_id)
            if len(hits) < min_obs:
                continue
            result = evaluate_var_forecasts(hits, confidence_level=level / 100.0)
            row = result.as_dict()
            row["config_id"] = config_id
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    ordered = ["config_id"] + [c for c in frame.columns if c != "config_id"]
    return frame[ordered]


def summarise_coverage(
    per_configuration: pd.DataFrame,
    group_by: str = "model",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Riduce i test per configurazione a una riga per gruppo.

    Riporta quante configurazioni sono state testate, la dispersione del
    tasso di violazione osservato, e la quota di configurazioni per cui
    l'ipotesi nulla di copertura corretta viene rigettata. Una quota e'
    il riassunto onesto di tanti test indipendenti; un singolo p-value
    ottenuto aggregandoli non lo e'.
    """
    if per_configuration.empty or group_by not in per_configuration.columns:
        return pd.DataFrame()

    rows = []
    for (group, level), part in per_configuration.groupby(
        [group_by, "confidence_level"]
    ):
        n = len(part)
        rows.append(
            {
                group_by: group,
                "confidence_level": level,
                "n_configurazioni": n,
                "osservazioni_per_configurazione": int(part["n_obs"].median()),
                "tasso_violazioni_mediano": float(part["observed_rate"].median()),
                "tasso_violazioni_min": float(part["observed_rate"].min()),
                "tasso_violazioni_max": float(part["observed_rate"].max()),
                "tasso_atteso": float(part["expected_rate"].iloc[0]),
                "quota_copertura_rifiutata": float(
                    (part["p_unconditional"] < alpha).mean()
                ),
                "quota_indipendenza_rifiutata": float(
                    (part["p_independence"] < alpha).mean()
                ),
                "p_mediano_copertura": float(part["p_unconditional"].median()),
                "zona_basel_prevalente": part["basel_zone"].mode().iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values([group_by, "confidence_level"])
