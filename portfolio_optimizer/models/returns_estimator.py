from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FF3_DISK_CACHE_TTL = 86400  # 1 giorno
_FF3_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"

# ---------------------------------------------------------------------------
# Fama-French a 3 fattori — mappatura dei dataset regionali
# ---------------------------------------------------------------------------

_CURRENCY_TO_FF3_DATASET: dict[str, str] = {
    "USD": "F-F_Research_Data_Factors_daily",
    "CAD": "F-F_Research_Data_Factors_daily",
    "EUR": "Europe_3_Factors_Daily",
    "GBP": "Europe_3_Factors_Daily",
    "CHF": "Europe_3_Factors_Daily",
    "SEK": "Europe_3_Factors_Daily",
    "NOK": "Europe_3_Factors_Daily",
    "DKK": "Europe_3_Factors_Daily",
    "JPY": "Japan_3_Factors_Daily",
    "AUD": "Asia_Pacific_ex_Japan_3_Factors_Daily",
    "NZD": "Asia_Pacific_ex_Japan_3_Factors_Daily",
}
_DEFAULT_FF3_DATASET = "F-F_Research_Data_Factors_daily"

# Cache a livello di modulo: {(dataset, start, end): DataFrame}
_FF3_CACHE: dict[str, pd.DataFrame] = {}


def bayes_stein_expected_returns(
    returns: pd.DataFrame,
    annualization: int = 252,
) -> pd.Series:
    """Stimatore con contrazione (shrinkage) di Bayes-Stein di Jorion (1986) per i rendimenti attesi.

    Contrae le medie storiche dei titoli verso il rendimento del portafoglio
    a varianza minima globale, con l'intensita' di contrazione calcolata in
    modo analitico. Riduce l'errore di stima nei rendimenti medi — la fonte
    dominante di instabilita' nell'ottimizzazione media-varianza (Winston,
    cap. 3).

    Parameters
    ----------
    returns : pd.DataFrame
        Matrice dei rendimenti log giornalieri (T × n), senza NaN.
    annualization : int
        Giorni di borsa in un anno.

    Returns
    -------
    pd.Series
        Rendimenti attesi contratti e annualizzati, stesso indice di
        returns.columns.
    """
    clean = returns.dropna()
    T, n = clean.shape
    if T < n + 2 or n == 0:
        # Dati insufficienti per la contrazione — si ripiega sulla media storica.
        return returns.mean() * annualization

    mu_sample = clean.mean()
    sigma = clean.cov()

    try:
        # Protezione numerica: pinv gestisce covarianze singolari o quasi singolari
        sigma_inv = np.linalg.pinv(sigma.values)
    except (np.linalg.LinAlgError, ValueError):
        return mu_sample * annualization

    ones = np.ones(n)

    # Target della contrazione: rendimento atteso del portafoglio a varianza minima globale.
    denom = float(ones @ sigma_inv @ ones)
    if denom <= 1e-14 or not np.isfinite(denom):
        return mu_sample * annualization
    w_mvp = sigma_inv @ ones / denom
    mu_target = float(w_mvp @ mu_sample.values)

    # Intensita' di contrazione ottima (Jorion 1986, eq. 14), in termini giornalieri (nativi).
    diff = mu_sample.values - mu_target * ones
    mahal = float(diff @ sigma_inv @ diff)
    if mahal < 1e-14 or not np.isfinite(mahal):
        phi = 1.0
    else:
        phi = (n + 2) / ((n + 2) + T * mahal)
        phi = float(np.clip(phi, 0.0, 1.0))

    mu_shrunk = (1.0 - phi) * mu_sample.values + phi * mu_target * ones
    return pd.Series(mu_shrunk * annualization, index=clean.columns, name="bayes_stein_returns")


# ---------------------------------------------------------------------------
# Modello Fama-French a 3 fattori
# ---------------------------------------------------------------------------


def _fetch_ff3_factors(dataset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Scarica i fattori giornalieri FF3 da Ken French via pandas_datareader.

    Torna un DataFrame con colonne Mkt-RF, SMB, HML, RF in forma decimale.
    I risultati sono cachati sia a livello di processo sia su disco (TTL
    24h). La cache su disco tiene tutta la storia; il taglio a
    [start_date, end_date] si fa dopo il caricamento, cosi' intervalli di
    date diversi condividono lo stesso file.
    """
    # Cache a livello di processo: la chiave e' solo il nome del dataset (storia intera)
    if dataset in _FF3_CACHE:
        df = _FF3_CACHE[dataset]
        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        return df[(df.index >= start_ts) & (df.index <= end_ts)]

    # Cache su disco
    _FF3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _FF3_CACHE_DIR / f"ff3_{dataset}.csv"
    try:
        if (
            cache_path.exists()
            and time.time() - cache_path.stat().st_mtime < _FF3_DISK_CACHE_TTL
        ):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            _FF3_CACHE[dataset] = df
            start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
            return df[(df.index >= start_ts) & (df.index <= end_ts)]
    except Exception:
        pass

    import pandas_datareader.data as web  # import ritardato — dipendenza opzionale

    # Scarica una finestra ampia (2000 → oggi) cosi' il file in cache copre tutte le richieste
    wide_start = "2000-01-01"
    wide_end = pd.Timestamp.today().strftime("%Y-%m-%d")
    raw = web.DataReader(dataset, "famafrench", wide_start, wide_end)
    df = raw[0] if isinstance(raw, dict) else raw
    df = df / 100.0  # percentuale → decimale
    if hasattr(df.index, "to_timestamp"):
        df.index = df.index.to_timestamp()
    else:
        df.index = pd.to_datetime(df.index)

    try:
        df.to_csv(cache_path)
    except Exception:
        pass

    _FF3_CACHE[dataset] = df
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    return df[(df.index >= start_ts) & (df.index <= end_ts)]


def _ols_betas(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS via equazioni normali: β = (X'X)^{-1} X'y. Torna il vettore dei coefficienti."""
    try:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1])


def fama_french_expected_returns(
    log_returns: pd.DataFrame,
    ticker_currencies: dict[str, str],
    start_date: str,
    end_date: str,
    annualization: int = 252,
    fallback_report: list[str] | None = None,
) -> pd.Series:
    """Rendimenti attesi Fama-French a 3 fattori, con selezione regionale dei fattori.

    Quando i fattori per un ticker non sono disponibili, la stima ripiega su
    Bayes-Stein SOLO PER QUEL TICKER. La sostituzione e' silenziosa per
    scelta progettuale — un dataset regionale mancante non deve far fallire
    tutto l'universo — ed e' esattamente per questo che va riportata:
    passando una lista come ``fallback_report`` si raccolgono i ticker che
    sono stati sostituiti, cosi' l'interfaccia puo' nominarli invece di
    presentare una stima Bayes-Stein sotto il nome di questa funzione.

    Per ogni ticker, seleziona il dataset regionale di Ken French che
    corrisponde alla valuta di negoziazione del ticker:

    - USD / CAD   → Nord America (F-F_Research_Data_Factors_daily)
    - EUR / GBP / CHF / SEK / NOK / DKK → Europa (Europe_3_Factors_Daily)
    - JPY         → Giappone (Japan_3_Factors_Daily)
    - AUD / NZD   → Asia Pacifico (Asia_Pacific_ex_Japan_3_Factors_Daily)

    Rendimenti attesi: E[r_i] = rf + β_mkt·E[Mkt-RF] + β_smb·E[SMB] + β_hml·E[HML]
    dove i premi fattoriali sono la media storica dei rispettivi fattori regionali.

    Ripiega su Bayes-Stein per qualsiasi ticker la cui regressione fallisce o
    per cui esistono meno di 60 osservazioni sovrapposte con i dati dei
    fattori.

    Solleva RuntimeError se pandas_datareader non e' disponibile.
    """
    # Raccoglie i dataset unici che servono
    dataset_per_ticker: dict[str, str] = {}
    for ticker in log_returns.columns:
        currency = ticker_currencies.get(ticker, "USD").upper()
        dataset_per_ticker[ticker] = _CURRENCY_TO_FF3_DATASET.get(
            currency, _DEFAULT_FF3_DATASET
        )

    unique_datasets = set(dataset_per_ticker.values())

    # Scarica i fattori (con cache)
    factors_by_dataset: dict[str, pd.DataFrame] = {}
    for ds in unique_datasets:
        try:
            factors_by_dataset[ds] = _fetch_ff3_factors(ds, start_date, end_date)
        except Exception as exc:
            logger.warning("Download dei fattori FF3 fallito per il dataset %s: %s", ds, exc)

    if not factors_by_dataset:
        raise RuntimeError("Tutti i download dei fattori FF3 sono falliti.")

    # Rendimenti attesi di ripiego (usati per singolo ticker se la regressione FF3 fallisce)
    bs_fallback = bayes_stein_expected_returns(log_returns, annualization)

    results: dict[str, float] = {}
    for ticker in log_returns.columns:
        ds = dataset_per_ticker[ticker]
        factors = factors_by_dataset.get(ds)

        if factors is None:
            results[ticker] = float(bs_fallback[ticker])
            if fallback_report is not None:
                fallback_report.append(ticker)
            continue

        # Allinea i rendimenti del ticker con i dati dei fattori sulle date comuni
        ticker_ret = log_returns[[ticker]].dropna()
        # Normalizza l'indice a sola data, per l'allineamento
        if hasattr(factors.index, "tz"):
            factors.index = factors.index.tz_localize(None)
        factors_daily = factors.reindex(ticker_ret.index).dropna()
        common_idx = ticker_ret.index.intersection(factors_daily.index)

        if len(common_idx) < 60:
            logger.debug(
                "FF3: sovrapposizione insufficiente per %s (%d oss.); uso Bayes-Stein.",
                ticker,
                len(common_idx),
            )
            results[ticker] = float(bs_fallback[ticker])
            if fallback_report is not None:
                fallback_report.append(ticker)
            continue

        r = ticker_ret.loc[common_idx, ticker].values
        mkt_rf = factors_daily.loc[common_idx, "Mkt-RF"].values
        smb = factors_daily.loc[common_idx, "SMB"].values
        hml = factors_daily.loc[common_idx, "HML"].values
        rf_d = factors_daily.loc[common_idx, "RF"].values

        # I fattori di Ken French sono rendimenti semplici in eccesso; convertiamo i log-rendimenti in semplici
        r_simple = np.expm1(r)
        excess_r = r_simple - rf_d
        X = np.column_stack([np.ones(len(excess_r)), mkt_rf, smb, hml])
        betas = _ols_betas(excess_r, X)
        # betas = [alpha, beta_mkt, beta_smb, beta_hml]

        # Premi fattoriali = media storica annualizzata dei fattori regionali.
        # Assunzione: la media storica e' un proxy non distorto dei premi futuri attesi
        # (database Fama-French, sito di Ken French). Nessuna correzione condizionale o di contrazione.
        mkt_premium = float(mkt_rf.mean() * annualization)
        smb_premium = float(smb.mean() * annualization)
        hml_premium = float(hml.mean() * annualization)
        rf_annual = float(rf_d.mean() * annualization)

        er = (
            rf_annual
            + betas[1] * mkt_premium
            + betas[2] * smb_premium
            + betas[3] * hml_premium
        )
        results[ticker] = float(er)

    return pd.Series(results, name="ff3_expected_returns")
