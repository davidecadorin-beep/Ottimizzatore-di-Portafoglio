from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from flask import redirect, render_template, request, session, url_for

from portfolio_optimizer.config import (
    BACKTEST_TEST_DAYS,
    BACKTEST_TRAIN_DAYS,
    COV_METHOD_OPTIONS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MODEL,
    MC_N_SIMS,
    MODEL_MAX_WEIGHT,
    MODEL_OPTIONS,
    RESULT_CACHE_TTL,
    RETURNS_METHOD_OPTIONS,
)
from portfolio_optimizer.data.data_loader import MarketDataLoader
from portfolio_optimizer.use_cases.load_market_data import LoadMarketDataUseCase
from portfolio_optimizer.use_cases.optimize_portfolio import (
    OptimizePortfolioUseCase,
    _estimate_covariance,
)
import portfolio_optimizer.data.ticker_resolver as ticker_resolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"^[A-Z0-9.\-\^]{1,12}$")

_METRIC_LABELS = {
    "return_annual": "Rendimento Ann.",
    "volatility_annual": "Volatilità Ann.",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "calmar_ratio": "Calmar Ratio",
    "omega_ratio": "Omega Ratio",
    "var_95": "VaR 95%",
    "cvar_95": "CVaR 95%",
    "max_drawdown": "Max Drawdown",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis in eccesso",
}

_LOOKBACK_OPTIONS = [1, 2, 3, 5, 7, 10]
_FORECAST_HORIZON_OPTIONS = [1, 2, 3, 5, 10]

_MODEL_OPTIONS = MODEL_OPTIONS

_BENCHMARKS = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ Composite"),
    ("^DJI", "Dow Jones"),
    ("URTH", "MSCI World (ETF)"),
    ("EEM", "MSCI Emerging Mkts"),
    ("^STOXX50E", "Euro Stoxx 50"),
    ("^RUT", "Russell 2000"),
    ("^FTSE", "FTSE 100"),
    ("^N225", "Nikkei 225"),
    ("^HSI", "Hang Seng"),
    ("GLD", "Oro (GLD ETF)"),
    ("SLV", "Argento (SLV ETF)"),
    ("BND", "Bond US Agg (BND)"),
    ("TLT", "US Treasury 20Y (TLT)"),
    ("HYG", "High Yield Bond (HYG)"),
    ("VNQ", "REITs USA (VNQ)"),
    ("DBC", "Commodity (DBC)"),
    ("", "Nessuno"),
]

# Tutti i criteri selezionabili, equipesato compreso. L'equipesato mancava, ed
# e' il termine di paragone dell'intero capitolo empirico: un confronto fra
# modelli che ometta il riferimento contro cui nessuno di essi risulta
# superiore non e' il confronto che la tesi svolge.
_COMPARE_MODELS = list(MODEL_OPTIONS)

# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

_MAX_TICKERS = int(os.environ.get("MAX_TICKERS_PER_REQUEST", "30"))

# Quanti titoli al massimo una sola richiesta di notizie puo' chiedere.
_NEWS_MAX_TICKERS = 20


def _parse_tickers(raw: str) -> list[str]:
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    invalid = [t for t in tickers if not _TICKER_RE.match(t)]
    if invalid:
        raise ValueError(
            f"Ticker non validi: {', '.join(invalid)}. "
            "Usa solo lettere, cifre, '.', '-', '^' (max 12 caratteri)."
        )
    unique = list(dict.fromkeys(tickers))
    if len(unique) < 2:
        raise ValueError("Inserisci almeno due ticker validi separati da virgola.")
    if len(unique) > _MAX_TICKERS:
        raise ValueError(
            f"Troppi ticker specificati ({len(unique)}). Il limite massimo è di {_MAX_TICKERS} titoli per richiesta."
        )
    return unique


def _parse_bl_views(raw: str) -> dict[str, float]:
    """Vecchio formato testo libero: una riga ``TICKER: rendimento%``.

    Tenuto perche' la pagina dei risultati porta avanti le view verso
    ``/export-csv`` e ``/compare`` proprio in questa forma. L'input nuovo
    arriva invece da ``_bl_views_from_rows``, che non si puo' scrivere male
    per errore di battitura.
    """
    views: dict[str, float] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        ticker_part, val_part = line.split(":", 1)
        ticker = ticker_part.strip().upper()
        if not ticker:
            continue
        try:
            val = float(val_part.strip().rstrip("%").replace(",", ".")) / 100.0
            views[ticker] = val
        except ValueError:
            logger.warning("BL views: cannot parse value for %s — skipped.", ticker)
    return views


def _bl_views_from_rows(form) -> tuple[dict[str, float], list[str]]:
    """Legge i campi delle view per ogni titolo, e dice quali non si leggono.

    La pagina invia due liste parallele, ``bl_view_ticker`` e
    ``bl_view_value``, una voce per titolo, col valore lasciato vuoto dove
    l'utente non ha un'opinione. Leggerle lato server vuol dire che le view
    non dipendono dal fatto che il JavaScript sia partito -- che e'
    esattamente come il modello era diventato inutilizzabile: un errore di
    sintassi fermava tutto lo script, la sezione restava nascosta, e
    Black-Litterman riceveva zero view senza che nessuno se ne accorgesse.

    Tutto quello che non si riesce a leggere viene RESTITUITO, non buttato
    via. Una view scritta come ``12,5%%`` o su un ticker che non e' piu' nel
    portafoglio spariva in silenzio, e un modello il cui input distintivo e'
    sparito senza far rumore restituisce il portafoglio di equilibrio pur
    dicendo di essere Black-Litterman.
    """
    tickers = form.getlist("bl_view_ticker")
    values = form.getlist("bl_view_value")
    views: dict[str, float] = {}
    problems: list[str] = []
    for ticker, raw in zip(tickers, values):
        ticker = (ticker or "").strip().upper()
        raw = (raw or "").strip().rstrip("%").replace(",", ".")
        if not ticker or not raw:
            continue
        try:
            views[ticker] = float(raw) / 100.0
        except ValueError:
            problems.append(f"{ticker}: «{raw}» non e' un numero")
    return views, problems


def _parse_tickers_raw(raw: str) -> list[str]:
    """Spezza, pulisce, maiuscolo, dedup (ordine mantenuto). Nessuna validazione regex."""
    seen = dict.fromkeys(t.strip().upper() for t in raw.split(",") if t.strip())
    return list(seen)


def _resolve_or_redirect(raw_tickers: str, target_url: str):
    """
    Ritorna (resolved_tokens, None) quando va tutto bene.
    Ritorna (None, response) se ambiguo o not_found — chi chiama deve ritornare response.
    """
    if request.form.get("_already_resolved"):
        return _parse_tickers_raw(raw_tickers), None

    tokens = _parse_tickers_raw(raw_tickers)
    if not tokens:
        return tokens, None

    try:
        resolve_result = ticker_resolver.resolve(tokens)
    except Exception as exc:
        logger.warning("Risoluzione ticker fallita: %s", exc)
        resolve_result = ticker_resolver.ResolveResult(not_found=tokens)

    if resolve_result.not_found:
        not_found_str = ", ".join(resolve_result.not_found)
        error = (
            f"Ticker non trovati: {not_found_str}. "
            "Verifica il simbolo o aggiungi il suffisso borsa (es. IS3N.AS per Euronext Amsterdam)."
        )
        return None, (
            render_template("index.html", error=error, form=request.form, **_index_template_ctx()),
            400,
        )

    if resolve_result.ambiguous:
        session["pending_resolve"] = {
            "resolved": resolve_result.resolved,
            "ambiguous": [asdict(a) for a in resolve_result.ambiguous],
            "warnings": resolve_result.warnings,
            "form": {k: vs if len(vs) > 1 else vs[0]
                     for k, vs in request.form.lists()
                     if k != "tickers"},
            "target": target_url,
        }
        return None, redirect(url_for("optimize.confirm_tickers"))

    if resolve_result.warnings:
        session["ticker_warnings"] = resolve_result.warnings

    return resolve_result.resolved, None


def _end_start_from_lookback(lookback_years: int) -> tuple[str, str]:
    """Ritorna (end_date, start_date) — end prima, nell'ordine che si aspetta chi chiama."""
    end = date.today()
    try:
        start = end.replace(year=end.year - lookback_years)
    except ValueError:
        start = end.replace(year=end.year - lookback_years, day=28)
    return end.isoformat(), start.isoformat()



def _index_template_ctx() -> dict:
    return dict(
        lookback_options=_LOOKBACK_OPTIONS,
        forecast_horizon_options=_FORECAST_HORIZON_OPTIONS,
        benchmarks=_BENCHMARKS,
        model_options=_MODEL_OPTIONS,
        default_model=DEFAULT_MODEL,
        model_max_weight=MODEL_MAX_WEIGHT,
        backtest_train=BACKTEST_TRAIN_DAYS,
        backtest_test=BACKTEST_TEST_DAYS,
        returns_method_options=RETURNS_METHOD_OPTIONS,
        cov_method_options=COV_METHOD_OPTIONS,
        mc_n_sims=MC_N_SIMS,
        mc_n_sims_formatted=f"{MC_N_SIMS:,}".replace(",", "."),
    )


# ---------------------------------------------------------------------------
# Esecuzione dell'ottimizzazione + cache dei risultati (thread-safe)
# ---------------------------------------------------------------------------

_CACHE_MAX_SIZE = 64
_RESULT_CACHE: OrderedDict[str, tuple[float, Any]] = OrderedDict()
_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Cache dei ticker FX
# ---------------------------------------------------------------------------

_FX_PAIRS = [
    ("EURUSD=X", "EUR/USD"),
    ("GBPUSD=X", "GBP/USD"),
    ("USDJPY=X", "USD/JPY"),
    ("USDCHF=X", "USD/CHF"),
    ("AUDUSD=X", "AUD/USD"),
    ("USDCAD=X", "USD/CAD"),
    ("BTC-USD", "BTC/USD"),
    ("ETH-USD", "ETH/USD"),
]
_FX_CACHE_TTL = 60
_fx_cache: dict = {"data": None, "ts": 0}
_fx_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Cache delle news
# ---------------------------------------------------------------------------

_NEWS_CACHE_TTL = 900  # 15 minuti
# Chiave: la lista dei ticker fornita dall'utente. Era un dict senza limite e
# senza sfratto -- le voci scadevano logicamente ma non venivano mai rimosse --
# su una rotta GET aperta, quindi bastava variare i ticker per farlo crescere
# indefinitamente. La capienza e' ora limitata e le voci scadute vengono tolte.
_NEWS_CACHE_MAX = 128
_NEWS_MAX_ITEMS = 40
_NEWS_CACHE: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_NEWS_CACHE_LOCK = threading.Lock()

_TICKER_COLOURS = [
    "#3b82f6",
    "#f59e0b",
    "#10b981",
    "#a78bfa",
    "#f43f5e",
    "#06b6d4",
    "#84cc16",
    "#fb923c",
]


def _relative_age(dt: datetime) -> str:
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = int((now - dt).total_seconds())
    if diff < 3600:
        return f"{max(diff // 60, 1)}min fa"
    if diff < 86400:
        return f"{diff // 3600}h fa"
    days = diff // 86400
    return "ieri" if days == 1 else f"{days} giorni fa"


def _parse_news_item(item: dict, ticker: str) -> "dict | None":
    """Fa il parsing di una news grezza di yfinance; torna None se non c'e' un timestamp usabile."""
    # yfinance >=0.2.50 mette tutto dentro item['content']
    content = item.get("content") or item
    pub_str = content.get("pubDate") or content.get("displayTime", "")
    ts_legacy = item.get("providerPublishTime", 0)
    if pub_str:
        try:
            dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.fromtimestamp(ts_legacy or 0, tz=timezone.utc)
    elif ts_legacy:
        dt = datetime.fromtimestamp(ts_legacy, tz=timezone.utc)
    else:
        return None
    provider = content.get("provider") or {}
    canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
    raw_link = str(canonical.get("url") or item.get("link") or "").strip()
    link = raw_link if raw_link.startswith(("http://", "https://")) else "#"
    return {
        "ticker": ticker,
        "title": content.get("title") or item.get("title", ""),
        "publisher": provider.get("displayName") or item.get("publisher", ""),
        "link": link,
        "published_at": dt,
        "age": _relative_age(dt),
    }


def _fetch_ticker_calendar_events(t_obj, ticker: str, today) -> list[dict]:
    """Ritorna gli eventi futuri di calendario (ex-div, dividendo, utili) per un titolo."""
    events: list[dict] = []
    # ── Calendario (ex-dividendo, dividendo) ──────────────────────────
    try:
        cal = t_obj.calendar
        if cal is not None:
            ex_div = cal.get("Ex-Dividend Date")
            if ex_div is not None and pd.notna(ex_div):
                ex_div_d = pd.Timestamp(ex_div).date()
                if ex_div_d >= today:
                    events.append({"ticker": ticker, "type": "ex_dividend", "date": ex_div_d, "value": None})
            div_date = cal.get("Dividend Date")
            div_rate = cal.get("Dividend Rate")
            if div_date is not None and pd.notna(div_date):
                div_d = pd.Timestamp(div_date).date()
                if div_d >= today:
                    events.append({
                        "ticker": ticker,
                        "type": "dividend",
                        "date": div_d,
                        "value": float(div_rate) if div_rate and pd.notna(div_rate) else None,
                    })
    except Exception:
        pass
    # ── Date degli utili ───────────────────────────────────────────────
    try:
        ed = t_obj.earnings_dates
        if ed is not None and not ed.empty:
            idx = ed.index
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_convert(None)
            future = [i for i in idx if pd.Timestamp(i).date() >= today]
            if future:
                events.append({"ticker": ticker, "type": "earnings", "date": pd.Timestamp(future[0]).date(), "value": None})
    except Exception:
        pass
    return events


def _fetch_one_ticker_news(ticker: str) -> tuple[list[dict], list[dict]]:
    """Scarica le news e gli eventi di calendario futuri per un titolo."""
    news_items: list[dict] = []
    today = date.today()
    try:
        t = yf.Ticker(ticker)
        for item in (t.news or [])[:8]:
            parsed = _parse_news_item(item, ticker)
            if parsed is not None:
                news_items.append(parsed)
        events = _fetch_ticker_calendar_events(t, ticker, today)
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", ticker, exc)
        events = []
    return news_items, events


def _run_parallel_news_fetch(tickers: list[str]) -> tuple[list[dict], list[dict]]:
    """Scarica le news per tutti i titoli in parallelo; ritorna (all_news, all_events)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_news: list[dict] = []
    all_events: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=min(len(tickers), 8))
    futures = {pool.submit(_fetch_one_ticker_news, t): t for t in tickers}
    try:
        for future in as_completed(futures, timeout=15):
            try:
                news_items, events = future.result()
                all_news.extend(news_items)
                all_events.extend(events)
            except Exception as exc:
                logger.warning("News future error for %s: %s", futures[future], exc)
    except TimeoutError:
        logger.warning("News fetch timed out after 15s, returning partial results")
    finally:
        for f in futures:
            f.cancel()
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)
    return all_news, all_events


def _quota_news(all_news: list[dict], tickers: list[str]) -> list[dict]:
    """Taglia all_news a _NEWS_MAX_ITEMS garantendo una quota minima per ogni titolo."""
    if len(all_news) <= _NEWS_MAX_ITEMS or not tickers:
        return all_news[:_NEWS_MAX_ITEMS]
    quota = max(1, _NEWS_MAX_ITEMS // len(tickers))
    per_ticker: dict[str, list] = {t: [] for t in tickers}
    remainder: list = []
    for item in all_news:
        t = item["ticker"]
        if t in per_ticker and len(per_ticker[t]) < quota:
            per_ticker[t].append(item)
        else:
            remainder.append(item)
    guaranteed = [item for t in tickers for item in per_ticker[t]]
    slots_left = _NEWS_MAX_ITEMS - len(guaranteed)
    result = guaranteed + remainder[:max(0, slots_left)]
    result.sort(key=lambda x: x["published_at"], reverse=True)
    return result


def _fetch_portfolio_news(tickers: list[str]) -> dict:
    """Ritorna news + eventi uniti e ordinati per la lista di titoli (cache 15 min).

    Chi chiama deve garantire che len(tickers) <= _NEWS_MAX_TICKERS; la
    logica della quota assume _NEWS_MAX_ITEMS // len(tickers) >= 1.
    """
    cache_key = ",".join(sorted(tickers))
    now = time.monotonic()
    with _NEWS_CACHE_LOCK:
        for stale in [k for k, (ts, _) in _NEWS_CACHE.items() if now - ts >= _NEWS_CACHE_TTL]:
            del _NEWS_CACHE[stale]
        cached = _NEWS_CACHE.get(cache_key)
        if cached:
            ts, payload = cached
            if now - ts < _NEWS_CACHE_TTL:
                _NEWS_CACHE.move_to_end(cache_key)
                return payload

    all_news, all_events = _run_parallel_news_fetch(tickers)
    all_news.sort(key=lambda x: x["published_at"], reverse=True)
    all_news = _quota_news(all_news, tickers)
    all_events.sort(key=lambda x: x["date"])

    tickers_in_news = {item["ticker"] for item in all_news}
    titoli_senza_notizie = [t for t in tickers if t not in tickers_in_news]
    payload = {"news": all_news, "events": all_events, "titoli_senza_notizie": titoli_senza_notizie}
    with _NEWS_CACHE_LOCK:
        _NEWS_CACHE[cache_key] = (now, payload)
        _NEWS_CACHE.move_to_end(cache_key)
        while len(_NEWS_CACHE) > _NEWS_CACHE_MAX:
            _NEWS_CACHE.popitem(last=False)
    return payload


def _universe_returns(result):
    """I rendimenti per titolo su cui si basa la diagnosi dell'universo."""
    frame = getattr(result, "asset_log_returns", None)
    if frame is None or frame.empty:
        raise ValueError("matrice dei rendimenti per titolo non disponibile")
    return frame


def _covariance_evidence(result, cov_method: str) -> dict:
    """Cosa hanno misurato i capitoli empirici sullo stimatore in uso.

    La scheda degli stimatori portava solo un badge col nome dello
    stimatore di covarianza e niente altro, mentre la covarianza e'
    proprio la dimensione su cui il lavoro empirico ha trovato il
    risultato piu' netto. La scheda confrontava i tre stimatori dei
    RENDIMENTI attesi e non diceva niente sui tre stimatori di
    covarianza, quindi la pagina stava zitta esattamente dove aveva
    qualcosa da dire.

    La volatilita' che ogni stimatore implica per l'allocazione
    effettivamente in mano viene calcolata qui; i numeri del bias
    vengono dall'esperimento e sono scritti come costanti in
    ``forecast_disclosure``, che tiene traccia della loro provenienza.
    """
    evidence = {
        "sample": (
            "Nessuna distorsione rilevabile della volatilita' dichiarata, a "
            "orizzonte mensile e trimestrale. Turnover intermedio."
        ),
        "ledoit_wolf": (
            "Nessuna distorsione rilevabile della volatilita' dichiarata. "
            "Turnover circa un quinto di quello di EWMA a parita' di "
            "volatilita' realizzata: e' lo stimatore che la raccomandazione propone."
        ),
        "ewma": (
            "Sottostima la volatilita' realizzata: rapporto 1,098 a "
            "ribilanciamento mensile e 1,134 a trimestrale, con p inferiore a "
            "0,0001. E' l'unico stimatore distorto, ed e' la ragione per cui la "
            "raccomandazione non lo propone: l'applicazione mostra all'utente la "
            "volatilita' attesa. Entro la propria memoria efficace, pero', e' il "
            "migliore dei tre sul rendimento — Sharpe medio 0,648 contro 0,620 di "
            "Ledoit-Wolf a ribilanciamento mensile — al prezzo di un turnover "
            "piu' che doppio."
        ),
    }
    rows = []
    try:
        universe = _universe_returns(result)
        weights = result.weights.reindex(universe.columns).fillna(0.0).to_numpy()
        for method, label in COV_METHOD_OPTIONS:
            try:
                matrix = _estimate_covariance(method, universe)
                variance = float(weights @ matrix.to_numpy() @ weights)
                rows.append(
                    {
                        "method": method,
                        "label": label,
                        "volatility": float(np.sqrt(max(variance, 0.0))),
                        "evidence": evidence.get(method, ""),
                        "selected": method == cov_method,
                        "biased": method == "ewma",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Covarianza %s non calcolabile: %s", method, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Confronto fra stimatori di covarianza non disponibile: %s", exc)
    realized = None
    try:
        realized = float(result.metrics.get("volatility_annual"))
    except Exception:  # noqa: BLE001
        pass
    return {"rows": rows, "realized": realized}


def _effective_max_weight(max_weight: float | None, optimization_model: str | None) -> float:
    """Il tetto effettivamente usato, risolto una volta sola e letto da chiunque ne ha bisogno."""
    if max_weight is not None:
        return max_weight
    return MODEL_MAX_WEIGHT.get(optimization_model or DEFAULT_MODEL, DEFAULT_MAX_WEIGHT)


def _effective_window(result, start_date: str, end_date: str) -> dict:
    """La finestra di stima effettivamente usata, affiancata a quella richiesta.

    ``compute_log_returns`` applica ``dropna(how="any")`` per far si' che
    la matrice dei rendimenti sia rettangolare. Un titolo con poca storia
    quindi tronca il campione di tutti gli altri, e l'intestazione
    continuava ad annunciare l'orizzonte richiesto dall'utente: verificato
    su dieci anni di dati con un titolo quotato da uno solo, il campione
    stimato scendeva da 2519 giorni a 299, un calo dell'88 per cento,
    senza che nulla in pagina lo dicesse.

    Troncare o escludere il titolo corto e' una decisione dell'utente e
    resta aperta (voce G9). Quello che non e' opzionale e' DICHIARARE su
    quale finestra sono stati calcolati davvero la covarianza e
    l'ottimizzazione.
    """
    frame = getattr(result, "asset_log_returns", None)
    if frame is None or frame.empty:
        return {
            "effective_start": None,
            "effective_end": None,
            "effective_days": None,
            "requested_days": None,
            "window_loss": None,
        }

    effective_days = int(len(frame))
    try:
        requested_days = int(
            len(pd.bdate_range(start=start_date, end=end_date))
        )
    except Exception:  # noqa: BLE001
        requested_days = None
    loss = (
        max(0.0, 1.0 - effective_days / requested_days)
        if requested_days
        else None
    )
    return {
        "effective_start": str(frame.index[0].date()),
        "effective_end": str(frame.index[-1].date()),
        "effective_days": effective_days,
        "requested_days": requested_days,
        # Sotto il cinque per cento la differenza e' fatta di festivita' non
        # coincidenti fra le borse, non di storia mancante.
        "window_loss": loss if loss is not None and loss > 0.05 else None,
    }


def _optimizer_cache_key(
    tickers: list[str],
    lookback_years: int = 5,
    forecast_horizon_years: float = 1.0,
    max_weight: float = 1.0,
    benchmark_ticker: str = "",
    optimization_model: str = "",
    bl_views: dict | None = None,
    returns_method: str = "bayes_stein",
    cov_method: str = "ledoit_wolf",
    reference_currency: str = "auto",
    run_backtest: bool = False,
) -> str:
    views_str = json.dumps(bl_views, sort_keys=True) if bl_views else ""
    raw = (
        f"{','.join(sorted(tickers))}|lb{lookback_years}|fc{forecast_horizon_years}|{max_weight}|"
        f"{benchmark_ticker}|{optimization_model}|{views_str}|"
        f"{returns_method}|{cov_method}|{reference_currency}|bt{int(run_backtest)}"
    )
    return hashlib.md5(raw.encode()).hexdigest()


def _run_optimizer(
    tickers,
    lookback_years: int = 5,
    forecast_horizon_years: float = 1.0,
    max_weight=None,
    benchmark_ticker="",
    optimization_model: str | None = None,
    run_backtest: bool = False,
    bl_views: dict | None = None,
    returns_method: str = "bayes_stein",
    cov_method: str = "ledoit_wolf",
    reference_currency: str = "auto",
):
    max_weight = _effective_max_weight(max_weight, optimization_model)
    key = _optimizer_cache_key(
        tickers=tickers,
        lookback_years=lookback_years,
        forecast_horizon_years=forecast_horizon_years,
        max_weight=max_weight,
        benchmark_ticker=benchmark_ticker or "",
        optimization_model=optimization_model or "",
        bl_views=bl_views,
        returns_method=returns_method,
        cov_method=cov_method,
        reference_currency=reference_currency,
        run_backtest=run_backtest,
    )
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _RESULT_CACHE.get(key)
        if entry is not None:
            ts, cached = entry
            if now - ts < RESULT_CACHE_TTL:
                logger.info("Cache hit for key %s", key[:8])
                _RESULT_CACHE.move_to_end(key)
                return cached
            else:
                del _RESULT_CACHE[key]

    logger.info(
        "Running optimizer: tickers=%s lookback=%sy forecast_horizon=%sy model=%s",
        tickers,
        lookback_years,
        forecast_horizon_years,
        optimization_model or DEFAULT_MODEL,
    )

    end_date, start_date = _end_start_from_lookback(lookback_years)
    loader = MarketDataLoader(cache_dir="cache")
    ticker_currencies = loader.get_ticker_currencies(tickers)
    data_use_case = LoadMarketDataUseCase(loader=loader)
    optimizer = OptimizePortfolioUseCase(market_data_use_case=data_use_case)
    result = optimizer.execute(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        max_weight=max_weight,
        forecast_horizon_years=float(forecast_horizon_years),
        ticker_currencies=ticker_currencies,
        benchmark_ticker=benchmark_ticker or None,
        optimization_model=optimization_model or None,
        run_backtest=run_backtest,
        bl_views=bl_views or None,
        returns_method=returns_method,
        cov_method=cov_method,
        reference_currency=reference_currency,
    )
    payload = (result, ticker_currencies, start_date, end_date)
    with _CACHE_LOCK:
        _RESULT_CACHE[key] = (now, payload)
        _RESULT_CACHE.move_to_end(key)
        while len(_RESULT_CACHE) > _CACHE_MAX_SIZE:
            _RESULT_CACHE.popitem(last=False)
    return payload


def _fetch_fx_data() -> list[dict]:
    now = time.monotonic()
    with _fx_lock:
        if _fx_cache["data"] is not None and now - _fx_cache["ts"] < _FX_CACHE_TTL:
            return _fx_cache["data"]

    tickers = [p[0] for p in _FX_PAIRS]
    try:
        raw = yf.download(
            tickers, period="6d", interval="1d", auto_adjust=True, progress=False
        )
        result = []
        for ticker, symbol in _FX_PAIRS:
            try:
                series = raw["Close"][ticker].dropna()
                if len(series) < 2:
                    raise ValueError("insufficient data")
                history = [round(float(v), 6) for v in series.iloc[-5:]]
                rate = history[-1]
                prev = history[-2]
                change_pct = round((rate - prev) / prev * 100, 3) if prev else 0.0
                direction = (
                    "up" if change_pct > 0 else ("down" if change_pct < 0 else "flat")
                )
                result.append(
                    {
                        "symbol": symbol,
                        "rate": round(rate, 5),
                        "change_pct": change_pct,
                        "direction": direction,
                        "history_5d": history,
                    }
                )
            except Exception:
                logger.warning("FX: failed to process %s", ticker)
                continue
    except Exception as exc:
        logger.warning("FX: yfinance download failed: %s", exc)
        # Restituire l'ultimo dato riuscito e' giusto; presentarlo come corrente
        # non lo e'. Il momento dell'ultimo aggiornamento viaggia con i dati,
        # cosi' che chi li mostra possa dire da quando non cambiano.
        with _fx_lock:
            stale = _fx_cache["data"] or []
            age = now - _fx_cache["ts"] if _fx_cache["ts"] else None
            return [
                {**row, "stale": True, "age_seconds": round(age) if age else None}
                for row in stale
            ]

    result = [{**row, "stale": False, "age_seconds": 0} for row in result]
    with _fx_lock:
        _fx_cache["data"] = result
        _fx_cache["ts"] = now
    return result
