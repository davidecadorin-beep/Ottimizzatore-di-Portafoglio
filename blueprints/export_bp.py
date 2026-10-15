from __future__ import annotations

import csv
import io
import json
import logging
import time

from flask import Blueprint, Response, current_app, request

from portfolio_optimizer.config import (
    COV_METHOD_OPTIONS,
    DEFAULT_MODEL,
    REFERENCE_CURRENCY_OPTIONS,
    RESULT_CACHE_TTL,
    RETURNS_METHOD_OPTIONS,
    SUPPORTED_OPTIMIZATION_MODELS,
)
from blueprints.shared import (
    _CACHE_LOCK,
    _METRIC_LABELS,
    _RESULT_CACHE,
    _effective_max_weight,
    _optimizer_cache_key,
    _parse_bl_views,
    _parse_tickers,
    _run_optimizer,
)
import blueprints.extensions as ext

logger = logging.getLogger(__name__)

export = Blueprint("export", __name__)


class ExportKeyMismatch(Exception):
    """Il ``result_key`` inviato non corrisponde all'hash dei parametri
    inviati: la pagina e il form descrivono due analisi diverse. Viene
    solo lanciata, non ritornata, cosi' ``_get_or_recompute_result`` resta
    una funzione normale e solo la route decide lo status HTTP (Parte
    XXIII dell'audit; lo stesso principio gia' applicato in A5/B1/S1/G11/G16)."""


def _resolve_export_params(form) -> dict:
    """Fa il parsing e valida ogni campo del form che serve a ``/export-csv``.

    Rispecchia esattamente quello che ``/optimize`` legge per costruire
    ``result_key`` (vedi la chiamata a ``_optimizer_cache_key`` in
    ``optimize_bp.py``): un default diverso qui rispetto a la' e' un
    disallineamento silenzioso della cache-key, la classe di difetto
    documentata nella Parte XXIII. Lancia eccezione sul primo campo non
    valido invece di tornare una risposta HTTP -- la route decide lo status.
    """
    raw_tickers = form.get("tickers", "")
    try:
        max_weight = float(form.get("max_weight", "") or "nan")
    except ValueError:
        max_weight = float("nan")
    if not (0.05 <= max_weight <= 1.0):
        max_weight = None
    # Stesso default di /optimize (optimize_bp.py): se il campo manca del
    # tutto dal POST, result_key e' stata calcolata su '^GSPC', non su
    # stringa vuota. Due default diversi per lo stesso campo erano un'altra
    # via per la chiave incoerente.
    benchmark_ticker = form.get("benchmark", "^GSPC")
    optimization_model = form.get("optimization_model") or None
    if optimization_model and optimization_model not in SUPPORTED_OPTIMIZATION_MODELS:
        raise ValueError(f"Modello di ottimizzazione non riconosciuto: '{optimization_model}'.")
    # result_key e' calcolato sulla view RICHIESTA, serializzata con
    # json.dumps in optimize_bp: solo un round-trip JSON e' bit-identico
    # (la percentuale a 4 decimali del vecchio formato testuale non lo era,
    # e riportava comunque la view APPLICATA invece di quella richiesta).
    # Il campo testuale resta come ripiego per form non ancora aggiornati.
    bl_views_json = form.get("bl_views_json", "")
    if bl_views_json:
        try:
            bl_views = json.loads(bl_views_json) or None
        except (TypeError, ValueError):
            bl_views = None
    else:
        bl_views_raw = form.get("bl_views", "")
        bl_views = _parse_bl_views(bl_views_raw) if bl_views_raw.strip() else None
    returns_method = form.get("returns_method", "bayes_stein")
    if returns_method not in {v for v, _ in RETURNS_METHOD_OPTIONS}:
        returns_method = "bayes_stein"
    cov_method = form.get("cov_method", "ledoit_wolf")
    if cov_method not in {v for v, _ in COV_METHOD_OPTIONS}:
        cov_method = "ledoit_wolf"
    reference_currency_raw = form.get("reference_currency", "auto").lower().strip()
    reference_currency = (
        reference_currency_raw if reference_currency_raw in REFERENCE_CURRENCY_OPTIONS else "auto"
    )
    try:
        lookback_years = int(form.get("lookback_years") or form.get("horizon_years") or "5")
    except ValueError:
        lookback_years = 5
    try:
        forecast_horizon_years = float(form.get("forecast_horizon_years") or "1.0")
    except ValueError:
        forecast_horizon_years = 1.0
    run_backtest = form.get("run_backtest") == "1"
    tickers = _parse_tickers(raw_tickers)

    return {
        "tickers": tickers,
        "max_weight": max_weight,
        "benchmark_ticker": benchmark_ticker,
        "optimization_model": optimization_model,
        "bl_views": bl_views,
        "returns_method": returns_method,
        "cov_method": cov_method,
        "reference_currency": reference_currency,
        "lookback_years": lookback_years,
        "forecast_horizon_years": forecast_horizon_years,
        "run_backtest": run_backtest,
        "posted_key": form.get("result_key", ""),
    }


def _get_or_recompute_result(params: dict):
    """Riusa il risultato indicato da ``params['posted_key']`` se e'
    ancora fresco in cache; ricalcola solo se e' scaduto.

    Lancia ``ExportKeyMismatch`` quando la chiave inviata e i parametri
    inviati non vanno d'accordo -- il difetto artefatto-diverso-da-quello
    -mostrato documentato come A5/B1/S1/G11/G16 -- invece di esportare in
    silenzio qualsiasi cosa producano i parametri disallineati.
    """
    expected_key = _optimizer_cache_key(
        tickers=params["tickers"],
        lookback_years=params["lookback_years"],
        forecast_horizon_years=params["forecast_horizon_years"],
        max_weight=_effective_max_weight(params["max_weight"], params["optimization_model"]),
        benchmark_ticker=params["benchmark_ticker"] or "",
        optimization_model=params["optimization_model"] or "",
        bl_views=params["bl_views"],
        returns_method=params["returns_method"],
        cov_method=params["cov_method"],
        reference_currency=params["reference_currency"],
        run_backtest=params["run_backtest"],
    )
    posted_key = params["posted_key"]
    if posted_key and posted_key != expected_key:
        logger.warning("Chiave del risultato incoerente con i parametri inviati.")
        raise ExportKeyMismatch(
            "I parametri dell'esportazione non corrispondono a quelli "
            "dell'analisi mostrata. Rieseguire l'analisi e riprovare."
        )

    with _CACHE_LOCK:
        entry = _RESULT_CACHE.get(expected_key)
        if entry is not None and time.monotonic() - entry[0] < RESULT_CACHE_TTL:
            _RESULT_CACHE.move_to_end(expected_key)
            return entry[1]

    return _run_optimizer(
        tickers=params["tickers"],
        lookback_years=params["lookback_years"],
        forecast_horizon_years=params["forecast_horizon_years"],
        max_weight=params["max_weight"],
        benchmark_ticker=params["benchmark_ticker"],
        optimization_model=params["optimization_model"],
        bl_views=params["bl_views"],
        returns_method=params["returns_method"],
        cov_method=params["cov_method"],
        reference_currency=params["reference_currency"],
        run_backtest=params["run_backtest"],
    )


def _format_csv_response(result, ticker_currencies, start_date, end_date, params: dict) -> Response:
    """Costruisce il CSV: prima l'intestazione con la configurazione
    effettiva, poi i pesi, poi le metriche. L'intestazione della
    configurazione esiste perche' due CSV dello stesso portafoglio
    calcolati con stimatori diversi non siano indistinguibili (Parte XII, G16)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["configurazione", "valore"])
    for label, value in (
        ("criterio", result.selected_model),
        ("stimatore_rendimenti_richiesto", params["returns_method"]),
        ("stimatore_rendimenti_usato", getattr(result, "returns_method", params["returns_method"])),
        ("stimatore_covarianza", params["cov_method"]),
        ("valuta_di_riferimento", result.reference_currency),
        ("tetto_per_titolo", _effective_max_weight(params["max_weight"], params["optimization_model"])),
        ("benchmark", params["benchmark_ticker"] or ""),
        ("lookback_anni", params["lookback_years"]),
        ("orizzonte_proiezione_anni", params["forecast_horizon_years"]),
        ("orizzonte_anni", params["lookback_years"]),
        ("periodo", f"{start_date} / {end_date}"),
    ):
        writer.writerow([label, value])
    writer.writerow([])

    writer.writerow(["ticker", "currency", "weight_pct", "risk_contribution_pct"])
    for t, w in result.weights.items():
        writer.writerow(
            [
                t,
                ticker_currencies.get(t, "USD"),
                round(float(w) * 100, 4),
                round(float(result.risk_contributions.get(t, 0)) * 100, 4),
            ]
        )
    writer.writerow([])
    writer.writerow(["metric", "value"])
    for k, v in result.metrics.items():
        writer.writerow([_METRIC_LABELS.get(k, k), round(v, 6)])

    filename = (
        f"portfolio_{params['optimization_model'] or DEFAULT_MODEL}_{params['returns_method']}_"
        f"{params['cov_method']}_{params['lookback_years']}y_{end_date}.csv"
    )
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@export.post("/export-csv")
def export_csv():
    """Scrive il CSV del portafoglio che e' a schermo, non di un altro.

    La route prima ri-lanciava l'ottimizzatore da un sottoinsieme dei
    campi del form: ``returns_method``, ``cov_method`` e
    ``reference_currency`` non venivano inviati dalla pagina ne' letti
    qui, quindi un utente che aveva scelto Fama-French, EWMA ed euro si
    ritrovava un file calcolato con Bayes-Stein, Ledoit-Wolf e la valuta
    automatica. Sul solo stimatore di covarianza l'allocazione a minima
    varianza si sposta fino a 8.4 punti percentuali su un titolo, e il
    benchmark aveva un default diverso sulle due route, quindi anche le
    metriche relative a lui erano diverse. Niente lo segnalava: il file
    si scaricava, i numeri erano plausibili, ed erano quelli di un
    portafoglio diverso.

    E' il difetto registrato come A5, B1, S1 e G11 -- l'artefatto
    prodotto non e' l'artefatto mostrato -- e aggiungere solo i tre campi
    mancanti rimanderebbe il problema fino a quando non si aggiunge un
    quarto parametro. Quindi l'export riusa il RISULTATO STESSO, tramite
    la cache key che la pagina dei risultati porta con se', e ricalcola
    solo quando quella chiave e' scaduta -- con la configurazione
    completa, che la pagina invia. La configurazione effettiva viene
    scritta nel file, cosi' un CSV si distingue da un altro CSV.

    Scomposto nella Parte XXIII dell'audit in parsing
    (``_resolve_export_params``), riuso/ricalcolo della cache
    (``_get_or_recompute_result``) e formattazione del CSV
    (``_format_csv_response``) -- lo stesso schema usato gia' per
    ``optimize_bp.optimize()``.
    """
    if ext._HAS_LIMITER and ext.limiter is not None:
        if not current_app.testing:
            ext.limiter.limit("10/minute")(lambda: None)()

    try:
        params = _resolve_export_params(request.form)
    except Exception as exc:
        return str(exc), 400

    try:
        result, ticker_currencies, start_date, end_date = _get_or_recompute_result(params)
    except ExportKeyMismatch as exc:
        return str(exc), 409
    except Exception as exc:
        return str(exc), 400

    return _format_csv_response(result, ticker_currencies, start_date, end_date, params)
