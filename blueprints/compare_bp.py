from __future__ import annotations

import logging

from flask import Blueprint, current_app, render_template, request

from portfolio_optimizer.config import (
    COV_METHOD_OPTIONS,
    DEFAULT_MAX_WEIGHT,
    REFERENCE_CURRENCY_OPTIONS,
    RETURNS_METHOD_OPTIONS,
)
from portfolio_optimizer.presentation.charts import (
    METRIC_FORMAT,
    _COMPARE_COLORS,
    chart_compare_weights,
    chart_oos_drawdown,
    chart_oos_equity,
)
from portfolio_optimizer.risk.metrics import HIGHER_IS_BETTER
from portfolio_optimizer.use_cases.compare_models import (
    BENCHMARK_MODEL,
    best_and_worst,
    compare_models_out_of_sample,
)
from portfolio_optimizer.use_cases.diagnose_universe import diagnose_universe
from portfolio_optimizer.use_cases.model_selector import assess_models

import blueprints.extensions as ext
from blueprints.shared import (
    _BENCHMARKS,
    _COMPARE_MODELS,
    _METRIC_LABELS,
    _index_template_ctx,
    _parse_tickers,
    _resolve_or_redirect,
    _run_optimizer,
    _universe_returns,
)

logger = logging.getLogger(__name__)

compare = Blueprint("compare", __name__)


def _parse_compare_params(form) -> dict:
    """Estrae e valida i parametri del form condivisi con ``/optimize``.

    La risoluzione dei ticker resta nella route (``_resolve_or_redirect``):
    puo' tornare un redirect, e una funzione helper qui non deve avere una
    via d'uscita HTTP.
    """
    try:
        max_weight = float(form.get("max_weight", "") or "nan")
    except ValueError:
        max_weight = float("nan")
    # Un solo tetto di concentrazione per tutti i criteri. Con il tetto
    # suggerito per criterio -- 1,00 per l'equipesato, 0,30 per la minima
    # varianza -- una differenza di risultato potrebbe venire dal vincolo
    # anziche' dal criterio, e il confronto non sarebbe piu' interpretabile.
    if not (0.05 <= max_weight <= 1.0):
        max_weight = DEFAULT_MAX_WEIGHT
    benchmark_ticker = form.get("benchmark", "^GSPC")
    reference_currency_raw = form.get("reference_currency", "auto").lower().strip()
    reference_currency = (
        reference_currency_raw if reference_currency_raw in REFERENCE_CURRENCY_OPTIONS else "auto"
    )
    returns_method = form.get("returns_method", "bayes_stein")
    if returns_method not in {v for v, _ in RETURNS_METHOD_OPTIONS}:
        returns_method = "bayes_stein"
    cov_method = form.get("cov_method", "ledoit_wolf")
    if cov_method not in {v for v, _ in COV_METHOD_OPTIONS}:
        cov_method = "ledoit_wolf"
    try:
        lookback_years = int(form.get("lookback_years") or form.get("horizon_years") or "5")
    except ValueError:
        lookback_years = 5
    try:
        forecast_horizon_years = float(form.get("forecast_horizon_years") or "1.0")
    except ValueError:
        forecast_horizon_years = 1.0
    return {
        "max_weight": max_weight,
        "benchmark_ticker": benchmark_ticker,
        "reference_currency": reference_currency,
        "returns_method": returns_method,
        "cov_method": cov_method,
        "lookback_years": lookback_years,
        "forecast_horizon_years": forecast_horizon_years,
    }


def _run_models_comparison(tickers, params: dict):
    """Fa girare ogni criterio di ``_COMPARE_MODELS`` sullo stesso universo.

    Un modello che fallisce viene registrato in ``errors`` ed escluso da
    ``model_results``, non solleva eccezione: la pagina confronta i
    modelli che sono stati calcolati, non li esclude tutti perche' uno e'
    andato male. ``ticker_currencies``, ``start_date`` e ``end_date``
    vengono dall'ultimo modello riuscito -- esattamente come prima che
    questa funzione esistesse: ogni modello condivide lo stesso universo,
    quindi qualsiasi run riuscito porta gli stessi valori.
    """
    model_results: dict[str, tuple] = {}
    errors: dict[str, str] = {}
    ticker_currencies: dict[str, str] = {}
    start_date = end_date = ""
    for model_key, model_label in _COMPARE_MODELS:
        try:
            res, tc, sd, ed = _run_optimizer(
                tickers=tickers,
                lookback_years=params["lookback_years"],
                forecast_horizon_years=params["forecast_horizon_years"],
                max_weight=params["max_weight"],
                benchmark_ticker=params["benchmark_ticker"],
                optimization_model=model_key,
                run_backtest=False,
                bl_views=None,
                returns_method=params["returns_method"],
                cov_method=params["cov_method"],
                reference_currency=params["reference_currency"],
            )
            model_results[model_key] = (res, model_label)
            ticker_currencies = tc
            start_date, end_date = sd, ed
        except Exception as exc:
            logger.warning("Compare: model %s failed: %s", model_key, exc)
            errors[model_key] = str(exc)
    return model_results, errors, ticker_currencies, start_date, end_date


def _build_in_sample_compare_data(model_results: dict) -> list:
    """L'allocazione di oggi e le metriche in campione per ogni criterio.

    Questa tabella non e' un verdetto: sul campione che l'ha prodotta, il
    portafoglio tangente vince per costruzione (vedi ``_build_oos_analysis``).
    """
    compare_data = []
    for model_key, (result, label) in model_results.items():
        metrics_d = {
            _METRIC_LABELS.get(k, k): round(float(v), 4)
            for k, v in result.metrics.items()
        }
        weights_d = {t: round(float(w) * 100, 2) for t, w in result.weights.items()}
        hhi = sum(float(w) ** 2 for w in result.weights.values)
        compare_data.append(
            dict(
                key=model_key,
                label=label,
                color=_COMPARE_COLORS.get(model_key, "#94a3b8"),
                metrics=metrics_d,
                weights=weights_d,
                hhi=hhi,
            )
        )
    return compare_data


def _build_oos_analysis(model_results: dict, params: dict) -> dict:
    """Confronto fuori campione: quello che decide davvero.

    Le metriche in campione descrivono l'allocazione di oggi sul campione
    che l'ha prodotta, dove il portafoglio tangente vince per costruzione
    perche' ottimizza esattamente la grandezza su cui viene poi giudicato.
    Ritorna un'analisi con tutte le chiavi vuote/None invece di solleva
    un'eccezione: la pagina mostra comunque la tabella in campione, che
    questo confronto sia riuscito a calcolarsi o no.
    """
    oos = None
    oos_paths: dict = {}
    oos_rows: list = []
    oos_best_worst: dict = {}
    guidance = diagnosis = None
    try:
        first_result = next(iter(model_results.values()))[0]
        universe = _universe_returns(first_result)
        rf_annual = float(first_result.rf_breakdown.get("weighted", 0.0))
        if not universe.empty:
            diagnosis = diagnose_universe(universe)
            guidance = assess_models(diagnosis)
            oos = compare_models_out_of_sample(
                universe,
                [(k, lbl) for k, lbl in _COMPARE_MODELS if k in model_results],
                max_weight=params["max_weight"],
                risk_free_annual=rf_annual,
                returns_method=params["returns_method"],
                cov_method=params["cov_method"],
            )
            for row in oos.rows:
                if row.ok and row.net_log_returns is not None:
                    oos_paths[row.model] = (row.net_log_returns, row.label)
                oos_rows.append(
                    dict(
                        key=row.model,
                        label=row.label,
                        color=_COMPARE_COLORS.get(row.model, "#94a3b8"),
                        ok=row.ok,
                        failure=row.failure,
                        is_benchmark=row.is_benchmark,
                        metrics={
                            _METRIC_LABELS.get(k, k): round(float(v), 4)
                            for k, v in row.metrics.items()
                        },
                        turnover=row.turnover_mean,
                        cost=row.cost_total,
                        n_rebalances=row.n_rebalances,
                        diff=row.sharpe_diff_vs_benchmark,
                        p_value=row.p_value_vs_benchmark,
                        p_value_onesided=row.p_value_single_onesided,
                        orderable=row.sharpe_orderable,
                        coincides=row.coincides_with_benchmark,
                        p_value_stepdown=row.p_value_stepdown_vs_benchmark,
                        significant=row.beats_benchmark_significantly,
                    )
                )
            for metric_key, higher in HIGHER_IS_BETTER.items():
                marks = best_and_worst(oos.rows, metric_key, higher)
                if marks:
                    oos_best_worst[_METRIC_LABELS.get(metric_key, metric_key)] = marks
    except Exception as exc:
        logger.warning("Confronto fuori campione non disponibile: %s", exc)
    return {
        "oos": oos,
        "oos_paths": oos_paths,
        "oos_rows": oos_rows,
        "oos_best_worst": oos_best_worst,
        "guidance": guidance,
        "diagnosis": diagnosis,
    }


@compare.post("/compare")
def compare_models():
    """Confronta ogni criterio di ottimizzazione sullo stesso universo.

    Scomposto nella Parte XXIII dell'audit in: parsing
    (``_parse_compare_params``), esecuzione per ogni modello
    (``_run_models_comparison``), formattazione in campione
    (``_build_in_sample_compare_data``) e verdetto fuori campione
    (``_build_oos_analysis``) -- lo stesso schema usato gia' per
    ``optimize_bp.optimize()`` e ``export_bp.export_csv()``.
    """
    if ext._HAS_LIMITER and ext.limiter is not None:
        if not current_app.testing:
            ext.limiter.limit("10/minute")(lambda: None)()

    raw_tickers = request.form.get("tickers", "")
    params = _parse_compare_params(request.form)

    resolved_tokens, redirect_response = _resolve_or_redirect(raw_tickers, "/compare")
    if redirect_response is not None:
        return redirect_response

    try:
        tickers = _parse_tickers(",".join(resolved_tokens))
    except Exception as exc:
        return render_template(
            "index.html", error=str(exc), form=request.form, **_index_template_ctx()
        ), 400

    model_results, errors, ticker_currencies, start_date, end_date = _run_models_comparison(
        tickers, params
    )

    if not model_results:
        return render_template(
            "index.html",
            error="Tutti i modelli hanno fallito durante il confronto.",
            form=request.form,
            **_index_template_ctx(),
        ), 400

    compare_data = _build_in_sample_compare_data(model_results)
    oos_ctx = _build_oos_analysis(model_results, params)

    charts = dict(
        oos_equity=chart_oos_equity(oos_ctx["oos_paths"]),
        oos_drawdown=chart_oos_drawdown(oos_ctx["oos_paths"]),
        compare_weights=chart_compare_weights(model_results),
    )

    return render_template(
        "compare.html",
        tickers=tickers,
        ticker_currencies=ticker_currencies,
        lookback_years=params["lookback_years"],
        forecast_horizon_years=params["forecast_horizon_years"],
        max_weight=params["max_weight"],
        returns_method=params["returns_method"],
        cov_method=params["cov_method"],
        start_date=start_date,
        end_date=end_date,
        compare_data=compare_data,
        all_metrics=list(_METRIC_LABELS.values()),
        charts=charts,
        metric_format=METRIC_FORMAT,
        errors=errors,
        benchmark_name=dict(_BENCHMARKS).get(params["benchmark_ticker"], params["benchmark_ticker"]),
        reference_currency=params["reference_currency"],
        oos=oos_ctx["oos"],
        oos_rows=oos_ctx["oos_rows"],
        oos_best_worst=oos_ctx["oos_best_worst"],
        guidance=oos_ctx["guidance"],
        diagnosis=oos_ctx["diagnosis"],
        benchmark_model=BENCHMARK_MODEL,
    )
