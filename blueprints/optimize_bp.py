from __future__ import annotations

import json
import logging

from flask import Blueprint, current_app, redirect, render_template, request, session, url_for

from portfolio_optimizer.config import (
    BACKTEST_TEST_DAYS,
    BACKTEST_TRAIN_DAYS,
    COV_METHOD_OPTIONS,
    DEFAULT_MAX_WEIGHT,
    REFERENCE_CURRENCY_OPTIONS,
    RETURNS_METHOD_OPTIONS,
    SUPPORTED_OPTIMIZATION_MODELS,
)
from portfolio_optimizer.presentation.charts import (
    METRIC_FORMAT,
    backtest_verdict,
    chart_backtest,
    chart_currency,
    chart_drawdown,
    chart_equity,
    chart_frontier,
    chart_monte_carlo,
    chart_returns_comparison,
    chart_risk_contribution,
    chart_rolling_sharpe,
    chart_stress,
    chart_weights_pie,
)
from portfolio_optimizer.risk.metrics import HIGHER_IS_BETTER
from portfolio_optimizer.use_cases.forecast_disclosure import disclosures_for
from portfolio_optimizer.use_cases.optimize_portfolio import OptimizePortfolioUseCase  # noqa: F401 — importato solo per il patching nei test

import blueprints.extensions as ext
from blueprints.shared import (
    _BENCHMARKS,
    _METRIC_LABELS,
    _TICKER_COLOURS,
    _bl_views_from_rows,
    _covariance_evidence,
    _effective_max_weight,
    _effective_window,
    _index_template_ctx,
    _optimizer_cache_key,
    _parse_bl_views,
    _parse_tickers,
    _resolve_or_redirect,
    _run_optimizer,
)

logger = logging.getLogger(__name__)

optimize_bp = Blueprint("optimize", __name__)


def _parse_optimize_params(form) -> dict:
    """Estrae e valida tutti i parametri del form per /optimize."""
    raw_tickers = form.get("tickers", "")
    try:
        max_weight = float(form.get("max_weight", "") or "nan")
    except ValueError:
        max_weight = float("nan")
    if not (0.05 <= max_weight <= 1.0):
        max_weight = None
    benchmark_ticker = form.get("benchmark", "^GSPC")
    optimization_model = form.get("optimization_model") or form.get("model") or None
    run_backtest = form.get("run_backtest") == "1"
    # Le righe per titolo hanno la precedenza; la casella di testo resta per le
    # richieste che rimbalzano dalla pagina dei risultati (esporta, confronta).
    bl_views, bl_view_problems = _bl_views_from_rows(form)
    if not bl_views:
        bl_views_raw = form.get("bl_views", "")
        bl_views = _parse_bl_views(bl_views_raw) if bl_views_raw.strip() else {}
    bl_views = bl_views or None
    returns_method = form.get("returns_method", "bayes_stein")
    if returns_method not in {v for v, _ in RETURNS_METHOD_OPTIONS}:
        returns_method = "bayes_stein"
    cov_method = form.get("cov_method", "ledoit_wolf")
    if cov_method not in {v for v, _ in COV_METHOD_OPTIONS}:
        cov_method = "ledoit_wolf"
    reference_currency_raw = form.get("reference_currency", "auto").lower().strip()
    reference_currency = reference_currency_raw if reference_currency_raw in REFERENCE_CURRENCY_OPTIONS else "auto"
    try:
        lookback_years = int(form.get("lookback_years") or form.get("horizon_years") or "5")
    except ValueError:
        lookback_years = 5
    try:
        forecast_horizon_years = float(form.get("forecast_horizon_years") or "1.0")
    except ValueError:
        forecast_horizon_years = 1.0
    return {
        "raw_tickers": raw_tickers,
        "max_weight": max_weight,
        "benchmark_ticker": benchmark_ticker,
        "optimization_model": optimization_model,
        "run_backtest": run_backtest,
        "bl_views": bl_views,
        "bl_view_problems": bl_view_problems,
        "returns_method": returns_method,
        "cov_method": cov_method,
        "reference_currency": reference_currency,
        "lookback_years": lookback_years,
        "forecast_horizon_years": forecast_horizon_years,
    }


def _build_render_ctx(
    result,
    tickers: list,
    ticker_currencies: dict,
    start_date: str,
    end_date: str,
    params: dict,
    ticker_warnings: list,
) -> dict:
    """Costruisce il dizionario completo di contesto per il template results.html."""
    max_weight = params["max_weight"]
    benchmark_ticker = params["benchmark_ticker"]
    optimization_model = params["optimization_model"]
    run_backtest = params["run_backtest"]
    bl_view_problems = params["bl_view_problems"]
    bl_views = params["bl_views"]
    returns_method = params["returns_method"]
    cov_method = params["cov_method"]
    reference_currency = params["reference_currency"]
    lookback_years = params["lookback_years"]
    forecast_horizon_years = params["forecast_horizon_years"]

    weights_table = [
        {
            "ticker": t,
            "weight": round(float(w) * 100, 2),
            "currency": ticker_currencies.get(t, "—"),
            "rc": round(float(result.risk_contributions.get(t, 0)) * 100, 2),
        }
        for t, w in result.weights.items()
    ]
    metrics_display = {_METRIC_LABELS.get(k, k): round(v, 4) for k, v in result.metrics.items()}
    bench_display = {_METRIC_LABELS.get(k, k): round(v, 4) for k, v in result.benchmark_metrics.items()}

    # Il simulatore a slider stima il CAGR con uno sviluppo di Taylor al
    # secondo ordine, exp(sum(w_i*mu_i) + (sum(w_i*sigma_i^2) - sigma_p^2)/2) - 1,
    # che presuppone mu LOGARITMICI: la correzione di Ito e' gia' dentro la
    # formula. Questa serie veniva letta da returns_comparison["Media Storica"],
    # che con la correzione G11 e' passata alla scala ARITMETICA per il grafico
    # di confronto fra stimatori — quindi la mezza varianza finiva contata due
    # volte. Verificato su quattro titoli con pesi 40/30/20/10: la deriva
    # logaritmica risultava gonfiata di esattamente 0,5*sum(w_i*sigma_i^2) =
    # 3,53 punti, che dopo la capitalizzazione diventano 4,46 punti sul CAGR,
    # da 23,92 a 28,38 per cento. Il salto era visibile all'utente perche' a
    # pesi iniziali la scheda mostra initial_metrics, esatte, e passa alla
    # formula solo al primo movimento di uno slider.
    if result.asset_log_returns is not None and not result.asset_log_returns.empty:
        mu_hist = {str(k): float(v) for k, v in (result.asset_log_returns.mean() * 252).items()}
    else:
        mu_hist = {k: float(v) for k, v in result.expected_returns.items()}

    slider_data = json.dumps(
        {
            "assets": list(result.expected_returns.index),
            "mu_hist": mu_hist,
            "cov": {
                r: {c: float(result.covariance.loc[r, c]) for c in result.covariance.columns}
                for r in result.covariance.index
            },
            "rf": result.rf_breakdown.get("weighted", 0.0),
            "weights": {k: float(v) for k, v in result.weights.items()},
            # Metriche esatte dell'ottimizzatore — usate ai pesi iniziali cosi' il
            # simulatore mostra gli stessi valori del riepilogo al caricamento pagina.
            "initial_metrics": {
                "ret": float(result.metrics.get("return_annual", 0.0)),
                "vol": float(result.metrics.get("volatility_annual", 0.0)),
                "sharpe": float(result.metrics.get("sharpe_ratio", 0.0)),
            },
        }
    )

    colour_map_opt = {
        t: _TICKER_COLOURS[i % len(_TICKER_COLOURS)]
        for i, t in enumerate(result.weights.index)
    }
    primary_color = list(colour_map_opt.values())[0] if colour_map_opt else "#3b82f6"

    charts = dict(
        frontier=chart_frontier(result, primary_color),
        equity=chart_equity(
            result.portfolio_returns,
            result.benchmark_returns,
            result.rf_breakdown.get("weighted", 0.0),
            primary_color,
        ),
        drawdown=chart_drawdown(result.portfolio_returns, result.benchmark_returns, primary_color),
        risk_contribution=chart_risk_contribution(result.risk_contributions, colour_map_opt, primary_color),
        stress=chart_stress(result.stress_results, primary_color),
        rolling_sharpe=chart_rolling_sharpe(result.rolling_sharpe, primary_color),
        weights_pie=chart_weights_pie(result.weights, colour_map_opt, primary_color),
        currency=chart_currency(ticker_currencies, result.weights),
        backtest=chart_backtest(result.backtest_returns, result.portfolio_returns),
        monte_carlo=chart_monte_carlo(result.mc_data),
        returns_comparison=chart_returns_comparison(result.returns_comparison),
    )

    bt_verdict = (
        backtest_verdict(
            result.backtest_returns,
            result.portfolio_returns,
            result.rf_breakdown.get("weighted", 0.0),
        )
        if run_backtest
        else None
    )

    # L'etichetta segue lo stimatore EFFETTIVAMENTE applicato. Mostrava quello
    # richiesto: con i dati fattoriali non disponibili la pagina dichiarava
    # Fama-French mentre l'ottimizzazione aveva usato Bayes-Stein.
    effective_returns_method = getattr(result, "returns_method", returns_method)
    returns_method_label = dict(RETURNS_METHOD_OPTIONS).get(effective_returns_method, effective_returns_method)
    estimate = getattr(result, "returns_estimate", None)
    returns_method_note = estimate.note() if estimate is not None else ""
    returns_method_requested_label = dict(RETURNS_METHOD_OPTIONS).get(returns_method, returns_method)
    cov_evidence = _covariance_evidence(result, cov_method)
    ff_fallback = (result.returns_comparison or {}).get("__fama_french_fallback__") or ()
    cov_method_label = dict(COV_METHOD_OPTIONS).get(cov_method, cov_method)

    # La raccomandazione fra criteri vive nella pagina di confronto, non qui.
    # Su questa pagina l'utente ha gia' scelto il criterio e guarda la propria
    # allocazione: un riquadro che gli dice che ne avrebbe potuto scegliere un
    # altro appartiene al luogo in cui i criteri si confrontano davvero, dove
    # accanto all'affermazione ci sono le misure fuori campione che la
    # sostengono. Qui resta la dichiarazione sull'accuratezza delle previsioni,
    # che riguarda i numeri effettivamente mostrati in questa pagina.
    forecast_notes = []
    try:
        forecast_notes = [
            d.as_dict()
            for d in disclosures_for(
                result.selected_model,
                result.cov_method,
                float(result.metrics.get("volatility_annual", 0.0)),
                var_95=result.metrics.get("var_95"),
            )
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Accuratezza previsiva non disponibile: %s", exc)

    return {
        "forecast_notes": forecast_notes,
        "selected_model": result.selected_model,
        "max_weight": max_weight,
        "tickers": tickers,
        "ticker_currencies": ticker_currencies,
        "lookback_years": lookback_years,
        "forecast_horizon_years": forecast_horizon_years,
        "start_date": start_date,
        "end_date": end_date,
        "weights_table": weights_table,
        "metrics": metrics_display,
        "benchmark_metrics": bench_display,
        "benchmark_name": dict(_BENCHMARKS).get(benchmark_ticker, benchmark_ticker),
        "rf_breakdown": result.rf_breakdown,
        "charts": charts,
        "primary_color": primary_color,
        "colour_map": colour_map_opt,
        "slider_data": slider_data,
        "run_backtest": run_backtest,
        "mc_summary": result.mc_data,
        "bl_views_used": result.bl_views_used,
        "bl_views_ignored": result.bl_views_ignored,
        # La view RICHIESTA, non quella applicata: e' quella su cui e' stata
        # calcolata result_key (vedi sotto), quindi e' quella che l'export deve
        # ritrasmettere bit per bit per ritrovare la stessa chiave. JSON invece
        # del formato testuale "TICKER: valore%" perche' round-trip esatto
        # (json.dumps/json.loads su float e' garantito bit-identico; percentuale
        # arrotondata a 4 decimali e ridivisa per 100 non lo e').
        "bl_views_requested_json": json.dumps(bl_views, sort_keys=True) if bl_views else "",
        "criterion_fallback": result.criterion_fallback,
        "bl_market_cap_failures": getattr(result, "bl_market_cap_failures", ()),
        "bl_market_cap_excluded": getattr(result, "bl_market_cap_excluded", ()),
        "bl_view_problems": bl_view_problems,
        "backtest_train": BACKTEST_TRAIN_DAYS,
        "backtest_test": BACKTEST_TEST_DAYS,
        "metric_format": METRIC_FORMAT,
        "bt_verdict": bt_verdict,
        "returns_method": returns_method,
        "returns_method_label": returns_method_label,
        "returns_method_note": returns_method_note,
        "returns_method_requested_label": returns_method_requested_label,
        "cov_evidence": cov_evidence,
        "ff_fallback": ff_fallback,
        "cov_method": cov_method,
        "cov_method_label": cov_method_label,
        "reference_currency": result.reference_currency,
        "ticker_warnings": ticker_warnings,
        "higher_is_better": HIGHER_IS_BETTER,
        **_effective_window(result, start_date, end_date),
        "result_key": _optimizer_cache_key(
            tickers=tickers,
            lookback_years=lookback_years,
            forecast_horizon_years=forecast_horizon_years,
            max_weight=_effective_max_weight(max_weight, optimization_model),
            benchmark_ticker=benchmark_ticker or "",
            optimization_model=optimization_model or "",
            bl_views=bl_views,
            returns_method=returns_method,
            cov_method=cov_method,
            reference_currency=reference_currency,
            run_backtest=run_backtest,
        ),
    }


@optimize_bp.get("/")
def index():
    return render_template("index.html", **_index_template_ctx())


@optimize_bp.post("/optimize")
def optimize():
    if ext._HAS_LIMITER and ext.limiter is not None:
        if not current_app.testing:
            ext.limiter.limit("10/minute")(lambda: None)()

    params = _parse_optimize_params(request.form)
    if params["optimization_model"] and params["optimization_model"] not in SUPPORTED_OPTIMIZATION_MODELS:
        return f"Modello di ottimizzazione non riconosciuto: '{params['optimization_model']}'.", 400

    resolved_tokens, redirect_response = _resolve_or_redirect(params["raw_tickers"], "/optimize")
    if redirect_response is not None:
        return redirect_response

    try:
        tickers = _parse_tickers(",".join(resolved_tokens))
        result, ticker_currencies, start_date, end_date = _run_optimizer(
            tickers=tickers,
            lookback_years=params["lookback_years"],
            forecast_horizon_years=params["forecast_horizon_years"],
            max_weight=params["max_weight"],
            benchmark_ticker=params["benchmark_ticker"],
            optimization_model=params["optimization_model"],
            run_backtest=params["run_backtest"],
            bl_views=params["bl_views"],
            returns_method=params["returns_method"],
            cov_method=params["cov_method"],
            reference_currency=params["reference_currency"],
        )
    except Exception as exc:
        logger.exception("Optimization failed")
        return render_template(
            "index.html", error=str(exc), form=request.form, **_index_template_ctx()
        ), 400

    ticker_warnings = session.pop("ticker_warnings", [])
    ctx = _build_render_ctx(result, tickers, ticker_currencies, start_date, end_date, params, ticker_warnings)
    return render_template("results.html", **ctx)


@optimize_bp.get("/confirm-tickers")
def confirm_tickers():
    pending = session.get("pending_resolve")
    if not pending:
        return redirect(url_for("optimize.index"))
    return render_template("confirm_tickers.html", pending=pending)
