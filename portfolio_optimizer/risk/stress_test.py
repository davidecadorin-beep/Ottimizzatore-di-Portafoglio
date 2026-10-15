from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimizer.risk.metrics import max_drawdown


SCENARIOS: dict[str, tuple[str, str]] = {
    "Dotcom Crash": ("2000-03-01", "2002-10-31"),
    "GFC 2008": ("2008-09-01", "2009-03-31"),
    "Covid 2020": ("2020-02-15", "2020-03-31"),
    "Bond Crisis 2022": ("2022-01-01", "2022-10-31"),
}


def stress_test_portfolio(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario_map: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Valuta la perdita cumulata nelle finestre storiche di crisi."""
    scenarios = SCENARIOS if scenario_map is None else scenario_map
    # Allineamento per ETICHETTA e non per posizione. Il prodotto fra la
    # matrice grezza e il vettore grezzo accoppiava il peso i-esimo con la
    # colonna i-esima: una strategia che restituisse i pesi in un ordine
    # diverso produceva in silenzio il portafoglio sbagliato. E' lo stesso
    # difetto registrato come B3 per il backtest rimosso.
    aligned = weights.reindex(returns.columns)
    if aligned.isna().any():
        mancanti = ", ".join(aligned[aligned.isna()].index.astype(str))
        raise ValueError(
            f"Pesi assenti per {mancanti}: lo stress test non puo' valutare un "
            "portafoglio di cui non conosce l'intera composizione."
        )
    simple = np.expm1(returns)
    port_simple = (simple.to_numpy() * aligned.to_numpy()).sum(axis=1)
    portfolio = pd.Series(np.log1p(port_simple), index=returns.index)
    results = []
    for name, (start, end) in scenarios.items():
        window = portfolio.loc[start:end]
        if window.empty:
            continue  # il periodo e' precedente ai dati disponibili — si omette invece di mostrare uno zero fuorviante
        # Una finestra coperta solo in parte descrive un episodio diverso da
        # quello che il suo nome dichiara: la crisi del 2008 valutata a partire
        # da novembre esclude Lehman e riporta un drawdown quasi dimezzato.
        requested_start = pd.Timestamp(start)
        requested_end = pd.Timestamp(end)
        covered_start = window.index[0]
        covered_end = window.index[-1]
        partial = (covered_start - requested_start).days > 7 or (
            requested_end - covered_end
        ).days > 7
        cum = np.exp(window.cumsum())
        mdd = max_drawdown(cum)
        results.append(
            {
                "scenario": name,
                "period_return": float(cum.iloc[-1] - 1),
                "max_drawdown": float(mdd),
                "copertura_parziale": bool(partial),
                "dal": str(covered_start.date()),
                "al": str(covered_end.date()),
            }
        )
    if not results:
        return pd.DataFrame(
            columns=[
                "period_return",
                "max_drawdown",
                "copertura_parziale",
                "dal",
                "al",
            ]
        )
    return pd.DataFrame(results).set_index("scenario")


def synthetic_stress(
    portfolio_returns: pd.Series,
    log_returns: pd.DataFrame,
    rf_annual: pd.Series,
    benchmark_returns: pd.Series | None,
) -> pd.DataFrame:
    """Scenari di stress sintetici: crollo di mercato, shock sui tassi, picco di volatilita'."""
    results = []

    # Il proxy di mercato e' il portafoglio equipesato, quindi va costruito
    # come tale: media dei rendimenti SEMPLICI riportata in logaritmi. La media
    # dei logaritmi e' il rendimento del portafoglio a media geometrica, un
    # oggetto diverso, e il divario cresce proprio quando la dispersione fra
    # titoli aumenta, cioe' nelle crisi che questa funzione vuole descrivere.
    equal_weighted = np.log1p(np.expm1(log_returns).mean(axis=1))
    market_proxy = equal_weighted.reindex(portfolio_returns.index).dropna()
    port_aligned = portfolio_returns.reindex(market_proxy.index).dropna()
    mkt = market_proxy.reindex(port_aligned.index)
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(port_aligned.index).dropna()
        if bench.var() > 0 and len(bench) > 5:
            beta = port_aligned.reindex(bench.index).dropna().cov(bench) / bench.var()
        else:
            beta = 1.0
    elif mkt.var() > 0:
        beta = port_aligned.cov(mkt) / mkt.var()
    else:
        beta = 1.0
    # Il limite inferiore era zero, il che azzerava per costruzione l'effetto
    # su qualunque portafoglio con beta negativo — esattamente i portafogli
    # che una caduta di mercato avvantaggia. Un universo che comprende
    # obbligazioni puo' averlo, e riportare zero non e' prudente ma sbagliato.
    beta = float(np.clip(beta, -1.5, 3.0))
    market_return = float(beta * (-0.20))
    results.append(
        {
            "scenario": "Caduta mercato -20%",
            "period_return": market_return,
            "max_drawdown": market_return,
        }
    )

    rf_monthly = rf_annual.resample("ME").last().dropna()
    rf_monthly_chg = rf_monthly.diff().dropna()
    port_monthly = portfolio_returns.resample("ME").sum()
    common_idx = rf_monthly_chg.index.intersection(port_monthly.index)
    if len(common_idx) >= 12 and rf_monthly_chg.reindex(common_idx).var() > 1e-10:
        rc = rf_monthly_chg.reindex(common_idx)
        pc = port_monthly.reindex(common_idx)
        emp_duration = float(-pc.cov(rc) / rc.var())
        rate_impact = float(np.clip(-emp_duration * 0.01, -0.50, 0.50))
    else:
        rate_impact = -0.01
    results.append(
        {
            "scenario": "Rialzo tassi +100bp",
            "period_return": rate_impact,
            "max_drawdown": min(rate_impact, 0.0),
        }
    )

    daily_vol = float(portfolio_returns.std(ddof=1))
    monthly_vol_1x = daily_vol * np.sqrt(21)
    monthly_vol_2x = monthly_vol_1x * 2
    monthly_log = portfolio_returns.resample("ME").sum()
    if len(monthly_log) >= 12 and monthly_vol_1x > 0:
        # Perdita al 5° percentile empirico, scalata al regime di volatilita' doppia
        # Si raddoppia lo SCOSTAMENTO dalla media, non il quantile intero: il
        # quantile contiene anche la deriva, e raddoppiarla significa
        # raddoppiare il rendimento atteso insieme alla volatilita', che non e'
        # cio' che uno scenario di volatilita' doppia descrive.
        p5 = float(np.percentile(monthly_log, 5))
        mu_monthly = float(np.mean(monthly_log))
        shocked_log = mu_monthly + 2.0 * (p5 - mu_monthly)
    else:
        # Ricaduta: due deviazioni standard nel regime a volatilita' doppia.
        shocked_log = float(-2.0 * monthly_vol_2x)
    # La colonna raccoglie rendimenti SEMPLICI — le altre due righe lo sono —
    # quindi il logaritmo va convertito prima di entrarci.
    var_spike = float(np.expm1(shocked_log))
    results.append(
        {
            "scenario": "Spike volatilità ×2",
            "period_return": var_spike,
            "max_drawdown": var_spike,
        }
    )

    return pd.DataFrame(results).set_index("scenario")
