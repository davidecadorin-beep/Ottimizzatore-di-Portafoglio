from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class PortfolioOptimizationResult:
    """Oggetto di dominio restituito dalla pipeline di ottimizzazione."""

    # --- dati principali ---
    selected_model: str
    weights: pd.Series
    frontier: pd.DataFrame
    expected_returns: pd.Series
    covariance: pd.DataFrame
    portfolio_returns: pd.Series
    metrics: dict[str, float]

    # --- portafogli di riferimento (calcolati sempre) ---
    min_var_weights: pd.Series
    max_sharpe_weights: pd.Series

    # --- scomposizione del rischio ---
    risk_contributions: pd.Series
    rolling_sharpe: pd.Series

    # --- scenari ---
    stress_results: pd.DataFrame

    # --- metadati ---
    rf_breakdown: dict[str, float]
    frontier_markers: dict[str, dict]

    # --- benchmark opzionale ---
    benchmark_returns: pd.Series | None = None
    benchmark_metrics: dict[str, float] = field(default_factory=dict)

    # --- curva equity del backtest walk-forward, opzionale ---
    backtest_returns: pd.Series | None = None

    # --- risultati della simulazione Monte Carlo ---
    mc_data: dict | None = None

    # --- view BL effettivamente applicate, e quelle che non si sono potute applicare ---
    bl_views_used: dict[str, float] | None = None
    # Le view scartate perche' il titolo non e' nel portafoglio, o perche' il
    # posteriore non e' stato calcolabile. Una view che sparisce in silenzio
    # trasforma Black-Litterman nel portafoglio di equilibrio senza dirlo.
    bl_views_ignored: dict[str, str] | None = None
    # Il criterio effettivamente applicato quando quello scelto non e' stato
    # calcolabile, con il motivo. Il minimo CVaR ripiegava sulla minima
    # varianza con un solo avviso nei log, e la pagina continuava a intitolarsi
    # al criterio richiesto.
    criterion_fallback: tuple[str, str] | None = None
    bl_market_cap_failures: tuple[str, ...] = ()
    bl_market_cap_excluded: tuple[str, ...] = ()

    # --- scomposizione del rendimento attivo rispetto al riferimento equipesato ---
    attribution: pd.DataFrame | None = None

    # --- metadati dello stimatore ---
    # ``returns_method`` e' lo stimatore EFFETTIVAMENTE applicato, non quello
    # richiesto: i due differiscono quando i dati fattoriali non sono
    # disponibili, e la pagina mostrava il secondo.
    returns_method: str = "bayes_stein"
    returns_estimate: object | None = None
    cov_method: str = "ledoit_wolf"
    reference_currency: str = "USD"

    # --- tutte e tre le stime dei rendimenti, per il grafico di confronto ---
    returns_comparison: dict | None = None  # {method_label: pd.Series}

    # --- rendimenti log per singolo titolo, servono per diagnosticare l'universo ---
    # La diagnosi descrive l'universo, non il portafoglio: serve la matrice per
    # titolo, che il risultato altrimenti non conserva.
    asset_log_returns: pd.DataFrame | None = None
