from __future__ import annotations

import numpy as np
import pandas as pd


def active_return_decomposition(
    portfolio_weights: pd.Series,
    log_returns: pd.DataFrame,
    benchmark_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Decomposizione per titolo del rendimento attivo rispetto a un portafoglio di riferimento.

    Calcola il contributo di ogni titolo allo scarto tra il portafoglio e un
    portafoglio di riferimento (benchmark) sull'intero periodo campionario.
    E' un'attribuzione a livello di singolo titolo: effetto di allocazione
    senza raggruppamento per settore (Brinson-Hood-Beebower, 1986,
    semplificato a livello di asset).

    Parametri
    ----------
    portfolio_weights : pd.Series
        Pesi del portafoglio ottimizzato (somma a 1, si assume solo long).
    log_returns : pd.DataFrame
        Matrice dei rendimenti logaritmici giornalieri sul periodo di
        valutazione.
    benchmark_weights : pd.Series, opzionale
        Pesi del portafoglio di riferimento. Se non passato, equipesato su
        tutti i titoli presenti in *portfolio_weights*.

    Restituisce
    -------
    pd.DataFrame
        Colonne:
        - portfolio_weight      : peso nel portafoglio ottimizzato
        - benchmark_weight      : peso nel portafoglio di riferimento
        - active_weight         : portfolio_weight − benchmark_weight
        - asset_return          : rendimento semplice totale sul periodo
        - portfolio_contribution: portfolio_weight × asset_return
        - benchmark_contribution: benchmark_weight × asset_return
        - active_contribution   : active_weight × asset_return
        - allocation_effect     : active_weight × (asset_return − benchmark_total_return)
        Indice: ticker.
        L'ultima riga ("Total") aggrega ogni colonna.
    """
    tickers = portfolio_weights.index.tolist()

    # Un titolo assente dalla matrice dei rendimenti riceveva rendimento zero,
    # non NaN: ``reindex`` crea una colonna interamente mancante e ``sum`` su
    # una colonna tutta NaN restituisce 0.0 per impostazione predefinita di
    # pandas, quindi il ``fillna`` piu' sotto non entrava nemmeno in gioco. Il
    # titolo compariva nella tavola come una posizione che ha reso esattamente
    # zero, e il totale del portafoglio risultava inferiore al vero di
    # w_i * R_i senza alcun segnale.
    unknown = [t for t in tickers if t not in log_returns.columns]
    if unknown:
        raise ValueError(
            "Rendimenti assenti per "
            + ", ".join(map(str, unknown))
            + ": l'attribuzione non puo' assegnare loro un contributo, e "
            "trattarli come titoli a rendimento nullo falserebbe il totale."
        )

    if benchmark_weights is None:
        n = len(tickers)
        benchmark_weights = pd.Series(np.ones(n) / n, index=tickers)

    bw = benchmark_weights.reindex(tickers).fillna(0.0)
    bw_total = bw.sum()
    if bw_total > 0:
        bw = bw / bw_total

    pw = portfolio_weights.reindex(tickers).fillna(0.0)

    # Rendimento semplice totale per titolo sull'intero periodo.
    # ``min_count=1`` distingue una colonna senza osservazioni da una che
    # somma a zero: la prima resta NaN e viene intercettata, la seconda e' un
    # rendimento nullo legittimo.
    log_aligned = log_returns.reindex(columns=tickers)
    totals = log_aligned.sum(axis=0, min_count=1)
    if totals.isna().any():
        vuoti = ", ".join(map(str, totals[totals.isna()].index))
        raise ValueError(
            f"Nessuna osservazione utilizzabile per {vuoti}: l'attribuzione "
            "richiede una storia per ogni titolo del portafoglio."
        )
    asset_returns = np.expm1(totals).reindex(tickers)

    benchmark_total_return = float((bw * asset_returns).sum())

    active_weight = pw - bw
    portfolio_contrib = pw * asset_returns
    benchmark_contrib = bw * asset_returns
    active_contrib = active_weight * asset_returns
    allocation_effect = active_weight * (asset_returns - benchmark_total_return)

    df = pd.DataFrame(
        {
            "portfolio_weight": pw,
            "benchmark_weight": bw,
            "active_weight": active_weight,
            "asset_return": asset_returns,
            "portfolio_contribution": portfolio_contrib,
            "benchmark_contribution": benchmark_contrib,
            "active_contribution": active_contrib,
            "allocation_effect": allocation_effect,
        },
        index=tickers,
    )

    totals = df.sum()
    totals["asset_return"] = float((bw * asset_returns).sum())  # media pesata col benchmark
    totals.name = "Total"
    df = pd.concat([df, totals.to_frame().T])

    return df
