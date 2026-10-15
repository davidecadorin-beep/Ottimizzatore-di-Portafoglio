"""Valutazione walk-forward con strumentazione per ogni ribilanciamento.

``walk_forward_backtest`` in questo package restituisce solo il percorso
di equity, che e' tutto cio' di cui l'applicazione web ha bisogno. I
capitoli empirici hanno bisogno di piu': i pesi scelti a ogni data di
ribilanciamento, il turnover che implicano, i costi di transazione che
comportano, e le stime che il modello aveva al momento della decisione —
perche' quelle stime sono le previsioni la cui accuratezza viene testata
dopo.

Questo modulo aggiunge quella strumentazione senza toccare la funzione
esistente, cosi' niente che dipende da essa cambia comportamento.

Nota di design. Il runner non sa cosa sia una previsione: espone ogni
evento di ribilanciamento tramite ``on_rebalance`` e lascia che sia chi
chiama a decidere cosa registrare. Questo mantiene il layer di backtest
indipendente dal layer di previsione, come richiedono i contratti di
import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cosa restituisce una strategia
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StrategyOutput:
    """I pesi, piu' le stime che li hanno prodotti.

    ``expected_returns`` e ``covariance`` sono opzionali: criteri come la
    parita' di rischio o il portafoglio equipesato non stimano affatto i
    rendimenti attesi, e quell'assenza e' di per se' informativa.
    """

    weights: pd.Series
    expected_returns: pd.Series | None = None
    covariance: pd.DataFrame | None = None


class StrategyCallable(Protocol):
    def __call__(self, train: pd.DataFrame) -> pd.Series | StrategyOutput: ...


@dataclass(slots=True)
class RebalanceContext:
    """Tutto quello che si puo' osservare a una data di ribilanciamento."""

    issue_date: pd.Timestamp
    horizon_days: int
    train: pd.DataFrame
    test: pd.DataFrame
    output: StrategyOutput
    previous_weights: pd.Series | None
    turnover: float
    cost: float

    @property
    def weights(self) -> pd.Series:
        return self.output.weights


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WalkForwardRunConfig:
    """Parametri della procedura walk-forward.

    ``step_days`` separa la frequenza di ribilanciamento dalla lunghezza
    della finestra di stima. Se lasciato non impostato la procedura
    avanza di ``test_days``, il che riproduce il comportamento del
    backtester originale. Impostarlo in modo indipendente e' quello che
    permette di confrontare ribilanciamenti mensili, trimestrali e
    annuali sulla stessa finestra di stima.
    """

    train_days: int = 252
    test_days: int = 63
    step_days: int | None = None
    transaction_cost: float = 0.001

    def effective_step(self) -> int:
        return self.step_days if self.step_days is not None else self.test_days


# ---------------------------------------------------------------------------
# Risultato
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WalkForwardResult:
    """Il percorso fuori campione e le decisioni che l'hanno prodotto."""

    net_simple: pd.Series
    gross_simple: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    n_rebalances: int = 0
    contexts: list[RebalanceContext] = field(default_factory=list, repr=False)

    @property
    def net_log(self) -> pd.Series:
        return pd.Series(
            np.log1p(self.net_simple.to_numpy()),
            index=self.net_simple.index,
            name="net_log",
        )

    @property
    def gross_log(self) -> pd.Series:
        return pd.Series(
            np.log1p(self.gross_simple.to_numpy()),
            index=self.gross_simple.index,
            name="gross_log",
        )

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net_simple).cumprod()

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    @property
    def average_turnover(self) -> float:
        return float(self.turnover.mean()) if len(self.turnover) else 0.0


# ---------------------------------------------------------------------------
# Esecutore
# ---------------------------------------------------------------------------


def _as_output(value: pd.Series | StrategyOutput) -> StrategyOutput:
    return value if isinstance(value, StrategyOutput) else StrategyOutput(weights=value)


def compute_turnover(new: pd.Series, previous: pd.Series | None) -> float:
    """Turnover one-way: meta' della somma dei cambi di peso in valore assoluto.

    La prima allocazione e' un acquisto completo e quindi ha turnover
    uno.

    Convenzione, e il suo limite. Il turnover si misura tra due
    allocazioni target successive, non tra il target e la posizione come
    si e' spostata durante il periodo di detenzione. Con questa
    convenzione una strategia con target costanti — il portafoglio
    equipesato — opera solo all'ingresso, mentre in pratica verrebbe
    riportata ai suoi target a ogni data di ribilanciamento. La
    convenzione quindi sottostima il costo delle strategie a peso
    costante ed e' quella giusta per confrontare quanto si muovono le
    *decisioni* di modelli diversi, che e' il punto della domanda
    sull'errore di stima. Confrontare regole di ribilanciamento richiede
    invece la posizione spostata, ed e' un esercizio diverso.
    """
    if previous is None:
        return 1.0
    aligned_new, aligned_old = new.align(previous, fill_value=0.0)
    return float(np.abs(aligned_new - aligned_old).sum()) / 2.0


def walk_forward_run(
    returns: pd.DataFrame,
    strategy_fn: StrategyCallable,
    config: WalkForwardRunConfig | None = None,
    on_rebalance: Callable[[RebalanceContext], None] | None = None,
) -> WalkForwardResult:
    """Esegue la procedura walk-forward e registra ogni decisione di ribilanciamento.

    ``returns`` sono rendimenti log giornalieri. La strategia vede solo
    la finestra di stima che precede ogni periodo di detenzione, quindi
    nessuna informazione dal periodo di detenzione puo' entrare nella
    decisione: l'assenza di look-ahead bias e' una proprieta' del ciclo,
    non qualcosa da controllare dopo.
    """
    cfg = config or WalkForwardRunConfig()
    if cfg.train_days < 2 or cfg.test_days < 1:
        raise ValueError("train_days must exceed 1 and test_days must be positive.")

    clean = returns.dropna(how="all")
    step = cfg.effective_step()
    if step <= 0:
        raise ValueError(f"step_days must be positive (got {step}).")
    if step > cfg.test_days:
        raise ValueError(
            f"step_days ({step}) cannot exceed test_days ({cfg.test_days}), "
            "otherwise observations are skipped creating gaps in the out-of-sample evaluation."
        )

    gross_parts: list[pd.Series] = []
    net_parts: list[pd.Series] = []
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    turnover_rows: dict[pd.Timestamp, float] = {}
    cost_rows: dict[pd.Timestamp, float] = {}
    contexts: list[RebalanceContext] = []

    previous_weights: pd.Series | None = None
    start = cfg.train_days

    while start < len(clean):
        train = clean.iloc[start - cfg.train_days : start]
        test = clean.iloc[start : start + min(step, cfg.test_days)]
        if test.empty:
            break

        output = _as_output(strategy_fn(train))
        weights = output.weights.reindex(clean.columns).fillna(0.0)
        issue_date = train.index[-1]

        turnover = compute_turnover(weights, previous_weights)
        cost = turnover * cfg.transaction_cost

        test_simple = np.expm1(test.to_numpy())
        gross = pd.Series(
            test_simple @ weights.to_numpy(), index=test.index, name="gross_simple"
        )
        net = gross.copy()
        net.iloc[0] = net.iloc[0] - cost

        gross_parts.append(gross)
        net_parts.append(net)
        weight_rows[issue_date] = weights
        turnover_rows[issue_date] = turnover
        cost_rows[issue_date] = cost

        context = RebalanceContext(
            issue_date=issue_date,
            horizon_days=len(test),
            train=train,
            test=test,
            output=output,
            previous_weights=previous_weights,
            turnover=turnover,
            cost=cost,
        )
        contexts.append(context)
        if on_rebalance is not None:
            on_rebalance(context)

        previous_weights = weights
        start += step

    if not net_parts:
        empty = pd.Series(dtype=float)
        return WalkForwardResult(
            net_simple=empty,
            gross_simple=empty,
            weights=pd.DataFrame(columns=clean.columns),
            turnover=pd.Series(dtype=float),
            costs=pd.Series(dtype=float),
        )

    return WalkForwardResult(
        net_simple=pd.concat(net_parts).sort_index(),
        gross_simple=pd.concat(gross_parts).sort_index(),
        weights=pd.DataFrame(weight_rows).T.sort_index(),
        turnover=pd.Series(turnover_rows).sort_index(),
        costs=pd.Series(cost_rows).sort_index(),
        n_rebalances=len(contexts),
        contexts=contexts,
    )
