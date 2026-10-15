"""Confronto fuori campione tra i criteri di ottimizzazione.

Perche' esiste questo modulo. La pagina di confronto prima classificava i
criteri su misure calcolate IN campione: ogni criterio veniva ottimizzato su
tutta la storia e poi valutato sulla stessa storia. Su un campione qualsiasi
il portafoglio tangente vince quella classifica per costruzione — ottimizza
esattamente la grandezza su cui poi viene giudicato — e la classifica si
rovescia quando si esce dal campione. La pagina quindi incoronava il
criterio che i capitoli empirici mostrano essere il meno affidabile, e
avrebbe fatto vedere il portafoglio equipesato sconfitto, contraddicendo il
risultato per cui niente lo batte davvero.

Cosa lo sostituisce. Ogni criterio viene fatto passare per la stessa
procedura walk-forward usata nei capitoli empirici: si stima su una finestra,
si tiene la posizione per il periodo seguente, e il periodo di detenzione non
viene mai guardato mentre si decide. Le misure si calcolano sul percorso
fuori campione risultante con la stessa funzione che l'applicazione usa per
un singolo portafoglio, cosi' le due tabelle vogliono dire la stessa cosa
sotto la stessa intestazione di colonna. La differenza di Sharpe contro il
portafoglio equipesato viene accompagnata dal suo p-value, perche una
differenza senza p-value non e' una prova di niente.

I valori in campione non vengono buttati via: lo scarto tra i due e' il
risultato che conta davvero, e una pagina che mostrasse solo il lato fuori
campione nasconderebbe la dimostrazione. Vengono calcolati dal chiamante, che
li ha gia' pronti, e riportati separatamente con l'etichetta giusta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portfolio_optimizer.backtest.strategies import StrategySpec, build_strategy
from portfolio_optimizer.backtest.walk_forward import (
    WalkForwardRunConfig,
    walk_forward_run,
)
from portfolio_optimizer.risk.inference import (
    compare_against_benchmark,
    romano_wolf_stepdown,
)
from portfolio_optimizer.risk.metrics import sharpe_is_orderable, summary_metrics

logger = logging.getLogger(__name__)


# L'applicazione e il layer sperimentale chiamano gli stessi criteri in modo
# diverso: l'applicazione li scrive per intero per l'interfaccia, lo
# sperimento li abbrevia per le intestazioni delle tabelle. La mappatura va
# scritta una volta sola, qui, perche' un disallineamento silenzioso non
# darebbe errore — confronterebbe semplicemente la colonna sbagliata senza
# che nessuno se ne accorga.
APP_TO_CRITERION: dict[str, str] = {
    "equal_weight": "equal_weight",
    "minimum_variance": "min_variance",
    "minimum_cvar": "min_cvar",
    "risk_parity": "risk_parity",
    "maximum_sharpe": "max_sharpe",
    "black_litterman": "black_litterman",
}

BENCHMARK_MODEL = "equal_weight"

# Numero minimo di osservazioni fuori campione sotto il quale il confronto
# non viene nemmeno tentato. Due ribilanciamenti da 63 giorni lasciano 126
# punti; una differenza di Sharpe stimata con meno di cosi ha un intervallo
# di confidenza largo abbastanza da contenere qualsiasi cosa.
MIN_OOS_OBSERVATIONS = 126


@dataclass(slots=True)
class ModelComparisonRow:
    """Un criterio, valutato fuori campione."""

    model: str
    label: str
    metrics: dict[str, float] = field(default_factory=dict)
    net_log_returns: pd.Series | None = None
    turnover_mean: float = float("nan")
    cost_total: float = float("nan")
    n_rebalances: int = 0
    sharpe_diff_vs_benchmark: float | None = None
    p_value_vs_benchmark: float | None = None
    # Il p-value corretto per il fatto che il confronto e' stato fatto su tutti
    # i criteri e poi si e' tenuto il migliore: e' quello su cui si decide.
    p_value_stepdown_vs_benchmark: float | None = None
    # Il p-value dello stesso confronto preso da solo, UNILATERALE come il
    # corretto. Serve per mostrare quanto costa la correzione: accostare il
    # bilaterale di Ledoit-Wolf al corretto unilaterale li renderebbe non
    # confrontabili, e la correzione sembrerebbe dimezzare i p-value invece
    # che alzarli.
    p_value_single_onesided: float | None = None
    # Se l'eccesso di rendimento del criterio e' negativo, il suo indice di
    # Sharpe non e' ordinabile e una differenza positiva non significa "meglio".
    sharpe_orderable: bool = True
    # Il percorso coincide con quello del riferimento (Black-Litterman senza
    # view e' il prior): confrontarlo con il riferimento e' confrontare una
    # serie con se' stessa, non un test.
    coincides_with_benchmark: bool = False
    is_benchmark: bool = False
    failure: str | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None and bool(self.metrics)

    @property
    def beats_benchmark_significantly(self) -> bool:
        """Una differenza positiva che resiste al test, e alla ricerca del migliore.

        La pagina confronta ogni criterio col portafoglio equipesato e poi
        riporta quello che vince. Questa e' una sola domanda fatta cinque
        volte, e il p-value di un confronto singolo non basta a risponderle:
        misurato sulle serie fuori campione di questo stesso progetto, sotto
        l'ipotesi che nessun criterio sia migliore, la probabilita' di
        trovare almeno un "significativo" al 5 per cento e' circa il 13 per
        cento — due volte e mezzo il livello nominale. E' lo stesso difetto
        corretto come G15 per la griglia sperimentale, rimasto in piedi nella
        pagina che l'utente legge davvero.

        A decidere e' quindi il p-value corretto. Quando manca — poche
        osservazioni, un bootstrap fallito — non si dichiara niente: tornare
        in silenzio al valore non corretto ripristinerebbe esattamente
        l'errore che la correzione dovrebbe togliere.

        Ci sono altre due condizioni. L'ordinamento per Sharpe deve avere
        SENSO: quando il rendimento in eccesso e' negativo il rapporto non e'
        piu' monotono nella desiderabilita', e una strategia che ha perso il
        trenta per cento mostra una differenza di +0.316 sopra una che ha
        perso il dieci (verificato su tre anni). E il criterio non deve
        essere lui stesso il benchmark: Black-Litterman senza view restituisce
        il prior, che senza capitalizzazioni di mercato e' il portafoglio
        equipesato stesso, quindi testarlo contro il benchmark vuol dire
        testare una serie contro se' stessa.
        """
        return (
            self.sharpe_diff_vs_benchmark is not None
            and self.sharpe_diff_vs_benchmark > 0
            and self.sharpe_orderable
            and not self.coincides_with_benchmark
            and self.p_value_stepdown_vs_benchmark is not None
            and self.p_value_stepdown_vs_benchmark < 0.05
        )


@dataclass(slots=True)
class ModelComparison:
    """Il confronto fuori campione, con quello che serve per leggerlo onestamente."""

    rows: list[ModelComparisonRow] = field(default_factory=list)
    benchmark: str = BENCHMARK_MODEL
    train_days: int = 252
    test_days: int = 63
    transaction_cost: float = 0.001
    max_weight: float = 1.0
    n_observations: int = 0
    start: str = ""
    end: str = ""
    unavailable: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable is None and any(r.ok for r in self.rows)

    def row(self, model: str) -> ModelComparisonRow | None:
        for r in self.rows:
            if r.model == model:
                return r
        return None

    @property
    def winners_over_benchmark(self) -> list[str]:
        return [r.model for r in self.rows if r.beats_benchmark_significantly]

    def verdict(self) -> str:
        """Una frase sola che dice cosa ha trovato il test, in un senso o nell'altro."""
        if not self.available:
            return ""
        winners = self.winners_over_benchmark
        tested = [r for r in self.rows if r.p_value_vs_benchmark is not None]
        if not tested:
            return ""
        unorderable = [
            r for r in tested if r.ok and not r.sharpe_orderable
        ]
        if unorderable and not winners:
            return (
                "Su questa finestra "
                + ("nessun criterio" if len(unorderable) == len(tested) else "alcuni criteri")
                + " ha un eccesso di rendimento positivo: dove si perde, l'indice di "
                "Sharpe non e' piu' ordinabile — una strategia piu' volatile ottiene "
                "un rapporto meno negativo pur avendo perso di piu' — quindi la "
                "classifica va letta sulla perdita e sul drawdown, non sullo Sharpe."
            )
        corrected = [
            r for r in tested if r.p_value_stepdown_vs_benchmark is not None
        ]
        if not corrected:
            return (
                "La correzione per i confronti multipli non e' stata calcolata su "
                "questa finestra, quindi nessun criterio puo' essere dichiarato "
                "superiore al portafoglio equipesato."
            )
        if winners:
            names = ", ".join(
                r.label for r in self.rows if r.model in winners
            )
            return (
                f"Su questo universo {names} supera il portafoglio equipesato con "
                "differenza significativa al 5 per cento, anche tenendo conto che "
                "il confronto e' stato fatto con tutti i criteri e si e' guardato "
                "il migliore. E' un esito che l'indagine empirica non ha mai "
                "osservato: su 56 configurazioni, quattro scenari e cinque regimi "
                f"non e' mai accaduto. Su una finestra di {self.n_observations} "
                "osservazioni la potenza del test e' bassa, quindi il risultato va "
                "trattato come indizio."
            )
        return (
            "Nessun criterio supera il portafoglio equipesato con differenza "
            "significativa al 5 per cento, tenuto conto che il confronto e' stato "
            "fatto con tutti i criteri e si e' guardato il migliore. E' lo stesso "
            "esito dell'indagine empirica su 56 configurazioni, quattro scenari e "
            "cinque regimi."
        )


def _metrics_from_path(net_log: pd.Series, risk_free_annual: float) -> dict[str, float]:
    return summary_metrics(net_log, risk_free_annual=risk_free_annual)


def compare_models_out_of_sample(
    log_returns: pd.DataFrame,
    models: list[tuple[str, str]],
    max_weight: float = 1.0,
    risk_free_annual: float = 0.0,
    returns_method: str = "bayes_stein",
    cov_method: str = "ledoit_wolf",
    train_days: int = 252,
    test_days: int = 63,
    transaction_cost: float = 0.001,
) -> ModelComparison:
    """Fa girare ogni criterio walk-forward sui medesimi dati e lo testa contro 1/N.

    ``models`` e' una lista di coppie (chiave applicazione, etichetta per
    l'utente). Ogni criterio vede le stesse finestre di stima, lo stesso tetto
    di concentrazione e lo stesso costo di transazione: usare un tetto unico
    per tutti e' quello che rende le colonne confrontabili, perche' un tetto
    diverso per ogni criterio farebbe venire la differenza di risultato dal
    vincolo e non dal criterio in se'.
    """
    frame = log_returns.dropna(axis=1, how="all").dropna()
    comparison = ModelComparison(
        train_days=train_days,
        test_days=test_days,
        transaction_cost=transaction_cost,
        max_weight=max_weight,
    )

    required = train_days + MIN_OOS_OBSERVATIONS
    if frame.shape[1] < 2 or len(frame) < required:
        comparison.unavailable = (
            f"Servono almeno {required} osservazioni per una valutazione fuori "
            f"campione con finestra di stima di {train_days} giorni; il campione "
            f"ne contiene {len(frame)}. Allunga l'orizzonte."
        )
        return comparison

    config = WalkForwardRunConfig(
        train_days=train_days,
        test_days=test_days,
        step_days=test_days,
        transaction_cost=transaction_cost,
    )
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / 252) - 1.0

    paths: dict[str, pd.Series] = {}
    for model_key, label in models:
        criterion = APP_TO_CRITERION.get(model_key)
        row = ModelComparisonRow(
            model=model_key, label=label, is_benchmark=model_key == BENCHMARK_MODEL
        )
        if criterion is None:
            row.failure = f"Criterio '{model_key}' non disponibile fuori campione."
            comparison.rows.append(row)
            continue
        try:
            spec = StrategySpec(
                criterion=criterion,
                returns_method=returns_method,
                cov_method=cov_method,
                max_weight=max_weight,
                risk_free_rate=risk_free_annual,
            )
            result = walk_forward_run(frame, build_strategy(spec), config)
            net_log = result.net_log
            if len(net_log) < MIN_OOS_OBSERVATIONS:
                raise ValueError(
                    f"Solo {len(net_log)} osservazioni fuori campione disponibili."
                )
            row.metrics = _metrics_from_path(net_log, risk_free_annual)
            row.net_log_returns = net_log
            row.turnover_mean = result.average_turnover
            row.cost_total = result.total_cost
            row.n_rebalances = result.n_rebalances
            paths[model_key] = net_log
        except Exception as exc:  # noqa: BLE001 - one criterion must not stop the rest
            logger.warning("Confronto fuori campione, %s: %s", model_key, exc)
            row.failure = str(exc)
        comparison.rows.append(row)

    if paths:
        first = next(iter(paths.values()))
        comparison.n_observations = len(first)
        comparison.start = str(first.index[0].date())
        comparison.end = str(first.index[-1].date())

    if BENCHMARK_MODEL in paths and len(paths) > 1:
        try:
            table = compare_against_benchmark(
                paths, BENCHMARK_MODEL, rf_daily=rf_daily, periods=252
            )
            for row in comparison.rows:
                if row.model in table.index:
                    row.sharpe_diff_vs_benchmark = float(
                        table.loc[row.model, "difference_annual"]
                    )
                    row.p_value_vs_benchmark = float(table.loc[row.model, "p_value"])
                elif row.model == BENCHMARK_MODEL and row.ok:
                    row.sharpe_diff_vs_benchmark = 0.0
                    row.p_value_vs_benchmark = 1.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Test sulla differenza di Sharpe non eseguito: %s", exc)

        # Ordinabilita' e coincidenza: due condizioni che precedono il test.
        reference_path = paths[BENCHMARK_MODEL]
        for row in comparison.rows:
            if row.net_log_returns is None or not row.ok:
                continue
            row.sharpe_orderable = sharpe_is_orderable(
                row.net_log_returns, risk_free_annual, 252
            )
            if row.model != BENCHMARK_MODEL:
                pair = pd.concat(
                    [row.net_log_returns, reference_path], axis=1
                ).dropna()
                row.coincides_with_benchmark = bool(
                    not pair.empty
                    and float((pair.iloc[:, 0] - pair.iloc[:, 1]).abs().max()) < 1e-10
                )

        # La stessa correzione applicata alla griglia sperimentale (G15). Qui la
        # famiglia e' piccola -- quattro o cinque criteri contro l'equipesato --
        # ma il principio non cambia, e la pagina di confronto e' il luogo in
        # cui il risultato viene letto.
        try:
            stepdown = romano_wolf_stepdown(
                paths, BENCHMARK_MODEL, rf_daily=rf_daily, periods=252
            )
            for row in comparison.rows:
                if row.model in stepdown.index:
                    row.p_value_stepdown_vs_benchmark = float(
                        stepdown.loc[row.model, "p_value_stepdown"]
                    )
                    row.p_value_single_onesided = float(
                        stepdown.loc[row.model, "p_value_single"]
                    )
                elif row.model == BENCHMARK_MODEL and row.ok:
                    row.p_value_stepdown_vs_benchmark = 1.0
                    row.p_value_single_onesided = 1.0
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Correzione per confronti multipli non eseguita (%s); nessun "
                "criterio potra' essere dichiarato superiore all'equipesato.",
                exc,
            )

    return comparison


def best_and_worst(
    rows: list[ModelComparisonRow], metric: str, higher_is_better: bool
) -> dict[str, str]:
    """Quale criterio vince e quale perde su una misura.

    Tenuto separato dal template perche' per decidere un vincitore serve
    conoscere la convenzione di segno della misura, e un template che la
    indovinasse dall'etichetta finirebbe per segnare il drawdown peggiore
    come il migliore.
    """
    values = [
        (r.model, r.metrics.get(metric))
        for r in rows
        if r.ok and r.metrics.get(metric) is not None
    ]
    values = [(k, v) for k, v in values if v is not None and not np.isnan(v)]
    if len(values) < 2:
        return {}
    ordered = sorted(values, key=lambda kv: kv[1], reverse=higher_is_better)
    return {"best": ordered[0][0], "worst": ordered[-1][0]}
