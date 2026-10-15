"""Validazione retrospettiva della distribuzione simulata.

Il VaR si valida contando quante volte la perdita ha superato un singolo
quantile previsto. Una simulazione prevede una distribuzione intera, quindi
contare gli sfondamenti di un solo quantile butta via gran parte di quello
che ha dichiarato. Il test che la usa tutta e' la trasformata integrale di
probabilita': se la distribuzione prevista F_t e' quella da cui l'esito e'
davvero estratto, allora u_t = F_t(y_t) e' uniforme su (0, 1), qualunque
forma abbia F_t (Rosenblatt 1952; Diebold, Gunther e Tay 1998). Un
istogramma di u_t che si accumula ai bordi dice che le bande sono troppo
stette; uno che si accumula al centro dice che sono troppo larghe; uno
che pende a destra dice che la deriva era troppo bassa.

Il test di uniformita' quindi sostituisce "la banda al novanta per cento
copre il novanta per cento" con "la forma intera e' giusta", e la
copertura di coda viene comunque riportata separatamente, perche' una
distribuzione puo' essere uniforme in media pur fallendo esattamente dove
un portafoglio viene giudicato.

Due proprieta' del design fanno il lavoro, ed entrambe sono le stesse su
cui si appoggia il confronto walk-forward. La distribuzione viene
calibrata su dati strettamente precedenti alla finestra su cui viene
valutata, quindi nessun esito realizzato puo' entrare nella propria
previsione. E le finestre di valutazione non si sovrappongono, quindi le
u_t sono indipendenti sotto l'ipotesi nulla e i test binomiali si
applicano senza correzione — finestre sovrapposte condividerebbero
giorni, e il numero effettivo di osservazioni sarebbe una frazione di
quello nominale, che e' l'errore registrato come D1 nell'audit per la
tabella di copertura del VaR.

Cosa questo non puo' fare. Validare una banda all'orizzonte h richiede
molti tratti non sovrapposti di lunghezza h, quindi una banda a dieci
anni richiederebbe secoli. L'orizzonte che si puo' validare su un
decennio di dati giornalieri si misura in settimane, e l'affermazione
onesta e' che la banda a breve orizzonte e' stata testata e quella a
lungo orizzonte no. Questa affermazione vale piu' di una banda non
testata presentata senza nessuna riserva.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portfolio_optimizer.forecast.var_backtesting import (
    christoffersen_independence,
    kupiec_pof,
)
from portfolio_optimizer.simulation.monte_carlo import (
    arithmetic_drift,
    calibrate_jump_diffusion,
    simulate_block_bootstrap,
    simulate_bootstrap,
    simulate_gbm,
    simulate_jump_diffusion,
)

# Sotto questo numero di finestre non sovrapposte i test non hanno potenza
# che valga la pena riportare: con venti osservazioni una coda al cinque
# per cento ci si aspetta che venga sfondata una volta, e una violazione
# contro due non e' prova di niente.
MIN_WINDOWS = 30

DEFAULT_LEVELS = (0.05, 0.95)

# Le tre procedure che l'applicativo mostra all'utente.
APPLICATION_METHODS = ("gbm", "block_bootstrap", "jump_diffusion")

# Le quattro che il capitolo empirico confronta. La quarta, l'estrazione
# indipendente, non e' un candidato che qualcuno adotterebbe: e' il CONTROLLO
# del bootstrap a blocchi. La ragione per cui il progetto campiona blocchi e'
# che l'estrazione indipendente distrugge il raggruppamento della volatilita',
# e senza il termine di paragone quella ragione resta una citazione anziche'
# una misura. Nell'interfaccia non compare, perche' chiederebbe all'utente di
# interpretare un metodo che non userebbe mai.
EVALUATION_METHODS = APPLICATION_METHODS + ("iid_bootstrap",)


@dataclass(slots=True)
class SimulationRecord:
    """Una distribuzione, emessa a una data, e quello che e' davvero successo."""

    method: str
    issue_date: pd.Timestamp
    horizon_days: int
    realized_log_return: float
    pit: float
    quantiles: dict[float, float] = field(default_factory=dict)

    def breaches(self, lower: float = 0.05, upper: float = 0.95) -> tuple[bool, bool]:
        return self.pit < lower, self.pit > upper


def _pit(simulated_terminal_log: np.ndarray, realized: float) -> float:
    """Dove e' caduto l'esito nella distribuzione prevista.

    La correzione a mid-rank tiene la trasformata lontana da esattamente
    zero e uno, che un campione finito di draw produrrebbe altrimenti
    ogni volta che l'esito cade fuori da tutti i percorsi simulati; un
    valore esattamente zero e' indistinguibile da uno di 0.0001 ma i
    test li trattano in modo diverso.
    """
    n = len(simulated_terminal_log)
    below = float(np.sum(simulated_terminal_log < realized))
    equal = float(np.sum(simulated_terminal_log == realized))
    return (below + 0.5 * equal + 0.5) / (n + 1.0)


def _terminal_log_returns(paths: np.ndarray) -> np.ndarray:
    return np.log(paths[:, -1] / paths[0, 0])


def simulate_terminal_distribution(
    train: pd.Series,
    method: str,
    horizon_days: int,
    n_sims: int = 4000,
    random_state: int = 42,
    steps_per_year: int = 252,
) -> np.ndarray:
    """La distribuzione del rendimento log all'orizzonte che ``method`` prevede.

    Ogni metodo vede solo ``train``. I parametri vengono ristimati a ogni
    data di emissione invece che fissati una volta, perche' una banda i
    cui parametri sono stati adattati sull'intero campione verrebbe
    validata contro dati che l'hanno plasmata.
    """
    years = horizon_days / steps_per_year
    if method == "gbm":
        mu_log = float(train.mean() * steps_per_year)
        sigma = float(train.std(ddof=1) * np.sqrt(steps_per_year))
        paths = simulate_gbm(
            100.0,
            arithmetic_drift(mu_log, sigma),
            sigma,
            years,
            n_sims=n_sims,
            steps_per_year=steps_per_year,
            random_state=random_state,
        )
    elif method == "iid_bootstrap":
        paths = simulate_bootstrap(
            train,
            100.0,
            years,
            n_sims=n_sims,
            steps_per_year=steps_per_year,
            random_state=random_state,
        )
    elif method == "block_bootstrap":
        paths = simulate_block_bootstrap(
            train,
            100.0,
            years,
            n_sims=n_sims,
            steps_per_year=steps_per_year,
            random_state=random_state,
        )
    elif method == "jump_diffusion":
        mu_log = float(train.mean() * steps_per_year)
        sigma = float(train.std(ddof=1) * np.sqrt(steps_per_year))
        calibration = calibrate_jump_diffusion(train)
        diffusive = (
            calibration.diffusive_sigma_annual
            if np.isfinite(calibration.diffusive_sigma_annual)
            else sigma
        )
        paths = simulate_jump_diffusion(
            100.0,
            arithmetic_drift(mu_log, sigma),
            diffusive,
            years,
            jump_lambda=calibration.lambda_annual,
            jump_mu=calibration.jump_mu,
            jump_sigma=calibration.jump_sigma,
            n_sims=n_sims,
            steps_per_year=steps_per_year,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Metodo di simulazione sconosciuto: '{method}'.")
    return _terminal_log_returns(paths)


def build_simulation_records(
    portfolio_log_returns: pd.Series,
    methods: tuple[str, ...] = EVALUATION_METHODS,
    train_days: int = 252,
    horizon_days: int = 21,
    n_sims: int = 4000,
    random_state: int = 42,
    quantile_levels: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> list[SimulationRecord]:
    """Avanza in walk-forward, emettendo una distribuzione a ogni passo e risolvendola.

    Il passo e' uguale all'orizzonte, cosi' le finestre di valutazione
    coprono il campione senza sovrapporsi.
    """
    series = portfolio_log_returns.dropna()
    if horizon_days < 1 or train_days < 2:
        raise ValueError("train_days deve superare 1 e horizon_days essere positivo.")

    records: list[SimulationRecord] = []
    start = train_days
    window_index = 0
    while start + horizon_days <= len(series):
        train = series.iloc[:start]
        future = series.iloc[start : start + horizon_days]
        realized = float(future.sum())
        for method in methods:
            simulated = simulate_terminal_distribution(
                train,
                method,
                horizon_days,
                n_sims=n_sims,
                # Un seme diverso per finestra: riusarne uno solo legherebbe
                # ogni previsione alla stessa realizzazione del rumore, e le
                # violazioni risulterebbero correlate per costruzione.
                random_state=random_state + 977 * window_index,
            )
            records.append(
                SimulationRecord(
                    method=method,
                    issue_date=series.index[start - 1],
                    horizon_days=horizon_days,
                    realized_log_return=realized,
                    pit=_pit(simulated, realized),
                    quantiles={
                        q: float(np.quantile(simulated, q)) for q in quantile_levels
                    },
                )
            )
        start += horizon_days
        window_index += 1
    return records


def _kolmogorov_smirnov_uniform(u: np.ndarray) -> tuple[float, float]:
    """Distanza fra la distribuzione empirica del PIT e l'uniformita'."""
    from scipy.stats import kstest

    statistic, p_value = kstest(u, "uniform")
    return float(statistic), float(p_value)


def evaluate_simulation_coverage(
    records: list[SimulationRecord],
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.DataFrame:
    """Una riga per metodo: la distribuzione prevista combacia con gli esiti?

    Riportate una accanto all'altra perche' falliscono in modo diverso.
    La quota dentro la banda risponde se la larghezza e' giusta. Le due
    code rispondono se la larghezza e' giusta *dove conta*, dato che una
    banda puo' coprire il novanta per cento in totale pur mancando tutto
    sul lato negativo. L'uniformita' della trasformata risponde se la
    forma intera e' giusta, incluse le parti su cui nessun quantile viene
    interrogato.
    """
    if not records:
        return pd.DataFrame()

    rows = []
    for method in dict.fromkeys(r.method for r in records):
        subset = [r for r in records if r.method == method]
        u = np.array([r.pit for r in subset], dtype=float)
        n = len(u)
        below = (u < lower).astype(int)
        above = (u > upper).astype(int)
        inside = 1 - below - above

        nominal_inside = upper - lower
        row: dict[str, object] = {
            "method": method,
            "n_windows": n,
            "horizon_days": subset[0].horizon_days,
            "share_inside": float(inside.mean()),
            "nominal_inside": float(nominal_inside),
            "share_below": float(below.mean()),
            "share_above": float(above.mean()),
            "mean_pit": float(u.mean()),
        }

        if n >= MIN_WINDOWS:
            # kupiec_pof prende il livello di CONFIDENZA e non il tasso atteso
            # di violazione: la frequenza attesa e' 1 - confidence_level. Per
            # una coda inferiore al 5 per cento il livello e' quindi 0,95.
            row["p_kupiec_below"] = kupiec_pof(below, 1.0 - lower)[1]
            row["p_kupiec_above"] = kupiec_pof(above, upper)[1]
            row["p_independence_below"] = christoffersen_independence(below)[1]
            statistic, p_uniform = _kolmogorov_smirnov_uniform(u)
            row["ks_statistic"] = statistic
            row["p_uniform"] = p_uniform
            row["verdict"] = _verdict(row)
        else:
            for key in (
                "p_kupiec_below",
                "p_kupiec_above",
                "p_independence_below",
                "ks_statistic",
                "p_uniform",
            ):
                row[key] = float("nan")
            row["verdict"] = (
                f"{n} finestre: sotto {MIN_WINDOWS} il test non ha potenza "
                "sufficiente perche' il risultato significhi qualcosa."
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("method")


def _verdict(row: dict) -> str:
    """Una frase che nomina la direzione del fallimento, non solo la sua presenza."""
    parts = []
    if row["p_uniform"] < 0.05:
        if row["share_inside"] < row["nominal_inside"] - 0.02:
            parts.append(
                "la distribuzione prevista e' troppo stretta: gli esiti cadono "
                "fuori dalla banda piu' spesso di quanto dichiari"
            )
        elif row["share_inside"] > row["nominal_inside"] + 0.02:
            parts.append(
                "la distribuzione prevista e' troppo larga: la banda copre piu' "
                "di quanto dichiari, quindi promette meno precisione di quella che ha"
            )
        else:
            parts.append(
                "la copertura complessiva e' corretta ma la forma della "
                "distribuzione non lo e'"
            )
    else:
        parts.append("la forma della distribuzione prevista non e' rifiutata")

    if row["p_kupiec_below"] < 0.05:
        direzione = "piu'" if row["share_below"] > 0.05 else "meno"
        parts.append(f"la coda inferiore e' violata {direzione} del dovuto")
    if row["p_kupiec_above"] < 0.05:
        direzione = "piu'" if row["share_above"] > 0.05 else "meno"
        parts.append(f"la coda superiore e' violata {direzione} del dovuto")
    if row["p_independence_below"] < 0.05:
        parts.append("le violazioni inferiori si raggruppano nel tempo")

    if row["mean_pit"] < 0.42:
        parts.append("la deriva prevista e' sistematicamente troppo alta")
    elif row["mean_pit"] > 0.58:
        parts.append("la deriva prevista e' sistematicamente troppo bassa")

    return "; ".join(parts) + "."


def pit_histogram(records: list[SimulationRecord], method: str, bins: int = 10) -> dict:
    """Conteggi della trasformata per bin, col conteggio che darebbe l'uniformita'.

    L'istogramma e' la diagnosi che il p-value riassume: bordi troppo
    pieni vuol dire bande troppo stette, il centro troppo pieno vuol
    dire bande troppo larghe, una pendenza vuol dire che la deriva non
    e' giusta.
    """
    u = np.array([r.pit for r in records if r.method == method], dtype=float)
    if len(u) == 0:
        return {"bins": [], "counts": [], "expected": 0.0}
    counts, edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
    return {
        "bins": [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(bins)],
        "counts": counts.tolist(),
        "expected": float(len(u) / bins),
        "n": int(len(u)),
    }
