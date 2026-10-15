"""Simulazione prospettica del valore del portafoglio.

Tre procedure, volutamente diverse in quello che assumono, mostrate una
accanto all'altra cosi' la differenza fra loro si legge bene: una diffusione
lognormale, un ricampionamento della storia stessa del portafoglio, e una
diffusione con salti.

Convenzioni, scritte una volta sola perche' due di queste erano state
applicate male in passato e nessuno dei due errori dava un avviso.

**Il drift.** ``simulate_gbm`` e ``simulate_jump_diffusion`` prendono
``mu_arithmetic``, il drift del rendimento ARITMETICO, che e' la
parametrizzazione standard del moto Browniano geometrico: il termine
``- sigma**2 / 2`` dentro l'esponente e' esattamente la conversione da
aritmetico a logaritmico. Chi chiama con una media annualizzata di
rendimenti LOG deve convertirla con ``arithmetic_drift`` prima di
passarla, altrimenti la conversione viene applicata due volte e la
mediana simulata cresce a ``mu - sigma**2 / 2`` invece di ``mu``. Questa
sottostima e' invisibile sul grafico — i percorsi sembrano perfettamente
normali — e cresce col quadrato della volatilita'.

**Rendimenti log dappertutto.** Le procedure di bootstrap prendono una
serie di rendimenti LOG giornalieri e li compongono con ``exp``, mai con
``1 + r``.

**Numeri casuali comuni.** Tutte le procedure usano lo stesso seed, quindi
gli shock gaussiani della diffusione e della diffusione con salti sono gli
stessi draw. La differenza fra quelle due curve e' quindi attribuibile
solo alla componente dei salti e non a una diversa realizzazione del
rumore. Confrontare run fatte con seed indipendenti non avrebbe questa
proprieta'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def arithmetic_drift(mu_log: float, sigma: float) -> float:
    """Converte un drift log annualizzato nel drift aritmetico che il GBM si aspetta.

    E[R_simple] ~= E[r_log] + sigma**2 / 2. E' la stessa correzione di Ito
    che l'ottimizzatore applica ai rendimenti attesi prima
    dell'ottimizzazione media-varianza; qui e' scritta come funzione con un
    nome proprio cosi' la conversione si vede esplicitamente dal
    chiamante invece di restare implicita.
    """
    return float(mu_log) + 0.5 * float(sigma) ** 2


def simulate_gbm(
    initial_value: float,
    mu_arithmetic: float,
    sigma: float,
    years: float,
    n_sims: int = 10000,
    steps_per_year: int = 252,
    random_state: int = 42,
) -> np.ndarray:
    """Moto Browniano geometrico.

    ``mu_arithmetic`` e' il drift del rendimento aritmetico: vedi il
    docstring del modulo. Passa ``arithmetic_drift(mu_log, sigma)`` se
    parti da una media annualizzata di rendimenti log.
    """
    dt = 1 / steps_per_year
    n_steps = int(years * steps_per_year)
    rng = np.random.default_rng(random_state)
    z = rng.normal(0, 1, (n_sims, n_steps))
    increments = np.exp((mu_arithmetic - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = initial_value
    paths[:, 1:] = initial_value * np.cumprod(increments, axis=1)
    return paths


def simulate_bootstrap(
    historical_returns: pd.Series,
    initial_value: float,
    years: float,
    n_sims: int = 10000,
    steps_per_year: int = 252,
    random_state: int = 42,
) -> np.ndarray:
    """Ricampionamento indipendente dei rendimenti log storici.

    Non assume niente sulla distribuzione ma distrugge ogni traccia di
    clustering della volatilita', ed e' per questo che l'applicazione usa
    la variante a blocchi qui sotto. Tenuta come riferimento rispetto al
    quale si giudica la variante a blocchi.
    """
    n_steps = int(years * steps_per_year)
    rng = np.random.default_rng(random_state)
    samples = historical_returns.dropna().values
    draws = rng.choice(samples, size=(n_sims, n_steps), replace=True)
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = initial_value
    # I draw in ingresso sono rendimenti log; si compongono con exp, non con (1+r).
    paths[:, 1:] = initial_value * np.cumprod(np.exp(draws), axis=1)
    return paths


def simulate_block_bootstrap(
    historical_returns: pd.Series,
    initial_value: float,
    years: float,
    n_sims: int = 10000,
    steps_per_year: int = 252,
    block_size: int = 21,
    random_state: int = 42,
) -> np.ndarray:
    """Bootstrap a blocchi circolare: blocchi contigui, che si avvolgono alla fine.

    Campionare blocchi contigui di ``block_size`` giorni invece di draw
    indipendenti preserva il clustering della volatilita', e vale la pena
    dire con precisione cosa questo comporta, perche' l'assunzione
    naturale al riguardo e' sbagliata. NON allarga la distribuzione
    all'orizzonte: il clustering e' dipendenza nei quadrati, mentre la
    varianza di una somma all'orizzonte dipende dall'autocovarianza dei
    livelli, che il clustering lascia a zero. Misurato su serie con
    clustering generate indipendentemente, la dispersione sotto i due
    schemi e' la stessa entro il suo errore standard.

    Quello che cambia e' la forma. I draw indipendenti lasciano che il
    teorema del limite centrale appiattisca sessantatre osservazioni
    verso una normale; i blocchi contigui tengono insieme i tratti
    agitati, cosi' la distribuzione all'orizzonte mantiene la sua curtosi
    in eccesso — 0.71 contro 0.02 su una serie con regimi di volatilita',
    contro una differenza di 0.003 su una serie senza nessuna dipendenza.
    Verificato in ``tests/test_simulation_evaluation.py``.

    Un avvertimento sulla lettura di un singolo portafoglio. Lo schema
    riproduce le autocovarianze empiriche del campione che gli viene
    dato, incluso il rumore di campionamento, quindi su una storia la
    dispersione relativa a un draw indipendente varia intorno a uno con
    una deviazione standard di circa 0.07 su sei anni di dati
    giornalieri. Differenze di quella dimensione fra le due curve su un
    grafico sono rumore, non prova di niente.

    I blocchi sono CIRCOLARI: gli indici si avvolgono modulo la lunghezza
    del campione. Senza l'avvolgimento, gli inizi di blocco estratti
    uniformemente in [0, T - b] danno a un'osservazione interna ``b``
    probabilita' di essere scelta mentre la prima e l'ultima ne hanno
    solo una, quindi la distribuzione ricampionata non e' quella empirica
    e il sottopeso cade sulle osservazioni piu' recenti — proprio quelle
    su cui una proiezione in avanti si appoggia di piu'. L'avvolgimento
    ripristina lo stesso peso per ogni osservazione. E' la stessa
    costruzione che ``risk.inference`` usa per il bootstrap studentizzato
    della differenza di Sharpe; adesso i due sono coerenti fra loro.
    """
    n_steps = int(years * steps_per_year)
    rng = np.random.default_rng(random_state)
    samples = historical_returns.dropna().values
    T = len(samples)
    if T < 2:
        raise ValueError("Serve piu' di un'osservazione per il bootstrap.")

    block = min(block_size, T)
    n_blocks = int(np.ceil(n_steps / block))
    starts = rng.integers(0, T, size=(n_sims, n_blocks))
    offsets = np.arange(block)
    # (n_sims, n_blocks, block) -> (n_sims, n_blocks * block) -> tagliato alla lunghezza giusta
    index = (starts[:, :, None] + offsets[None, None, :]) % T
    draws = samples[index.reshape(n_sims, -1)[:, :n_steps]]

    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = initial_value
    paths[:, 1:] = initial_value * np.cumprod(np.exp(draws), axis=1)
    return paths


@dataclass(frozen=True, slots=True)
class JumpCalibration:
    """Cosa dice la serie storica sui salti, se dice qualcosa.

    ``diffusive_sigma_annual`` e' la volatilita' della serie coi giorni di
    salto rimossi, ed e' quella che va passata alla simulazione: la
    volatilita' totale contiene gia' i salti, quindi usarla e poi
    aggiungere i salti sopra conta la loro varianza due volte.
    """

    lambda_annual: float
    jump_mu: float
    jump_sigma: float
    diffusive_sigma_annual: float
    n_jumps: int
    threshold: float
    detected: bool

    def describe(self) -> str:
        if not self.detected:
            return (
                "Nessun movimento oltre la soglia di rilevazione: il processo di "
                "salto e' inattivo e la simulazione coincide con la diffusione pura."
            )
        return (
            f"{self.n_jumps} movimenti oltre {self.threshold:.2%} giornaliero, "
            f"pari a {self.lambda_annual:.1f} l'anno; ampiezza media {self.jump_mu:.2%}, "
            f"dispersione {self.jump_sigma:.2%}."
        )


def calibrate_jump_diffusion(
    historical_returns: pd.Series,
    n_sigma: float = 4.0,
    min_jumps: int = 5,
) -> JumpCalibration:
    """Identifica i salti come movimenti troppo grandi per essere spiegati dalla diffusione.

    La soglia e' ``n_sigma`` volte una stima ROBUSTA della deviazione
    standard giornaliera, ottenuta dalla deviazione assoluta mediana cosi'
    i salti stessi non gonfiano la grandezza usata per rilevarli. Quattro
    deviazioni standard robuste corrispondono a circa un giorno su
    quindicimila sotto normalita', quindi un movimento oltre quella soglia
    si adatta male alla diffusione ed e' un candidato ragionevole per un
    salto.

    Cosa sostituisce questo, e perche'. La versione precedente usava una
    soglia ASSOLUTA di -3 per cento, slegata dalla volatilita' del
    portafoglio che si stava simulando. Quanti "salti" trovava era quindi
    deciso da quella volatilita' e non dalla presenza reale di salti: su
    serie puramente normali, i portafogli sotto il 20 per cento di
    volatilita' annua non arrivavano mai al numero minimo e ricevevano
    tre costanti scritte a mano, mentre quelli sopra saturavano il tetto
    massimo di cinque salti all'anno rietichettando la coda sinistra della
    gaussiana, che a quella volatilita' sta a meno di due deviazioni
    standard di distanza. Guardava anche solo i movimenti negativi, e i
    giorni di salto restavano nel campione da cui veniva stimata la
    volatilita' diffusiva.

    Seguono tre cambiamenti. La soglia si scala con la serie. Si
    considerano entrambe le direzioni, cosi' una serie che salta verso
    l'alto non viene forzata in un modello solo verso il basso. E la
    volatilita' diffusiva viene misurata sui giorni rimanenti, cosi' la
    varianza dei salti si aggiunge invece di essere contata due volte.

    Quando troppo pochi movimenti qualificano, la risposta onesta e' che
    questa serie non mostra salti: la calibrazione restituisce un
    processo inattivo, e la diffusione con salti degenera nella
    diffusione semplice. Inventare una frequenza di salti metterebbe sullo
    schermo un numero che non viene da niente nei dati.
    """
    clean = historical_returns.dropna()
    T = len(clean)
    values = clean.to_numpy()

    if T < 63:
        return JumpCalibration(0.0, 0.0, 0.0, float("nan"), 0, float("nan"), False)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = float(np.std(values, ddof=1))
    threshold = n_sigma * robust_sigma

    is_jump = np.abs(values - median) > threshold
    n_jumps = int(is_jump.sum())
    diffusive = values[~is_jump]
    diffusive_sigma = float(np.std(diffusive, ddof=1) * np.sqrt(252)) if len(diffusive) > 1 else float("nan")

    if n_jumps < min_jumps:
        return JumpCalibration(
            0.0, 0.0, 0.0, diffusive_sigma, n_jumps, threshold, False
        )

    jumps = values[is_jump]
    return JumpCalibration(
        lambda_annual=float(n_jumps / T * 252),
        jump_mu=float(jumps.mean()),
        jump_sigma=float(jumps.std(ddof=1)) if n_jumps > 1 else 0.0,
        diffusive_sigma_annual=diffusive_sigma,
        n_jumps=n_jumps,
        threshold=threshold,
        detected=True,
    )


def simulate_jump_diffusion(
    initial_value: float,
    mu_arithmetic: float,
    sigma: float,
    years: float,
    jump_lambda: float = 0.2,
    jump_mu: float = -0.1,
    jump_sigma: float = 0.2,
    n_sims: int = 10000,
    steps_per_year: int = 252,
    random_state: int = 42,
) -> np.ndarray:
    """Diffusione con salti di Merton (1976).

    ``mu_arithmetic`` segue la stessa convenzione di ``simulate_gbm``, e
    ``sigma`` e' la volatilita' DIFFUSIVA — la volatilita' della serie una
    volta esclusi i giorni di salto. Passare la volatilita' totale conta
    i salti due volte, perche' la componente dei salti aggiunge la sua
    varianza sopra.

    Il drift porta il compensatore di Merton, ``lambda * (exp(mu_J +
    sigma_J**2 / 2) - 1)``, cosi' aggiungere i salti ispessisce le code
    senza spostare anche il valore atteso.

    Quando piu' salti capitano nello stesso step le loro dimensioni si
    sommano come draw indipendenti — media ``n * mu_J`` e varianza ``n *
    sigma_J**2``. Moltiplicare un singolo draw per ``n``, come faceva la
    versione precedente, darebbe varianza ``n**2 * sigma_J**2``. A una
    frequenza di salti di cinque all'anno il caso si presenta circa due
    volte su diecimila step, quindi l'effetto e' trascurabile; viene
    corretto perche' costa una riga e perche' una formula sbagliata, anche
    se raramente, resta sbagliata.
    """
    dt = 1 / steps_per_year
    n_steps = int(years * steps_per_year)
    rng = np.random.default_rng(random_state)
    z = rng.normal(size=(n_sims, n_steps))
    n_jumps = rng.poisson(jump_lambda * dt, size=(n_sims, n_steps))
    z_jump = rng.normal(size=(n_sims, n_steps))
    jump_part = n_jumps * jump_mu + np.sqrt(n_jumps) * jump_sigma * z_jump
    jump_compensation = jump_lambda * (np.exp(jump_mu + 0.5 * jump_sigma**2) - 1)
    increments = np.exp(
        (mu_arithmetic - 0.5 * sigma**2 - jump_compensation) * dt
        + sigma * np.sqrt(dt) * z
        + jump_part
    )
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = initial_value
    paths[:, 1:] = initial_value * np.cumprod(increments, axis=1)
    return paths


def summarize_simulation(
    paths: np.ndarray,
    target_return: float = 0.5,
    loss_threshold: float = 0.2,
) -> dict[str, float]:
    """Percentili del valore terminale e due probabilita' di soglia.

    Accetta sia la matrice completa dei percorsi sia una ridotta a un
    sottoinsieme delle sue colonne, a patto che la prima e l'ultima colonna
    siano tenute: solo quelle due vengono lette.
    """
    final_values = paths[:, -1]
    initial = paths[0, 0]
    pct = np.percentile(final_values, [5, 25, 50, 75, 95])
    total_ret = final_values / initial - 1
    return {
        "p5": float(pct[0]),
        "p25": float(pct[1]),
        "p50": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
        "prob_target": float(np.mean(total_ret >= target_return)),
        "prob_loss_exceeds_threshold": float(
            np.mean(total_ret <= -abs(loss_threshold))
        ),
    }


def simulate_downsampled(
    simulate,
    indices: list[int],
    n_sims: int,
    batch_size: int = 2000,
    random_state: int = 42,
) -> np.ndarray:
    """Esegue una simulazione a lotti, tenendo solo le colonne che servono davvero.

    La matrice completa dei percorsi e' il motivo per cui il numero di
    simulazioni andava tagliato: duemila percorsi su un orizzonte di dieci
    anni occupano quaranta megabyte a procedura. Di quelle 2.521 colonne
    l'applicazione ne legge circa centoventi, per il grafico, e l'ultima,
    per il riepilogo. Tenere solo quelle rende il costo in memoria venti
    volte piu' piccolo, ed e' proprio questo che permette di far
    risalire il numero di simulazioni — ed e' il numero di simulazioni, non
    il numero di colonne salvate, a decidere se i percentili mostrati sono
    stabili.

    ``simulate`` viene chiamata come ``simulate(n_sims, random_state)`` e
    deve restituire la matrice completa dei percorsi per quel lotto. Ogni
    lotto ha il suo seed, derivato da ``random_state``: riusare un solo
    seed produrrebbe lotti identici e non moltiplicherebbe niente tranne
    il tempo di esecuzione.
    """
    parts = []
    produced = 0
    batch_index = 0
    while produced < n_sims:
        size = min(batch_size, n_sims - produced)
        paths = simulate(size, random_state + 1000 * batch_index)
        parts.append(paths[:, indices])
        produced += size
        batch_index += 1
    return np.vstack(parts)
