from __future__ import annotations

# Criterio predefinito e vincoli di concentrazione, derivati dal confronto
# empirico e non decisi a tavolino. La regola completa, con la diagnostica
# dell'universo e le motivazioni per ciascun criterio, e' in
# use_cases/model_selector.py.
#
# Non esiste piu' una mappatura da profilo di rischio a criterio. Un profilo e'
# un'etichetta, e tradurla in un modello nasconde l'unica cosa che conta: che
# cosa ciascun criterio dimostrabilmente offre e che cosa no. Su 56
# configurazioni, quattro scenari e cinque regimi nessun criterio ha superato il
# portafoglio equipesato con differenza significativa; cio' che differisce e' il
# rischio realizzato. L'utente sceglie il criterio, il sistema gli dice che cosa
# comporta sceglierlo.
DEFAULT_MODEL: str = "equal_weight"
DEFAULT_MAX_WEIGHT: float = 0.40

# Tetto di concentrazione suggerito per criterio. I criteri che concentrano per
# costruzione ricevono il vincolo piu' stretto: e' l'unica cosa fra un
# ottimizzatore e un portafoglio di due titoli.
MODEL_MAX_WEIGHT: dict[str, float] = {
    "equal_weight": 1.00,
    "minimum_variance": 0.30,
    "minimum_cvar": 0.30,
    "risk_parity": 0.40,
    "maximum_sharpe": 0.50,
    "black_litterman": 0.50,
}

MODEL_OPTIONS: list[tuple[str, str]] = [
    ("equal_weight", "Equipesato (1/N)"),
    ("minimum_variance", "Minima Varianza"),
    ("minimum_cvar", "Minimo CVaR"),
    ("risk_parity", "Parita' di rischio (ERC)"),
    ("maximum_sharpe", "Portafoglio tangente (Max Sharpe)"),
    ("black_litterman", "Black-Litterman"),
]

SUPPORTED_OPTIMIZATION_MODELS: frozenset[str] = frozenset(
    [key for key, _ in MODEL_OPTIONS]
)

BACKTEST_TRAIN_DAYS: int = 252
BACKTEST_TEST_DAYS: int = 63
BACKTEST_TRANSACTION_COST: float = 0.001

ROLLING_SHARPE_WINDOW: int = 126

# Il numero di simulazioni determina la stabilita' dei percentili mostrati.
# Con duemila, su venti semi diversi il quinto percentile a cinque anni
# oscillava di sei punti e la probabilita' di perdita di un fattore sei, mentre
# l'interfaccia la riporta con un decimale. Era stato ridotto per contenere la
# memoria, vincolo rimosso conservando le sole colonne effettivamente lette:
# la matrice completa dei percorsi non viene piu' materializzata.
# Diecimila e' il punto in cui la stabilita' dei percentili smette di
# migliorare quanto il tempo di calcolo peggiora: dimezza l'oscillazione del
# quinto percentile fra semi diversi rispetto a duemila, e costa 2,3 secondi
# sull'orizzonte piu' lungo contro i 5,6 di ventimila.
MC_N_SIMS: int = 10_000
# Percorsi simulati per lotto. Ogni lotto materializza la propria matrice
# completa e ne conserva le sole colonne campionate, quindi il lotto e non il
# totale determina la memoria di picco.
MC_BATCH_SIZE: int = 2_000
MC_STEPS_PER_YEAR: int = 252

# Verifica retrospettiva della copertura delle bande simulate. La finestra di
# stima e' la stessa del walk-forward; le simulazioni sono meno numerose
# perche' qui serve la forma della distribuzione e non il suo quinto
# percentile con tre cifre.
SIM_COVERAGE_TRAIN_DAYS: int = 252
SIM_COVERAGE_N_SIMS: int = 2_000
MC_DOWNSAMPLE_POINTS: int = 120
MC_TARGET_RETURN: float = 0.50
MC_LOSS_THRESHOLD: float = 0.20

RESULT_CACHE_TTL: int = 300

RETURNS_METHOD_OPTIONS: list[tuple[str, str]] = [
    ("bayes_stein", "Bayes-Stein (default)"),
    ("historical", "Media Storica"),
    ("fama_french", "Fama-French 3 (regionale)"),
]

COV_METHOD_OPTIONS: list[tuple[str, str]] = [
    ("ledoit_wolf", "Ledoit-Wolf (default)"),
    ("ewma", "EWMA λ=0.94 (RiskMetrics)"),
    ("sample", "Campionaria"),
]

REFERENCE_CURRENCY_OPTIONS: frozenset[str] = frozenset({"auto", "usd", "eur"})
