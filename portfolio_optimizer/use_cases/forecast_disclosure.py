"""Cosa dire accanto a una previsione che il sistema mostra.

L'applicazione mostra all'utente una volatilita' attesa, un Value at Risk e
una distribuzione simulata. Nel capitolo empirico abbiamo misurato quanto
ognuna di queste si e' allontanata da quello che poi e' successo davvero.
Mostrare il numero senza quella misura riprodurrebbe esattamente la prassi
che il lavoro critica: una stima puntuale presentata con una sicurezza che
non si e' guadagnata.

Le correzioni qui sotto vengono dalla tabella incrociata criterio per
stimatore di covarianza, non da quelle marginali, perche un bias che
appartiene a uno stimatore si vede su tutti i criteri che lo hanno usato.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Rapporto tra volatilita' realizzata e dichiarata, misurato su tutto lo
# storico fuori campione nei quattro scenari. Un valore sopra 1 vuol dire
# che il modello ha sottostimato la volatilita' che poi si e' realizzata.
VOLATILITY_BIAS: dict[tuple[str, str], float] = {
    ("minimum_variance", "ewma"): 1.21,
    ("maximum_sharpe", "ewma"): 1.14,
    ("risk_parity", "ewma"): 1.06,
    ("minimum_variance", "ledoit_wolf"): 0.98,
    ("minimum_variance", "sample"): 1.02,
    ("maximum_sharpe", "ledoit_wolf"): 1.04,
    ("maximum_sharpe", "sample"): 1.04,
    ("risk_parity", "ledoit_wolf"): 0.97,
    ("risk_parity", "sample"): 0.96,
    ("minimum_cvar", "ledoit_wolf"): 0.96,
    ("equal_weight", "ledoit_wolf"): 0.94,
}

# Sotto questa distanza da 1 il bias non vale la pena di dirlo all'utente:
# e' dentro il rumore della stima e segnalarlo darebbe un'idea di precisione
# che la misura in realta' non ha.
MATERIAL_BIAS = 0.05

# Tasso di violazione osservato del VaR storico rispetto al suo livello
# nominale, aggregato su tutte le configurazioni. La copertura e' stata
# rigettata quasi dappertutto, pochissime eccezioni.
VAR_OBSERVED_RATE = {0.95: 0.065, 0.99: 0.024}


@dataclass(frozen=True, slots=True)
class ForecastDisclosure:
    """Una previsione mostrata all'utente, con quel che sappiamo sulla sua accuratezza."""

    quantity: str
    declared: float
    corrected: float | None
    note: str
    reliable: bool = True
    interval: tuple[float, float] | None = field(default=None)

    def as_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "declared": self.declared,
            "corrected": self.corrected,
            "note": self.note,
            "reliable": self.reliable,
            "interval": list(self.interval) if self.interval else None,
        }


def volatility_disclosure(
    declared_volatility: float, criterion: str, cov_method: str
) -> ForecastDisclosure:
    """Affianca alla volatilita' dichiarata il bias che abbiamo misurato.

    Quando il bias e' rilevante, il valore corretto viene affiancato a quello
    dichiarato e non lo sostituisce: l'utente ha diritto a vedere sia cosa ha
    detto il modello sia cosa suggerisce lo storico.
    """
    ratio = VOLATILITY_BIAS.get((criterion, cov_method))
    if ratio is None:
        return ForecastDisclosure(
            quantity="volatilita_attesa",
            declared=declared_volatility,
            corrected=None,
            note=(
                "Nessuna misura di accuratezza disponibile per questa "
                "combinazione di criterio e stimatore."
            ),
            reliable=False,
        )

    if abs(ratio - 1.0) < MATERIAL_BIAS:
        return ForecastDisclosure(
            quantity="volatilita_attesa",
            declared=declared_volatility,
            corrected=None,
            note=(
                "Nel periodo esaminato la volatilita' dichiarata da questa "
                "combinazione non si e' discostata in modo apprezzabile da quella "
                "realizzata."
            ),
        )

    corrected = declared_volatility * ratio
    verso = "sottostimata" if ratio > 1 else "sovrastimata"
    scostamento = abs(ratio - 1.0)
    return ForecastDisclosure(
        quantity="volatilita_attesa",
        declared=declared_volatility,
        corrected=corrected,
        note=(
            f"Nel periodo esaminato questa combinazione ha {verso} la volatilita' "
            f"del {scostamento:.0%}. Il valore corretto per la distorsione "
            f"osservata e' {corrected:.1%}."
        ),
        reliable=False,
        interval=(min(declared_volatility, corrected), max(declared_volatility, corrected)),
    )


def var_disclosure(declared_var: float, confidence_level: float = 0.95) -> ForecastDisclosure:
    """Affianca a un Value at Risk la sua copertura osservata.

    Il test di un VaR non e' se e' piccolo ma se viene superato tante volte
    quante dichiara. Su questo storico viene superato molto piu spesso, e un
    utente che leggesse il numero come un limite verrebbe tratto in inganno.
    """
    nominal = 1.0 - confidence_level
    observed = VAR_OBSERVED_RATE.get(round(confidence_level, 2))
    if observed is None:
        return ForecastDisclosure(
            quantity="var",
            declared=declared_var,
            corrected=None,
            note="Copertura non misurata per questo livello di confidenza.",
            reliable=False,
        )

    factor = observed / nominal
    return ForecastDisclosure(
        quantity="var",
        declared=declared_var,
        corrected=None,
        note=(
            f"Questa soglia dichiara di essere superata nell'{nominal:.0%} dei "
            f"giorni. Nel periodo esaminato e' stata superata nel {observed:.1%}, "
            f"cioe' {factor:.1f} volte piu' spesso. Va letta come un ordine di "
            "grandezza della perdita di coda, non come un limite."
        ),
        reliable=False,
    )


def disclosures_for(
    criterion: str,
    cov_method: str,
    volatility: float,
    var_95: float | None = None,
) -> list[ForecastDisclosure]:
    """Tutto quello da mostrare accanto alle previsioni di una ottimizzazione."""
    out = [volatility_disclosure(volatility, criterion, cov_method)]
    if var_95 is not None:
        out.append(var_disclosure(var_95, 0.95))
    return out
