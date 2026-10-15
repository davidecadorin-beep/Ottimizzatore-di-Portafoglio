"""Dall'evidenza empirica al criterio che il sistema propone.

Questo modulo e' il punto in cui i risultati dei capitoli empirici entrano
nell'applicazione. Ogni frase scritta qui sotto porta con se' il risultato
che la giustifica, cosi' una raccomandazione si puo' sempre ricondurre alla
tabella da cui viene, e se un risultato cambia la regola deve cambiare
insieme a lui.

La regola si appoggia su una sola conclusione, che i dati hanno imposto piu
che confermato: su quattro scenari, cinque regimi di mercato e centododici
configurazioni, **nessun criterio batte il portafoglio equipesato con un
margine distinguibile da zero**. Quello che invece cambia, in modo
sistematico e in ogni regime, e' il rischio che ciascun criterio si porta
dietro e quanto e' affidabile la volatilita' che dichiara di se' stesso.

Per questo l'applicazione non chiede all'utente di dichiarare un profilo di
rischio per poi restituirgli un modello in cambio. Un profilo e' solo
un'etichetta, e far corrispondere un'etichetta a un criterio nasconde
l'unica cosa che conta davvero: cosa ciascun criterio offre dimostrabilmente
e cosa no. E' l'utente che scelglie il criterio; questo modulo si limita a
dire, per l'universo che ha davanti, cosa significa scegliere quel criterio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from portfolio_optimizer.use_cases.diagnose_universe import UniverseDiagnosis


# ---------------------------------------------------------------------------
# L'evidenza, scritta una volta sola (cosi non si dimentica di aggiornarla)
# ---------------------------------------------------------------------------

EVIDENCE: dict[str, str] = {
    "1_su_N": (
        "Nessuna delle 56 configurazioni esaminate supera il portafoglio "
        "equipesato con differenza significativa al 5 per cento, in nessuno dei "
        "quattro scenari e in nessuno dei cinque regimi di mercato. Nei mercati "
        "in rialzo molte gli sono significativamente inferiori."
    ),
    "rischio_mantenuto": (
        "I criteri a minima varianza e a minimo CVaR realizzano la volatilita', "
        "il drawdown e il CVaR piu' bassi in ognuno dei cinque regimi esaminati. "
        "Mantengono cio' che promettono: meno rischio, pagato in rendimento."
    ),
    "regime": (
        "L'ordinamento fra criteri si inverte fra regimi: nei mercati in rialzo "
        "guidano equipesato e parita' di rischio, nelle due crisi esaminate "
        "guida il portafoglio tangente. Su un campione unico l'alternanza si annulla."
    ),
    "ewma_distorta": (
        "Lo stimatore a medie mobili esponenziali sottostima la volatilita' "
        "realizzata dal 6 al 25 per cento a seconda del criterio, in quattro "
        "regimi su cinque. L'effetto e' massimo per la minima varianza, che "
        "sfrutta la stima di covarianza piu' di ogni altro criterio. La "
        "distorsione non dipende dall'orizzonte di valutazione: verificata "
        "anche a ribilanciamento mensile, cioe' entro la memoria efficace "
        "dello stimatore, resta del 9,8 per cento con p inferiore a 0,0001."
    ),
    "ewma_orizzonte": (
        "Il suo svantaggio di rendimento invece dipendeva dall'orizzonte. A "
        "ribilanciamento trimestrale l'indice di Sharpe medio di EWMA e' 0,609 "
        "contro 0,623 di Ledoit-Wolf; a ribilanciamento mensile, entro la sua "
        "memoria, l'ordine si inverte e diventa 0,648 contro 0,620. Resta "
        "sconsigliato per la distorsione della volatilita' dichiarata, non per "
        "il rendimento, e al prezzo di un turnover piu' che doppio."
    ),
    "finestra_lunga": (
        "Gli stimatori su finestra a 252 giorni — campionaria e Ledoit-Wolf — "
        "sbagliano in entrambe le direzioni a seconda del regime: sovrastimano "
        "del 24 per cento nell'anno che segue una crisi, sottostimano del 25 "
        "nell'anno che segue un periodo quieto. In aggregato gli errori si "
        "compensano e la distorsione sembra assente."
    ),
    "turnover": (
        "Il turnover distingue gli stimatori piu' del rendimento: a parita' di "
        "volatilita' realizzata, Ledoit-Wolf muove il portafoglio circa un "
        "quinto rispetto a EWMA."
    ),
    "var_copertura": (
        "La copertura del VaR storico e' rifiutata su quasi tutte le "
        "configurazioni: il tasso di violazione osservato e' fra due e tre "
        "volte quello nominale al 99 per cento."
    ),
}


@dataclass(frozen=True, slots=True)
class ModelAssessment:
    """Cosa un criterio offre in modo dimostrabile su questo universo."""

    criterion: str
    label: str
    delivers: str
    does_not_deliver: str
    when_preferable: str
    recommended: bool = False
    suggested_max_weight: float = 0.40
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "label": self.label,
            "delivers": self.delivers,
            "does_not_deliver": self.does_not_deliver,
            "when_preferable": self.when_preferable,
            "recommended": self.recommended,
            "suggested_max_weight": self.suggested_max_weight,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ModelGuidance:
    """La raccomandazione e la valutazione di tutte le alternative."""

    recommended: str
    headline: str
    cov_method: str
    returns_method: str
    assessments: list[ModelAssessment] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def for_criterion(self, criterion: str) -> ModelAssessment | None:
        for assessment in self.assessments:
            if assessment.criterion == criterion:
                return assessment
        return None

    def as_dict(self) -> dict:
        return {
            "recommended": self.recommended,
            "headline": self.headline,
            "cov_method": self.cov_method,
            "returns_method": self.returns_method,
            "assessments": [a.as_dict() for a in self.assessments],
            "caveats": list(self.caveats),
        }


# Tetto di concentrazione suggerito per ogni criterio. I criteri che per
# costruzione tendono a concentrarsi prendono il tetto piu' stretto: e'
# l'unico vincolo che sta fra l'ottimizzatore e un portafoglio fatto di due
# soli titoli.
SUGGESTED_MAX_WEIGHT: dict[str, float] = {
    "equal_weight": 1.00,
    "minimum_variance": 0.30,
    "minimum_cvar": 0.30,
    "risk_parity": 0.40,
    "maximum_sharpe": 0.50,
    "black_litterman": 0.50,
}

DEFAULT_MODEL = "equal_weight"


def covariance_for(diagnosis: UniverseDiagnosis) -> tuple[str, list[str]]:
    """Scieglie lo stimatore di covarianza, e dice perche'.

    EWMA non viene mai proposto: e' l'unico stimatore la cui volatilita'
    dichiarata risulta distorta in quasi tutti i regimi, e l'applicazione
    quella volatilita' la mostra all'utente.
    """
    notes = [EVIDENCE["ewma_distorta"], EVIDENCE["ewma_orizzonte"], EVIDENCE["turnover"]]
    if diagnosis.estimation_is_stretched:
        notes.append(
            "Il campione e' corto rispetto al numero di parametri da stimare: "
            "la covarianza campionaria e' mal condizionata e la contrazione non "
            "e' facoltativa."
        )
    return "ledoit_wolf", notes


def assess_models(diagnosis: UniverseDiagnosis) -> ModelGuidance:
    """Descrive cosa offre ogni criterio su questo specifico universo."""
    cov_method, cov_notes = covariance_for(diagnosis)
    fat_tails = diagnosis.has_fat_tails

    if fat_tails:
        tail_note = (
            f"La curtosi in eccesso mediana e' {diagnosis.median_excess_kurtosis:.1f}: "
            "le code sono marcatamente piu' spesse di quelle di una normale, "
            "quindi una misura di coda descrive il rischio meglio della varianza."
        )
    else:
        tail_note = (
            f"La curtosi in eccesso mediana e' {diagnosis.median_excess_kurtosis:.1f}: "
            "le code non giustificano il costo di un criterio costruito su di esse."
        )

    risk_criterion = "minimum_cvar" if fat_tails else "minimum_variance"

    assessments = [
        ModelAssessment(
            criterion="equal_weight",
            label="Equipesato (1/N)",
            delivers=(
                "Nessun vantaggio dimostrabile da battere e nessuna stima da "
                "sbagliare. Turnover praticamente nullo."
            ),
            does_not_deliver=(
                "Nessun controllo del rischio: la volatilita' e' quella che "
                "l'universo produce, senza alcun contenimento."
            ),
            when_preferable=(
                "Sempre, salvo che si voglia esplicitamente contenere il rischio "
                "accettando un rendimento inferiore."
            ),
            recommended=True,
            suggested_max_weight=SUGGESTED_MAX_WEIGHT["equal_weight"],
            notes=[EVIDENCE["1_su_N"]],
        ),
        ModelAssessment(
            criterion=risk_criterion,
            label="Minimo CVaR" if fat_tails else "Minima Varianza",
            delivers=(
                "La perdita di coda piu' bassa in ognuno dei cinque regimi "
                "esaminati, e il drawdown minore."
                if fat_tails
                else "La volatilita' realizzata piu' bassa in ognuno dei cinque "
                "regimi esaminati, e il drawdown minore."
            ),
            does_not_deliver=(
                "Un rendimento superiore. Nei dati esaminati rende meno "
                "dell'equipesato in quattro regimi su cinque."
            ),
            when_preferable=(
                "Quando l'obiettivo e' contenere la perdita e si accetta di "
                "pagarlo in rendimento."
            ),
            suggested_max_weight=SUGGESTED_MAX_WEIGHT[risk_criterion],
            notes=[EVIDENCE["rischio_mantenuto"], tail_note] + cov_notes,
        ),
        ModelAssessment(
            criterion="risk_parity",
            label="Parita' di rischio (ERC)",
            delivers=(
                "Contributi al rischio equalizzati e turnover contenuto. "
                "Seconda in classifica nei mercati in rialzo."
            ),
            does_not_deliver=(
                "Un contenimento del rischio paragonabile ai criteri costruiti "
                "su di esso, ne' un vantaggio di rendimento."
            ),
            when_preferable=(
                "Quando si vuole diversificare il rischio anziche' il capitale, "
                "senza stimare rendimenti attesi."
            ),
            suggested_max_weight=SUGGESTED_MAX_WEIGHT["risk_parity"],
            notes=cov_notes,
        ),
        ModelAssessment(
            criterion="maximum_sharpe",
            label="Portafoglio tangente (Max Sharpe)",
            delivers=(
                "Il primo posto nelle due crisi esaminate, dove ha perso "
                "sensibilmente meno degli altri criteri."
            ),
            does_not_deliver=(
                "Un risultato competitivo nei mercati in rialzo, dove e' "
                "risultato ultimo fra i cinque criteri, ne' un turnover "
                "sostenibile: e' il piu' alto del gruppo."
            ),
            when_preferable=(
                "Quando si accetta la concentrazione e la movimentazione in "
                "cambio di un comportamento migliore nelle fasi di stress, che "
                "non sono pero' prevedibili in anticipo."
            ),
            suggested_max_weight=SUGGESTED_MAX_WEIGHT["maximum_sharpe"],
            notes=[EVIDENCE["regime"]] + cov_notes,
        ),
        ModelAssessment(
            criterion="black_litterman",
            label="Black-Litterman",
            delivers=(
                "Allocazioni stabili e la possibilita' di incorporare opinioni "
                "esplicite dell'investitore."
            ),
            does_not_deliver=(
                "Un risultato verificato su questi dati: in assenza di opinioni "
                "restituisce il portafoglio di equilibrio, e le opinioni non "
                "sono state oggetto del confronto empirico."
            ),
            when_preferable=(
                "Quando si vogliono esprimere view soggettive in forma relativa "
                "senza che l'ottimizzatore le amplifichi."
            ),
            suggested_max_weight=SUGGESTED_MAX_WEIGHT["black_litterman"],
            notes=cov_notes,
        ),
    ]

    caveats = [EVIDENCE["var_copertura"], EVIDENCE["finestra_lunga"]]
    if diagnosis.volatility_regime == "elevato":
        caveats.append(
            "La volatilita' recente e' sensibilmente superiore alla sua media "
            "storica. Nei regimi di crisi esaminati l'ordinamento fra criteri "
            "si inverte rispetto ai mercati in rialzo: la raccomandazione e' "
            "meno affidabile del solito."
        )
    if diagnosis.is_highly_correlated:
        caveats.append(
            f"La correlazione media fra i titoli e' {diagnosis.mean_correlation:.2f}: "
            "il beneficio della diversificazione e' limitato e la scelta del "
            "criterio incide poco sul risultato."
        )

    return ModelGuidance(
        recommended=DEFAULT_MODEL,
        headline=(
            "Equipesato: su questi dati nessun criterio lo batte in modo "
            "dimostrabile. Gli altri vanno scelti per il rischio che contengono, "
            "non per il rendimento che promettono."
        ),
        cov_method=cov_method,
        returns_method="bayes_stein",
        assessments=assessments,
        caveats=caveats,
    )


def recommend(log_returns) -> tuple[UniverseDiagnosis, ModelGuidance]:
    """Diagnostica l'universo e valuta tutti i criteri in un'unica chiamata."""
    from portfolio_optimizer.use_cases.diagnose_universe import diagnose_universe

    diagnosis = diagnose_universe(log_returns)
    return diagnosis, assess_models(diagnosis)
