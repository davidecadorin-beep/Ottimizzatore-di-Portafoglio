from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# (yf_ticker, invert)
# invert=True: la coppia e' quotata come base/local (es. USDJPY=X), quindi 1/rate da' local/base
_FX_TO_USD: dict[str, tuple[str, bool]] = {
    "EUR": ("EURUSD=X", False),
    "GBP": ("GBPUSD=X", False),
    "AUD": ("AUDUSD=X", False),
    "NZD": ("NZDUSD=X", False),
    "JPY": ("USDJPY=X", True),
    "CAD": ("USDCAD=X", True),
    "CHF": ("USDCHF=X", True),
    "SEK": ("USDSEK=X", True),
    "NOK": ("USDNOK=X", True),
    "DKK": ("USDDKK=X", True),
    "HKD": ("USDHKD=X", True),
    "SGD": ("USDSGD=X", True),
    "CNY": ("USDCNY=X", True),
    "INR": ("USDINR=X", True),
    "BRL": ("USDBRL=X", True),
    "MXN": ("USDMXN=X", True),
    "ZAR": ("USDZAR=X", True),
}

# Tutte le coppie incrociate EUR: EURXXX=X da' il prezzo di 1 EUR in XXX → si inverte per avere local/EUR
_FX_TO_EUR: dict[str, tuple[str, bool]] = {
    "USD": ("EURUSD=X", True),
    "GBP": ("EURGBP=X", True),
    "AUD": ("EURAUD=X", True),
    "NZD": ("EURNZD=X", True),
    "JPY": ("EURJPY=X", True),
    "CAD": ("EURCAD=X", True),
    "CHF": ("EURCHF=X", True),
    "SEK": ("EURSEK=X", True),
    "NOK": ("EURNOK=X", True),
    "DKK": ("EURDKK=X", True),
    "HKD": ("EURHKD=X", True),
    "SGD": ("EURSGD=X", True),
    "CNY": ("EURCNY=X", True),
    "INR": ("EURINR=X", True),
    "BRL": ("EURBRL=X", True),
    "MXN": ("EURMXN=X", True),
    "ZAR": ("EURZAR=X", True),
}

_PAIRS_MAP: dict[str, dict[str, tuple[str, bool]]] = {
    "USD": _FX_TO_USD,
    "EUR": _FX_TO_EUR,
}


def detect_reference_currency(ticker_currencies: dict[str, str]) -> str:
    """Torna 'USD' o 'EUR' in base alla maggioranza dei ticker; pareggio o nessun match → 'USD'."""
    counts = Counter(c for c in ticker_currencies.values() if c in ("USD", "EUR"))
    return "EUR" if counts.get("EUR", 0) > counts.get("USD", 0) else "USD"


def download_fx_rates(
    currencies: set[str],
    base: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Scarica i tassi di cambio giornalieri per ogni valuta diversa dalla base.

    Torna un DataFrame indicizzato per giorno di calendario, colonne =
    codici valuta. I valori sono il prezzo di 1 unita' di quella valuta in
    `base`. I buchi di weekend/festivita' vengono riportati avanti e poi
    indietro.
    """
    pairs_map = _PAIRS_MAP.get(base, _FX_TO_USD)
    non_base = {c for c in currencies if c != base}
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    result = pd.DataFrame(index=full_range, dtype=float)

    for currency in non_base:
        if currency not in pairs_map:
            logger.warning(
                "No FX pair configured for %s/%s — skipping conversion.", currency, base
            )
            continue
        yf_ticker, invert = pairs_map[currency]
        try:
            raw = yf.download(
                yf_ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                raise ValueError(f"yfinance returned no data for {yf_ticker}")

            # Gestisce sia colonne flat che MultiIndex
            if isinstance(raw.columns, pd.MultiIndex):
                series = raw["Close"].iloc[:, 0]
            else:
                series = raw["Close"]
            series = series.squeeze()

            if invert:
                series = 1.0 / series

            # Normalizza a date senza fuso orario
            if series.index.tz is not None:
                series.index = series.index.tz_convert("UTC").tz_localize(None)
            series.index = series.index.normalize()

            result[currency] = series.reindex(full_range).ffill().bfill()

        except Exception as exc:
            logger.warning(
                "FX download failed for %s/%s (%s) — skipping conversion.",
                currency,
                base,
                exc,
            )

    return result


def convert_log_returns(
    log_returns: pd.DataFrame,
    fx_rates: pd.DataFrame,
    ticker_currencies: dict[str, str],
    base: str,
) -> pd.DataFrame:
    """Convertire i rendimenti log nella valuta di riferimento, aggiungendo il log-rendimento del cambio.

    Formula (esatta in tempo continuo):
        log_return_base[t] = log_return_local[t] + log(FX_t / FX_{t-1})

    I ticker che sono gia' nella valuta base o senza dati FX passano invariati.
    """
    converted = log_returns.copy()
    for ticker in log_returns.columns:
        currency = ticker_currencies.get(ticker, base)
        if currency == base or currency not in fx_rates.columns:
            continue
        fx = fx_rates[currency].reindex(log_returns.index).ffill().bfill()
        fx_log_return = np.log(fx / fx.shift(1)).fillna(0.0)
        converted[ticker] = log_returns[ticker] + fx_log_return
    return converted


@dataclass(frozen=True, slots=True)
class ConversionIssue:
    """Un ticker la cui conversione valutaria non e' affidabile."""

    ticker: str
    currency: str
    volatility_local: float
    volatility_converted: float
    retention: float
    correlation_with_fx: float
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.ticker} ({self.currency}): volatilita' locale "
            f"{self.volatility_local:.2%}, convertita {self.volatility_converted:.2%} "
            f"({self.retention:.0%} della locale), correlazione fra rendimento del "
            f"titolo e cambio applicato {self.correlation_with_fx:+.3f} — {self.detail}"
        )


# Normalmente un'esposizione valutaria AGGIUNGE varianza a un titolo. Puo'
# toglierne un po' quando il titolo e la valuta vanno in direzioni opposte —
# un esportatore le cui azioni salgono mentre la valuta di casa si svaluta —
# ma l'effetto e' di secondo ordine: con una correlazione di -0.3 fra un
# titolo al 25% e una coppia valutaria all'8%, la volatilita' convertita e'
# ancora il 95% di quella locale. Una conversione che toglie un terzo o piu'
# della volatilita' di un titolo implica quindi una correlazione che nessuna
# coppia valutaria reale ha con un singolo titolo, e vuol dire che la serie
# usata come tasso di cambio non e' un tasso di cambio.
MIN_VOLATILITY_RETENTION = 0.60
MAX_NEGATIVE_CORRELATION = -0.80
MIN_OBSERVATIONS = 60

# Una serie contaminata puo' anche AGGIUNGERE varianza invece di toglierla, e
# anche questo guasto e' silenzioso allo stesso modo: una vecchia cache di
# questo progetto conserva un'esecuzione dove la serie applicata a tre titoli
# italiani in un portafoglio in dollari aveva una volatilita' del 24 per
# cento, mentre l'euro contro il dollaro sta fra il sette e l'otto. I
# rendimenti non erano distrutti, erano gonfiati, e le misure di rischio
# costruite sopra erano sbagliate nella direzione opposta. Nessuna coppia
# nella tabella di mappatura — le principali piu' qualche valuta emergente —
# sostiene il trenta per cento annualizzato per anni, quindi un tasso sopra
# quella soglia non e' un tasso.
MAX_FX_VOLATILITY = 0.30


def verify_conversion(
    local_log_returns: pd.DataFrame,
    converted: pd.DataFrame,
    ticker_currencies: dict[str, str],
    base: str,
    periods: int = 252,
    fx_columns: object = None,
) -> list[ConversionIssue]:
    """Controlla che nessuna conversione abbia distrutto il rendimento a cui e' stata applicata.

    Il test non dipende dal sapere perche' una conversione e' andata male.
    Confronta quello che e' entrato con quello che e' uscito: la differenza
    fra il rendimento convertito e quello locale e' il rendimento del cambio
    effettivamente applicato, e se quella serie rispecchia il titolo invece
    di una valuta il controllo scatta. Cosi' si prende un download
    contaminato, un calendario disallineato, una coppia sbagliata nella
    tabella di mappatura e qualsiasi altra causa non ancora pensata.
    """
    issues: list[ConversionIssue] = []
    scale = np.sqrt(periods)
    available = set(fx_columns or ())

    for ticker in local_log_returns.columns:
        currency = ticker_currencies.get(ticker, base)
        if currency == base or ticker not in converted.columns:
            continue

        # Il caso che nessun confronto fra ingresso e uscita puo' rilevare:
        # la conversione non e' stata TENTATA. Senza il tasso, il titolo passa
        # invariato, la differenza applicata e' identicamente nulla e ogni
        # soglia sotto e' superata -- ritenzione esattamente 1, cambio
        # applicato a volatilita' zero, correlazione zero. La serie sembra
        # perfetta proprio perche' non e' successo niente, e un titolo in JPY
        # finisce sommato a uno in USD al cambio 1:1. Va quindi verificato
        # prima, e sulla disponibilita' del tasso, non sul risultato.
        if fx_columns is not None and currency not in available:
            local_only = local_log_returns[ticker].dropna()
            vol = (
                float(local_only.std(ddof=1) * scale) if len(local_only) > 1 else float("nan")
            )
            issues.append(
                ConversionIssue(
                    ticker=str(ticker),
                    currency=currency,
                    volatility_local=vol,
                    volatility_converted=vol,
                    retention=1.0,
                    correlation_with_fx=0.0,
                    detail=(
                        f"nessun tasso {currency}/{base} disponibile: il titolo e' "
                        f"rimasto in {currency} e verrebbe sommato agli altri come "
                        "se il cambio fosse uno a uno"
                    ),
                )
            )
            continue

        pair = pd.concat(
            [local_log_returns[ticker], converted[ticker]], axis=1
        ).dropna()
        if len(pair) < MIN_OBSERVATIONS:
            continue

        local = pair.iloc[:, 0]
        after = pair.iloc[:, 1]
        applied = after - local

        vol_local = float(local.std(ddof=1) * scale)
        vol_after = float(after.std(ddof=1) * scale)
        if vol_local <= 0:
            continue
        retention = vol_after / vol_local

        if float(applied.std(ddof=1)) > 0:
            correlation = float(np.corrcoef(local, applied)[0, 1])
        else:
            correlation = 0.0

        vol_applied = float(applied.std(ddof=1) * scale)

        if retention < MIN_VOLATILITY_RETENTION:
            detail = (
                "la conversione ha rimosso oltre il quaranta per cento della "
                "volatilita' del titolo, effetto che nessun cambio reale produce"
            )
        elif vol_applied > MAX_FX_VOLATILITY:
            detail = (
                f"il cambio applicato ha una volatilita' del {vol_applied:.1%} "
                "annuo, incompatibile con qualunque coppia valutaria presente "
                "nella tavola di conversione"
            )
        elif vol_applied > vol_local:
            detail = (
                f"il cambio applicato ({vol_applied:.1%}) e' piu' volatile del "
                f"titolo che converte ({vol_local:.1%}): non e' un cambio"
            )
        elif correlation < MAX_NEGATIVE_CORRELATION:
            detail = (
                "il cambio applicato e' quasi perfettamente opposto al rendimento "
                "del titolo: la serie usata come cambio non e' un cambio"
            )
        else:
            continue

        issues.append(
            ConversionIssue(
                ticker=str(ticker),
                currency=currency,
                volatility_local=vol_local,
                volatility_converted=vol_after,
                retention=retention,
                correlation_with_fx=correlation,
                detail=detail,
            )
        )
    return issues


def convert_log_returns_checked(
    log_returns: pd.DataFrame,
    fx_rates: pd.DataFrame,
    ticker_currencies: dict[str, str],
    base: str,
    on_failure: str = "raise",
) -> tuple[pd.DataFrame, list[ConversionIssue]]:
    """Converte, poi verifica, e agisce in base alla verifica.

    ``on_failure`` decide cosa fare di un ticker la cui conversione viene
    rigettata: ``raise`` si ferma, ``drop`` rimuove il ticker, ``keep``
    procede e registra solo il problema. Il default si ferma, perche' una
    metrica di portafoglio calcolata su una serie di rendimenti distrutta e'
    peggio di nessuna metrica: sembra corretta, e l'utente non ha modo di
    saperlo.
    """
    if on_failure not in {"raise", "drop", "keep"}:
        raise ValueError("on_failure must be 'raise', 'drop' or 'keep'.")

    converted = convert_log_returns(log_returns, fx_rates, ticker_currencies, base)
    issues = verify_conversion(
        log_returns,
        converted,
        ticker_currencies,
        base,
        fx_columns=list(fx_rates.columns),
    )

    for issue in issues:
        logger.error("Conversione valutaria non affidabile — %s", issue)

    if issues and on_failure == "raise":
        # Il motivo va riportato per titolo: una conversione mai tentata e una
        # conversione che ha distrutto la serie sono guasti diversi, e dire
        # all'utente il secondo quando e' avvenuto il primo lo manda a
        # cercare nel posto sbagliato.
        detail = "; ".join(f"{i.ticker} ({i.currency}): {i.detail}" for i in issues)
        raise ValueError(
            f"Conversione valutaria non affidabile — {detail}. "
            "Le metriche di rischio calcolate su queste serie non avrebbero "
            "significato. Verificare con audit-finanziario/diagnostica_fx.py."
        )
    if issues and on_failure == "drop":
        converted = converted.drop(columns=[i.ticker for i in issues])

    return converted, issues


def latest_fx_rate(currency: str, base: str) -> float | None:
    """Prezzo di una unita' di ``currency`` in ``base``, osservazione piu' recente.

    Si usa quando serve una conversione puntuale e non una serie intera —
    le capitalizzazioni di mercato, che Yahoo riporta nella valuta di
    negoziazione propria di ogni strumento e che quindi non sono additive
    su un universo multi-valuta.

    Torna ``None`` invece di 1.0 quando la coppia non e' conosciuta o il
    download fallisce. Tornare uno vorrebbe dire "non serve conversione",
    che e' un'affermazione sulle valute; l'assenza di un tasso e'
    un'affermazione sui dati, e chi chiama deve poterle distinguere.
    """
    currency = (currency or "").upper()
    base = (base or "USD").upper()
    if not currency or currency == base:
        return 1.0

    pairs_map = _PAIRS_MAP.get(base, _FX_TO_USD)
    if currency not in pairs_map:
        logger.warning("Nessuna coppia configurata per %s/%s.", currency, base)
        return None

    symbol, invert = pairs_map[currency]
    try:
        raw = yf.download(
            symbol, period="5d", progress=False, threads=False, auto_adjust=False
        )
        if raw is None or raw.empty:
            return None
        column = "Adj Close" if "Adj Close" in raw.columns else "Close"
        series = raw[column]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.dropna()
        if series.empty:
            return None
        rate = float(series.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cambio %s/%s non ottenuto: %s", currency, base, exc)
        return None

    if rate <= 0:
        return None
    return 1.0 / rate if invert else rate
