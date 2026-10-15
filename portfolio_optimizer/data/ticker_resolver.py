from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

_EXCHANGE_SUFFIXES: list[tuple[str, str]] = [
    (".AS", "Euronext Amsterdam"),
    (".DE", "XETRA Frankfurt"),
    (".MI", "Borsa Italiana"),
    (".L", "London Stock Exchange"),
    (".SW", "SIX Swiss Exchange"),
    (".PA", "Euronext Paris"),
    (".TO", "Toronto Stock Exchange"),
    (".T", "Tokyo Stock Exchange"),
]

_PREFERRED_EXCHANGES = {
    "NYSE", "NASDAQ", "NGM", "PCX", "XETRA", "MIL", "LSE", "AMS", "SWX", "PAR"
}
_OTC_EXCHANGES = {"PNK", "OTC", "GREY"}
_TICKER_RE = re.compile(r"^[A-Z0-9.\-\^]{1,12}$")
_PROBE_TIMEOUT = 3.0
_CACHE_TTL = 600  # 10 minuti

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TickerCandidate:
    symbol: str
    exchange: str
    name: str


@dataclass
class AmbiguousTicker:
    input: str
    candidates: list[TickerCandidate]


@dataclass
class ResolveResult:
    resolved: list[str] = field(default_factory=list)
    ambiguous: list[AmbiguousTicker] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache in memoria
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, Optional[TickerCandidate]]] = {}
_cache_lock = threading.Lock()


def _cache_lookup(symbol: str) -> tuple[bool, Optional[TickerCandidate]]:
    """Torna (is_cached, result). Se result e' None vuol dire che e' stato provato e non trovato."""
    with _cache_lock:
        entry = _cache.get(symbol)
    if entry is None:
        return False, None
    ts, val = entry
    if time.time() - ts >= _CACHE_TTL:
        return False, None
    return True, val


def _cache_store(symbol: str, result: Optional[TickerCandidate]) -> None:
    with _cache_lock:
        _cache[symbol] = (time.time(), result)


def _verify_suffixed(symbol: str) -> Optional[TickerCandidate]:
    """Controlla che un ticker gia' con suffisso esista davvero su yfinance."""
    found, cached = _cache_lookup(symbol)
    if found:
        return cached
    try:
        fast = yf.Ticker(symbol).fast_info
        currency = getattr(fast, "currency", None)
        if not currency or str(currency).upper() in ("NONE", ""):
            _cache_store(symbol, None)
            return None
        exchange = str(getattr(fast, "exchange", "") or "")
        candidate = TickerCandidate(symbol=symbol, exchange=exchange, name=symbol)
        _cache_store(symbol, candidate)
        return candidate
    except Exception:
        _cache_store(symbol, None)
        return None


def _probe_suffixes(bare: str) -> list[TickerCandidate]:
    """Sonda il ticker nudo + tutti i suffissi di borsa conosciuti, in parallelo."""
    _suffix_map = {bare + s: label for s, label in _EXCHANGE_SUFFIXES}
    _suffix_map[bare] = bare  # sonda anche il ticker nudo, senza suffisso
    candidates_to_probe = list(_suffix_map.keys())

    results: list[TickerCandidate] = []

    def probe(symbol: str) -> Optional[TickerCandidate]:
        found, cached = _cache_lookup(symbol)
        if found:
            return cached
        try:
            fast = yf.Ticker(symbol).fast_info
            currency = getattr(fast, "currency", None)
            if not currency or str(currency).upper() in ("NONE", ""):
                _cache_store(symbol, None)
                return None
            exchange_label = _suffix_map.get(symbol, str(getattr(fast, "exchange", "") or ""))
            candidate = TickerCandidate(symbol=symbol, exchange=exchange_label, name=symbol)
            _cache_store(symbol, candidate)
            return candidate
        except Exception:
            _cache_store(symbol, None)
            return None

    pool = ThreadPoolExecutor(max_workers=len(candidates_to_probe))
    futures = {pool.submit(probe, s): s for s in candidates_to_probe}
    try:
        for future in as_completed(futures, timeout=_PROBE_TIMEOUT * 2):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                pass
    except (TimeoutError, Exception) as exc:
        logger.warning("Suffix probing timed out or failed for %s: %s", bare, exc)
    finally:
        for f in futures:
            f.cancel()
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)

    # Se il ticker nudo si risolve da solo, non c'e' ambiguita' — si ignorano le varianti con suffisso.
    bare_match = next((r for r in results if r.symbol == bare), None)
    if bare_match is not None:
        return [bare_match]

    return results


def _search_company(query: str) -> list[TickerCandidate]:
    """Cerca un ticker per nome societa' usando yf.Search."""
    try:
        search = yf.Search(query, max_results=8)
        quotes = getattr(search, "quotes", []) or []
    except Exception:
        return []

    candidates: list[TickerCandidate] = []
    for q in quotes:
        qtype = str(q.get("quoteType", "")).upper()
        if qtype not in ("EQUITY", "ETF", "MUTUALFUND"):
            continue
        exchange = str(q.get("exchange", "") or "")
        if exchange in _OTC_EXCHANGES:
            continue
        symbol = q.get("symbol", "")
        if not symbol:
            continue
        name = q.get("shortname", "") or q.get("longname", "") or symbol
        candidates.append(TickerCandidate(symbol=symbol, exchange=exchange, name=name))

    # Ordina: le borse preferite vanno prima delle altre
    candidates.sort(key=lambda c: 0 if c.exchange in _PREFERRED_EXCHANGES else 1)
    return candidates


def resolve(tokens: list[str]) -> ResolveResult:
    """
    Risolve gli input grezzi dell'utente in simboli yfinance validi.

    Ogni token viene classificato cosi':
    - gia' con suffisso (contiene '.', soddisfa _TICKER_RE) → _verify_suffixed
    - ticker nudo (niente '.', soddisfa _TICKER_RE) → _probe_suffixes
    - nome societa' (non soddisfa _TICKER_RE) → _search_company
    """
    result = ResolveResult()

    for token in tokens:
        token = token.strip().upper()
        if not token:
            continue

        if _TICKER_RE.match(token):
            if "." in token:
                candidate = _verify_suffixed(token)
                if candidate:
                    result.resolved.append(token)
                else:
                    result.not_found.append(token)
            else:
                try:
                    candidates = _probe_suffixes(token)
                except Exception as exc:
                    logger.warning("Error probing suffixes for %s: %s", token, exc)
                    candidates = []
                if candidates:
                    if len(candidates) == 1:
                        symbol = candidates[0].symbol
                        result.resolved.append(symbol)
                        if symbol != token:
                            result.warnings.append(f"{token} → {symbol}")
                    else:
                        result.ambiguous.append(
                            AmbiguousTicker(input=token, candidates=candidates)
                        )
                    continue

                # Un nome societario in maiuscolo -- APPLE, NESTLE, TOTAL --
                # soddisfa _TICKER_RE come qualunque simbolo, e finiva qui
                # dichiarato inesistente senza che la ricerca per nome fosse mai
                # tentata. La regex distingue la FORMA di un simbolo, non il
                # fatto che quel simbolo esista: solo il sondaggio dei suffissi
                # lo stabilisce, e quando fallisce l'ipotesi "e' un ticker" e'
                # esaurita, non confermata.
                #
                # Il risultato pero' NON viene accettato in silenzio, nemmeno
                # quando e' unico. Il token aveva la forma di un simbolo, quindi
                # puo' benissimo essere un simbolo digitato male, e sostituirgli
                # d'ufficio la societa' che piu' gli somiglia significherebbe
                # ottimizzare un portafoglio diverso da quello chiesto senza
                # dirlo. Va alla schermata di conferma.
                candidates = _search_company(token)
                if not candidates:
                    result.not_found.append(token)
                else:
                    result.ambiguous.append(
                        AmbiguousTicker(input=token, candidates=candidates)
                    )
        else:
            candidates = _search_company(token)
            if not candidates:
                result.not_found.append(token)
            elif len(candidates) == 1:
                symbol = candidates[0].symbol
                result.resolved.append(symbol)
                result.warnings.append(f"{token} → {symbol}")
            else:
                result.ambiguous.append(
                    AmbiguousTicker(input=token, candidates=candidates)
                )

    # Rimuove i duplicati dai risolti, mantenendo l'ordine originale
    seen: set[str] = set()
    deduped: list[str] = []
    for s in result.resolved:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    result.resolved = deduped

    return result
