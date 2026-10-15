from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from portfolio_optimizer.data.data_loader import MarketDataLoader, analyze_history_impact
from portfolio_optimizer.domain.market_data import MarketDataset
from portfolio_optimizer.fx.converter import (
    convert_log_returns_checked,
    detect_reference_currency,
    download_fx_rates,
)

logger = logging.getLogger(__name__)

_CACHE_MAX_AGE_SECONDS = 24 * 3600
CACHE_SCHEMA_VERSION = "v2"
FX_CONVERSION_VERSION = "v1"
PRICE_TYPE = "adj_close"


def _cache_file_fresh(path) -> bool:
    try:
        return path.exists() and (
            time.time() - path.stat().st_mtime < _CACHE_MAX_AGE_SECONDS
        )
    except OSError:
        return False


def _resolve_currencies(
    ticker_list: list[str],
    ticker_currencies: dict[str, str] | None,
    reference_currency: str,
    loader: MarketDataLoader,
    on_conversion_failure: str = "raise",
) -> tuple[dict[str, str], str, dict[str, dict]]:
    currencies, metadata = loader.resolve_ticker_currencies(
        ticker_list, declared_currencies=ticker_currencies
    )
    unresolved = [t for t in ticker_list if not currencies.get(t)]
    if unresolved:
        if on_conversion_failure == "raise":
            raise ValueError(
                f"Impossibile determinare la valuta per i seguenti titoli: {', '.join(unresolved)}. "
                "Specificare esplicitamente 'ticker_currencies' per questi asset."
            )
        for t in unresolved:
            currencies[t] = "USD"
            metadata[t] = {
                "currency": "USD",
                "source": "assumed_usd",
                "is_confirmed": False,
            }
            logger.warning(
                "Valuta non disponibile per %s; applicato fallback ipotizzato a USD (non confermato).",
                t,
            )

    resolved = {t: str(currencies[t]).upper() for t in ticker_list}
    if reference_currency == "auto":
        reference_currency = detect_reference_currency(resolved)
    return resolved, reference_currency.upper(), metadata


def _fetch_fresh_dataset(
    loader: MarketDataLoader,
    ticker_list: list[str],
    ticker_currencies: dict[str, str],
    currency_metadata: dict[str, dict],
    reference_currency: str,
    start_date: str,
    end_date: str,
    horizon_years: float,
    use_cache: bool,
    prices_file: str,
    returns_file: str,
    rf_file: str,
    manifest_file: str,
    canonical_manifest: dict,
    on_conversion_failure: str = "raise",
) -> MarketDataset:
    currencies_needed = set(ticker_currencies.values())

    # I tre download sono seriali di proposito, non per pigrizia. yfinance
    # tiene uno stato globale a livello di processo (una sessione condivisa
    # con cookie e crumb, piu' una cache della timezone su disco) e niente di
    # tutto questo e' thread-safe. Se lanci prezzi, cambi e curva risk-free in
    # parallelo le richieste si sporcano a vicenda: il sintomo era una
    # richiesta per sette titoli che tornava con cinque senza nessun errore, e
    # una serie di cambio che tornava con dentro i dati di un altro strumento,
    # il che falsava completamente i rendimenti a cui veniva applicata.
    # Entrambi i problemi erano silenziosi e capitavano a intermittenza, che e'
    # esattamente la faccia di una race condition. Il parallelismo faceva
    # risparmiare un paio di secondi; la correttezza vale di piu'.
    raw = loader.download_price_data(ticker_list, start_date, end_date)
    # Un portafoglio a cui manca un titolo senza dirlo e' peggio di un errore:
    # sembra corretto, e l'utente non ha modo di accorgersene. Lo stesso
    # ragionamento che rende l'eccezione il comportamento predefinito quando la
    # verifica della conversione valutaria fallisce.
    missing = tuple(getattr(loader, "last_missing_tickers", ()) or ())
    if missing:
        raise ValueError(
            "Nessun dato di prezzo per: "
            + ", ".join(missing)
            + ". I titoli sono stati richiesti una seconda volta separatamente "
            "senza esito. Verifica i simboli oppure riprova: puo' trattarsi di "
            "un'indisponibilita' temporanea della fonte."
        )
    fx_rates = download_fx_rates(
        currencies_needed, reference_currency, start_date, end_date
    )
    risk_free_annual = loader.get_risk_free_rate(
        start_date, end_date, reference_currency, horizon_years
    )

    prices = loader.extract_adjusted_close(raw)
    dividends, splits = loader.extract_corporate_actions(raw)
    local_log_returns = loader.compute_log_returns(prices)
    log_returns, conversion_issues = convert_log_returns_checked(
        local_log_returns,
        fx_rates,
        ticker_currencies,
        reference_currency,
        on_failure=on_conversion_failure,
    )
    if conversion_issues:
        logger.warning(
            "Conversione valutaria non affidabile per %s; i titoli sono stati esclusi.",
            ", ".join(i.ticker for i in conversion_issues),
        )

    unconverted = {c for c in currencies_needed if c != reference_currency} - set(
        fx_rates.columns
    )
    if unconverted:
        logger.warning(
            "FX rates unavailable for %s — returns for affected tickers remain in "
            "local currency. Portfolio metrics may be currency-mismatched.",
            sorted(unconverted),
        )

    risk_free_daily = loader.annual_to_daily_rate(risk_free_annual)
    history_impact = analyze_history_impact(prices)

    manifest_data = {
        **canonical_manifest,
        "saved_at": time.time(),
        "currency_metadata": currency_metadata,
        "history_impact": history_impact,
    }

    if use_cache:
        loader.save_dataframe_csv(prices, prices_file)
        loader.save_dataframe_csv(log_returns, returns_file)
        loader.save_dataframe_csv(risk_free_annual.to_frame("risk_free_rate"), rf_file)
        manifest_path = loader.cache_dir / manifest_file
        try:
            manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Impossibile salvare il manifest di provenienza.", exc_info=True)

    return MarketDataset(
        prices=prices,
        log_returns=log_returns,
        dividends=dividends,
        splits=splits,
        risk_free_rate_annual=risk_free_annual,
        risk_free_rate_daily=risk_free_daily,
        reference_currency=reference_currency,
        ticker_currencies=ticker_currencies,
        currency_metadata=currency_metadata,
        provenance_manifest=manifest_data,
        history_impact=history_impact,
    )


@dataclass(slots=True)
class LoadMarketDataUseCase:
    """Caso d'uso per scaricare i dati, convertire le valute e la cache CSV opzionale."""

    loader: MarketDataLoader

    def execute(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
        reference_currency: str = "auto",
        horizon_years: float = 1.0,
        ticker_currencies: dict[str, str] | None = None,
        use_cache: bool = True,
        cache_prefix: str = "market",
        on_conversion_failure: str = "raise",
    ) -> MarketDataset:
        ticker_list = sorted(set(str(t).upper() for t in tickers))
        ticker_currencies, reference_currency, currency_metadata = _resolve_currencies(
            ticker_list,
            ticker_currencies,
            reference_currency,
            self.loader,
            on_conversion_failure,
        )

        canonical_manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fx_version": FX_CONVERSION_VERSION,
            "price_type": PRICE_TYPE,
            "tickers": sorted(ticker_list),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "reference_currency": str(reference_currency).upper(),
            "horizon_years": round(float(horizon_years), 4),
            "ticker_currencies": {t: ticker_currencies[t] for t in sorted(ticker_currencies)},
            "on_conversion_failure": str(on_conversion_failure),
        }
        manifest_hash = hashlib.sha256(
            json.dumps(canonical_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        cache_key = (
            f"{cache_prefix}_{CACHE_SCHEMA_VERSION}_{reference_currency}"
            f"_{start_date}_{end_date}_{manifest_hash}"
        )
        prices_file = f"{cache_key}_prices.csv"
        returns_file = f"{cache_key}_returns.csv"
        rf_file = f"{cache_key}_risk_free.csv"
        manifest_file = f"{cache_key}_manifest.json"

        def _load_if_fresh(filename: str):
            if not use_cache:
                return None
            path = self.loader.cache_dir / filename
            return (
                self.loader.load_dataframe_csv(filename)
                if _cache_file_fresh(path)
                else None
            )

        prices = _load_if_fresh(prices_file)
        log_returns = _load_if_fresh(returns_file)
        risk_free = _load_if_fresh(rf_file)

        if prices is None or log_returns is None or risk_free is None:
            return _fetch_fresh_dataset(
                self.loader,
                ticker_list,
                ticker_currencies,
                currency_metadata,
                reference_currency,
                start_date,
                end_date,
                horizon_years,
                use_cache,
                prices_file,
                returns_file,
                rf_file,
                manifest_file,
                canonical_manifest,
                on_conversion_failure,
            )

        risk_free_annual = risk_free.iloc[:, 0].rename("risk_free_rate")
        risk_free_daily = self.loader.annual_to_daily_rate(risk_free_annual)

        manifest_data = {}
        manifest_path = self.loader.cache_dir / manifest_file
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        return MarketDataset(
            prices=prices,
            log_returns=log_returns,
            dividends=pd.DataFrame(
                index=prices.index, columns=prices.columns, dtype=float
            ).fillna(0.0),
            splits=pd.DataFrame(
                index=prices.index, columns=prices.columns, dtype=float
            ).fillna(0.0),
            risk_free_rate_annual=risk_free_annual,
            risk_free_rate_daily=risk_free_daily,
            reference_currency=reference_currency,
            ticker_currencies=ticker_currencies,
            currency_metadata=manifest_data.get("currency_metadata", currency_metadata),
            provenance_manifest=manifest_data,
            history_impact=manifest_data.get("history_impact", {}),
        )
