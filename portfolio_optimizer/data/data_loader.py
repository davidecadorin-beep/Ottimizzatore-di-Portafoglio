from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CURRENCY_CACHE_TTL = 30 * 86400  # 30 giorni — le valute cambiano di rado
_RF_CACHE_TTL = 86400  # 1 giorno — i tassi risk-free si aggiornano ogni giorno


# ---------------------------------------------------------------------------
# Scalette di scadenza per scegliere il tasso privo di rischio
# Ogni voce: (soglia_orizzonte_anni, chiave_sorgente)
# Si prende la prima soglia che e' >= horizon_years.
# ---------------------------------------------------------------------------
_USD_MATURITIES: list[tuple[float, str]] = [
    (0.5, "DGS3MO"),  # ≤ 6 mesi   → Treasury USA a 3 mesi
    (1.5, "DGS1"),  # ≤ 18 mesi  → Treasury USA a 1 anno
    (3.0, "DGS2"),  # ≤ 3 anni   → Treasury USA a 2 anni
    (7.0, "DGS5"),  # ≤ 7 anni   → Treasury USA a 5 anni
    (float("inf"), "DGS10"),  # > 7 anni   → Treasury USA a 10 anni
]

# Tassi spot sui titoli di stato AAA dell'area euro (modello Svensson, giornaliero)
# Il suffisso della chiave va dentro: B.U2.EUR.4F.G_N_A.SV_C_YM.<key>
_EUR_MATURITIES: list[tuple[float, str]] = [
    (0.5, "SR_3M"),  # ≤ 6 mesi   → Bund spot a 3 mesi
    (1.5, "SR_1Y"),  # ≤ 18 mesi  → Bund spot a 1 anno
    (3.0, "SR_2Y"),  # ≤ 3 anni   → Bund spot a 2 anni
    (7.0, "SR_5Y"),  # ≤ 7 anni   → Bund spot a 5 anni
    (float("inf"), "SR_10Y"),  # > 7 anni   → Bund spot a 10 anni
]

_CURRENCY_MATURITIES: dict[str, list[tuple[float, str]]] = {
    "USD": _USD_MATURITIES,
    "EUR": _EUR_MATURITIES,
}



def _naive_index(index: pd.Index) -> pd.Index:
    """Togliere il fuso orario e l'orario dall'indice dei prezzi.

    I prezzi sono giornalieri, quindi l'orario non porta nessuna informazione;
    il fuso orario porta solo la borsa da cui viene la serie, che e' diversa
    fra il download in blocco e il recupero per singolo titolo. Lasciarli
    dentro fa si' che due frame con le stesse date non si allineino.
    """
    if isinstance(index, pd.DatetimeIndex):
        if index.tz is not None:
            index = index.tz_localize(None)
        return index.normalize()
    return index


class MarketDataLoader:
    """Servizio di infrastruttura per scaricare dati di mercato e gestirne la cache locale."""

    def __init__(self, cache_dir: str | Path = "cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Simboli richiesti per i quali l'ultimo download non ha restituito
        # dati utilizzabili, dopo il tentativo separato di recupero.
        self.last_missing_tickers: tuple[str, ...] = ()
        # Titoli per i quali la capitalizzazione non e' stata ottenuta.
        self.last_market_cap_failures: tuple[str, ...] = ()
        self.last_market_cap_excluded: tuple[str, ...] = ()
        self.last_price_type_used: str = "Adj Close"
        self.last_unresolved_currencies: tuple[str, ...] = ()
        self.yf_cache_dir = self.cache_dir / "yfinance_tz_cache"
        self.yf_cache_dir.mkdir(parents=True, exist_ok=True)
        self._currency_lock = threading.Lock()
        self._currency_cache: dict[str, str] = self._load_currency_cache()
        self._configure_yfinance_cache()

    def _load_currency_cache(self) -> dict[str, str]:
        path = self.cache_dir / "ticker_currencies.json"
        try:
            if path.exists() and (
                time.time() - path.stat().st_mtime < _CURRENCY_CACHE_TTL
            ):
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_currency_cache(self) -> None:
        """Salva la cache delle valute in modo atomico, protetta da un lock.

        ``get_ticker_currencies`` risolve fino a otto ticker in parallelo e
        ogni thread che ne risolve uno chiama questo metodo, sullo stesso
        file. Una semplice ``write_text`` non e' ne' atomica ne' seriale:
        due thread possono troncare e riscrivere lo stesso path insieme, e
        ``json.dumps`` puo' pure ricevere un dizionario in cui un altro
        thread sta ancora inserendo roba. Il risultato e' un file troncato,
        e dato che ``_load_currency_cache`` ingoia qualsiasi eccezione, il
        danno si vede solo come una cache che silenziosamente non viene mai
        riusata.

        Scrivere su un file temporaneo nella stessa cartella e poi chiamare
        ``os.replace`` rende la sostituzione atomica sia su POSIX che su
        Windows; il lock serializza anche lo snapshot del dizionario.
        """
        path = self.cache_dir / "ticker_currencies.json"
        try:
            with self._currency_lock:
                payload = json.dumps(dict(self._currency_cache))
                handle = tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=str(self.cache_dir),
                    prefix=".ticker_currencies-",
                    suffix=".tmp",
                    delete=False,
                )
                try:
                    with handle as stream:
                        stream.write(payload)
                    os.replace(handle.name, path)
                except BaseException:
                    Path(handle.name).unlink(missing_ok=True)
                    raise
        except Exception:  # noqa: BLE001 - la cache e' un'ottimizzazione
            logger.debug("Impossibile salvare la cache delle valute.", exc_info=True)

    # ------------------------------------------------------------------
    # Dati di prezzo
    # ------------------------------------------------------------------

    def download_price_data(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
        auto_adjust: bool = False,
    ) -> pd.DataFrame:
        """
        Scarica dati OHLCV piu' dividendi/frazionamenti per piu' ticker insieme.

        Parameters
        ----------
        tickers : Iterable[str]
            Simboli da scaricare (es. ['AAPL', 'MSFT']).
        start_date : str
            Data di inizio in formato YYYY-MM-DD.
        end_date : str
            Data di fine in formato YYYY-MM-DD.
        auto_adjust : bool
            Se True, yfinance rettifica i prezzi OHLC. Lasciarlo False per
            tenere i prezzi grezzi e le colonne esplicite delle corporate action.
        """
        ticker_list = list(tickers)
        if not ticker_list:
            raise ValueError("tickers cannot be empty")

        try:
            data = yf.download(
                ticker_list,
                start=start_date,
                end=end_date,
                actions=True,
                auto_adjust=auto_adjust,
                group_by="column",
                progress=False,
                threads=False,
            )
        except Exception:
            data = self._download_price_data_fallback(
                ticker_list=ticker_list,
                start_date=start_date,
                end_date=end_date,
                auto_adjust=auto_adjust,
            )

        if data.empty:
            data = self._download_price_data_fallback(
                ticker_list=ticker_list,
                start_date=start_date,
                end_date=end_date,
                auto_adjust=auto_adjust,
            )

        # Ricaduta sui simboli SINGOLI mancanti, non solo sul fallimento totale.
        #
        # yfinance restituisce senza errore anche download parzialmente
        # riusciti: cinque serie su sette richieste, nessuna eccezione, colonne
        # presenti ma vuote per le due assenti. La ricaduta sottostante si
        # attivava solo quando l'intera chiamata falliva o tornava vuota,
        # quindi la perdita parziale passava in silenzio e il portafoglio
        # veniva costruito su meno titoli di quelli chiesti, con le tavole che
        # portavano comunque l'etichetta dell'universo completo.
        #
        # Il difetto e' documentato come D5 nell'audit: nella prima esecuzione
        # sperimentale fece sparire MSFT e NESN.SW, entrambi recuperati al
        # primo tentativo separato, il che dimostra che non si trattava di
        # indisponibilita' dei titoli.
        if not data.empty:
            data.index = _naive_index(data.index)

        missing = [t for t in ticker_list if t not in self._tickers_present(data, ticker_list)]
        if missing and len(missing) < len(ticker_list):
            logger.warning(
                "Download parziale: %s assenti dalla richiesta cumulativa; "
                "nuovo tentativo separato.",
                ", ".join(missing),
            )
            recovered = self._download_price_data_fallback(
                ticker_list=missing,
                start_date=start_date,
                end_date=end_date,
                auto_adjust=auto_adjust,
            )
            if not recovered.empty:
                # Normalizzato qui e non solo alla sorgente: la fusione non
                # deve dipendere da quale percorso abbia prodotto la serie.
                recovered.index = _naive_index(recovered.index)
                # Le colonne vuote dei simboli recuperati vanno rimosse prima
                # della fusione, altrimenti ogni titolo recuperato resta
                # presente due volte — una colonna vuota dalla richiesta
                # cumulativa e una piena dal tentativo separato — e ogni
                # estrazione a valle riceve un duplicato al posto di una serie.
                if isinstance(data.columns, pd.MultiIndex) and data.columns.nlevels == 2:
                    absent = set(missing)
                    data = data.loc[:, [c for c in data.columns if c[1] not in absent]]
                data = pd.concat([data, recovered], axis=1).sort_index(axis=1)
                still = [
                    t for t in missing if t not in self._tickers_present(data, ticker_list)
                ]
                logger.info(
                    "Recuperati %d titoli su %d.", len(missing) - len(still), len(missing)
                )
                missing = still

        # Registrato sull'istanza perche' il chiamante decida: restituire una
        # matrice a cui mancano titoli senza dirlo e' esattamente il guasto che
        # questa correzione rimuove.
        self.last_missing_tickers = tuple(missing)
        if missing:
            logger.warning(
                "Titoli senza dati dopo il nuovo tentativo: %s.", ", ".join(missing)
            )

        if data.empty:
            raise ValueError(
                "No data returned by yfinance for the requested range. "
                "This may indicate invalid tickers, delisted assets, or a temporary upstream issue."
            )

        return data.sort_index()

    def extract_adjusted_close(
        self, market_data: pd.DataFrame, allow_close_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Estrae i prezzi di chiusura rettificati da un DataFrame multi-ticker di yfinance.
        """
        import warnings

        if "Adj Close" in market_data.columns:
            self.last_price_type_used = "Adj Close"
            adj_close = market_data["Adj Close"]
            return self._to_dataframe(adj_close)

        if "Close" in market_data.columns:
            if not allow_close_fallback:
                raise ValueError("Adj Close not found in market_data and allow_close_fallback is False.")
            warnings.warn(
                "Colonna 'Adj Close' non presente nei dati di mercato. Ricaduta su 'Close' non rettificato: "
                "dividendi e frazionamenti azionari non sono inclusi nei prezzi.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Colonna 'Adj Close' non presente nei dati di mercato. Ricaduta su 'Close' non rettificato."
            )
            self.last_price_type_used = "Close (unadjusted fallback)"
            close = market_data["Close"]
            return self._to_dataframe(close)

        raise ValueError("Unable to find 'Adj Close' or 'Close' in market_data.")

    def extract_corporate_actions(
        self, market_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Torna i DataFrame di dividendi e frazionamenti azionari (se disponibili).
        """
        dividends = (
            self._to_dataframe(market_data["Dividends"])
            if "Dividends" in market_data.columns
            else pd.DataFrame(index=market_data.index)
        )
        splits = (
            self._to_dataframe(market_data["Stock Splits"])
            if "Stock Splits" in market_data.columns
            else pd.DataFrame(index=market_data.index)
        )
        return dividends.fillna(0.0), splits.fillna(0.0)

    # ------------------------------------------------------------------
    # Rendimenti
    # ------------------------------------------------------------------

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola i rendimenti log giornalieri: log(P_t / P_{t-1}).

        I prezzi mancanti all'interno della serie vengono riportati avanti
        (forward-fill) prima di calcolare i rendimenti. Le righe che restano
        NaN dopo questo passaggio (cioe' i NaN iniziali che non si possono
        riempire) vengono tolte dall'output, insieme alla prima riga di
        osservazione (che non ha un prezzo precedente con cui fare la
        differenza).

        Parameters
        ----------
        prices : pd.DataFrame
            Matrice dei prezzi con i ticker come colonne.

        Returns
        -------
        pd.DataFrame
            Matrice dei rendimenti log. L'indice inizia un'osservazione
            dopo quello di *prices* e non contiene nessun NaN.
        """
        if prices.empty:
            raise ValueError("prices DataFrame is empty")

        # Riporta avanti i buchi (es. festivita' / titolo sospeso),
        # poi butta le righe che sono tutte NaN (es. prima che partisse un ticker qualsiasi).
        clean_prices = prices.ffill().dropna(how="all")

        # Butta i ticker la cui storia prezzi e' tutta NaN (es. titolo delistato,
        # suffisso di borsa sbagliato tipo IS3N invece di IS3N.AS).
        all_nan_cols = clean_prices.columns[clean_prices.isna().all()]
        if len(all_nan_cols) > 0:
            logger.warning(
                "Dropping tickers with no price data (delisted or wrong symbol): %s",
                list(all_nan_cols),
            )
            clean_prices = clean_prices.drop(columns=all_nan_cols)

        if clean_prices.empty:
            raise ValueError(
                "All tickers had missing price data. "
                "Check that the symbols are valid and try adding the exchange suffix "
                "(e.g. IS3N.AS for Euronext Amsterdam)."
            )

        log_returns = np.log(clean_prices / clean_prices.shift(1))

        # Butta ogni riga che contiene almeno un NaN. Questo toglie:
        #   * la prima riga prodotta da shift(1), e
        #   * le righe iniziali dove un ticker non aveva ancora un prezzo.
        return log_returns.dropna(how="any")

    # ------------------------------------------------------------------
    # Tasso privo di rischio — API pubblica
    # ------------------------------------------------------------------

    def get_risk_free_rate(
        self,
        start_date: str,
        end_date: str,
        currency: str = "USD",
        horizon_years: float = 1.0,
    ) -> pd.Series:
        """Scarica il tasso privo di rischio per USD (FRED) o EUR (ECB).

        Solleva ValueError per valute non supportate. horizon_years sceglie
        la scadenza: ≤0.5y → 3M, ≤1.5y → 1Y, ≤3y → 2Y, ≤7y → 5Y, >7y → 10Y.
        """
        currency = currency.upper()
        if currency not in ("USD", "EUR"):
            raise ValueError(
                f"Unsupported reference currency '{currency}'. "
                "Only 'USD' and 'EUR' are accepted."
            )
        if horizon_years <= 0:
            raise ValueError("horizon_years must be > 0.")

        maturity_key = self._select_maturity(currency, horizon_years)

        if currency == "EUR":
            return self._fetch_ecb_series(maturity_key, start_date, end_date)

        try:
            return self._fetch_fred_series(maturity_key, start_date, end_date)
        except Exception as exc:
            # La ricaduta restituiva la curva in EURO sotto il nome del tasso
            # richiesto. Un tasso in un'altra valuta non e' una stima
            # degradata dello stesso oggetto ma un oggetto diverso: a fine
            # 2023 il decennale statunitense rendeva il 3,88 per cento e
            # l'equivalente euro il 2,0, novanta punti base che entrano
            # direttamente nel numeratore di ogni indice di Sharpe e nel
            # portafoglio tangente.
            raise RuntimeError(
                f"Tasso privo di rischio {maturity_key} non disponibile ({exc}). "
                "Non viene sostituito con quello di un'altra valuta: sarebbe una "
                "grandezza diversa, non una sua approssimazione."
            ) from exc

    # ------------------------------------------------------------------
    # Pesi per capitalizzazione di mercato
    # ------------------------------------------------------------------

    def get_market_cap_weights(
        self, tickers: list[str], reference_currency: str = "USD"
    ) -> pd.Series:
        """Pesi per capitalizzazione di mercato di *tickers*, in un'unica valuta.

        Yahoo riporta ogni capitalizzazione nella VALUTA DI NEGOZIAZIONE
        propria dello strumento. Sommarle direttamente vuol dire aggiungere
        yen a dollari: su un universo con Apple e Toyota i pesi veri sono
        circa 93 e 7 per cento, mentre la somma non convertita da' 8 e 92 —
        l'ordine si inverte. Questi pesi sono il prior di equilibrio di
        Black-Litterman, quindi ogni rendimento implicito
        ``Pi = delta * Sigma * w`` si porta dietro l'errore. Per questo ogni
        capitalizzazione viene convertita in ``reference_currency`` prima di
        normalizzare.

        ETF e fondi comuni riportano l'AUM, che non e' paragonabile a una
        capitalizzazione azionaria, e vengono esclusi. Anche un ticker la cui
        ricerca FALLISCE viene escluso — ma le due cose si registrano
        separatamente, perche' un timeout di rete e un'esclusione vera sono
        due fatti diversi, e prima erano indistinguibili: finivano entrambi
        come un peso zero silenzioso.
        """
        from portfolio_optimizer.fx.converter import latest_fx_rate

        caps: dict[str, float] = {}
        failed: list[str] = []
        excluded: list[str] = []
        for ticker in tickers:
            try:
                fast = yf.Ticker(ticker).fast_info
                quote_type = str(getattr(fast, "quote_type", "") or "").upper()
                if quote_type in ("ETF", "MUTUALFUND", "FUND"):
                    excluded.append(ticker)
                    caps[ticker] = 0.0
                    continue
                mc = getattr(fast, "market_cap", None)
                if not mc or mc <= 0:
                    failed.append(ticker)
                    caps[ticker] = 0.0
                    continue
                # La valuta viene da ``get_ticker_currency``, che e' la fonte
                # gia' usata dal resto del progetto e ha una cache, anziche'
                # dal campo ``currency`` di fast_info, che puo' mancare.
                local = self.get_ticker_currency(ticker)
                if not local:
                    failed.append(ticker)
                    caps[ticker] = 0.0
                    continue
                rate = latest_fx_rate(local, reference_currency)
                if rate is None:
                    failed.append(ticker)
                    caps[ticker] = 0.0
                    continue
                caps[ticker] = float(mc) * rate
            except Exception as exc:  # noqa: BLE001
                logger.warning("Capitalizzazione non ottenuta per %s: %s", ticker, exc)
                failed.append(ticker)
                caps[ticker] = 0.0

        if failed:
            logger.warning(
                "Capitalizzazione mancante per %s: esclusi dal prior di equilibrio.",
                ", ".join(failed),
            )
        if excluded:
            logger.info(
                "Fondi ed ETF esclusi dal prior di equilibrio: %s.", ", ".join(excluded)
            )
        self.last_market_cap_failures = tuple(failed)
        self.last_market_cap_excluded = tuple(excluded)

        series = pd.Series(caps, dtype=float)
        total = series.sum()
        if total <= 0:
            return pd.Series(
                np.ones(len(tickers)) / len(tickers), index=tickers, dtype=float
            )
        return series / total

    # ------------------------------------------------------------------
    # Rilevamento della valuta
    # ------------------------------------------------------------------

    def get_ticker_currency(
        self,
        ticker: str,
        default: str | None = None,
        max_retries: int = 2,
    ) -> str | None:
        """Torna la valuta di negoziazione di un ticker (cache su disco, usa fast_info).

        Se non si riesce a risolverla dopo i tentativi, torna `default` (che
        di default e' None). Una valuta di ripiego o non risolta NON viene
        MAI salvata nella cache persistente.
        """
        if ticker in self._currency_cache:
            return self._currency_cache[ticker]

        currency: str | None = None
        for attempt in range(max_retries):
            try:
                fast = yf.Ticker(ticker).fast_info
                raw = getattr(fast, "currency", None)
                if not raw or str(raw).upper() == "NONE":
                    info = getattr(yf.Ticker(ticker), "info", {}) or {}
                    raw = info.get("currency")
                if raw and str(raw).upper() != "NONE":
                    cand = str(raw).strip().upper()
                    if cand and len(cand) == 3 and cand.isalpha():
                        currency = cand
                        break
            except Exception as exc:
                logger.debug(
                    "Tentativo %d/%d fallito per valuta di %s: %s",
                    attempt + 1,
                    max_retries,
                    ticker,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(0.1)

        if currency:
            self._currency_cache[ticker] = currency
            self._save_currency_cache()
            return currency

        return default

    def resolve_ticker_currencies(
        self,
        tickers: list[str],
        declared_currencies: dict[str, str] | None = None,
    ) -> tuple[dict[str, str | None], dict[str, dict]]:
        """Risolve le valute di tutti i ticker, tornando (currencies, metadata).

        I metadati tengono traccia per ogni ticker di:
        - currency: codice valuta risolto (o None)
        - source: 'user_declared' | 'provider_confirmed' | 'unresolved'
        - is_confirmed: bool
        """
        declared = declared_currencies or {}
        currencies: dict[str, str | None] = {}
        metadata: dict[str, dict] = {}
        uncached: list[str] = []

        for t in tickers:
            if t in declared and declared[t]:
                cur = str(declared[t]).strip().upper()
                currencies[t] = cur
                metadata[t] = {
                    "currency": cur,
                    "source": "user_declared",
                    "is_confirmed": True,
                }
            elif t in self._currency_cache:
                cur = self._currency_cache[t]
                currencies[t] = cur
                metadata[t] = {
                    "currency": cur,
                    "source": "provider_confirmed",
                    "is_confirmed": True,
                }
            else:
                uncached.append(t)

        if uncached:
            workers = min(len(uncached), 8)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self.get_ticker_currency, t): t for t in uncached
                }
                for f in as_completed(futures):
                    t = futures[f]
                    try:
                        cur = f.result()
                    except Exception:
                        cur = None
                    currencies[t] = cur
                    if cur is not None:
                        metadata[t] = {
                            "currency": cur,
                            "source": "provider_confirmed",
                            "is_confirmed": True,
                        }
                    else:
                        metadata[t] = {
                            "currency": None,
                            "source": "unresolved",
                            "is_confirmed": False,
                        }

        return currencies, metadata

    def get_ticker_currencies(
        self,
        tickers: list[str],
        default: str | None = None,
    ) -> dict[str, str | None]:
        """Torna {ticker: currency} per tutti i ticker, scaricati in parallelo."""
        currencies, _ = self.resolve_ticker_currencies(tickers)
        if default is not None:
            return {t: (currencies.get(t) or default) for t in tickers}
        return currencies

    def annual_to_daily_rate(
        self, annual_rate: pd.Series, trading_days: int = 252
    ) -> pd.Series:
        """
        Convertire un tasso decimale annualizzato nel suo equivalente giornaliero.

        Formula: ``(1 + r_annual) ** (1 / trading_days) - 1``
        """
        if trading_days <= 0:
            raise ValueError("trading_days must be > 0")
        return (1.0 + annual_rate) ** (1.0 / trading_days) - 1.0

    # ------------------------------------------------------------------
    # Helper per la cache
    # ------------------------------------------------------------------

    def save_dataframe_csv(self, data: pd.DataFrame, filename: str) -> Path:
        """Salva il DataFrame nella cartella di cache e torna il path salvato."""
        file_path = self.cache_dir / filename
        data.to_csv(file_path)
        return file_path

    def load_dataframe_csv(
        self,
        filename: str,
        parse_dates: bool = True,
        index_col: int = 0,
    ) -> Optional[pd.DataFrame]:
        """Carica il DataFrame dalla cache se c'e', altrimenti torna None."""
        file_path = self.cache_dir / filename
        if not file_path.exists():
            return None
        return pd.read_csv(file_path, index_col=index_col, parse_dates=parse_dates)

    # ------------------------------------------------------------------
    # Helper privati
    # ------------------------------------------------------------------

    @staticmethod
    def _select_maturity(currency: str, horizon_years: float) -> str:
        """Scieglie la scadenza che corrisponde all'ORIZZONTE DI INVESTIMENTO prospettico.

        L'applicazione passa qui ``forecast_horizon_years``, non la lunghezza
        del campione storico. L'obbligazione viene quindi accoppiata
        all'orizzonte di pianificazione dell'investitore, non alla finestra
        di stima; e' la scelta difendibile quando l'indice di Sharpe deve
        confrontare il portafoglio con l'alternativa priva di rischio sullo
        stesso arco temporale futuro.

        Vale la pena dire la conseguenza pratica: cambiare l'orizzonte di
        investimento cambia anche il tasso privo di rischio usato al
        denominatore. Passando da un orizzonte di un anno a uno di dieci
        anni il tasso si spostava dal 4,79 al 3,88 per cento a fine 2023,
        il che su un portafoglio che rende il 10 per cento con il 15 per
        cento di volatilita' sposta l'indice di Sharpe da 0,347 a 0,408 —
        un cambio del 18 per cento prodotto da una scelta sull'orizzonte,
        non dal portafoglio.
        """
        table = _CURRENCY_MATURITIES.get(currency, _USD_MATURITIES)
        for threshold, key in table:
            if horizon_years <= threshold:
                return key
        return table[-1][1]

    def _configure_yfinance_cache(self) -> None:
        """Usa una cache del fuso orario di yfinance locale, per evitare problemi con quella globale corrotta."""
        set_cache = getattr(yf, "set_tz_cache_location", None)
        if callable(set_cache):
            set_cache(str(self.yf_cache_dir))


    @staticmethod
    def _tickers_present(data: pd.DataFrame, ticker_list: list[str]) -> set[str]:
        """Quali dei simboli richiesti portano davvero prezzi utilizzabili.

        Non basta che la colonna esista: un download in blocco riuscito solo
        in parte torna la colonna con tutti i valori mancanti, ed e' cosi'
        che la perdita restava silenziosa. Il test e' se la serie di prezzi
        ha almeno un'osservazione finita.
        """
        if data.empty:
            return set()
        field = "Adj Close" if "Adj Close" in data.columns else "Close"
        if field not in data.columns:
            return set()
        frame = data[field]
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(name=ticker_list[0] if ticker_list else "value")
        present = set()
        for ticker in ticker_list:
            if ticker not in frame.columns:
                continue
            # Difensivo: se una colonna arrivasse comunque duplicata, il
            # confronto deve restare uno scalare anziche' sollevare.
            column = frame[ticker]
            has_data = column.notna().any()
            if bool(has_data if np.isscalar(has_data) else has_data.any()):
                present.add(ticker)
        return present

    def _download_price_data_fallback(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        """
        Percorso di ripiego che scarica un ticker alla volta.

        E' piu' robusto di `yf.download(...)` quando la cache locale del
        database dei fusi orari di Yahoo si corrompe sulla macchina host.
        """
        frames: list[pd.DataFrame] = []
        valid_tickers: list[str] = []

        for ticker in ticker_list:
            try:
                ticker_data = yf.Ticker(ticker).history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust,
                    actions=True,
                )
            except Exception:
                continue

            if ticker_data.empty:
                continue

            renamed = ticker_data.rename_axis(index="Date").copy()
            renamed.index = _naive_index(renamed.index)
            renamed.columns = pd.MultiIndex.from_product([renamed.columns, [ticker]])
            frames.append(renamed)
            valid_tickers.append(ticker)

        if not frames:
            return pd.DataFrame()

        # Gli indici devono essere confrontabili prima di essere uniti.
        # ``yf.download`` restituisce un indice SENZA fuso orario, mentre
        # ``Ticker.history`` lo restituisce localizzato al fuso della borsa.
        # Concatenare i due solleva TypeError, e il recupero introdotto per la
        # voce D5 — che unisce proprio il risultato delle due chiamate — si
        # interrompeva quindi ogni volta che serviva davvero.
        combined = pd.concat(frames, axis=1).sort_index()
        combined.index = _naive_index(combined.index)
        combined = combined.swaplevel(axis=1).sort_index(axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(field, ticker) for ticker, field in combined.columns],
            names=["Price", "Ticker"],
        )
        return combined

    def _rf_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"rf_{key}.csv"

    def _rf_cache_fresh(self, path: Path) -> bool:
        try:
            return path.exists() and (
                time.time() - path.stat().st_mtime < _RF_CACHE_TTL
            )
        except Exception:
            return False

    def _fetch_fred_series(
        self, series_id: str, start_date: str, end_date: str
    ) -> pd.Series:
        """
        Scarica una serie FRED e torna un tasso decimale annualizzato giornaliero.

        La serie FRED grezza e' in punti percentuali (es. 5.0 = 5 %). I
        valori mancanti (pubblicati come '.' da FRED) vengono forzati a NaN
        e poi riportati avanti.
        """
        cache_path = self._rf_cache_path(f"fred_{series_id}")
        if self._rf_cache_fresh(cache_path):
            try:
                cached = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[
                    :, 0
                ]
                start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
                sliced = cached[(cached.index >= start_ts) & (cached.index <= end_ts)]
                if not sliced.empty:
                    return sliced.rename("risk_free_rate")
            except Exception:
                pass

        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        fred_df = pd.read_csv(url)

        if fred_df.empty:
            raise ValueError(f"No data returned from FRED for series '{series_id}'.")

        # FRED ha rinominato la colonna della data — gestiamo sia il formato vecchio che il nuovo.
        date_col = next(
            (c for c in fred_df.columns if c.lower() in ("date", "observation_date")),
            fred_df.columns[0],
        )
        fred_df[date_col] = pd.to_datetime(fred_df[date_col])

        # Colonna dei valori = series_id, ma mettiamo una guardia per se il nome colonna cambia.
        value_col = series_id if series_id in fred_df.columns else fred_df.columns[-1]
        values = pd.to_numeric(fred_df[value_col], errors="coerce")
        full_rate = (
            pd.Series(
                (values / 100.0).values,
                index=fred_df[date_col].values,
                name="risk_free_rate",
            )
            .ffill()
            .dropna()
        )

        # Salva la serie intera su disco cosi' le prossime richieste vengono servite dalla cache
        try:
            full_rate.to_frame().to_csv(cache_path)
        except Exception:
            pass

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        sliced = full_rate[(full_rate.index >= start_ts) & (full_rate.index <= end_ts)]
        if sliced.empty:
            raise ValueError(
                f"FRED series '{series_id}' returned no data "
                f"for the range {start_date} – {end_date}."
            )
        return sliced

    def _fetch_ecb_series(
        self, maturity_key: str, start_date: str, end_date: str
    ) -> pd.Series:
        """
        Scarica un tasso spot sui Bund tedeschi dal Data Warehouse Statistico della BCE.

        Dataset: YC — curve dei rendimenti per l'area euro
        Chiave del flusso: ``B.U2.EUR.4F.G_N_A.SV_C_YM.<maturity_key>``

        La BCE pubblica i tassi in punti percentuali nei giorni lavorativi
        TARGET2. Il risultato viene reindicizzato su ogni giorno di
        calendario e riportato avanti, cosi' si allinea con una serie di
        prezzi giornaliera.
        """
        cache_path = self._rf_cache_path(f"ecb_{maturity_key}")
        if self._rf_cache_fresh(cache_path):
            try:
                cached = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[
                    :, 0
                ]
                start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
                sliced = cached[(cached.index >= start_ts) & (cached.index <= end_ts)]
                if not sliced.empty:
                    return sliced.rename("risk_free_rate")
            except Exception:
                pass

        ecb_flow_key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.{maturity_key}"
        # Scarica un intervallo storico ampio cosi' la cache copre tante richieste future
        url = (
            "https://data-api.ecb.europa.eu/service/data/"
            f"YC/{ecb_flow_key}"
            f"?startPeriod=2000-01-01&endPeriod={end_date}&format=csvdata"
        )
        ecb_df = pd.read_csv(url)

        if ecb_df.empty:
            raise ValueError(
                f"No data returned from ECB for maturity key '{maturity_key}'."
            )

        if "TIME_PERIOD" not in ecb_df.columns or "OBS_VALUE" not in ecb_df.columns:
            raise ValueError(
                "ECB response is missing expected columns "
                "'TIME_PERIOD' and/or 'OBS_VALUE'."
            )

        ecb_df["TIME_PERIOD"] = pd.to_datetime(ecb_df["TIME_PERIOD"])
        values = pd.to_numeric(ecb_df["OBS_VALUE"], errors="coerce")
        full_rate = pd.Series(
            (values / 100.0).values,
            index=ecb_df["TIME_PERIOD"].values,
            name="risk_free_rate",
        )
        # Riporta avanti i buchi dei giorni lavorativi su tutto il calendario
        wide_range = pd.date_range(start="2000-01-01", end=end_date, freq="D")
        full_rate = full_rate.reindex(wide_range).ffill().dropna()
        full_rate.index.name = "DATE"

        try:
            full_rate.to_frame().to_csv(cache_path)
        except Exception:
            pass

        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        sliced = full_rate[(full_rate.index >= start_ts) & (full_rate.index <= end_ts)]
        if sliced.empty:
            raise ValueError(
                f"ECB series '{maturity_key}' returned no data "
                f"for the range {start_date} – {end_date}."
            )
        return sliced

    @staticmethod
    def _to_dataframe(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            return data.to_frame()
        return data


# ---------------------------------------------------------------------------
# Analisi della copertura storica
# ---------------------------------------------------------------------------


def analyze_history_impact(prices: pd.DataFrame) -> dict:
    """Analizza la copertura e identifica i ticker che restringono la finestra comune.

    Torna un dict con:
    - common_window_days: giorni nella finestra comune (dropna(how='any'))
    - full_window_days: righe totali nell'indice di prices
    - days_lost: full_window_days - common_window_days
    - limiting_tickers: ticker la cui presenza restringe la finestra comune
    - per_ticker: dict di {ticker: {start_date, end_date, valid_days, days_without_this}}
    """
    if prices.empty:
        return {
            "common_window_days": 0,
            "full_window_days": 0,
            "days_lost": 0,
            "limiting_tickers": [],
            "per_ticker": {},
        }

    full_window_days = len(prices)
    common_df = prices.dropna(how="any")
    common_window_days = len(common_df)
    days_lost = full_window_days - common_window_days

    per_ticker: dict[str, dict] = {}
    limiting_tickers: list[str] = []

    for col in prices.columns:
        s = prices[col].dropna()
        if s.empty:
            per_ticker[str(col)] = {
                "start_date": None,
                "end_date": None,
                "valid_days": 0,
                "days_without_this": common_window_days,
            }
            continue

        without_col = (
            prices.drop(columns=[col]).dropna(how="any")
            if prices.shape[1] > 1
            else pd.DataFrame()
        )
        days_without = len(without_col) if not without_col.empty else full_window_days
        per_ticker[str(col)] = {
            "start_date": (
                str(s.index.min().date())
                if hasattr(s.index.min(), "date")
                else str(s.index.min())
            ),
            "end_date": (
                str(s.index.max().date())
                if hasattr(s.index.max(), "date")
                else str(s.index.max())
            ),
            "valid_days": len(s),
            "days_without_this": days_without,
        }
        if days_without > common_window_days:
            limiting_tickers.append(str(col))

    if not limiting_tickers and days_lost > 0:
        limiting_tickers = list(prices.columns)

    return {
        "common_window_days": common_window_days,
        "full_window_days": full_window_days,
        "days_lost": days_lost,
        "limiting_tickers": limiting_tickers,
        "per_ticker": per_ticker,
    }
