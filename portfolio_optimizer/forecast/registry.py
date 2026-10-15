"""Registro delle previsioni ex-ante e delle loro controparti realizzate.

L'ottimizzatore produce gia', a ogni data di ribilanciamento, una previsione
di cosa fara' il portafoglio: rendimento atteso, volatilita' attesa, Value
at Risk ed Expected Shortfall. Oggi questi numeri vengono mostrati e poi
buttati via. Questo modulo li salva, accoppia ciascuno col valore
effettivamente realizzato nella finestra successiva, e passa il campione
risultante ai moduli di valutazione.

Senza questo registro non si puo' dire niente sull'accuratezza delle
previsioni, perche' le previsioni stesse non sono osservabili a posteriori.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ForecastRecord:
    """Una previsione emessa in un dato momento, col suo esito realizzato."""

    # --- identificazione ---
    issue_date: pd.Timestamp
    horizon_days: int
    model: str
    returns_method: str
    cov_method: str
    step_days: int = 0
    scenario: str = ""
    config_id: str = ""

    @property
    def config_signature(self) -> str:
        """Identificatore immutabile della configurazione che genera la previsione."""
        return f"{self.scenario}|{self.model}|{self.returns_method}|{self.cov_method}|h{self.horizon_days}|s{self.step_days}"


    # --- previsione ex-ante ---
    expected_return_annual: float = np.nan
    expected_volatility_annual: float = np.nan
    var_95_daily: float = np.nan
    cvar_95_daily: float = np.nan
    var_99_daily: float = np.nan
    cvar_99_daily: float = np.nan

    # --- realizzato sull'orizzonte (riempito da resolve) ---
    realized_return_annual: float = np.nan
    realized_volatility_annual: float = np.nan
    realized_worst_daily: float = np.nan
    violations_95: int = -1
    violations_99: int = -1
    observations: int = 0

    # --- rendimenti giornalieri realizzati dell'orizzonte, tenuti per l'analisi di coda ---
    realized_daily: list[float] = field(default_factory=list, repr=False)
    realized_dates: list[pd.Timestamp] = field(default_factory=list, repr=False)

    def as_row(self) -> dict:
        row = asdict(self)
        row.pop("realized_daily", None)
        row.pop("realized_dates", None)
        return row


class ForecastRegistry:
    """Collezione di previsioni solo in append, risolvibile contro i dati realizzati."""

    def __init__(self) -> None:
        self._records: list[ForecastRecord] = []

    # -- costruzione ----------------------------------------------------------

    def append(self, record: ForecastRecord) -> None:
        self._records.append(record)

    def extend(self, records: list[ForecastRecord]) -> None:
        self._records.extend(records)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[ForecastRecord]:
        return list(self._records)

    # -- risoluzione ------------------------------------------------------------

    def resolve(
        self,
        record: ForecastRecord,
        realized_log_returns: pd.Series,
        periods: int = 252,
    ) -> ForecastRecord:
        """Riempie i campi realizzati di un record dalla finestra osservata.

        ``realized_log_returns`` sono i rendimenti log giornalieri del
        portafoglio sull'orizzonte di previsione. L'annualizzazione del
        rendimento realizzato usa la convenzione geometrica, in modo
        coerente col resto del codice.
        """
        series = pd.Series(realized_log_returns).dropna()
        n = len(series)
        record.observations = n
        if n == 0:
            return record

        total_log = float(series.sum())
        record.realized_return_annual = float(np.exp(total_log) ** (periods / n) - 1.0)
        record.realized_volatility_annual = float(
            series.std(ddof=1) * np.sqrt(periods)
        ) if n > 1 else np.nan
        record.realized_worst_daily = float(series.min())
        record.realized_daily = [float(v) for v in series.to_numpy()]
        record.realized_dates = [pd.Timestamp(d) for d in series.index]

        # Una violazione e' un giorno la cui perdita supera il VaR previsto. Il
        # VaR e' tenuto come perdita positiva, quindi il confronto e' contro il suo negativo.
        if np.isfinite(record.var_95_daily):
            record.violations_95 = int((series < -record.var_95_daily).sum())
        if np.isfinite(record.var_99_daily):
            record.violations_99 = int((series < -record.var_99_daily).sum())
        return record

    # -- esportazione -----------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(
                columns=[
                    f.name
                    for f in fields(ForecastRecord)
                    if f.name not in ("realized_daily", "realized_dates")
                ]
            )
        return pd.DataFrame([r.as_row() for r in self._records])

    @property
    def config_ids(self) -> tuple[str, ...]:
        """Le configurazioni presenti nel registro, nell'ordine in cui sono apparse."""
        seen: dict[str, None] = {}
        for record in self._records:
            seen.setdefault(record.config_id, None)
        return tuple(seen)

    def hit_series(self, level: int = 95, config_id: str | None = None) -> pd.Series:
        """Serie binaria delle violazioni di VaR per UNA sola configurazione.

        Un'osservazione per ogni giorno di ogni orizzonte di previsione
        risolto, uguale a 1 quando la perdita realizzata ha superato il
        VaR previsto per quel giorno.

        ``config_id`` va passato ogni volta che il registro contiene piu'
        di una configurazione. Mettere insieme piu' configurazioni
        metterebbe tante osservazioni sui medesimi giorni di calendario:
        quelle violazioni sono per lo piu' gli stessi eventi visti
        attraverso portafogli leggermente diversi, quindi la serie
        aggregata viola l'assunzione di indipendenza dei test di
        copertura e gonfia il campione effettivo per il numero di
        configurazioni. Il vincolo e' imposto e non solo documentato,
        perche' la serie aggregata sembra perfettamente utilizzabile.
        """
        if level not in (95, 99):
            raise ValueError("level must be 95 or 99")
        var_attr = f"var_{level}_daily"

        available = self.config_ids
        if config_id is None:
            if len(available) > 1:
                raise ValueError(
                    "The registry holds several configurations "
                    f"({len(available)}); pass config_id to test one at a time. "
                    "Pooling them would break the independence assumption of "
                    "the coverage tests."
                )
            config_id = available[0] if available else None

        hits: list[int] = []
        index: list[pd.Timestamp] = []
        for record in self._records:
            if config_id is not None and record.config_id != config_id:
                continue
            var_value = getattr(record, var_attr)
            if not np.isfinite(var_value) or not record.realized_daily:
                continue
            if record.realized_dates and len(record.realized_dates) == len(record.realized_daily):
                for dt, value in zip(record.realized_dates, record.realized_daily):
                    hits.append(int(value < -var_value))
                    index.append(pd.Timestamp(dt))
            else:
                start = pd.Timestamp(record.issue_date)
                for offset, value in enumerate(record.realized_daily, start=1):
                    hits.append(int(value < -var_value))
                    index.append(start + pd.Timedelta(days=offset))
        return pd.Series(
            hits, index=pd.DatetimeIndex(index), name=f"hit_{level}"
        ).sort_index()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)
        return path

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ForecastRegistry":
        registry = cls()
        known = {f.name for f in fields(ForecastRecord)}
        for row in frame.to_dict(orient="records"):
            payload = {k: v for k, v in row.items() if k in known}
            payload["issue_date"] = pd.Timestamp(payload["issue_date"])
            registry.append(ForecastRecord(**payload))
        return registry


# ---------------------------------------------------------------------------
# Costruire una previsione da un modello stimato
# ---------------------------------------------------------------------------


def build_forecast(
    issue_date: pd.Timestamp,
    horizon_days: int,
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    train_returns: pd.DataFrame,
    model: str,
    returns_method: str,
    cov_method: str,
    step_days: int = 0,
    scenario: str = "",
    config_id: str = "",
    periods: int = 252,
) -> ForecastRecord:
    """Assembla la previsione ex-ante implicata da un modello stimato.

    Rendimento atteso e volatilita' vengono dalle stime del modello stesso,
    ed e' proprio questo il punto: il test e' se quello che il modello ha
    dichiarato di se' si confermi vero o no. VaR ed Expected Shortfall
    vengono dalla distribuzione empirica dei rendimenti del portafoglio
    in campione, la stessa convenzione usata altrove nel codice.
    """
    aligned = weights.reindex(covariance.index).fillna(0.0)
    variance = float(aligned.to_numpy() @ covariance.to_numpy() @ aligned.to_numpy())
    expected_vol = float(np.sqrt(max(variance, 0.0)))
    expected_ret = float(aligned.to_numpy() @ expected_returns.reindex(aligned.index).to_numpy())

    simple = np.expm1(train_returns[aligned.index].to_numpy())
    portfolio_simple = simple @ aligned.to_numpy()
    portfolio_log = np.log1p(portfolio_simple)

    def _var_cvar(alpha: float) -> tuple[float, float]:
        quantile = float(np.quantile(portfolio_log, 1.0 - alpha))
        tail = portfolio_log[portfolio_log <= quantile]
        cvar = float(-tail.mean()) if tail.size else float(-quantile)
        return -quantile, cvar

    var_95, cvar_95 = _var_cvar(0.95)
    var_99, cvar_99 = _var_cvar(0.99)

    return ForecastRecord(
        issue_date=pd.Timestamp(issue_date),
        horizon_days=horizon_days,
        model=model,
        returns_method=returns_method,
        cov_method=cov_method,
        step_days=step_days,
        scenario=scenario,
        config_id=config_id,
        expected_return_annual=expected_ret,
        expected_volatility_annual=expected_vol,
        var_95_daily=var_95,
        cvar_95_daily=cvar_95,
        var_99_daily=var_99,
        cvar_99_daily=cvar_99,
    )
