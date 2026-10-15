"""Archiviazione e valutazione delle previsioni.

L'ottimizzatore emette previsioni a ogni data di ribilanciamento; questo
package le salva e misura, a posteriori, se si sono poi rivelate giuste.
"""

from portfolio_optimizer.forecast.registry import (
    ForecastRecord,
    ForecastRegistry,
    build_forecast,
)

__all__ = ["ForecastRecord", "ForecastRegistry", "build_forecast"]
