"""Portafoglio equipesato.

Promosso a criterio di prima classe e non lasciato come riferimento
implicito: e' il benchmark con cui la letteratura misura ogni ottimizzatore,
e DeMiguel, Garlappi e Uppal (2009) dimostrano che e' difficile battere fuori
campione una volta messi in conto l'errore di stima e i costi di transazione.
"""

from __future__ import annotations

import pandas as pd


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """Pesi del portafoglio 1/N sulle colonne di *returns*."""
    columns = returns.columns
    n = len(columns)
    if n == 0:
        raise ValueError("At least one asset is required.")
    return pd.Series(1.0 / n, index=columns, name="equal_weight")
