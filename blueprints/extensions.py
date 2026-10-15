from __future__ import annotations

try:
    import flask_limiter as _flask_limiter  # noqa: F401 — serve solo a controllare che ci sia

    _HAS_LIMITER = True
except ImportError:
    _HAS_LIMITER = False

# Viene impostato da app.py dopo la creazione dell'app Flask; i blueprint lo leggono quando arriva la richiesta.
limiter = None
