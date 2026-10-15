from __future__ import annotations

import logging
import os

import yfinance as yf
from flask import Flask

from blueprints.shared import (  # noqa: F401 — ri-esportati, servono ai test
    _METRIC_LABELS, _NEWS_CACHE, _bl_views_from_rows,
    _covariance_evidence, _effective_window, _end_start_from_lookback,
    _fetch_one_ticker_news, _fetch_portfolio_news, _fx_cache, _optimizer_cache_key,
    _parse_bl_views, _parse_tickers,
)
import blueprints.extensions as _ext

_HAS_LIMITER = _ext._HAS_LIMITER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_ENV = os.environ.get("FLASK_ENV", os.environ.get("APP_ENV", "development")).lower()
_IS_PRODUCTION = _ENV == "production" or os.environ.get("PRODUCTION", "0") == "1"

_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")
if _IS_PRODUCTION:
    if not _SECRET_KEY or _SECRET_KEY in ("dev-only-change-in-production", "change-me", "secret"):
        raise RuntimeError(
            "FLASK_SECRET_KEY non configurata o non sicura per l'ambiente di produzione. "
            "Imposta una chiave segreta crittografica valida nella variabile di ambiente FLASK_SECRET_KEY."
        )
    app.secret_key = _SECRET_KEY
else:
    app.secret_key = _SECRET_KEY or "dev-only-change-in-production"

app.config.update(
    IS_PRODUCTION=_IS_PRODUCTION,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=_IS_PRODUCTION or os.environ.get("SESSION_COOKIE_SECURE", "0").lower() in ("1", "true"),
    MAX_CONTENT_LENGTH=int(os.environ.get("MAX_CONTENT_LENGTH", 2 * 1024 * 1024)),
)

if os.environ.get("USE_PROXYFIX", "0").lower() in ("1", "true"):
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

yf.set_tz_cache_location(os.path.join(os.path.dirname(__file__), "cache", "yf-tz"))

_RATELIMIT_STORAGE = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")
if _HAS_LIMITER:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        app=app, key_func=get_remote_address,
        default_limits=["200/hour"], storage_uri=_RATELIMIT_STORAGE,
    )
    _ext.limiter = limiter

from blueprints.secondary_bp import secondary
from blueprints.export_bp import export
from blueprints.compare_bp import compare
from blueprints.optimize_bp import optimize_bp
from blueprints.hooks import _MESI_IT, init_hooks  # noqa: F401 — _MESI_IT ri-esportato, serve ai test
app.register_blueprint(secondary)
app.register_blueprint(export)
app.register_blueprint(compare)
app.register_blueprint(optimize_bp)
init_hooks(app)

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
