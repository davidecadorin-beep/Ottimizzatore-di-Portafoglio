from __future__ import annotations

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

import blueprints.extensions as ext
from blueprints.shared import (
    _NEWS_MAX_TICKERS,
    _TICKER_COLOURS,
    _TICKER_RE,
    _fetch_fx_data,
    _fetch_portfolio_news,
)

secondary = Blueprint("secondary", __name__)


@secondary.get("/api/fx")
def api_fx():
    return jsonify(_fetch_fx_data())


@secondary.get("/info")
def info():
    return render_template("info.html")


@secondary.get("/news")
def news():
    if ext._HAS_LIMITER and ext.limiter is not None:
        if not current_app.testing:
            ext.limiter.limit("20/minute")(lambda: None)()

    raw = request.args.get("tickers", "").strip()
    if not raw:
        return redirect(url_for("optimize.index"))
    raw_tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    tickers = [t for t in raw_tickers if _TICKER_RE.match(t)]
    seen: set[str] = set()
    tickers = [t for t in tickers if not (t in seen or seen.add(t))]
    if not tickers:
        return redirect(url_for("optimize.index"))
    troncati = len(tickers) > _NEWS_MAX_TICKERS
    tickers = tickers[:_NEWS_MAX_TICKERS]
    colour_map = {
        t: _TICKER_COLOURS[i % len(_TICKER_COLOURS)] for i, t in enumerate(tickers)
    }
    payload = _fetch_portfolio_news(tickers)
    return render_template(
        "news.html",
        news=payload["news"],
        events=payload["events"],
        tickers=tickers,
        colour_map=colour_map,
        troncati=troncati,
        max_tickers=_NEWS_MAX_TICKERS,
        titoli_senza_notizie=payload.get("titoli_senza_notizie", []),
    )
