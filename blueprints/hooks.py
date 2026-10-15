from __future__ import annotations

import os
import secrets
from datetime import date

from flask import Response, current_app, request, session


def _generate_csrf_token() -> str:
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


def _inject_csrf():
    return {"csrf_token": _generate_csrf_token}


def _set_security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.plot.ly https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://query1.finance.yahoo.com;"
    )
    return response


def _csrf_protect():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        if current_app.testing or current_app.config.get("TESTING"):
            return
        if os.environ.get("ENABLE_CSRF", "1") != "1":
            return
        if request.path.startswith("/api/"):
            return
        token = request.form.get("csrf_token") or request.headers.get("X-CSRFToken")
        expected = session.get("_csrf_token")
        if not token or not expected or not secrets.compare_digest(token, expected):
            return "Token CSRF non valido o mancante. Ricarica la pagina e riprova.", 400


_MESI_IT = {
    1: "gen", 2: "feb", 3: "mar", 4: "apr", 5: "mag", 6: "giu",
    7: "lug", 8: "ago", 9: "set", 10: "ott", 11: "nov", 12: "dic",
}


def _format_event_date(d) -> str:
    """Ritorna 'DD mmm' in italiano, aggiungendo l'anno se e' diverso da quello corrente."""
    mese = _MESI_IT[d.month]
    s = f"{d.day:02d} {mese}"
    if d.year != date.today().year:
        s += f" {d.year}"
    return s


def init_hooks(app) -> None:
    app.context_processor(_inject_csrf)
    app.after_request(_set_security_headers)
    app.before_request(_csrf_protect)
    app.template_filter("_format_event_date")(_format_event_date)
