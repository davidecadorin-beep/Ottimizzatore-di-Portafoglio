"""Costruttori di grafici Plotly e utility di presentazione per l'app Flask.

Estratti da app.py per tenere i route handler liberi dalla logica dei
grafici. Tutte le funzioni pubbliche restituiscono una stringa di
frammento HTML (o stringa vuota quando Plotly non e' disponibile).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimizer.risk.metrics import drawdown_series

try:
    import plotly.graph_objects as go

    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ---------------------------------------------------------------------------
# Costanti visive
# ---------------------------------------------------------------------------

_COMPARE_COLORS: dict[str, str] = {
    # Il colore chiaro e' riservato all'equipesato: e' il riferimento del
    # confronto, non uno dei criteri confrontati.
    "equal_weight": "#e2e8f0",
    "maximum_sharpe": "#38bdf8",
    "minimum_variance": "#c084fc",
    "risk_parity": "#00e5a0",
    "black_litterman": "#fde047",
    "minimum_cvar": "#fb7185",
}

_PROF_COLORS: list[str] = [
    "#38bdf8",
    "#c084fc",
    "#fde047",
    "#00e5a0",
    "#fb7185",
    "#67e8f9",
    "#fdba74",
    "#e879f9",
    "#bef264",
    "#f9a8d4",
]

_NAVY = "#06091a"  # navy profondo — area del grafico
_CYAN = "#7fdfff"  # ciano tenue — assi, tick, etichette
_CYAN_GRID = "rgba(127,223,255,0.05)"  # griglia quasi invisibile
_CYAN_LINE = "rgba(127,223,255,0.18)"  # linee di bordo degli assi
_CYAN_TICK = "rgba(127,223,255,0.35)"  # tacche degli assi
_FONT = "'Helvetica Neue', 'SF Pro Display', 'Inter', ui-sans-serif, sans-serif"

_PLOTLY_LAYOUT = dict(
    autosize=True,
    paper_bgcolor="rgba(0,0,0,0)",  # trasparente — lascia vedere la tinta della card come vignettatura
    font=dict(color=_CYAN, size=11, family=_FONT),
    legend=dict(
        bgcolor="rgba(6,9,26,0.92)",
        bordercolor=_CYAN_LINE,
        borderwidth=1,
        font=dict(size=11, color=_CYAN, family=_FONT),
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.15,
        yanchor="top",
    ),
)
_DEFAULT_MARGIN = dict(l=58, r=36, t=52, b=82)

_AXIS_STYLE = dict(
    gridcolor=_CYAN_GRID,
    gridwidth=0.5,
    linecolor=_CYAN_LINE,
    linewidth=0.5,
    showgrid=True,
    zeroline=False,
    tickfont=dict(size=10, color=_CYAN, family=_FONT),
    tickcolor=_CYAN_TICK,
    title_font=dict(size=11, color=_CYAN, family=_FONT),
)

_TITLE_STYLE = dict(
    font=dict(size=16, color="#ffffff", family=_FONT),
    x=0.0,
    xanchor="left",
    pad=dict(b=6),
)

METRIC_FORMAT: dict[str, str] = {
    "Rendimento Ann.": "pct",
    "Volatilità Ann.": "pct",
    "VaR 95%": "pct",
    "CVaR 95%": "pct",
    "Max Drawdown": "pct",
    "Sharpe Ratio": "ratio",
    "Sortino Ratio": "ratio",
    "Calmar Ratio": "ratio",
    "Omega Ratio": "ratio",
    "Skewness": "stat",
    # L'etichetta deve coincidere con quella prodotta da _METRIC_LABELS: dalla
    # convenzione C1 la metrica e' la curtosi IN ECCESSO, e una chiave rimasta
    # su "Kurtosis" lasciava la voce senza formato in ogni tabella.
    "Kurtosis in eccesso": "stat",
}

# ---------------------------------------------------------------------------
# Funzioni di supporto interne
# ---------------------------------------------------------------------------


def _tint(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _title_style(primary_color: str | None = None) -> dict:
    return dict(
        font=dict(size=16, color="#ffffff", family=_FONT),
        x=0.0,
        xanchor="left",
        pad=dict(b=6),
    )


def _style(fig, primary_color: str | None = None) -> None:
    fig.update_layout(**_PLOTLY_LAYOUT, margin=_DEFAULT_MARGIN, plot_bgcolor=_NAVY)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)


def _html(fig) -> str:
    return fig.to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True}
    )


def _make_donut(
    labels: list,
    values: list,
    title: str,
    colors: list,
    primary_color: str | None = None,
) -> str:
    total = sum(values)
    n = len(labels)
    base_colors = colors[:n]
    fill_colors = [_tint(c, 0.60) for c in base_colors]  # riempimento fluo
    legend_labels = [f"{label}  {v / total:.1%}" for label, v in zip(labels, values)]

    fig = go.Figure(
        go.Pie(
            labels=legend_labels,
            values=values,
            text=labels,
            hole=0,  # torta piena — niente buco centrale
            textinfo="text",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont=dict(
                size=12, color=base_colors, family=_FONT
            ),  # colore del ticker per ogni spicchio
            hovertemplate="<b>%{text}</b>: %{percent}<extra></extra>",
            marker=dict(
                colors=fill_colors,
                line=dict(color=base_colors, width=1),  # bordo neon 1px — stile badge
            ),
            showlegend=True,
        )
    )

    fig.update_layout(
        title={**_title_style(primary_color), "text": f"<b>{title}</b>"},
        height=360,
        margin=dict(l=10, r=150, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_NAVY,
        font=dict(color=_CYAN, size=11, family=_FONT),
        legend=dict(
            bgcolor="rgba(6,9,26,0.95)",
            bordercolor=_CYAN_LINE,
            borderwidth=1,
            font=dict(size=11, color=_CYAN, family=_FONT),
            orientation="v",
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            itemsizing="constant",
        ),
    )
    return _html(fig)


# ---------------------------------------------------------------------------
# Costruttori di grafici — risultato singolo
# ---------------------------------------------------------------------------


def chart_frontier(result, primary_color: str | None = None) -> str:
    if not _PLOTLY or result.frontier.empty:
        return ""
    frontier = result.frontier
    markers = result.frontier_markers
    rf = result.rf_breakdown.get("weighted", 0.0)

    sharpe_vals = (frontier["return"] - rf) / frontier["volatility"].clip(lower=1e-8)

    assets = list(result.expected_returns.index)
    hover = []
    for _, row in frontier.iterrows():
        w = row.get("weights", None)
        w_text = ""
        if w is not None and len(w) == len(assets):
            pairs = sorted(zip(assets, w), key=lambda x: -x[1])[:4]
            w_text = "<br>" + " | ".join(f"{a}: {v:.1%}" for a, v in pairs)
        hover.append(
            f"Ret: {row['return']:.2%}  Vol: {row['volatility']:.2%}<br>"
            f"Sharpe: {(row['return'] - rf) / max(row['volatility'], 1e-8):.3f}"
            f"{w_text}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier["volatility"],
            y=frontier["return"],
            mode="markers",
            marker=dict(
                color=sharpe_vals,
                colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                reversescale=False,
                size=6,
                opacity=0.75,
                line=dict(width=0),
                showscale=True,
                colorbar=dict(
                    title=dict(text="Sharpe", font=dict(size=11, color=_CYAN)),
                    thickness=10,
                    x=1.01,
                    tickfont=dict(size=10, color=_CYAN),
                    outlinewidth=0,
                    bgcolor="rgba(0,0,0,0)",
                ),
            ),
            text=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    _mc = [
        ("min_variance", "Min Variance", "#60a5fa", "diamond", 14),
        ("max_sharpe",   "Max Sharpe",   "#f59e0b", "star",    17),
        ("selected",     "Portafoglio",  "#10b981", "circle",  13),
    ]
    for key, label, color, symbol, sz in _mc:
        if key not in markers:
            continue
        m = markers[key]
        fig.add_trace(
            go.Scatter(
                x=[m["volatility"]],
                y=[m["return"]],
                mode="markers",
                marker=dict(
                    color=_tint(color, 0.45),   # riempimento semi-trasparente — coerente con gli altri grafici
                    size=sz,
                    symbol=symbol,
                    line=dict(color=color, width=2),  # bordo neon
                ),
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Vol: {m['volatility']:.2%}<br>"
                    f"Ret: {m['return']:.2%}<extra></extra>"
                ),
            )
        )

    _style(fig, primary_color)
    fig.update_layout(
        title={**_title_style(primary_color), "text": "<b>Frontiera Efficiente</b>"},
        xaxis=dict(tickformat=".1%", title="Volatilità annualizzata"),
        yaxis=dict(tickformat=".1%", title="Rendimento atteso ann."),
        height=460,
        margin=dict(l=60, r=100, t=52, b=82),
    )
    return _html(fig)


def chart_equity(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    rf_annual: float,
    primary_color: str | None = None,
) -> str:
    if not _PLOTLY:
        return ""

    port_cum = (np.exp(portfolio_returns.cumsum()) - 1) * 100
    line_col = primary_color if primary_color else "#3b82f6"
    fill_col = _tint(line_col, 0.35)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=port_cum.index,
            y=port_cum.values,
            mode="lines",
            name="Portafoglio",
            line=dict(color=line_col, width=2),
            fill="tozeroy",
            fillcolor=fill_col,
        )
    )

    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_cum = (np.exp(benchmark_returns.cumsum()) - 1) * 100
        bench_aligned = bench_cum.reindex(port_cum.index).ffill()
        fig.add_trace(
            go.Scatter(
                x=bench_aligned.index,
                y=bench_aligned.values,
                mode="lines",
                name=str(benchmark_returns.name or "Benchmark"),
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
            )
        )

    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    rf_cum = ((1 + rf_daily) ** np.arange(len(port_cum)) - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=port_cum.index,
            y=rf_cum,
            mode="lines",
            name="Risk-free",
            line=dict(color="#475569", width=1, dash="dot"),
        )
    )

    _style(fig, primary_color)
    fig.update_layout(
        title={
            **_title_style(primary_color),
            "text": "<b>Rendimento Cumulativo (%)</b>",
        },
        xaxis=dict(title="Data"),
        yaxis=dict(title="Rendimento (%)", ticksuffix="%"),
    )
    return _html(fig)


def chart_drawdown(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    primary_color: str | None = None,
) -> str:
    if not _PLOTLY:
        return ""

    def _dd(ret: pd.Series) -> pd.Series:
        cum = np.exp(ret.cumsum())
        return drawdown_series(cum) * 100

    line_col = primary_color if primary_color else "#3b82f6"
    fill_col = _tint(line_col, 0.40)
    fig = go.Figure()
    dd = _dd(portfolio_returns)
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name="Portafoglio",
            line=dict(color=line_col, width=1.5),
            fill="tozeroy",
            fillcolor=fill_col,
        )
    )

    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_dd = _dd(benchmark_returns.reindex(portfolio_returns.index).dropna())
        fig.add_trace(
            go.Scatter(
                x=bench_dd.index,
                y=bench_dd.values,
                mode="lines",
                name=str(benchmark_returns.name or "Benchmark"),
                line=dict(color="#94a3b8", width=1, dash="dash"),
            )
        )

    _style(fig, primary_color)
    fig.update_layout(
        title={**_title_style(primary_color), "text": "<b>Drawdown (%)</b>"},
        xaxis=dict(title="Data"),
        yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
    )
    return _html(fig)


def chart_risk_contribution(
    risk_contributions: pd.Series,
    colour_map: dict[str, str] | None = None,
    primary_color: str | None = None,
) -> str:
    if not _PLOTLY:
        return ""

    rc = risk_contributions.sort_values()
    n = len(rc)
    equal = 1.0 / n
    if colour_map:
        base_colors = [
            colour_map.get(t, _PROF_COLORS[i % len(_PROF_COLORS)])
            for i, t in enumerate(rc.index)
        ]
    else:
        base_colors = [
            "#fb7185"
            if v > equal * 1.2
            else "#00e5a0"
            if v < equal * 0.8
            else "#38bdf8"
            for v in rc.values
        ]
    fill_colors = [
        _tint(c, 0.35) for c in base_colors
    ]  # riempimento acceso — coerente con i grafici di performance

    fig = go.Figure(
        go.Bar(
            x=rc.values * 100,
            y=rc.index,
            orientation="h",
            marker_color=fill_colors,
            marker_line=dict(color=base_colors, width=1),
            text=[f"{v:.1%}" for v in rc.values],
            textfont=dict(size=10, color=base_colors),
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )

    fig.add_vline(
        x=equal * 100,
        line_dash="dot",
        line_color="#475569",
        line_width=1.5,
        annotation_text=f"Equal {equal:.1%}",
        annotation_font=dict(size=10, color="#64748b"),
        annotation_position="top",
    )
    _style(fig, primary_color)
    fig.update_layout(
        title={
            **_title_style(primary_color),
            "text": "<b>Contributo al Rischio (% varianza portafoglio)</b>",
        },
        xaxis=dict(title="Contributo (%)", ticksuffix="%"),
        yaxis=dict(title=""),
        height=max(300, n * 46),
        margin=dict(l=90, r=80, t=52, b=42),
        showlegend=False,
        barmode="overlay",
    )
    return _html(fig)


def chart_stress(stress_results: pd.DataFrame, primary_color: str | None = None) -> str:
    if not _PLOTLY or stress_results.empty:
        return ""

    scenarios = stress_results.index.tolist()
    ret_vals = (stress_results["period_return"] * 100).values
    dd_vals = (stress_results["max_drawdown"] * 100).values

    bar_colors = ["#ef4444" if v < 0 else "#10b981" for v in ret_vals]
    fill_colors = [_tint(c, 0.35) for c in bar_colors]
    hover_texts = [
        f"<b>{s}</b><br>Rendimento periodo: {r:.1f}%<br>Max Drawdown: {d:.1f}%"
        for s, r, d in zip(scenarios, ret_vals, dd_vals)
    ]

    n = len(scenarios)
    fig = go.Figure(
        go.Bar(
            y=scenarios,
            x=ret_vals,
            orientation="h",
            marker_color=fill_colors,
            marker_line=dict(color=bar_colors, width=1),
            hovertext=hover_texts,
            hoverinfo="text",
            width=0.55,
        )
    )
    fig.add_vline(x=0, line_color="#334155", line_width=1.5)

    # Le annotazioni stanno sempre a DESTRA dello zero — evita sovrapposizioni con le etichette dell'asse y.
    # Barre negative: il testo parte appena a destra dello zero; barre positive: il testo parte dopo la fine della barra.
    _vmax = float(max(abs(v) for v in ret_vals)) if len(ret_vals) > 0 else 5.0
    _gap = _vmax * 0.03
    for s, r, d in zip(scenarios, ret_vals, dd_vals):
        ann_x = (float(r) + _gap) if float(r) >= 0 else _gap
        fig.add_annotation(
            x=ann_x,
            y=s,
            text=f"{float(r):+.1f}%  (DD {float(d):.1f}%)",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10, color="#cbd5e1", family=_FONT),
        )

    _min_x = min(float(min(ret_vals)), 0) * 1.1 if len(ret_vals) > 0 else -10
    _max_x = max(float(max(ret_vals)), 0) if len(ret_vals) > 0 else 10
    _span = max(_max_x - _min_x, _vmax)
    _x_max = _max_x + _span * 0.65  # spazio per il testo dell'annotazione a destra

    _style(fig, primary_color)
    fig.update_layout(
        title={
            **_title_style(primary_color),
            "text": "<b>Stress Test — Storici e Sintetici</b>",
        },
        xaxis=dict(
            title="Rendimento periodo (%)",
            ticksuffix="%",
            zeroline=False,
            range=[_min_x, _x_max],
        ),
        yaxis=dict(title="", autorange="reversed"),
        height=max(380, n * 56),
        margin=dict(l=200, r=36, t=52, b=42),
        showlegend=False,
    )
    return _html(fig)


def chart_rolling_sharpe(
    rolling_sharpe: pd.Series, primary_color: str | None = None
) -> str:
    if not _PLOTLY or rolling_sharpe.empty:
        return ""

    line_col = primary_color if primary_color else "#6366f1"
    fill_col = _tint(line_col, 0.35)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode="lines",
            name="Sharpe 6M",
            line=dict(color=line_col, width=2),
            fill="tozeroy",
            fillcolor=fill_col,
        )
    )
    fig.add_hline(y=0, line_color="#334155", line_width=1.5, line_dash="solid")
    fig.add_hline(
        y=1,
        line_color="#10b981",
        line_dash="dot",
        line_width=1,
        annotation_text="Sharpe = 1",
        annotation_font=dict(size=10, color="#10b981"),
        annotation_position="right",
    )
    _style(fig, primary_color)
    fig.update_layout(
        title={
            **_title_style(primary_color),
            "text": "<b>Sharpe Ratio Rolling 6 Mesi (Stabilità Parametri)</b>",
        },
        xaxis=dict(title="Data"),
        yaxis=dict(title="Sharpe annualizzato"),
        height=320,
    )
    return _html(fig)


def chart_weights_pie(
    weights: pd.Series,
    colour_map: dict[str, str] | None = None,
    primary_color: str | None = None,
) -> str:
    if not _PLOTLY:
        return ""
    labels = list(weights.index)
    colors = [
        colour_map[t]
        if colour_map and t in colour_map
        else _PROF_COLORS[i % len(_PROF_COLORS)]
        for i, t in enumerate(labels)
    ]
    pc = primary_color or (colors[0] if colors else None)
    return _make_donut(
        labels=labels,
        values=[float(v) for v in weights.values],
        title="Composizione Portafoglio",
        colors=colors,
        primary_color=pc,
    )


def chart_currency(ticker_currencies: dict[str, str], weights: pd.Series) -> str:
    if not _PLOTLY:
        return ""
    from collections import defaultdict

    cur_weight: dict[str, float] = defaultdict(float)
    for ticker, cur in ticker_currencies.items():
        if ticker in weights.index:
            cur_weight[cur] += float(weights[ticker])
    return _make_donut(
        labels=list(cur_weight.keys()),
        values=list(cur_weight.values()),
        title="Esposizione Valutaria",
        colors=["#3b82f6", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6", "#06b6d4"],
    )


def chart_backtest(
    backtest_returns: pd.Series | None, portfolio_returns: pd.Series
) -> str:
    if not _PLOTLY or backtest_returns is None or backtest_returns.empty:
        return ""

    bt_cum = (
        (1 + backtest_returns).cumprod() - 1
    ) * 100  # rendimenti semplici dal walk-forward
    is_cum = (
        np.exp(portfolio_returns.cumsum()) - 1
    ) * 100  # rendimenti log in-sample

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt_cum.index,
            y=bt_cum.values,
            mode="lines",
            name="Walk-forward (out-of-sample)",
            line=dict(color="#6366f1", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=is_cum.index,
            y=is_cum.values,
            mode="lines",
            name="In-sample (ottimizzato)",
            line=dict(color="#3b82f6", width=1.5, dash="dot"),
            opacity=0.5,
        )
    )
    _style(fig)
    fig.update_layout(
        title={
            **_TITLE_STYLE,
            "text": "<b>Walk-Forward Backtest vs. In-Sample (costi transazione inclusi)</b>",
        },
        xaxis=dict(title="Data"),
        yaxis=dict(title="Rendimento Cumulativo (%)", ticksuffix="%"),
        height=400,
    )
    return _html(fig)


def chart_monte_carlo(mc_data: dict | None) -> str:
    if not _PLOTLY or mc_data is None:
        return ""

    n_points = mc_data.get("n_points", 121)
    horizon = float(mc_data.get("horizon_years", 1.0))
    x = [i * horizon / (n_points - 1) for i in range(n_points)]

    fig = go.Figure()

    def _add_band(paths: dict, fill_rgb: str, line_color: str, label: str) -> None:
        p95, p75, p50, p25, p5 = (
            paths.get("p95", []),
            paths.get("p75", []),
            paths.get("p50", []),
            paths.get("p25", []),
            paths.get("p5", []),
        )
        if not p50:
            return
        xf = x + x[::-1]
        fig.add_trace(
            go.Scatter(
                x=xf,
                y=p95 + p5[::-1],
                fill="toself",
                fillcolor=f"rgba({fill_rgb},0.22)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xf,
                y=p75 + p25[::-1],
                fill="toself",
                fillcolor=f"rgba({fill_rgb},0.42)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p50,
                mode="lines",
                name=f"{label} — mediana (P50)",
                line=dict(color=line_color, width=2),
                hovertemplate=f"Anno: %{{x:.1f}}<br>{label} P50: %{{y:.1f}}<extra></extra>",
            )
        )

    if "gbm_paths" in mc_data:
        _add_band(mc_data["gbm_paths"], "59,130,246", "#3b82f6", "GBM")
    if "bootstrap_paths" in mc_data:
        p50_b = mc_data["bootstrap_paths"].get("p50", [])
        if p50_b:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=p50_b,
                    mode="lines",
                    name="Bootstrap a blocchi — mediana (P50)",
                    line=dict(color="#8b5cf6", width=2, dash="dash"),
                    hovertemplate="Anno: %{x:.1f}<br>Bootstrap a blocchi P50: %{y:.1f}<extra></extra>",
                )
            )
    if "jd_paths" in mc_data:
        p50_j = mc_data["jd_paths"].get("p50", [])
        if p50_j:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=p50_j,
                    mode="lines",
                    name="Jump-Diffusion — mediana (P50)",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    hovertemplate="Anno: %{x:.1f}<br>JD P50: %{y:.1f}<extra></extra>",
                )
            )

    fig.add_hline(
        y=100,
        line_color="#334155",
        line_width=1.5,
        line_dash="dot",
        annotation_text="Valore iniziale (base 100)",
        annotation_font=dict(size=10, color="#64748b"),
        annotation_position="bottom right",
    )

    _style(fig)
    fig.update_layout(
        title={
            **_TITLE_STYLE,
            "text": "<b>Simulazione Monte Carlo — Proiezione Valore Portafoglio (Base 100)</b>",
        },
        xaxis=dict(title="Anni"),
        yaxis=dict(title="Valore (base 100)"),
        height=520,
    )
    return _html(fig)


# ---------------------------------------------------------------------------
# Costruttore di grafico — confronto stimatori dei rendimenti
# ---------------------------------------------------------------------------


def chart_returns_comparison(returns_comparison: dict | None) -> str:
    """Grafico a barre raggruppate: rendimento atteso per titolo, per ogni stimatore."""
    if not _PLOTLY or not returns_comparison:
        return ""

    # La chiave di servizio non e' una serie da disegnare.
    methods = [k for k in returns_comparison if not k.startswith("__")]
    tickers = list(returns_comparison[methods[0]].index)
    method_colors = {
        "Media Storica": "#94a3b8",
        "Bayes-Stein": "#3b82f6",
        "Fama-French": "#10b981",
    }

    fig = go.Figure()
    for method in methods:
        base_col = method_colors.get(method, "#f59e0b")
        fill_col = _tint(base_col, 0.35)
        values = [
            float(returns_comparison[method].get(t, float("nan"))) * 100
            for t in tickers
        ]
        fig.add_trace(
            go.Bar(
                name=method,
                x=tickers,
                y=values,
                marker_color=fill_col,
                marker_line=dict(color=base_col, width=1),
                hovertemplate=f"<b>{method}</b> — %{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_color="#334155", line_width=1)
    _style(fig)
    fig.update_layout(
        title={
            **_TITLE_STYLE,
            "text": "<b>Rendimenti Attesi — Confronto Stimatori</b>",
        },
        barmode="group",
        xaxis=dict(title=""),
        yaxis=dict(title="Rendimento atteso ann. (%)", ticksuffix="%"),
        height=420,
    )
    return _html(fig)


# ---------------------------------------------------------------------------
# Costruttori di grafici — confronto tra modelli
# ---------------------------------------------------------------------------


def chart_compare_weights(model_results: dict) -> str:
    if not _PLOTLY or not model_results:
        return ""
    model_keys = list(model_results.keys())
    model_labels = [model_results[k][1] for k in model_keys]
    all_tickers = list(next(iter(model_results.values()))[0].weights.index)

    fig = go.Figure()
    for i, ticker in enumerate(all_tickers):
        values = [
            float(model_results[k][0].weights.get(ticker, 0)) * 100 for k in model_keys
        ]
        fig.add_trace(
            go.Bar(
                name=ticker,
                x=model_labels,
                y=values,
                marker_color=_PROF_COLORS[i % len(_PROF_COLORS)],
                hovertemplate=f"<b>{ticker}</b>: %{{y:.1f}}%<extra></extra>",
            )
        )
    _style(fig)
    fig.update_layout(
        title={**_TITLE_STYLE, "text": "<b>Allocazione per Modello</b>"},
        barmode="stack",
        xaxis=dict(title=""),
        yaxis=dict(title="Peso (%)", ticksuffix="%", range=[0, 102]),
        height=360,
        margin=dict(l=58, r=36, t=52, b=82),
    )
    return _html(fig)


def backtest_verdict(
    backtest_returns: pd.Series | None, portfolio_returns: pd.Series, rf_annual: float
) -> dict | None:
    if backtest_returns is None or backtest_returns.empty:
        return None
    n_years = max(len(backtest_returns) / 252, 0.1)
    bt_cum = float((1 + backtest_returns).prod() - 1)  # rendimenti semplici
    bt_cagr = float((1 + bt_cum) ** (1 / n_years) - 1)

    # Le misure vengono dalla definizione condivisa, sulla stessa serie
    # convertita in logaritmi: la scheda ne calcolava di proprie, e lo Sharpe
    # risultava del 3,4 per cento diverso da quello riportato ovunque altrove
    # nella stessa pagina.
    from portfolio_optimizer.risk.metrics import summary_metrics

    bt_log = pd.Series(np.log1p(backtest_returns.to_numpy()), index=backtest_returns.index)
    bt_metrics = summary_metrics(bt_log, risk_free_annual=rf_annual)
    bt_vol = bt_metrics["volatility_annual"]
    bt_sharpe = bt_metrics["sharpe_ratio"]
    bt_max_dd = bt_metrics["max_drawdown"]

    # Il divario fra dentro e fuori campione va misurato sulle STESSE date.
    #
    # Confrontava il cumulato in campione sull'intera storia con quello
    # walk-forward, che comincia dopo la prima finestra di stima: un anno di
    # rendimenti in piu' da un lato. Su una strategia equipesata, che non ha
    # parametri e non puo' sovradattarsi per costruzione, il divario cosi'
    # calcolato risultava del 35 per cento su due anni di storia e del 109 su
    # dieci, e la scheda dichiarava "segnali di sovradattamento" a chiunque
    # avesse piu' di quattro anni di dati. Allineare le date lascia soltanto
    # cio' che il sovradattamento produce davvero: la differenza fra un
    # portafoglio scelto conoscendo il periodo e uno scelto prima.
    aligned = portfolio_returns.reindex(backtest_returns.index).dropna()
    if len(aligned) >= 2:
        is_cum = float(np.exp(aligned.sum()) - 1)
    else:
        is_cum = float(np.exp(portfolio_returns.sum()) - 1)
    gap = is_cum - bt_cum

    if bt_cum >= 0.05 and bt_sharpe >= 0.5 and gap < 0.25:
        verdict, label, color = "positive", "Strategia solida", "emerald"
        desc = (
            f"Rendimento WF cumulativo {bt_cum:.1%} (CAGR {bt_cagr:.1%}), "
            f"Sharpe {bt_sharpe:.2f} e gap in-sample/out-of-sample contenuto ({gap:.1%}). "
            f"Il modello generalizza bene fuori campione senza segnali evidenti di overfitting."
        )
    elif bt_cum >= 0 and gap < 0.40:
        verdict, label, color = "neutral", "Performance accettabile", "amber"
        desc = (
            f"Rendimento WF positivo ma modesto ({bt_cum:.1%}, CAGR {bt_cagr:.1%}), "
            f"Sharpe {bt_sharpe:.2f}. Gap in-sample/out-of-sample {gap:.1%}: "
            f"il modello funziona ma con margini ridotti. Valuta un orizzonte di training più lungo."
        )
    elif bt_cum >= 0 and gap >= 0.40:
        verdict, label, color = "warning", "Segnali di overfitting", "orange"
        desc = (
            f"Rendimento WF {bt_cum:.1%} ma gap in-sample/out-of-sample elevato ({gap:.1%}): "
            f"il modello si adatta eccessivamente ai dati storici. "
            f"Considera ridurre il numero di asset o usare un modello più robusto (es. Risk Parity)."
        )
    else:
        verdict, label, color = "negative", "Strategia debole out-of-sample", "rose"
        desc = (
            f"Rendimento WF negativo ({bt_cum:.1%}, CAGR {bt_cagr:.1%}), "
            f"Sharpe {bt_sharpe:.2f}, max drawdown {bt_max_dd:.1%}. "
            f"La strategia non generalizza fuori campione: considera modello diverso o periodo dati più ampio."
        )

    return dict(
        verdict=verdict,
        label=label,
        color=color,
        desc=desc,
        bt_cum=bt_cum,
        bt_cagr=bt_cagr,
        bt_sharpe=bt_sharpe,
        bt_vol=bt_vol,
        bt_max_dd=bt_max_dd,
        is_cum=is_cum,
        gap=gap,
    )


def _oos_traces(paths: dict, transform) -> list:
    """Corpo condiviso dei due grafici fuori campione.

    ``paths`` mappa una chiave di modello a ``(rendimenti log netti,
    etichetta)``. Il portafoglio equipesato viene disegnato ultimo e piu'
    spesso cosi' si legge come la linea di riferimento e non come una
    serie fra sei: il confronto e' proprio contro di lui.
    """
    traces = []
    ordered = sorted(paths.items(), key=lambda kv: kv[0] == "equal_weight")
    for model_key, (net_log, label) in ordered:
        if net_log is None or len(net_log) == 0:
            continue
        y = transform(net_log)
        is_benchmark = model_key == "equal_weight"
        traces.append(
            go.Scatter(
                x=y.index,
                y=y.values,
                mode="lines",
                name=label,
                line=dict(
                    color=_COMPARE_COLORS.get(model_key, "#94a3b8"),
                    width=3.0 if is_benchmark else 1.7,
                ),
                hovertemplate=f"<b>{label}</b>  %{{y:.2f}}%<extra></extra>",
            )
        )
    return traces


def chart_oos_equity(paths: dict) -> str:
    """Rendimento cumulativo fuori campione di ogni criterio.

    La curva e' il percorso walk-forward netto dei costi di transazione,
    non l'allocazione finale applicata all'indietro su tutta la storia:
    ogni suo punto e' stato prodotto da una decisione che poteva vedere
    solo cio' che la precedeva.
    """
    if not _PLOTLY or not paths:
        return ""
    fig = go.Figure()
    for trace in _oos_traces(paths, lambda s: (np.exp(s.cumsum()) - 1) * 100):
        fig.add_trace(trace)
    if not fig.data:
        return ""
    fig.add_hline(y=0, line=dict(color="#334155", width=1))
    _style(fig)
    fig.update_layout(
        title={
            **_TITLE_STYLE,
            "text": "<b>Rendimento Cumulativo Fuori Campione</b> — netto costi",
        },
        xaxis=dict(title="Data"),
        yaxis=dict(title="Rendimento cumulativo (%)", ticksuffix="%"),
        height=420,
    )
    return _html(fig)


def chart_oos_drawdown(paths: dict) -> str:
    """Drawdown lungo gli stessi percorsi fuori campione."""
    if not _PLOTLY or not paths:
        return ""

    def _dd(series):
        cumulative = np.exp(series.cumsum())
        return drawdown_series(cumulative) * 100

    fig = go.Figure()
    for trace in _oos_traces(paths, _dd):
        fig.add_trace(trace)
    if not fig.data:
        return ""
    _style(fig)
    fig.update_layout(
        title={**_TITLE_STYLE, "text": "<b>Drawdown Fuori Campione</b>"},
        xaxis=dict(title="Data"),
        yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
        height=420,
    )
    return _html(fig)
