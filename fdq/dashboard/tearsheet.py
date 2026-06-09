"""Static HTML experiment tearsheet."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
from matplotlib.figure import Figure

from fdq.validation.metrics import sharpe

if TYPE_CHECKING:
    from fdq.experiment.runner import StrategyRunResult

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _viability_chart(runs: list[StrategyRunResult]) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    for run in runs:
        tiers = sorted(run.tier_results.keys())
        sharpes = [sharpe(run.tier_results[t].returns) for t in tiers]
        label = f"{run.name} {run.params.get('symbol', '')}".strip()
        ax.plot(tiers, sharpes, marker="o", label=label)
        for t, s in zip(tiers, sharpes, strict=True):
            if abs(s) < 0.05:
                ax.axvline(t, color="gray", linestyle="--", alpha=0.3)
    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Starting Capital ($)")
    ax.set_ylabel("Net Sharpe (annualized)")
    ax.set_title("Capital-Viability Frontier")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _cost_ledger_chart(runs: list[StrategyRunResult], tier: float = 5.0) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for run in runs:
        if tier not in run.tier_results:
            continue
        result = run.tier_results[tier]
        if result.cost_timeseries.empty:
            continue
        ts = result.cost_timeseries
        label = f"{run.name} {run.params.get('symbol', '')}".strip()
        ax.plot(ts.index, ts["total_cents"], label=label)
    ax.set_ylabel("Cumulative Friction (¢)")
    ax.set_title(f"Cost Ledger at ${int(tier)}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _waterfall_chart(run: StrategyRunResult, tier: float = 5.0) -> str:
    if tier not in run.tier_results:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _fig_to_base64(fig)

    result = run.tier_results[tier]
    gross_return = 0.0
    if len(result.equity_curve) >= 2:
        eq = result.equity_curve
        gross_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    cost = result.cost_ledger
    spread_pct = cost.spread_cents / (result.starting_equity * 100)
    fees_pct = (cost.sec_fees_cents + cost.finra_taf_cents) / (result.starting_equity * 100)
    net_return = gross_return

    components = ["Gross", "Spread", "Fees", "Net"]
    values = [gross_return * 100, -spread_pct * 100, -fees_pct * 100, net_return * 100]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]
    ax.bar(components, values, color=colors)
    ax.set_ylabel("Return / Cost (%)")
    label = f"{run.name} {run.params.get('symbol', '')} @ ${int(tier)}"
    ax.set_title(f"Gross vs Net Waterfall — {label}")
    ax.axhline(0, color="black", linewidth=0.5)
    return _fig_to_base64(fig)


def _equity_chart(run: StrategyRunResult, tier: float = 5.0) -> str:
    fig, ax = plt.subplots(figsize=(9, 3))
    if tier in run.tier_results:
        eq = run.tier_results[tier].equity_curve
        ax.plot(eq.index, list(eq.values), linewidth=1.2)
    ax.set_ylabel("Equity ($)")
    label = f"{run.name} {run.params.get('symbol', '')} @ ${int(tier)}"
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def write_experiment_tearsheet(
    cfg: dict[str, Any],
    runs: list[StrategyRunResult],
    out_path: Path,
) -> Path:
    charts = {
        "viability": _viability_chart(runs),
        "cost_ledger": _cost_ledger_chart(runs, tier=5.0),
        "waterfalls": [_waterfall_chart(r, 5.0) for r in runs],
        "equity_curves": [_equity_chart(r, 5.0) for r in runs],
    }

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("experiment.html")
    html = template.render(
        title=f"{cfg['id']} — {cfg['title']}",
        friction_version=cfg.get("friction_version", "1.0.0"),
        charts=charts,
        runs=[
            {
                "name": f"{r.name} {r.params.get('symbol', '')}".strip(),
                "params": r.params,
            }
            for r in runs
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
