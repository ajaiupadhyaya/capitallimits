"""Experiment pipeline: config → backtest → results → tearsheet → report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fdq.backtest.engine import BacktestConfig, BacktestResult, run_backtest, run_capital_sweep
from fdq.dashboard.tearsheet import write_experiment_tearsheet
from fdq.data.bars import BarRequest
from fdq.data.ensure import load_validated_bars, load_validated_macro
from fdq.frictions.config import load_friction_config
from fdq.strategies.base import Strategy
from fdq.strategies.benchmarks import build_strategy
from fdq.util.settings import Settings
from fdq.validation.bootstrap import bootstrap_ruin_analysis
from fdq.validation.dsr import deflated_sharpe, probabilistic_sharpe
from fdq.validation.metrics import cagr, max_drawdown, sharpe, sortino, total_return, turnover
from fdq.validation.pbo import probability_backtest_overfitting
from fdq.validation.walkforward import in_sample_return_matrix, walk_forward


@dataclass
class StrategyRunResult:
    name: str
    params: dict[str, Any]
    tier_results: dict[float, BacktestResult]
    stress_results: dict[float, dict[int, BacktestResult]]


def _parse_strategy_entry(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(entry) != 1:
        msg = f"Invalid strategy entry: {entry}"
        raise ValueError(msg)
    name = next(iter(entry))
    params = dict(entry[name])
    return name, params


def _load_experiment_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        doc: dict[str, Any] = yaml.safe_load(f)
    return doc


def _experiment_dir(config_path: Path) -> Path:
    return config_path.parent


def run_experiment(
    config_path: Path,
    tearsheet: bool = True,
    report: bool = True,
    data_dir: Path | None = None,
) -> Path:
    cfg = _load_experiment_config(config_path)
    exp_dir = _experiment_dir(config_path)
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    window = cfg["window"]
    start = date.fromisoformat(window["start"])
    end = date.fromisoformat(window["end"])
    tiers: list[float] = [float(t) for t in cfg.get("capital_tiers", [5])]
    stress_multipliers: list[int] = [int(s) for s in cfg.get("stress_multipliers", [1])]
    seed = int(cfg.get("seed", 42))

    symbols = _collect_symbols(cfg["strategies"])
    bars = load_validated_bars(
        BarRequest(symbols=list(symbols), start=start, end=end),
        data_dir=data_dir,
    )
    settings = Settings()
    macro = load_validated_macro(data_dir)
    if macro.empty and settings.fred_api_key:
        from fdq.data.ensure import ensure_macro_cached

        ensure_macro_cached(start, end, settings=settings)
        macro = load_validated_macro(data_dir)

    if cfg.get("mode") == "walkforward":
        return _run_walkforward_experiment(
            cfg, exp_dir, results_dir, bars, macro, start, end, tiers, seed, report
        )

    all_runs: list[StrategyRunResult] = []
    summary: dict[str, Any] = {
        "id": cfg["id"],
        "title": cfg["title"],
        "friction_model_version": cfg.get("friction_version", "1.0.0"),
        "window": window,
        "strategies": [],
    }

    for entry in cfg["strategies"]:
        name, params = _parse_strategy_entry(entry)
        friction = load_friction_config(stress_multiplier=1.0)

        def make_strategy(n: str = name, p: dict[str, Any] = params) -> Strategy:
            return build_strategy(n, p)

        tier_results = run_capital_sweep(
            make_strategy,
            bars,
            tiers,
            friction,
            macro,
            start,
            end,
            seed,
        )

        stress_results: dict[float, dict[int, BacktestResult]] = {}
        for mult in stress_multipliers:
            if mult == 1:
                continue
            f_stress = load_friction_config(stress_multiplier=float(mult))
            stress_results[float(tiers[0])] = stress_results.get(float(tiers[0]), {})
            stress_results[float(tiers[0])][mult] = run_backtest(
                build_strategy(name, params),
                bars,
                BacktestConfig(starting_capital=tiers[0], friction=f_stress, seed=seed),
                macro,
                start,
                end,
            )

        run = StrategyRunResult(name, params, tier_results, stress_results)
        all_runs.append(run)

        strat_summary = _summarize_strategy(run, friction)
        summary["strategies"].append(strat_summary)

        for tier, result in tier_results.items():
            result.equity_curve.to_frame("equity").to_parquet(
                results_dir / f"{name}_{params.get('symbol', 'combo')}_tier{int(tier)}_equity.parquet"
            )
        if not tier_results[tiers[0]].trades.empty:
            tier_results[tiers[0]].trades.to_parquet(
                results_dir / f"{name}_{params.get('symbol', 'combo')}_trades.parquet"
            )

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if tearsheet:
        write_experiment_tearsheet(cfg, all_runs, exp_dir / "tearsheet.html")

    if report:
        _write_report(cfg, all_runs, exp_dir / "report.md")

    return results_dir


def _collect_symbols(strategies: list[dict[str, Any]]) -> set[str]:
    symbols: set[str] = set()
    for entry in strategies:
        name, params = _parse_strategy_entry(entry)
        if "symbol" in params:
            symbols.add(str(params["symbol"]))
        if "symbols" in params:
            symbols.update(str(s) for s in params["symbols"])
        if name == "balanced_6040" and "symbols" not in params:
            symbols.update(["SPY", "TLT"])
    return symbols


def _summarize_strategy(run: StrategyRunResult, friction: Any) -> dict[str, Any]:
    tier_stats: dict[str, Any] = {}
    for tier, result in run.tier_results.items():
        eq = result.equity_curve
        ret = result.returns
        trial_sharpes = np.array([0.0])
        tier_stats[str(int(tier))] = {
            "starting_equity": result.starting_equity,
            "ending_equity": result.ending_equity,
            "total_return": total_return(eq),
            "cagr": cagr(eq),
            "sharpe": sharpe(ret),
            "sortino": sortino(ret),
            "max_drawdown": max_drawdown(eq),
            "turnover": turnover(result.trades, eq),
            "psr": probabilistic_sharpe(ret, sr_benchmark=0.0),
            "dsr": deflated_sharpe(ret, trial_sharpes),
            "cost_ledger": result.cost_ledger.as_dict(),
            "n_trades": len(result.trades),
            "n_rejections": result.cost_ledger.rejections,
            "bootstrap": bootstrap_ruin_analysis(
                ret,
                starting_capital=tier,
                ruin_threshold=friction.ruin_capital_threshold,
            ).__dict__,
        }
    label = f"{run.name}_{run.params.get('symbol', 'combo')}"
    return {"name": label, "params": run.params, "tiers": tier_stats}


def _write_report(cfg: dict[str, Any], runs: list[StrategyRunResult], path: Path) -> None:
    lines = [
        f"# {cfg['id']} — {cfg['title']}",
        "",
        "## Hypothesis",
        "",
        "Even passive buy-and-hold on liquid ETFs incurs meaningful friction drag at $5 "
        "starting capital due to spread costs on entry and regulatory fee floors on exit. "
        "Monthly 60/40 rebalancing may fail the $1 minimum order constraint or bleed to fees.",
        "",
        "## Data & Window",
        "",
        "- Universe: Tier 0 ETFs",
        f"- Window: {cfg['window']['start']} → {cfg['window']['end']} (in-sample)",
        f"- Friction model version: {cfg.get('friction_version', '1.0.0')}",
        "",
        "## Methodology",
        "",
        "- Strategies: buy-and-hold (SPY, QQQ, IWM), 60/40 SPY/TLT monthly rebalance",
        "- Execution: signal at close T, fill at open T+1",
        "- Capital tiers: " + ", ".join(f"${int(t)}" for t in cfg.get("capital_tiers", [])),
        "- Trial count: 1 (no hyperparameter search)",
        f"- Stress multipliers: {cfg.get('stress_multipliers', [1])}",
        "",
        "## Results",
        "",
    ]

    for run in runs:
        label = f"{run.name} ({run.params})"
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Tier | CAGR | Sharpe | Max DD | Total Cost (¢) | Trades | Rejections |")
        lines.append("|------|------|--------|--------|----------------|--------|------------|")
        for tier, result in sorted(run.tier_results.items()):
            eq = result.equity_curve
            ret = result.returns
            cost = result.cost_ledger.total_cents
            lines.append(
                f"| ${int(tier)} | {cagr(eq):.2%} | {sharpe(ret):.2f} | {max_drawdown(eq):.2%} "
                f"| {cost:.1f} | {len(result.trades)} | {result.cost_ledger.rejections} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Statistical Assessment",
            "",
            "- DSR with trial_count=1 reduces to PSR vs zero benchmark",
            "- PBO/CSCV: N/A for Tier 0 (no parameter search)",
            "- Bootstrap ruin analysis at $5 tier reported in summary.json",
            "",
            "## Capital-Viability Finding",
            "",
            "See tearsheet.html for capital-viability frontier (Sharpe vs starting capital, log scale).",
            "Break-even capital is the tier where net Sharpe crosses zero after friction.",
            "",
            "## Limitations & Threats to Validity",
            "",
            "- Spread model uses static bps estimates, not historical NBBO",
            "- No market impact model (irrelevant at micro scale)",
            "- Locked test set (2024-2026) not used in this experiment",
            "",
            "## Conclusion & Next Steps",
            "",
            "EXP-001 establishes the friction baseline for passive strategies. "
            "Phase 1 will add Tier 1 rule-based strategies with walk-forward validation.",
            "",
        ]
    )

    path.write_text("\n".join(lines))


def _run_walkforward_experiment(
    cfg: dict[str, Any],
    exp_dir: Path,
    results_dir: Path,
    bars: Any,
    macro: Any,
    start: date,
    end: date,
    tiers: list[float],
    seed: int,
    report: bool,
) -> Path:
    n_folds = int(cfg.get("n_folds", 4))
    friction = load_friction_config(stress_multiplier=1.0)
    summary: dict[str, Any] = {
        "id": cfg["id"],
        "title": cfg["title"],
        "friction_model_version": cfg.get("friction_version", "1.0.0"),
        "window": cfg["window"],
        "mode": "walkforward",
        "strategies": [],
    }
    report_runs: list[dict[str, Any]] = []
    for entry in cfg["strategies"]:
        name, params = _parse_strategy_entry(entry)
        grid = dict(params.pop("grid", {}))
        matrix, trial_sharpes = in_sample_return_matrix(
            name, params, grid, bars, tiers[0], friction, macro, start, end, seed
        )
        pbo = probability_backtest_overfitting(matrix) if matrix.shape[1] >= 2 else 0.0
        tier_stats: dict[str, Any] = {}
        for tier in tiers:
            wf = walk_forward(
                name, params, grid, bars, tier, friction, macro, n_folds=n_folds, seed=seed
            )
            eq = wf.oos_equity
            ret = wf.oos_returns
            tier_stats[str(int(tier))] = {
                "starting_equity": tier,
                "ending_equity": float(eq.iloc[-1]) if len(eq) else tier,
                "cagr": cagr(eq),
                "sharpe": sharpe(ret),
                "sortino": sortino(ret),
                "max_drawdown": max_drawdown(eq),
                "dsr": deflated_sharpe(ret, trial_sharpes),
                "pbo": pbo,
                "n_trials": int(wf.n_trials),
                "fold_params": wf.fold_params,
            }
            eq.to_frame("equity").to_parquet(
                results_dir / f"{name}_{params.get('symbol', 'combo')}_tier{int(tier)}_equity.parquet"
            )
        label = f"{name}_{params.get('symbol', 'combo')}"
        summary["strategies"].append({"name": label, "params": params, "tiers": tier_stats})
        report_runs.append({"label": f"{name} ({params})", "tiers": tier_stats})

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if report:
        _write_walkforward_report(cfg, report_runs, exp_dir / "report.md")
    return results_dir


def _write_walkforward_report(
    cfg: dict[str, Any], runs: list[dict[str, Any]], path: Path
) -> None:
    lines = [
        f"# {cfg['id']} — {cfg['title']}",
        "",
        "## Methodology",
        "",
        f"- Mode: walk-forward ({cfg.get('n_folds', 4)} expanding folds), grid-search on train only",
        f"- Window (in-sample): {cfg['window']['start']} → {cfg['window']['end']}",
        f"- Friction model version: {cfg.get('friction_version', '1.0.0')}",
        "- Every reported result is net-of-friction; DSR deflates by trial count; PBO via CSCV.",
        "",
        "## Results",
        "",
    ]
    for run in runs:
        lines.append(f"### {run['label']}")
        lines.append("")
        lines.append("| Tier | CAGR | Sharpe | Max DD | DSR | PBO | Trials |")
        lines.append("|------|------|--------|--------|-----|-----|--------|")
        for tier, s in sorted(run["tiers"].items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| ${tier} | {s['cagr']:.2%} | {s['sharpe']:.2f} | {s['max_drawdown']:.2%} "
                f"| {s['dsr']:.3f} | {s['pbo']:.3f} | {s['n_trials']} |"
            )
        lines.append("")
    lines += ["## Limitations", "", "- Locked test set (2024-2026) not used here.", ""]
    path.write_text("\n".join(lines))
