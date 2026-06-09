"""Daily bar fetcher: Alpaca primary, yfinance backup, parquet cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from fdq.data.provenance import write_provenance
from fdq.util.settings import Settings

_BAR_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class BarRequest:
    symbols: list[str]
    start: date
    end: date


def _cache_path(symbol: str, data_dir: Path) -> Path:
    return data_dir / "raw" / f"{symbol}.parquet"


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index, name="timestamp")
    df.index = df.index.as_unit("s")
    return df.dropna(how="all")


def _write_cache(
    df: pd.DataFrame,
    path: Path,
    *,
    source: str,
    symbol: str,
    start: date,
    end: date,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.dropna(how="all").to_parquet(path)
    write_provenance(
        path,
        source=source,  # type: ignore[arg-type]
        data_kind="historical",
        start=start,
        end=end,
        symbol=symbol,
    )


def _merge_cache(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _fetch_alpaca(
    symbols: list[str], start: date, end: date, settings: Settings
) -> dict[str, pd.DataFrame]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.max.time()),
    )
    bars = client.get_stock_bars(req)
    raw: pd.DataFrame = bars.df  # type: ignore[union-attr]

    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for sym in symbols:
        if sym not in raw.index.get_level_values(0):
            continue
        sym_df: pd.DataFrame = raw.xs(sym, level=0).copy()  # type: ignore[assignment]
        sym_df.index = pd.DatetimeIndex(sym_df.index.date, name="timestamp")  # type: ignore[attr-defined]
        cols = [c for c in _BAR_COLUMNS if c in sym_df.columns]
        sym_df = sym_df[cols]
        if "close" in sym_df.columns:
            sym_df["close_adj"] = sym_df["close"]
        out[sym] = sym_df
    return out


def _fetch_yfinance(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        tickers=symbols,
        start=start.isoformat(),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        tickers_in_response = set(raw.columns.get_level_values(0))
        for sym in symbols:
            if sym not in tickers_in_response:
                continue
            df = raw[sym].copy()
            df.columns = [str(c).lower() for c in df.columns]
            df.index = pd.DatetimeIndex(df.index.date, name="timestamp")
            cols = [c for c in _BAR_COLUMNS if c in df.columns]
            df = df[cols]
            if "adj close" in raw[sym].columns:
                df["close_adj"] = raw[sym]["Adj Close"].values
            elif "close" in df.columns:
                df["close_adj"] = df["close"]
            out[sym] = df
        return out

    df = raw.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index.date, name="timestamp")
    cols = [c for c in _BAR_COLUMNS if c in df.columns]
    df = df[cols]
    if "adj close" in raw.columns:
        df["close_adj"] = raw["Adj Close"]
    elif "close" in df.columns:
        df["close_adj"] = df["close"]
    out[symbols[0]] = df
    return out


def _log_discrepancy(
    symbol: str, field: str, d1: date, v_alpaca: float, v_yf: float, quality_dir: Path
) -> None:
    pct = abs(v_alpaca - v_yf) / max(abs(v_yf), 1e-9)
    if pct <= 0.005:
        return
    quality_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "symbol": symbol,
        "field": field,
        "date": d1.isoformat(),
        "alpaca": v_alpaca,
        "yfinance": v_yf,
        "pct_diff": pct,
    }
    with open(quality_dir / "discrepancies.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def fetch_symbol(
    symbol: str,
    start: date,
    end: date,
    settings: Settings | None = None,
    cross_validate: bool = True,
) -> pd.DataFrame:
    settings = settings or Settings()
    path = _cache_path(symbol, settings.data_dir)

    source = "yfinance"
    fetched: dict[str, pd.DataFrame] = {}
    if settings.alpaca_api_key and settings.alpaca_secret_key:
        try:
            fetched = _fetch_alpaca([symbol], start, end, settings)
            if symbol in fetched and not fetched[symbol].empty:
                source = "alpaca"
        except Exception:
            fetched = {}

    if symbol not in fetched or fetched.get(symbol) is None or fetched[symbol].empty:
        fetched = _fetch_yfinance([symbol], start, end)
        source = "yfinance"

    if symbol not in fetched or fetched[symbol].empty:
        msg = (
            f"No real historical data for {symbol} from Alpaca or yfinance. "
            "Check symbol, date range, and API credentials."
        )
        raise ValueError(msg)

    df = fetched[symbol]
    if cross_validate:
        try:
            yf_data = _fetch_yfinance([symbol], start, end).get(symbol)
            if yf_data is not None and not yf_data.empty and "close" in df.columns:
                common = df.index.intersection(yf_data.index)
                for ts in common[-5:]:
                    _log_discrepancy(
                        symbol,
                        "close",
                        ts.date(),
                        float(df.loc[ts, "close"]),
                        float(yf_data.loc[ts, "close"]),
                        settings.quality_dir,
                    )
        except Exception:
            pass

    merged = _merge_cache(_read_cache(path), df) if path.exists() else df
    cache_start = merged.index.min().date() if len(merged) else start
    cache_end = merged.index.max().date() if len(merged) else end
    _write_cache(merged, path, source=source, symbol=symbol, start=cache_start, end=cache_end)
    mask = (merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))
    return merged.loc[mask]


def fetch_universe(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    settings = Settings()
    return {sym: fetch_symbol(sym, start, end, settings) for sym in symbols}


def get_bars(req: BarRequest, data_dir: Path | None = None) -> pd.DataFrame:
    """Wide DataFrame with MultiIndex columns (symbol, field)."""
    settings = Settings()
    base = data_dir or settings.data_dir
    frames: dict[str, pd.DataFrame] = {}
    for sym in req.symbols:
        path = _cache_path(sym, base)
        if not path.exists():
            fetch_symbol(sym, req.start, req.end, settings)
        df = _read_cache(path)
        mask = (df.index >= pd.Timestamp(req.start)) & (df.index <= pd.Timestamp(req.end))
        frames[sym] = df.loc[mask]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)
