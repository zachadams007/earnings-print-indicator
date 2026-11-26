"""Synthetic 1-minute RTH bar fixtures for Stage A tests and the runner
script's no-data fallback.

NOT real market data. Used only to (a) exercise the pipeline end-to-end in
unit tests without a live SQLite store, and (b) let
scripts/run_scalp_stage_a.py produce a smoke-test report before SPY/QQQ 1m
bars have landed in data/sqlite/scalp.db.
"""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from ..data.models import Bar

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def make_session_bars(
    symbol: str,
    trade_date: date,
    *,
    base_price: float = 450.0,
    n_minutes: int = 390,
    seed: int = 0,
    drift_bps_per_min: float = 0.0,
    vol_bps: float = 8.0,
    base_volume: int = 50_000,
) -> list[Bar]:
    """Generate one synthetic RTH session (9:30 ET start) of ``n_minutes`` 1m bars.

    A simple random walk with optional per-minute drift. Volume gets a crude
    opening spike and midday lull so RVOL-TOD has something to differentiate.
    """
    rng = random.Random(seed ^ hash((symbol, trade_date)))
    session_open = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, tzinfo=ET)
    price = base_price
    bars: list[Bar] = []
    now = datetime.now(UTC)

    for i in range(n_minutes):
        ts = session_open + timedelta(minutes=i)
        ret = drift_bps_per_min / 10_000.0 + rng.gauss(0, vol_bps / 10_000.0)
        o = price
        price = max(price * (1 + ret), 0.01)
        c = price
        wick = abs(rng.gauss(0, vol_bps / 20_000.0))
        h = max(o, c) * (1 + wick)
        l = min(o, c) * (1 - wick)

        if i < 15:
            vol_mult = 2.5
        elif 120 < i < 240:
            vol_mult = 0.5
        else:
            vol_mult = 1.0
        volume = max(1, int(base_volume * vol_mult * (1 + rng.gauss(0, 0.2))))

        bars.append(
            Bar(
                symbol=symbol,
                ts=ts,
                timeframe="1m",
                open=round(o, 4),
                high=round(h, 4),
                low=round(l, 4),
                close=round(c, 4),
                volume=volume,
                source="synthetic_fixture",
                ingested_ts=now,
            )
        )
    return bars


def make_multi_day_sessions(
    symbol: str,
    start_date: date,
    n_days: int,
    *,
    seed: int = 0,
    **kwargs,
) -> dict[date, list[Bar]]:
    """``n_days`` consecutive weekday sessions starting at ``start_date`` (skips weekends)."""
    out: dict[date, list[Bar]] = {}
    d = start_date
    count = 0
    while count < n_days:
        if d.weekday() < 5:
            out[d] = make_session_bars(symbol, d, seed=seed + count, **kwargs)
            count += 1
        d = d + timedelta(days=1)
    return out
