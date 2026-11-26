"""Shared deterministic bar-building helpers for tests/scalp/*."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from bot.data.models import Bar

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def bar(symbol: str, ts: datetime, o: float, h: float, l: float, c: float, v: int = 1000) -> Bar:
    return Bar(
        symbol=symbol,
        ts=ts,
        timeframe="1m",
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        source="test_fixture",
        ingested_ts=datetime.now(UTC),
    )


def session_open(trade_date: date) -> datetime:
    return datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, tzinfo=ET)


def make_flat_session(
    symbol: str,
    trade_date: date,
    n_minutes: int,
    price: float = 100.0,
    volume: int = 1000,
) -> list[Bar]:
    """A boring, perfectly flat session - every bar OHLC == price. Useful as a
    deterministic base for tests that only care about volume/derivation math.
    """
    start = session_open(trade_date)
    return [
        bar(symbol, start + timedelta(minutes=i), price, price, price, price, volume)
        for i in range(n_minutes)
    ]


def make_walk_session(
    symbol: str,
    trade_date: date,
    closes: list[float],
    volume: int = 1000,
) -> list[Bar]:
    """Build a session where bar i's OHLC is derived from a caller-supplied close
    path: open = previous close (or first close), high/low pad +-0.5, close = closes[i].
    """
    start = session_open(trade_date)
    bars = []
    prev_close = closes[0]
    for i, c in enumerate(closes):
        o = prev_close
        h = max(o, c) + 0.5
        l = min(o, c) - 0.5
        bars.append(bar(symbol, start + timedelta(minutes=i), o, h, l, c, volume))
        prev_close = c
    return bars
