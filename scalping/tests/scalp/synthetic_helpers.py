"""Deterministic 1m session fixtures for sweep-v2 integration tests
(tests/scalp/test_evaluate_v2.py) -- not used by the sweep script itself."""
from __future__ import annotations

from datetime import date

from bot.data.models import Bar

from .helpers import bar, session_open


def make_band_touch_session(symbol: str, trade_date: date, n_minutes: int = 200) -> list[Bar]:
    """A mostly-flat session with one sharp, high-volume dip-and-recover around
    minute 60 (well inside the 9:45-11:30 main1 trade window), deep enough to
    pierce any plausible +-2sigma VWAP band and close back near VWAP -- built
    to reliably fire the band-reversal trigger regardless of the exact
    cumulative-sigma value, without hand-computing it.
    """
    start = session_open(trade_date)
    bars: list[Bar] = []
    price = 450.0
    for i in range(n_minutes):
        if i == 60:
            o, h, l, c, vol = 450.0, 450.2, 300.0, 300.0, 5000
        elif i == 61:
            o, h, l, c, vol = 300.0, 300.2, 250.0, 250.0, 6000
        elif i == 62:
            o, h, l, c, vol = 250.0, 446.0, 249.5, 445.0, 7000
        else:
            # tiny deterministic wobble so VWAP/sigma aren't perfectly degenerate
            wobble = 0.05 if i % 2 == 0 else -0.05
            o, h, l, c, vol = price, price + 0.2, price - 0.2, price + wobble, 1000
            price = c
        bars.append(bar(symbol, start.replace() + _minutes(i), o, h, l, c, vol))
    return bars


def _minutes(i: int):
    from datetime import timedelta

    return timedelta(minutes=i)
