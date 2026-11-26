from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bot.data.models import Bar
from bot.scalp.pattern_lab import (
    ResearchSignal,
    build_session_points,
    evaluate_signal,
    rsi_allows,
    summarize,
    wilder_rsi,
)

ET = ZoneInfo("America/New_York")


def _bars(closes):
    start = datetime(2025, 1, 2, 9, 31, tzinfo=ET)
    out = []
    prev = closes[0]
    for i, close in enumerate(closes):
        out.append(
            Bar(
                symbol="IWM",
                ts=start + timedelta(minutes=i),
                timeframe="1m",
                open=prev,
                high=max(prev, close) + 0.01,
                low=min(prev, close) - 0.01,
                close=close,
                volume=1000 + i,
                source="test",
                ingested_ts=start,
            )
        )
        prev = close
    return out


def test_wilder_rsi_uses_full_range_for_monotonic_paths():
    assert wilder_rsi(list(range(30)), 14)[-1] == 100.0
    assert wilder_rsi(list(range(30, 0, -1)), 14)[-1] == 0.0


def test_wilder_rsi_validates_period():
    with pytest.raises(ValueError):
        wilder_rsi([1.0, 2.0], 1)


def test_session_points_are_point_in_time():
    bars = _bars([100.0 + i for i in range(25)])
    points = build_session_points(bars)
    assert points[10].session_high == bars[10].high
    assert points[10].session_high < points[-1].session_high
    assert points[-1].rsi == 100.0


def test_rsi_extreme_and_hook_variants():
    points = build_session_points(_bars([100.0] * 15 + [99, 98, 97, 96, 95, 110]))
    assert rsi_allows(points[-2], "long", "extreme70")
    assert rsi_allows(points[-1], "long", "hook70")


def test_signal_fills_next_bar_and_delay_moves_fill():
    bars = _bars([100, 100, 110, 120, 130])
    signal = ResearchSignal("x", "x1", "IWM", bars[0].ts.date(), 1, bars[1].ts.time(), "long", 2)
    normal = evaluate_signal(signal, bars)
    delayed = evaluate_signal(signal, bars, extra_delay_minutes=1)
    assert normal is not None and delayed is not None
    assert normal.gross_bps > delayed.gross_bps


def test_summary_charges_round_trip_cost():
    bars = _bars([100, 100, 100.1, 100.2])
    signal = ResearchSignal("x", "x1", "IWM", bars[0].ts.date(), 0, bars[0].ts.time(), "long", 2)
    trade = evaluate_signal(signal, bars)
    summary = summarize([trade])
    assert summary.net_bps == pytest.approx(summary.gross_bps - 2.0)
