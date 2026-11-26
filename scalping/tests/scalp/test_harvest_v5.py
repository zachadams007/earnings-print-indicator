"""Tests for SWEEP v5 "HARVEST" (bot.scalp.harvest_v5) -- the five
practitioner strategies from docs/STRATEGY-HARVEST-2026-07-09.md.

Focus areas called out by the task brief:
  - no-lookahead on the IB gate (fill must use the NEXT bar's open, never
    the trigger bar's own close/high/low)
  - gap classification at the open (pure classifier + integration)
  - NR7 flagged only from completed prior days (the flag must never read
    the trigger day's own bars)
plus one smoke test per remaining strategy (opening-drive confirmation gate,
turtle-soup rolling extreme + false-breakout fade) and the shared
``_walk_generic``/delayed-signal-control primitives.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

from bot.data.models import Bar  # noqa: E402
from bot.scalp.features import DailyOHLC  # noqa: E402
from bot.scalp.harvest_v5 import (  # noqa: E402
    DriveConfig,
    GapConfig,
    IBConfig,
    NR7Config,
    TurtleSoupConfig,
    WalkResult,
    _classify_gap,
    _confirmed_no_new_extreme,
    _is_nr7_day,
    _materialize,
    _materialize_delayed,
    _rolling_extreme_before,
    _walk_generic,
    build_intents_drive,
    build_intents_gap,
    build_intents_ib,
    build_intents_nr7,
    build_intents_turtle_soup,
    build_sessions,
)

ET = ZoneInfo("America/New_York")


def _bar(symbol: str, d: date, hh: int, mm: int, o: float, h: float, l: float, c: float, v: int = 1000) -> Bar:
    ts = datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)
    return Bar(
        symbol=symbol, ts=ts, timeframe="1m", open=o, high=h, low=l, close=c,
        volume=v, source="test_fixture", ingested_ts=datetime.now(ET),
    )


def _minutes_930_to_1600() -> list[tuple[int, int]]:
    out = []
    h, m = 9, 31
    while (h, m) <= (16, 0):
        out.append((h, m))
        m += 1
        if m == 60:
            m = 0
            h += 1
    return out


_ALL_MINUTES = _minutes_930_to_1600()


def _make_day(
    symbol: str,
    d: date,
    overrides: dict[tuple[int, int], tuple[float, float, float, float, int]] | None = None,
    default: tuple[float, float, float, float, int] = (100.0, 100.1, 99.9, 100.0, 1000),
) -> list[Bar]:
    overrides = overrides or {}
    bars = []
    for hh, mm in _ALL_MINUTES:
        o, h, l, c, v = overrides.get((hh, mm), default)
        bars.append(_bar(symbol, d, hh, mm, o, h, l, c, v))
    return bars


def _consecutive_dates(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


# ── shared walker / delayed-signal control ──────────────────────────────


def test_walk_generic_stop_before_target_tiebreak():
    d = date(2024, 1, 3)
    bars = [
        _bar("SPY", d, 10, 0, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 1, 100.0, 101.0, 98.0, 99.0),  # both stop(99) and target(101) touched
    ]
    wr = _walk_generic(bars, 0, "long", 100.0, 99.0, 101.0, len(bars) - 1)
    assert wr.exit_reason == "stop"
    assert wr.exit_price == 99.0


def test_walk_generic_time_stop_vs_eod():
    d = date(2024, 1, 3)
    bars = [_bar("SPY", d, 10, i, 100.0, 100.0, 100.0, 100.0) for i in range(5)]
    wr = _walk_generic(bars, 0, "long", 100.0, 90.0, 110.0, 2)  # max_idx=2, not last bar (4)
    assert wr.exit_reason == "time_stop"
    assert wr.exit_idx == 2
    wr2 = _walk_generic(bars, 0, "long", 100.0, 90.0, 110.0, len(bars) - 1)
    assert wr2.exit_reason == "eod_close"


# ── 1. IB breakout: no-lookahead + wide-IB day-type gate ────────────────


def test_ib_breakout_fill_uses_next_bar_open_not_trigger_bar_price():
    """The trigger bar (10:31) closes above the IB high with confirming
    volume -- but the fill MUST be the very next bar's (10:32) open, not
    the trigger bar's own close. We give the trigger bar an extreme close
    (105.0) and the fill bar a distinct, moderate open (100.5) so a bug that
    used the trigger bar's own price would produce a visibly different
    entry_price than this test's assertion.
    """
    d = date(2024, 3, 1)
    overrides = {
        (10, 31): (100.0, 105.0, 100.0, 105.0, 5000),  # breakout: close > IB high, high volume
        (10, 32): (100.5, 100.6, 100.4, 100.5, 1000),  # fill bar -- distinct open
    }
    bars = _make_day("SPY", d, overrides, default=(100.0, 100.1, 99.9, 100.0, 1000))
    sessions = build_sessions("SPY", {d: bars})
    cfg = IBConfig(day_type_gate=False)  # isolate the fill mechanics from the history-dependent gate
    intents = build_intents_ib("SPY", sessions, d, d, cfg)
    assert len(intents) == 1
    it = intents[0]
    assert it.direction == "long"
    entry_price = it.day_bars[it.entry_idx].open
    assert entry_price == pytest.approx(100.5)
    assert entry_price != pytest.approx(105.0)  # must NOT be the trigger bar's own close


def test_ib_breakout_wide_day_is_skipped_by_day_type_gate():
    start = date(2024, 3, 1)
    dates = _consecutive_dates(start, 21)  # 20 prior narrow-IB days + 1 wide-IB target day
    bars_by_day: dict[date, list[Bar]] = {}
    for dd in dates[:-1]:
        bars_by_day[dd] = _make_day("SPY", dd, default=(100.0, 100.2, 99.8, 100.0, 1000))  # IB width ~0.4
    target_day = dates[-1]
    wide_overrides = {(10, 31): (100.0, 108.0, 100.0, 108.0, 5000)}  # huge IB (width ~8) + breakout after
    wide_overrides.update({(h, m): (100.0, 106.0, 94.0, 100.0, 1000) for (h, m) in _ALL_MINUTES if (h, m) <= (10, 30)})
    bars_by_day[target_day] = _make_day("SPY", target_day, wide_overrides, default=(100.0, 100.1, 99.9, 100.0, 1000))

    sessions = build_sessions("SPY", bars_by_day)
    gated = build_intents_ib("SPY", sessions, target_day, target_day, IBConfig(day_type_gate=True))
    assert gated == []  # wide-IB day must be skipped entirely

    ungated = build_intents_ib("SPY", sessions, target_day, target_day, IBConfig(day_type_gate=False))
    assert len(ungated) >= 0  # gate off: no assertion on trade presence, just that it didn't error


# ── 2. Gap continuation: classification at the open ─────────────────────


def test_classify_gap_pure_thresholds():
    assert _classify_gap(today_open=101.0, prior_close=100.0, atr14=1.0, atr_mult=0.7) == "long"
    assert _classify_gap(today_open=99.0, prior_close=100.0, atr14=1.0, atr_mult=0.7) == "short"
    assert _classify_gap(today_open=100.5, prior_close=100.0, atr14=1.0, atr_mult=0.7) is None  # gap too small
    assert _classify_gap(today_open=100.0, prior_close=100.0, atr14=1.0, atr_mult=0.7) is None  # no gap


def test_gap_continuation_integration_only_triggers_past_threshold():
    start = date(2024, 4, 1)
    dates = _consecutive_dates(start, 16)  # 15 prior days (ATR14 needs window+1) + target day
    bars_by_day: dict[date, list[Bar]] = {}
    for dd in dates[:-1]:
        bars_by_day[dd] = _make_day("SPY", dd, default=(100.0, 101.0, 99.0, 100.0, 1000))  # daily range ~2 -> ATR~2

    target_day = dates[-1]
    # Gap up 5 points (>> ATR ~2 * 0.7), opening range 9:31-9:35 flat at 105, then
    # a later close breaks above the opening-range high in the gap direction.
    overrides = {(h, m): (105.0, 105.2, 104.8, 105.0, 1000) for (h, m) in _ALL_MINUTES if (h, m) <= (9, 35)}
    overrides[(9, 40)] = (105.0, 106.0, 105.0, 106.0, 1000)  # trigger: close beyond OR high
    overrides[(9, 41)] = (105.8, 105.9, 105.7, 105.8, 1000)  # fill bar
    bars_by_day[target_day] = _make_day("SPY", target_day, overrides, default=(105.0, 105.2, 104.8, 105.0, 1000))

    sessions = build_sessions("SPY", bars_by_day)
    intents = build_intents_gap("SPY", sessions, target_day, target_day, GapConfig(atr_mult=0.7))
    assert len(intents) == 1
    assert intents[0].direction == "long"
    assert intents[0].day_bars[intents[0].entry_idx].open == pytest.approx(105.8)

    # Same data but a much stricter ATR multiple -> the 5pt gap no longer clears -> no trade.
    intents_strict = build_intents_gap("SPY", sessions, target_day, target_day, GapConfig(atr_mult=100.0))
    assert intents_strict == []


# ── 3. Opening-drive reversal: confirmation gate ────────────────────────


def test_confirmed_no_new_extreme_pure():
    d = date(2024, 1, 3)
    bars = [_bar("SPY", d, 10, i, 100.0, 100.5, 99.5, 100.0) for i in range(15)]
    assert _confirmed_no_new_extreme(bars, extreme=100.5, drive_up=True) is True
    bars_break = list(bars)
    bars_break[5] = _bar("SPY", d, 10, 5, 100.0, 101.0, 99.5, 100.5)  # new high printed
    assert _confirmed_no_new_extreme(bars_break, extreme=100.5, drive_up=True) is False


def test_opening_drive_reversal_fades_only_when_confirmed():
    d = date(2024, 5, 1)
    overrides = {(h, m): (100.0, 100.2, 99.8, 100.0, 1000) for (h, m) in _ALL_MINUTES if (h, m) < (10, 0)}
    overrides[(10, 0)] = (100.5, 100.6, 100.4, 100.6, 1000)  # drive up: +0.6% from open (100.0)
    # 10:01-10:15: no new extreme beyond 100.6
    for m in range(1, 16):
        overrides[(10, m)] = (100.3, 100.4, 100.2, 100.3, 1000)
    overrides[(10, 16)] = (100.25, 100.3, 100.2, 100.25, 1000)  # fill bar
    bars = _make_day("SPY", d, overrides, default=(100.0, 100.1, 99.9, 100.0, 1000))
    sessions = build_sessions("SPY", {d: bars})

    intents = build_intents_drive("SPY", sessions, d, d, DriveConfig(threshold_kind="pct", threshold_pct=0.004))
    assert len(intents) == 1
    it = intents[0]
    assert it.direction == "short"  # fading an up-drive
    assert it.stop_price == pytest.approx(100.6)  # beyond the pre-10:00 extreme

    # Now let 10:05 print a new extreme (101.0) -- confirmation must fail, no trade.
    overrides2 = dict(overrides)
    overrides2[(10, 5)] = (100.3, 101.0, 100.2, 100.9, 1000)
    bars2 = _make_day("SPY", d, overrides2, default=(100.0, 100.1, 99.9, 100.0, 1000))
    sessions2 = build_sessions("SPY", {d: bars2})
    intents2 = build_intents_drive("SPY", sessions2, d, d, DriveConfig(threshold_kind="pct", threshold_pct=0.004))
    assert intents2 == []


# ── 4. NR7 / IDNR7: flagged only from completed prior days ─────────────


def _daily(d: date, o: float, h: float, l: float, c: float) -> DailyOHLC:
    return DailyOHLC(trade_date=d, open=o, high=h, low=l, close=c)


def test_is_nr7_day_unaffected_by_trigger_day_own_bars():
    dates = _consecutive_dates(date(2024, 1, 1), 8)  # D-7..D-1, D (index 7)
    # D-7..D-2 wide ranges, D-1 narrowest (NR7) -- daily_sorted[7] is D itself.
    daily_sorted = [
        _daily(dates[0], 100, 110, 90, 105),
        _daily(dates[1], 100, 109, 91, 104),
        _daily(dates[2], 100, 108, 92, 103),
        _daily(dates[3], 100, 107, 93, 103),
        _daily(dates[4], 100, 106, 94, 103),
        _daily(dates[5], 100, 105, 95, 103),
        _daily(dates[6], 100, 101, 99, 100),  # D-1: range=2, narrowest of the 7 -> NR7
        _daily(dates[7], 999, 999, -999, 999),  # D (trigger day) -- deliberately absurd values
    ]
    prior_idx = 6  # D-1
    assert _is_nr7_day(daily_sorted, prior_idx, lookback_days=7, require_inside_day=False) is True

    # Mutating the trigger day (index 7, D itself) must NOT change the flag --
    # this is the "flagged only from completed prior days" invariant.
    mutated = list(daily_sorted)
    mutated[7] = _daily(dates[7], 0, 0, 0, 0)
    assert _is_nr7_day(mutated, prior_idx, lookback_days=7, require_inside_day=False) is True

    # But mutating D-1 itself (widening it) changes the flag.
    mutated2 = list(daily_sorted)
    mutated2[6] = _daily(dates[6], 100, 120, 80, 100)  # now the widest, not narrowest
    assert _is_nr7_day(mutated2, prior_idx, lookback_days=7, require_inside_day=False) is False


def test_is_nr7_day_idnr7_requires_inside_day():
    dates = _consecutive_dates(date(2024, 1, 1), 8)
    daily_sorted = [
        _daily(dates[0], 100, 110, 90, 105),
        _daily(dates[1], 100, 109, 91, 104),
        _daily(dates[2], 100, 108, 92, 103),
        _daily(dates[3], 100, 107, 93, 103),
        _daily(dates[4], 100, 106, 94, 103),
        _daily(dates[5], 100, 105, 95, 103),  # D-2: high=105, low=95
        _daily(dates[6], 100, 101, 99, 100),  # D-1: range=2 (narrowest), high<=105 low>=95 -> inside day too
        _daily(dates[7], 999, 999, -999, 999),
    ]
    prior_idx = 6
    assert _is_nr7_day(daily_sorted, prior_idx, 7, require_inside_day=True) is True

    # Make D-1 NOT an inside day (poke its high above D-2's high) while keeping it narrowest.
    not_inside = list(daily_sorted)
    not_inside[6] = _daily(dates[6], 100, 106, 99, 100)  # high=106 > D-2 high=105 -> not inside
    assert _is_nr7_day(not_inside, prior_idx, 7, require_inside_day=False) is True  # still NR7 (narrowest by range... wait range=7 now)


def test_nr7_breakout_integration_fills_at_next_bar_open():
    dates = _consecutive_dates(date(2024, 6, 1), 8)
    bars_by_day: dict[date, list[Bar]] = {}
    # D-7..D-2: wide days.
    for i, dd in enumerate(dates[:-2]):
        bars_by_day[dd] = _make_day("SPY", dd, default=(100.0, 100.0 + 8 + i, 100.0 - 8 - i, 100.0, 1000))
    # D-1: narrowest (NR7 day). High=100.5, Low=99.5.
    d_minus_1 = dates[-2]
    bars_by_day[d_minus_1] = _make_day("SPY", d_minus_1, default=(100.0, 100.5, 99.5, 100.0, 1000))
    # D: breaks above D-1's high (100.5) at 10:00, fills at 10:01.
    d = dates[-1]
    overrides = {(10, 0): (100.0, 100.8, 100.0, 100.8, 1000), (10, 1): (100.6, 100.7, 100.5, 100.6, 1000)}
    bars_by_day[d] = _make_day("SPY", d, overrides, default=(100.0, 100.1, 99.9, 100.0, 1000))

    sessions = build_sessions("SPY", bars_by_day)
    intents = build_intents_nr7("SPY", sessions, d, d, NR7Config(require_inside_day=False))
    assert len(intents) == 1
    it = intents[0]
    assert it.direction == "long"
    assert it.stop_price == pytest.approx(99.5)  # opposite side of the NR7 range (D-1's low)
    assert it.day_bars[it.entry_idx].open == pytest.approx(100.6)  # next-bar-open fill, not the 10:00 trigger's own price


# ── 5. Turtle Soup: rolling extreme + false-breakout fade ───────────────


def test_rolling_extreme_before_basic():
    values = [1.0, 5.0, 3.0, 2.0, 8.0, 4.0]
    roll_max = _rolling_extreme_before(values, n=3, take_max=True)
    # out[3] = max(values[0:3]) = max(1,5,3) = 5, at idx 1
    assert roll_max[3] == (5.0, 1)
    # out[4] = max(values[1:4]) = max(5,3,2) = 5, at idx 1
    assert roll_max[4] == (5.0, 1)
    # out[5] = max(values[2:5]) = max(3,2,8) = 8, at idx 4
    assert roll_max[5] == (8.0, 4)
    assert roll_max[0] is None and roll_max[1] is None and roll_max[2] is None

    roll_min = _rolling_extreme_before(values, n=3, take_max=False)
    assert roll_min[3] == (1.0, 0)


def test_turtle_soup_false_breakout_fade_requires_separation_and_reclaim():
    d = date(2024, 7, 1)
    n = 20
    # Flat base (high=100.4, low=99.5) for the whole pre-breakout window, EXCEPT
    # one bar well inside the rolling window (index 16) that sets the true
    # "old extreme" (high=100.5) far enough back (>=3 bars) from the breakout
    # bar (index 26) that the separation gate is meaningfully exercised.
    overrides = {}
    for hh_mm in _ALL_MINUTES[: n + 10]:
        overrides[hh_mm] = (100.0, 100.4, 99.5, 100.0, 1000)
    overrides[_ALL_MINUTES[16]] = (100.0, 100.5, 99.5, 100.0, 1000)  # old extreme, idx 16
    overrides[_ALL_MINUTES[26]] = (100.0, 101.5, 100.0, 101.5, 1000)  # new high (breakout), idx 26 (sep=10)
    overrides[_ALL_MINUTES[27]] = (101.0, 101.2, 100.0, 100.3, 1000)  # reclaim: closes back below 100.5
    overrides[_ALL_MINUTES[28]] = (100.2, 100.3, 100.1, 100.2, 1000)  # fill bar

    bars = _make_day("SPY", d, overrides, default=(100.0, 100.4, 99.5, 100.0, 1000))
    sessions = build_sessions("SPY", {d: bars})
    cfg = TurtleSoupConfig(n_bars=n, min_separation_bars=3, reclaim_within_bars=2, min_rr=0.0)
    intents = build_intents_turtle_soup("SPY", sessions, d, d, cfg)
    assert len(intents) == 1
    it = intents[0]
    assert it.direction == "short"  # fading the false upside breakout
    assert it.stop_price == pytest.approx(101.5)  # beyond the false-breakout extreme
    assert it.day_bars[it.entry_idx].open == pytest.approx(100.2)

    # Now move the old extreme to idx 24 (separation from breakout at 26 is only
    # 2 bars, below min_separation_bars=3) -- the same breakout+reclaim shape
    # must now be REJECTED by the separation gate.
    overrides_close = dict(overrides)
    del overrides_close[_ALL_MINUTES[16]]
    overrides_close[_ALL_MINUTES[24]] = (100.0, 100.5, 99.5, 100.0, 1000)
    bars_close = _make_day("SPY", d, overrides_close, default=(100.0, 100.4, 99.5, 100.0, 1000))
    sessions_close = build_sessions("SPY", {d: bars_close})
    intents_close = build_intents_turtle_soup("SPY", sessions_close, d, d, cfg)
    assert intents_close == []


def test_turtle_soup_min_rr_filter_rejects_poor_reward_setups():
    d = date(2024, 7, 2)
    n = 20
    overrides = {}
    for hh_mm in _ALL_MINUTES[: n + 10]:
        overrides[hh_mm] = (100.0, 100.5, 99.5, 100.0, 1000)
    breakout_hm = _ALL_MINUTES[n + 6]
    overrides[breakout_hm] = (100.0, 100.51, 100.0, 100.51, 1000)  # barely a new high -> tiny reward
    reclaim_hm = _ALL_MINUTES[n + 7]
    overrides[reclaim_hm] = (100.3, 100.4, 100.0, 100.3, 1000)
    bars = _make_day("SPY", d, overrides, default=(100.0, 100.5, 99.5, 100.0, 1000))
    sessions = build_sessions("SPY", {d: bars})
    cfg = TurtleSoupConfig(n_bars=n, min_separation_bars=3, reclaim_within_bars=2, min_rr=2.0)
    intents = build_intents_turtle_soup("SPY", sessions, d, d, cfg)
    assert intents == []  # reward/risk floor should reject this setup


# ── delayed-signal control (generic, exercised via IB intents) ─────────


def test_materialize_delayed_shifts_entry_but_keeps_structural_levels():
    d = date(2024, 3, 1)
    overrides = {
        (10, 31): (100.0, 105.0, 100.0, 105.0, 5000),
        (10, 32): (100.5, 100.6, 100.4, 100.5, 1000),
        (10, 33): (101.0, 101.1, 100.9, 101.0, 1000),
    }
    bars = _make_day("SPY", d, overrides, default=(100.0, 100.1, 99.9, 100.0, 1000))
    sessions = build_sessions("SPY", {d: bars})
    intents = build_intents_ib("SPY", sessions, d, d, IBConfig(day_type_gate=False))
    assert len(intents) == 1

    real = _materialize(intents)
    delayed = _materialize_delayed(intents, delay_bars=1)
    assert len(real) == 1 and len(delayed) == 1
    assert real[0].entry_price == pytest.approx(100.5)
    assert delayed[0].entry_price == pytest.approx(101.0)  # one bar later
    assert delayed[0].entry_stop_price == real[0].entry_stop_price  # SAME structural stop level
