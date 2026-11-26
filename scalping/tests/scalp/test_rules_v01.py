"""Decision model v0.1 tests: day-type classification, trigger families,
core gate combination, boosters, vetoes."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bot.scalp.features import (
    AggBar,
    DailyOHLC,
    DayType,
    FeatureSnapshot,
    OpeningRange,
    StructureState,
    TimeWindow,
    build_feature_snapshots,
)
from bot.scalp.rules import Recommendation
from bot.scalp.rules_v01 import (
    CoreGateConfig,
    TriggerFamily,
    _confirming_candle,
    _core_bias,
    _core_context,
    _trig_failed_break_reversal,
    _trig_orb,
    _trig_rotation_fade,
    _trig_trend_pullback,
    _trig_vwap_midline,
    decide_core_gate,
)
from tests.scalp.helpers import bar, make_flat_session, session_open

ET = ZoneInfo("America/New_York")
DAY = date(2026, 6, 1)


def _agg5(close: float, ts_close_minute: int = 40, o=None, h=None, l=None, vol=1000) -> AggBar:
    o = 100.0 if o is None else o
    ts_close = session_open(DAY) + timedelta(minutes=ts_close_minute)
    return AggBar(
        ts_open=ts_close - timedelta(minutes=5),
        ts_close=ts_close,
        open=o,
        high=max(o, close) + 1.0 if h is None else h,
        low=min(o, close) - 1.0 if l is None else l,
        close=close,
        volume=vol,
        n_minutes=5,
    )


def base_snapshot(**overrides) -> FeatureSnapshot:
    decision_ts = datetime(2026, 6, 1, 10, 45, tzinfo=ET)  # main1, after the 10:30 day-type floor
    defaults = dict(
        decision_ts=decision_ts,
        idx=75,
        window=TimeWindow.main1,
        last_close=101.0,
        vwap=100.0,
        vwap_slope_5m=0.1,
        rvol_tod=2.0,
        structure_15m=StructureState.higher_highs_lows,
        opening_range=OpeningRange(high=100.5, low=99.0, width=1.5, width_vs_atr=0.5),
        prior_day=DailyOHLC(trade_date=date(2026, 5, 29), open=99.0, high=101.0, low=97.0, close=100.0),
        atr_14=1.0,
        bars_3m=[],
        bars_5m=[_agg5(101.0)],
        bars_15m=[],
        day_type=DayType.trend_up,
        vwap_sigma=1.0,
        ema9_5m=100.2,
        ema20_5m=99.8,
        ema9_3m=100.0,
    )
    defaults.update(overrides)
    return FeatureSnapshot(**defaults)


# ── day-type classification (via build_feature_snapshots) ──────────────────


def test_day_type_undecided_before_1030():
    # One-sided session (always above a flat VWAP-ish price), but the decision
    # point is still before 10:30 ET.
    closes = [100.0 + 0.01 * i for i in range(65)]  # 9:30 .. 10:34
    bars = [
        bar("SPY", session_open(DAY) + timedelta(minutes=i), closes[i] - 0.1, closes[i] + 0.2, closes[i] - 0.2, closes[i])
        for i in range(len(closes))
    ]
    snaps = build_feature_snapshots(bars, [])
    snap_1025 = next(s for s in snaps if s.decision_ts.time() == datetime(2026, 6, 1, 10, 25).time())
    assert snap_1025.day_type == DayType.undecided


def test_day_type_trend_up_when_one_sided():
    # Price ramps steadily up all session -> should sit above VWAP almost the
    # whole post-10:00 window -> trend_up by early afternoon.
    n = 240  # 9:30 .. 13:29
    closes = [100.0 + 0.05 * i for i in range(n)]
    bars = [
        bar("SPY", session_open(DAY) + timedelta(minutes=i), closes[i] - 0.05, closes[i] + 0.1, closes[i] - 0.1, closes[i])
        for i in range(n)
    ]
    snaps = build_feature_snapshots(bars, [])
    snap_late = next(s for s in snaps if s.decision_ts.time() == datetime(2026, 6, 1, 13, 0).time())
    assert snap_late.day_type == DayType.trend_up


def test_day_type_rotation_when_oscillating():
    # Oscillate tightly around 100 all session -> roughly balanced above/below
    # a slow-moving VWAP -> rotation by early afternoon.
    n = 240
    closes = [100.0 + (0.3 if i % 4 < 2 else -0.3) for i in range(n)]
    bars = [
        bar("SPY", session_open(DAY) + timedelta(minutes=i), closes[i] - 0.05, closes[i] + 0.4, closes[i] - 0.4, closes[i])
        for i in range(n)
    ]
    snaps = build_feature_snapshots(bars, [])
    snap_late = next(s for s in snaps if s.decision_ts.time() == datetime(2026, 6, 1, 13, 0).time())
    assert snap_late.day_type == DayType.rotation


# ── core gate: context + bias ───────────────────────────────────────────────


def test_core_context_matches_structure():
    snap = base_snapshot(structure_15m=StructureState.higher_highs_lows)
    assert _core_context(snap) == (True, False)
    snap2 = base_snapshot(structure_15m=StructureState.lower_highs_lows)
    assert _core_context(snap2) == (False, True)
    snap3 = base_snapshot(structure_15m=StructureState.mixed)
    assert _core_context(snap3) == (False, False)


def test_core_bias_uses_last_completed_5m_close_and_slope():
    long_ok, short_ok = _core_bias(base_snapshot(bars_5m=[_agg5(101.0)], vwap=100.0, vwap_slope_5m=0.1))
    assert (long_ok, short_ok) == (True, False)

    long_ok, short_ok = _core_bias(base_snapshot(bars_5m=[_agg5(99.0)], vwap=100.0, vwap_slope_5m=-0.1))
    assert (long_ok, short_ok) == (False, True)

    # Adverse slope kills the bias even though price is on the right side.
    long_ok, short_ok = _core_bias(base_snapshot(bars_5m=[_agg5(101.0)], vwap=100.0, vwap_slope_5m=-0.5))
    assert (long_ok, short_ok) == (False, False)


# ── trigger families ─────────────────────────────────────────────────────────


def test_trend_pullback_fires_on_hold_at_ema9_trend_up_only():
    # Bar dips into EMA9 (100.2) then closes back above it -> a hold.
    snap = base_snapshot(
        day_type=DayType.trend_up,
        ema9_5m=100.2,
        ema20_5m=99.5,
        vwap=98.0,
        vwap_sigma=0.5,
        bars_5m=[_agg5(101.0, o=100.5, h=101.5, l=100.0)],
    )
    long_ok, short_ok = _trig_trend_pullback(snap)
    assert long_ok is True
    assert short_ok is False

    # Same bar shape but trend_down day-type -> long side must not fire
    # (family is gated to the matching trend direction).
    snap_wrong_day = base_snapshot(day_type=DayType.trend_down, bars_5m=snap.bars_5m, ema9_5m=100.2, ema20_5m=99.5)
    long_ok2, _ = _trig_trend_pullback(snap_wrong_day)
    assert long_ok2 is False


def test_rotation_fade_only_on_rotation_days():
    # Bar pierces -2sigma (100 - 2*1 = 98) then closes back above it.
    snap = base_snapshot(day_type=DayType.rotation, vwap=100.0, vwap_sigma=1.0, bars_5m=[_agg5(98.5, o=99.0, h=99.5, l=97.5)])
    long_ok, short_ok = _trig_rotation_fade(snap)
    assert long_ok is True
    assert short_ok is False

    snap_trend = base_snapshot(day_type=DayType.trend_up, vwap=100.0, vwap_sigma=1.0, bars_5m=snap.bars_5m)
    assert _trig_rotation_fade(snap_trend) == (False, False)


def test_vwap_midline_reclaim_and_reject():
    snap_reclaim = base_snapshot(vwap=100.0, bars_5m=[_agg5(99.0, ts_close_minute=35), _agg5(101.0, ts_close_minute=40)])
    assert _trig_vwap_midline(snap_reclaim) == (True, False)

    snap_reject = base_snapshot(vwap=100.0, bars_5m=[_agg5(101.0, ts_close_minute=35), _agg5(99.0, ts_close_minute=40)])
    assert _trig_vwap_midline(snap_reject) == (False, True)


def test_orb_break_either_direction():
    orange = OpeningRange(high=100.5, low=99.0, width=1.5, width_vs_atr=0.5)
    snap_up = base_snapshot(opening_range=orange, bars_5m=[_agg5(101.0)])
    assert _trig_orb(snap_up) == (True, False)

    snap_down = base_snapshot(opening_range=orange, bars_5m=[_agg5(98.5)])
    assert _trig_orb(snap_down) == (False, True)


def test_failed_break_reversal_at_prior_day_levels():
    prior = DailyOHLC(trade_date=date(2026, 5, 29), open=99.0, high=101.0, low=97.0, close=100.0)
    # Long: pokes below prior-day low (97) then closes back above it.
    snap_long = base_snapshot(prior_day=prior, bars_5m=[_agg5(97.5, o=98.0, h=98.5, l=96.5)])
    assert _trig_failed_break_reversal(snap_long) == (True, False)

    # Short: pokes above prior-day high (101) then closes back below it.
    snap_short = base_snapshot(prior_day=prior, bars_5m=[_agg5(100.5, o=100.0, h=101.5, l=99.5)])
    assert _trig_failed_break_reversal(snap_short) == (False, True)


# ── boosters ─────────────────────────────────────────────────────────────────


def test_confirming_candle_requires_direction_and_ema9_side():
    bars3_long = [AggBar(session_open(DAY), session_open(DAY) + timedelta(minutes=3), open=100.0, high=100.6, low=99.9, close=100.5, volume=100, n_minutes=3)]
    snap = base_snapshot(bars_3m=bars3_long, ema9_3m=100.1)
    assert _confirming_candle(snap) == (True, False)

    bars3_flat = [AggBar(session_open(DAY), session_open(DAY) + timedelta(minutes=3), open=100.0, high=100.1, low=99.9, close=100.0, volume=100, n_minutes=3)]
    snap_flat = base_snapshot(bars_3m=bars3_flat, ema9_3m=100.1)
    assert _confirming_candle(snap_flat) == (False, False)


# ── decide_core_gate integration ────────────────────────────────────────────


def test_decide_core_gate_full_agreement_yields_long():
    snap = base_snapshot(
        structure_15m=StructureState.higher_highs_lows,
        vwap=98.0,
        vwap_sigma=0.5,
        vwap_slope_5m=0.1,
        day_type=DayType.trend_up,
        ema9_5m=100.2,
        ema20_5m=99.5,
        bars_5m=[_agg5(101.0, o=100.5, h=101.5, l=100.0)],  # bias long (101 > vwap 98) + trend pullback hold
    )
    cfg = CoreGateConfig(trigger_families=frozenset({TriggerFamily.trend_pullback}))
    d = decide_core_gate(snap, cfg)
    assert d.recommendation == Recommendation.long
    assert "trend_pullback" in d.long_confirmations
    assert d.vetoes == ()


def test_decide_core_gate_context_bias_conflict_yields_wait():
    # Structure says long (HH/HL) but VWAP bias says short.
    snap = base_snapshot(
        structure_15m=StructureState.higher_highs_lows,
        vwap=100.0,
        vwap_slope_5m=-0.5,
        bars_5m=[_agg5(99.0)],  # below vwap -> bias short, conflicts with structure long
    )
    cfg = CoreGateConfig()
    d = decide_core_gate(snap, cfg)
    assert d.recommendation == Recommendation.wait


def test_decide_core_gate_rvol_veto_floor_forces_wait():
    snap = base_snapshot(
        structure_15m=StructureState.higher_highs_lows,
        vwap=98.0,
        vwap_sigma=0.5,
        vwap_slope_5m=0.1,
        day_type=DayType.trend_up,
        ema9_5m=100.2,
        bars_5m=[_agg5(101.0, o=100.5, h=101.5, l=100.0)],
        rvol_tod=0.3,
    )
    cfg = CoreGateConfig(trigger_families=frozenset({TriggerFamily.trend_pullback}), rvol_veto_threshold=1.0)
    d = decide_core_gate(snap, cfg)
    assert d.recommendation == Recommendation.wait
    assert "rvol_below_veto_floor" in d.vetoes


def test_decide_core_gate_rvol_booster_blocks_when_below_threshold():
    snap = base_snapshot(
        structure_15m=StructureState.higher_highs_lows,
        vwap=98.0,
        vwap_sigma=0.5,
        vwap_slope_5m=0.1,
        day_type=DayType.trend_up,
        ema9_5m=100.2,
        bars_5m=[_agg5(101.0, o=100.5, h=101.5, l=100.0)],
        rvol_tod=1.2,  # above veto floor (1.0) but below the booster threshold
    )
    cfg = CoreGateConfig(trigger_families=frozenset({TriggerFamily.trend_pullback}), rvol_booster_threshold=1.5)
    d = decide_core_gate(snap, cfg)
    assert d.recommendation == Recommendation.wait  # booster required and not met


def test_decide_core_gate_trade_window_filter_toggle():
    snap = base_snapshot(
        window=TimeWindow.lunch,
        structure_15m=StructureState.higher_highs_lows,
        vwap=98.0,
        vwap_sigma=0.5,
        vwap_slope_5m=0.1,
        day_type=DayType.trend_up,
        ema9_5m=100.2,
        bars_5m=[_agg5(101.0, o=100.5, h=101.5, l=100.0)],
    )
    cfg_on = CoreGateConfig(trigger_families=frozenset({TriggerFamily.trend_pullback}), trade_window_filter_enabled=True)
    assert decide_core_gate(snap, cfg_on).recommendation == Recommendation.wait

    cfg_off = CoreGateConfig(trigger_families=frozenset({TriggerFamily.trend_pullback}), trade_window_filter_enabled=False)
    assert decide_core_gate(snap, cfg_off).recommendation == Recommendation.long


def test_decide_core_gate_no_enabled_trigger_family_never_fires():
    snap = base_snapshot(
        structure_15m=StructureState.higher_highs_lows,
        vwap=98.0,
        vwap_sigma=0.5,
        vwap_slope_5m=0.1,
        day_type=DayType.trend_up,
        ema9_5m=100.2,
        bars_5m=[_agg5(101.0, o=100.5, h=101.5, l=100.0)],
    )
    cfg = CoreGateConfig(trigger_families=frozenset())  # nothing enabled
    d = decide_core_gate(snap, cfg)
    assert d.recommendation == Recommendation.wait
