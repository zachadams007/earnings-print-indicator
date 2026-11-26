"""Decision model v0.3 tests: the ONE new trigger family (trend_pullback_hold,
wiring v01's already-tested _trig_trend_pullback through a v02-style single-
family core gate), plus confirmation that band_reversal / vwap_midline_retest
are reused UNCHANGED via decide_core_gate_v3, and the RVOL-filter-as-toggle
knob (rvol_veto_threshold=0.0 disables the veto; nonzero re-enables it)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from bot.scalp.features import (
    AggBar,
    DailyOHLC,
    DayType,
    FeatureSnapshot,
    OpeningRange,
    StructureState,
    TimeWindow,
)
from bot.scalp.rules import Recommendation
from bot.scalp.rules_v03 import CoreGateConfigV3, TriggerFamilyV3, decide_core_gate_v3
from tests.scalp.helpers import session_open

ET = ZoneInfo("America/New_York")
DAY = date(2026, 6, 1)


def _agg(close: float, minutes_offset: int, o=None, h=None, l=None, vol=1000, n_minutes=5) -> AggBar:
    o = 100.0 if o is None else o
    ts_close = session_open(DAY) + timedelta(minutes=minutes_offset)
    return AggBar(
        ts_open=ts_close - timedelta(minutes=n_minutes),
        ts_close=ts_close,
        open=o,
        high=max(o, close) + 1.0 if h is None else h,
        low=min(o, close) - 1.0 if l is None else l,
        close=close,
        volume=vol,
        n_minutes=n_minutes,
    )


def base_snapshot(**overrides) -> FeatureSnapshot:
    decision_ts = datetime(2026, 6, 1, 10, 45, tzinfo=ET)
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
        bars_1m=[],
        bars_3m=[],
        bars_5m=[],
        bars_15m=[],
        day_type=DayType.trend_up,
        vwap_sigma=1.0,
        ema9_5m=100.5,
        ema20_5m=100.2,
        ema9_3m=100.6,
    )
    defaults.update(overrides)
    return FeatureSnapshot(**defaults)


# ── trend_pullback_hold: the one genuinely new v0.3 family ─────────────────


def test_trend_pullback_hold_fires_long_on_trend_up_pullback_to_ema20():
    # Last completed 5m bar's range touches EMA20 (100.2) and closes back
    # above it on a trend_up day, with context/bias agreeing (longs favored).
    last = _agg(100.4, 45, o=100.6, h=100.7, l=100.1)
    snap = base_snapshot(bars_5m=[last], day_type=DayType.trend_up, ema20_5m=100.2, ema9_5m=100.6)
    cfg = CoreGateConfigV3(family=TriggerFamilyV3.trend_pullback_hold)
    d = decide_core_gate_v3(snap, cfg)
    assert d.recommendation == Recommendation.long
    assert "trend_pullback_hold" in d.long_confirmations


def test_trend_pullback_hold_requires_core_context_and_bias_agreement():
    # Trigger geometry is identical to the passing test above, but structure
    # disagrees (LH/LL) -- core gate must refuse even though the pullback
    # trigger itself would fire.
    last = _agg(100.4, 45, o=100.6, h=100.7, l=100.1)
    snap = base_snapshot(
        bars_5m=[last],
        day_type=DayType.trend_up,
        ema20_5m=100.2,
        ema9_5m=100.6,
        structure_15m=StructureState.lower_highs_lows,
    )
    cfg = CoreGateConfigV3(family=TriggerFamilyV3.trend_pullback_hold)
    d = decide_core_gate_v3(snap, cfg)
    assert d.recommendation == Recommendation.wait


def test_trend_pullback_hold_does_not_fire_on_a_rotation_day():
    # _trig_trend_pullback (reused from rules_v01) is day-type-gated to
    # trend_up/trend_down internally -- a rotation day must never fire it.
    last = _agg(100.4, 45, o=100.6, h=100.7, l=100.1)
    snap = base_snapshot(bars_5m=[last], day_type=DayType.rotation, ema20_5m=100.2, ema9_5m=100.6)
    cfg = CoreGateConfigV3(family=TriggerFamilyV3.trend_pullback_hold)
    d = decide_core_gate_v3(snap, cfg)
    assert d.recommendation == Recommendation.wait


# ── band_reversal / vwap_midline_retest: reused unchanged from rules_v02 ────


def test_band_reversal_reused_bypasses_context_and_bias_like_v02():
    history = [_agg(97.0, 30 + 3 * i, vol=500, n_minutes=3) for i in range(6)]
    reversal_bar = _agg(98.5, 48, o=97.5, h=98.6, l=97.8, vol=1500, n_minutes=3)
    snap = base_snapshot(
        vwap=100.0,
        vwap_sigma=1.0,
        vwap_slope_5m=-0.5,  # falling -- would refuse a long under a bias-gated family
        structure_15m=StructureState.lower_highs_lows,  # disagrees with a long -- irrelevant to band_reversal
        day_type=DayType.trend_down,
        bars_3m=history + [reversal_bar],
    )
    cfg = CoreGateConfigV3(family=TriggerFamilyV3.band_reversal, band_reversal_confirm_tf="3m")
    d = decide_core_gate_v3(snap, cfg)
    assert d.recommendation == Recommendation.long
    assert "band_reversal" in d.long_confirmations


def test_midline_retest_reused_still_requires_core_context_and_bias():
    bars = [
        _agg(99.0, 35, n_minutes=5),
        _agg(101.0, 40, n_minutes=5),
        _agg(100.3, 45, o=100.5, h=100.6, l=99.8, n_minutes=5),
    ]
    snap = base_snapshot(
        vwap=100.0,
        vwap_slope_5m=0.1,
        structure_15m=StructureState.lower_highs_lows,  # disagrees with the long retest signal
        bars_5m=bars,
    )
    cfg = CoreGateConfigV3(family=TriggerFamilyV3.vwap_midline_retest)
    d = decide_core_gate_v3(snap, cfg)
    assert d.recommendation == Recommendation.wait


# ── RVOL-filter-as-toggle (rvol_veto_threshold 0.0 = off, 1.0 = on) ─────────


def test_rvol_veto_threshold_zero_disables_the_filter():
    last = _agg(100.4, 45, o=100.6, h=100.7, l=100.1)
    snap = base_snapshot(bars_5m=[last], day_type=DayType.trend_up, ema20_5m=100.2, ema9_5m=100.6, rvol_tod=0.1)

    cfg_on = CoreGateConfigV3(family=TriggerFamilyV3.trend_pullback_hold, rvol_veto_threshold=1.0)
    d_on = decide_core_gate_v3(snap, cfg_on)
    assert d_on.recommendation == Recommendation.wait
    assert "rvol_below_veto_floor" in d_on.vetoes

    cfg_off = CoreGateConfigV3(family=TriggerFamilyV3.trend_pullback_hold, rvol_veto_threshold=0.0)
    d_off = decide_core_gate_v3(snap, cfg_off)
    assert d_off.recommendation == Recommendation.long
    assert d_off.vetoes == ()
