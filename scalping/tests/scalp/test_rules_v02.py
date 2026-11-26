"""Decision model v0.2 (sweep v2) tests: band-reversal fires against the
current VWAP side (the v1 spec-bug fix), retest-entry never fires on the
crossing bar, and the two families combine correctly with CoreGateConfigV2's
hard vetoes."""
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
from bot.scalp.rules_v02 import (
    CoreGateConfigV2,
    TriggerFamilyV2,
    _trig_band_reversal,
    _trig_vwap_midline_retest,
    decide_core_gate_v2,
)
from tests.scalp.helpers import session_open

ET = ZoneInfo("America/New_York")
DAY = date(2026, 6, 1)


def _agg(close: float, minutes_offset: int, o=None, h=None, l=None, vol=1000, n_minutes=3) -> AggBar:
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
    decision_ts = datetime(2026, 6, 1, 10, 45, tzinfo=ET)  # main1, after the 10:30 day-type floor
    defaults = dict(
        decision_ts=decision_ts,
        idx=75,
        window=TimeWindow.main1,
        last_close=97.0,
        vwap=100.0,
        vwap_slope_5m=-0.1,
        rvol_tod=2.0,
        structure_15m=StructureState.lower_highs_lows,  # favors SHORT -- opposite of the band-reversal LONG we test
        opening_range=OpeningRange(high=100.5, low=99.0, width=1.5, width_vs_atr=0.5),
        prior_day=DailyOHLC(trade_date=date(2026, 5, 29), open=99.0, high=101.0, low=97.0, close=100.0),
        atr_14=1.0,
        bars_1m=[],
        bars_3m=[],
        bars_5m=[],
        bars_15m=[],
        day_type=DayType.trend_down,  # dominant side is DOWN -- the band-reversal LONG fights this on purpose
        vwap_sigma=1.0,
        ema9_5m=97.5,
        ema20_5m=98.0,
        ema9_3m=97.2,
    )
    defaults.update(overrides)
    return FeatureSnapshot(**defaults)


# ── band reversal: fires below VWAP for longs, against bias/context/day_type ─


def test_band_reversal_fires_long_below_vwap_with_volume():
    # Price sits below VWAP (100) and -2sigma (98); a 3m bar pierces -2sigma
    # (low <= 98) and closes back above it (98.5) on above-average volume.
    history = [_agg(97.0, 30 + 3 * i, vol=500) for i in range(6)]  # quiet history, avg vol 500
    reversal_bar = _agg(98.5, 48, o=97.5, h=98.6, l=97.8, vol=1500)  # pierces 98, closes back above, high volume
    snap = base_snapshot(vwap=100.0, vwap_sigma=1.0, bars_3m=history + [reversal_bar])
    long_ok, short_ok = _trig_band_reversal(
        snap, CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="3m")
    )
    assert long_ok is True
    assert short_ok is False


def test_band_reversal_requires_above_average_volume():
    history = [_agg(97.0, 30 + 3 * i, vol=1000) for i in range(6)]
    weak_reversal_bar = _agg(98.5, 48, o=97.5, h=98.6, l=97.8, vol=200)  # same shape, LOW volume
    snap = base_snapshot(vwap=100.0, vwap_sigma=1.0, bars_3m=history + [weak_reversal_bar])
    long_ok, short_ok = _trig_band_reversal(
        snap, CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="3m")
    )
    assert (long_ok, short_ok) == (False, False)


def test_band_reversal_decide_ignores_core_context_and_bias_and_day_type():
    """The actual v1 spec-bug fix: structure=LH/LL, VWAP bias=short (price
    below VWAP, falling slope), day_type=trend_down -- every v1 core-gate
    check disagrees with a LONG. The band-reversal family must still fire
    LONG, because it does not consult context/bias/day_type at all."""
    history = [_agg(97.0, 30 + 3 * i, vol=500) for i in range(6)]
    reversal_bar = _agg(98.5, 48, o=97.5, h=98.6, l=97.8, vol=1500)
    snap = base_snapshot(
        vwap=100.0,
        vwap_sigma=1.0,
        vwap_slope_5m=-0.5,  # falling -- v1 bias gate would refuse a long here
        structure_15m=StructureState.lower_highs_lows,  # v1 context gate would refuse a long here
        day_type=DayType.trend_down,  # v1's rotation_fade would not even be eligible (not a rotation day)
        bars_3m=history + [reversal_bar],
    )
    cfg = CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="3m")
    d = decide_core_gate_v2(snap, cfg)
    assert d.recommendation == Recommendation.long
    assert "band_reversal" in d.long_confirmations
    assert d.vetoes == ()


def test_band_reversal_1sigma_variant_fires_at_1sigma_when_enabled():
    # -1sigma level = 99.0; 2sigma-only variant must NOT fire here, 1s+2s must.
    history = [_agg(99.5, 30 + 3 * i, vol=500) for i in range(6)]
    reversal_bar = _agg(99.3, 48, o=98.8, h=99.4, l=98.7, vol=1500)  # pierces 99.0, closes back above
    snap = base_snapshot(vwap=100.0, vwap_sigma=1.0, bars_3m=history + [reversal_bar])

    cfg_2s_only = CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="3m", band_reversal_use_1sigma=False)
    assert _trig_band_reversal(snap, cfg_2s_only) == (False, False)

    cfg_1s2s = CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="3m", band_reversal_use_1sigma=True)
    long_ok, short_ok = _trig_band_reversal(snap, cfg_1s2s)
    assert long_ok is True
    assert short_ok is False


def test_band_reversal_confirm_tf_selects_1m_vs_3m_bars():
    history_3m = [_agg(97.0, 30 + 3 * i, vol=500) for i in range(6)]
    reversal_3m = _agg(98.5, 48, o=97.5, h=98.6, l=97.8, vol=1500)
    # 1m bars tell a DIFFERENT (non-firing) story -- proves the confirm_tf knob
    # actually switches which list is consulted, not just which one exists.
    history_1m = [_agg(97.0, 30 + i, vol=500, n_minutes=1) for i in range(6)]
    flat_1m = _agg(97.0, 36, o=97.0, h=97.2, l=96.9, vol=1500, n_minutes=1)  # never touches the band
    snap = base_snapshot(vwap=100.0, vwap_sigma=1.0, bars_3m=history_3m + [reversal_3m], bars_1m=history_1m + [flat_1m])

    cfg_3m = CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="3m")
    assert _trig_band_reversal(snap, cfg_3m) == (True, False)

    cfg_1m = CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="1m")
    assert _trig_band_reversal(snap, cfg_1m) == (False, False)


# ── retest-entry midline: never fires on the crossing bar itself ────────────


def test_retest_never_fires_when_last_bar_is_the_cross():
    # X below VWAP, Y (LAST bar) crosses above -- this is the crossing bar
    # itself; the retest rule must NOT enter here (v1's cross-entry bug).
    bars = [_agg(99.0, 35, n_minutes=5), _agg(101.0, 40, n_minutes=5)]
    snap = base_snapshot(vwap=100.0, bars_5m=bars)
    assert _trig_vwap_midline_retest(snap) == (False, False)


def test_retest_fires_on_later_pullback_hold_after_cross():
    # X below, Y crosses above (CROSS bar), Z pulls back to touch VWAP and
    # holds (closes back above) -- entry belongs on Z, not Y.
    bars = [
        _agg(99.0, 35, n_minutes=5),
        _agg(101.0, 40, n_minutes=5),  # cross bar
        _agg(100.3, 45, o=100.5, h=100.6, l=99.8, n_minutes=5),  # retest: touches 100 (low<=100<=high), closes above
    ]
    snap = base_snapshot(vwap=100.0, bars_5m=bars)
    long_ok, short_ok = _trig_vwap_midline_retest(snap)
    assert long_ok is True
    assert short_ok is False


def test_retest_does_not_fire_if_reclaimed_side_was_lost_after_cross():
    # X below, Y crosses above (CROSS), W falls back below VWAP (loses the
    # hold), Z touches VWAP and closes above again -- the ORIGINAL cross no
    # longer has an unbroken "held" run, so this must not count as its retest.
    bars = [
        _agg(99.0, 35, n_minutes=5),
        _agg(101.0, 40, n_minutes=5),  # cross bar
        _agg(99.5, 45, n_minutes=5),  # lost the hold -- back below VWAP
        _agg(100.3, 50, o=100.5, h=100.6, l=99.8, n_minutes=5),  # looks like a retest but the run was broken
    ]
    snap = base_snapshot(vwap=100.0, bars_5m=bars)
    long_ok, short_ok = _trig_vwap_midline_retest(snap)
    assert long_ok is False


def test_retest_short_side_mirrors_long():
    bars = [
        _agg(101.0, 35, n_minutes=5),
        _agg(99.0, 40, n_minutes=5),  # cross bar (above -> below)
        _agg(99.7, 45, o=99.5, h=100.2, l=99.4, n_minutes=5),  # retest: touches 100, closes below
    ]
    snap = base_snapshot(vwap=100.0, bars_5m=bars)
    long_ok, short_ok = _trig_vwap_midline_retest(snap)
    assert (long_ok, short_ok) == (False, True)


# ── decide_core_gate_v2 integration: midline families keep the core gate ────


def test_midline_retest_still_requires_core_context_and_bias():
    bars = [
        _agg(99.0, 35, n_minutes=5),
        _agg(101.0, 40, n_minutes=5),
        _agg(100.3, 45, o=100.5, h=100.6, l=99.8, n_minutes=5),
    ]
    # Structure disagrees (LH/LL) with the long retest signal -> core gate WAIT.
    snap = base_snapshot(
        vwap=100.0,
        vwap_slope_5m=0.1,
        structure_15m=StructureState.lower_highs_lows,
        bars_5m=bars,
    )
    cfg = CoreGateConfigV2(family=TriggerFamilyV2.vwap_midline_retest)
    d = decide_core_gate_v2(snap, cfg)
    assert d.recommendation == Recommendation.wait


def test_decide_v2_respects_rvol_veto_and_trade_window_for_band_reversal():
    history = [_agg(97.0, 30 + 3 * i, vol=500) for i in range(6)]
    reversal_bar = _agg(98.5, 48, o=97.5, h=98.6, l=97.8, vol=1500)
    snap = base_snapshot(vwap=100.0, vwap_sigma=1.0, bars_3m=history + [reversal_bar], rvol_tod=0.3)
    cfg = CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="3m", rvol_veto_threshold=1.0)
    d = decide_core_gate_v2(snap, cfg)
    assert d.recommendation == Recommendation.wait
    assert "rvol_below_veto_floor" in d.vetoes
