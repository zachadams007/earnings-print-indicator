"""Decision-rule tests: confluence counting, vetoes, Wait defaults."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bot.scalp.features import (
    AggBar,
    FeatureSnapshot,
    OpeningRange,
    StructureState,
    TimeWindow,
)
from bot.scalp.rules import Recommendation, RuleConfig, decide

ET = ZoneInfo("America/New_York")
DECISION_TS = datetime(2026, 6, 1, 10, 0, tzinfo=ET)  # inside main1 window


def _agg3(body: float, atr_frac_ok: bool = True) -> AggBar:
    o = 100.0
    c = o + body
    return AggBar(
        ts_open=DECISION_TS,
        ts_close=DECISION_TS + timedelta(minutes=3),
        open=o,
        high=max(o, c) + 0.1,
        low=min(o, c) - 0.1,
        close=c,
        volume=1000,
        n_minutes=3,
    )


def base_snapshot(**overrides) -> FeatureSnapshot:
    defaults = dict(
        decision_ts=DECISION_TS,
        idx=50,
        window=TimeWindow.main1,
        last_close=101.0,
        vwap=100.0,
        vwap_slope_5m=0.1,
        rvol_tod=2.0,
        structure_15m=StructureState.higher_highs_lows,
        opening_range=OpeningRange(high=100.5, low=99.0, width=1.5, width_vs_atr=0.5),
        prior_day=None,
        atr_14=1.0,
        bars_3m=[_agg3(0.5)],  # body 0.5, atr_frac default 0.15 -> threshold 0.15 -> confirms long
        bars_5m=[],
        bars_15m=[],
    )
    defaults.update(overrides)
    return FeatureSnapshot(**defaults)


def test_all_five_confirmations_yields_long():
    snap = base_snapshot()  # last_close > vwap, above OR high, structure HH/HL, rvol ok, impulse up
    d = decide(snap, RuleConfig())
    assert d.recommendation == Recommendation.long
    assert d.confirmations_met == 5
    assert d.vetoes == ()


def test_symmetric_setup_yields_short():
    snap = base_snapshot(
        last_close=98.0,
        vwap=100.0,
        vwap_slope_5m=-0.1,
        structure_15m=StructureState.lower_highs_lows,
        opening_range=OpeningRange(high=100.5, low=99.0, width=1.5, width_vs_atr=0.5),
        bars_3m=[_agg3(-0.5)],
    )
    d = decide(snap, RuleConfig())
    assert d.recommendation == Recommendation.short
    assert d.confirmations_met == 5


def test_only_two_confirmations_yields_wait():
    snap = base_snapshot(
        structure_15m=StructureState.mixed,      # category 1 fails both sides
        bars_3m=[_agg3(0.0)],                    # category 5 fails (no impulse either way)
        rvol_tod=1.2,                            # above veto floor (1.0) but below confirm (1.5)
    )
    # Only vwap_bias_5m + setup_trigger_5m confirm -> 2 < default min_confirmations (3)
    d = decide(snap, RuleConfig())
    assert d.recommendation == Recommendation.wait
    assert d.confirmations_met == 2
    assert d.vetoes == ()  # Wait here is "insufficient", not a hard veto


def test_rvol_below_veto_floor_forces_wait_even_with_5_of_5():
    snap = base_snapshot(rvol_tod=0.5)
    d = decide(snap, RuleConfig())
    assert d.recommendation == Recommendation.wait
    assert "rvol_below_veto_floor" in d.vetoes


def test_outside_trade_window_forces_wait():
    snap = base_snapshot(window=TimeWindow.lunch)
    d = decide(snap, RuleConfig())
    assert d.recommendation == Recommendation.wait
    assert "outside_trade_window" in d.vetoes


def test_structure_vwap_conflict_is_a_veto():
    # 15m structure says long (HH/HL) but VWAP bias says short (below + falling slope).
    snap = base_snapshot(
        structure_15m=StructureState.higher_highs_lows,
        last_close=98.0,
        vwap=100.0,
        vwap_slope_5m=-0.2,
    )
    d = decide(snap, RuleConfig())
    assert d.recommendation == Recommendation.wait
    assert "structure_vwap_conflict" in d.vetoes


def test_macro_blackout_forces_wait():
    snap = base_snapshot()
    d = decide(snap, RuleConfig(macro_blackout=True))
    assert d.recommendation == Recommendation.wait
    assert "macro_blackout" in d.vetoes


def test_min_confirmations_threshold_is_configurable():
    snap = base_snapshot(
        structure_15m=StructureState.mixed,   # category 1 fails
        bars_3m=[_agg3(0.0)],                 # category 5 fails
    )
    # 3 categories still confirm (vwap_bias, setup_trigger, volume) -> long_hits == 3
    cfg_default = RuleConfig()
    d_default = decide(snap, cfg_default)
    assert d_default.recommendation == Recommendation.long
    assert d_default.confirmations_met == 3

    cfg_strict = RuleConfig(min_confirmations=4)
    d_strict = decide(snap, cfg_strict)
    assert d_strict.recommendation == Recommendation.wait


def test_required_categories_can_exclude_a_check():
    # Drop "execution_timing" from required categories entirely; with structure
    # mixed (fails) and 3 remaining hits, min_confirmations=3 should still pass.
    snap = base_snapshot(structure_15m=StructureState.mixed, bars_3m=[_agg3(0.0)])
    cfg = RuleConfig(required_categories=frozenset({"vwap_bias_5m", "setup_trigger_5m", "volume_rvol"}))
    d = decide(snap, cfg)
    assert d.recommendation == Recommendation.long
    assert d.confirmations_met == 3


def test_missing_vwap_data_never_confirms_vwap_category():
    snap = base_snapshot(vwap=None)
    d = decide(snap, RuleConfig())
    assert "vwap_bias_5m" not in d.long_confirmations
    assert "vwap_bias_5m" not in d.short_confirmations
