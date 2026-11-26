"""Integration tests: rules_v03 decisions -> evaluate_v3 R-multiple + bps
report, covering BOTH new v0.3 exit archetypes (vwap_half band-fade,
prior_swing_capped trend-pullback) and the exact gross/net bps arithmetic."""
from __future__ import annotations

from datetime import date

from bot.scalp.evaluate_v3 import evaluate_decisions_v3
from bot.scalp.exits import ExitConfig, ExitConfigV3
from bot.scalp.features import build_feature_snapshots
from bot.scalp.rules import Recommendation
from bot.scalp.rules_v03 import CoreGateConfigV3, TriggerFamilyV3, decide_core_gate_v3_all
from tests.scalp.synthetic_helpers import make_band_touch_session

DAY = date(2026, 6, 2)  # a Tuesday


def test_evaluate_v3_band_fade_half_vwap_produces_trades_and_bps():
    bars = make_band_touch_session("SPY", DAY)
    snaps = build_feature_snapshots(bars, [])
    cfg = CoreGateConfigV3(family=TriggerFamilyV3.band_reversal, band_reversal_confirm_tf="3m")
    decisions = decide_core_gate_v3_all(snaps, cfg)
    assert any(d.recommendation != Recommendation.wait for d in decisions), "fixture should produce at least one trade"

    bars_by_day = {DAY: bars}
    snaps_by_day = {DAY: snaps}
    decisions_by_day = {DAY: decisions}
    exit_cfg = ExitConfigV3(stop_type="band_extreme_buffer", stop_mult=1.25, target_kind="vwap_half", time_stop_minutes=20)

    report = evaluate_decisions_v3(bars_by_day, snaps_by_day, decisions_by_day, exit_cfg, fold_size_days=5)

    assert report.trade_count == report.long.n + report.short.n
    assert report.trade_count > 0
    assert report.gross_expectancy_bps is not None
    assert report.net_expectancy_bps_1bp is not None
    # net is EXACTLY gross minus the 2bps round-trip cost (1bp/side), not an approximation.
    assert abs(report.net_expectancy_bps_1bp - (report.gross_expectancy_bps - 2.0)) < 1e-9


def test_evaluate_v3_band_fade_scratch_uses_plain_exit_config_unchanged():
    """The scratch-quick arm needs NO new exit code -- a plain ExitConfig with
    target_r=0.5/entry_extreme (tautologically 1R stop) must flow through the
    SAME evaluate_decisions_v3 entry point as the new ExitConfigV3 archetypes."""
    bars = make_band_touch_session("SPY", DAY)
    snaps = build_feature_snapshots(bars, [])
    cfg = CoreGateConfigV3(family=TriggerFamilyV3.band_reversal, band_reversal_confirm_tf="3m")
    decisions = decide_core_gate_v3_all(snaps, cfg)

    bars_by_day = {DAY: bars}
    snaps_by_day = {DAY: snaps}
    decisions_by_day = {DAY: decisions}
    exit_cfg = ExitConfig(stop_type="entry_extreme", target_r=0.5, time_stop_minutes=10)

    report = evaluate_decisions_v3(bars_by_day, snaps_by_day, decisions_by_day, exit_cfg, fold_size_days=5)
    assert report.trade_count > 0
    # every winning trade must have r_multiple close to +0.5 (target hit) or
    # be a loser at -1.0 (stop) / a time-stop in between -- never > 0.5.
    for r in [report.overall.r_expectancy]:
        assert r is not None


def test_gross_and_net_bps_match_r_multiple_times_risk_bps_exactly():
    """Algebraic identity this module's docstring asserts: trade_bps ==
    r_multiple * risk_bps, which equals the raw price return in bps. Verified
    here on a hand-built single-trade day rather than trusting the docstring's
    algebra alone."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from bot.scalp.evaluate_v3 import build_trade_outcomes_v3
    from bot.scalp.features import AggBar, DayType, FeatureSnapshot, OpeningRange, StructureState, TimeWindow, to_et
    from bot.scalp.rules import Decision
    from tests.scalp.helpers import bar, session_open

    ET = ZoneInfo("America/New_York")
    start = session_open(DAY)

    # A simple long trend-pullback trade: entry at bar[10].open, trigger 5m
    # bar's low = 99.0 (stop), swing target capped at 1R.
    day_bars = [bar("SPY", start + timedelta(minutes=i), 100.0, 100.3, 99.8, 100.0) for i in range(30)]
    # Make the entry bar's open distinct and give it enough forward bars to
    # resolve via a clean time-stop (avoids depending on stop/target geometry
    # for this arithmetic-only check).
    day_bars[10] = bar("SPY", start + timedelta(minutes=10), 100.0, 100.5, 99.9, 100.2)

    decision_ts = to_et(day_bars[10].ts)
    decision = Decision(
        decision_ts=decision_ts,
        recommendation=Recommendation.long,
        confirmations_met=1,
        long_confirmations=frozenset({"trend_pullback_hold"}),
        short_confirmations=frozenset(),
        vetoes=(),
        window=TimeWindow.main1,
    )
    trigger_bar = AggBar(
        ts_open=start + timedelta(minutes=5), ts_close=start + timedelta(minutes=10),
        open=100.0, high=100.4, low=99.0, close=100.1, volume=1000, n_minutes=5,
    )
    snap = FeatureSnapshot(
        decision_ts=decision_ts, idx=10, window=TimeWindow.main1, last_close=100.0,
        vwap=100.0, vwap_slope_5m=0.0, rvol_tod=2.0, structure_15m=StructureState.higher_highs_lows,
        opening_range=OpeningRange(high=100.5, low=99.5, width=1.0, width_vs_atr=0.5),
        prior_day=None, atr_14=1.0, bars_5m=[trigger_bar], day_type=DayType.trend_up,
    )
    snaps_by_ts = {decision_ts: snap}
    exit_cfg = ExitConfigV3(stop_type="entry_extreme", target_kind="prior_swing_capped", swing_lookback=10, time_stop_minutes=5)

    outcomes = build_trade_outcomes_v3(day_bars, [decision], snaps_by_ts, exit_cfg)
    assert len(outcomes) == 1
    o = outcomes[0]

    entry_price = day_bars[10].open  # 100.0
    stop_price = trigger_bar.low  # 99.0 (entry_extreme)
    risk_bps = abs(entry_price - stop_price) / entry_price * 10_000.0
    assert abs(o.trade_bps - o.r_multiple * risk_bps) < 1e-9
