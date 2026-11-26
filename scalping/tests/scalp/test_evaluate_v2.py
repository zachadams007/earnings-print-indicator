"""Integration test: rules_v02 decisions -> evaluate_v2 R-multiple report,
including the long/short split and the fixed-horizon comparability column."""
from __future__ import annotations

from datetime import date

from bot.scalp.evaluate_v2 import evaluate_decisions_v2
from bot.scalp.exits import ExitConfig
from bot.scalp.features import build_feature_snapshots, to_et
from bot.scalp.rules import Recommendation
from bot.scalp.rules_v02 import CoreGateConfigV2, TriggerFamilyV2, decide_core_gate_v2_all
from tests.scalp.synthetic_helpers import make_band_touch_session

DAY = date(2026, 6, 2)  # a Tuesday


def test_evaluate_v2_produces_long_short_split_and_fixed_horizon():
    bars = make_band_touch_session("SPY", DAY)
    snaps = build_feature_snapshots(bars, [])
    cfg = CoreGateConfigV2(family=TriggerFamilyV2.band_reversal, band_reversal_confirm_tf="3m")
    decisions = decide_core_gate_v2_all(snaps, cfg)

    assert any(d.recommendation != Recommendation.wait for d in decisions), "fixture should produce at least one trade"

    bars_by_day = {DAY: bars}
    snaps_by_day = {DAY: snaps}
    decisions_by_day = {DAY: decisions}
    exit_cfg = ExitConfig(stop_type="entry_extreme", target_r=1.5, time_stop_minutes=30)

    report = evaluate_decisions_v2(bars_by_day, snaps_by_day, decisions_by_day, exit_cfg, fold_size_days=5)

    assert report.trade_count == report.long.n + report.short.n
    assert report.overall.n == report.trade_count
    if report.trade_count > 0:
        assert report.overall.r_expectancy is not None
        assert report.fixed_horizon_expectancy_bps is not None
        assert report.trades_per_day == report.trade_count / 1
