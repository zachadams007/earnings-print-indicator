"""Negative control tests: shuffled bars, delayed signal, matched-frequency random labels."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from bot.scalp.controls import (
    delay_decisions,
    random_decisions_matched_frequency,
    run_negative_controls,
    shuffle_bars_within_day,
)
from bot.scalp.evaluate import HORIZONS_MIN
from bot.scalp.features import TimeWindow, build_feature_snapshots
from bot.scalp.rules import Decision, Recommendation, RuleConfig, decide_all
from bot.scalp.synthetic import make_multi_day_sessions
from tests.scalp.helpers import bar, session_open

DAY = date(2026, 6, 1)


def _decision(ts, rec, window=TimeWindow.main1):
    return Decision(
        decision_ts=ts,
        recommendation=rec,
        confirmations_met=3,
        long_confirmations=frozenset(),
        short_confirmations=frozenset(),
        vetoes=(),
        window=window,
    )


def test_shuffle_bars_within_day_preserves_ts_grid_but_permutes_content():
    """The ts sequence (ascending, one-per-minute) MUST be preserved exactly -
    that's what keeps build_feature_snapshots' incremental bucket aggregator
    correct (see shuffle_bars_within_day docstring). Only the OHLCV content
    is permuted across that fixed grid."""
    bars = [
        bar("SPY", session_open(DAY) + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100 + i, 100 + i)
        for i in range(30)
    ]
    shuffled = shuffle_bars_within_day(bars, seed=1)
    # Same timestamps, same order, one-per-minute grid untouched.
    assert [b.ts for b in shuffled] == [b.ts for b in bars]
    # But the content (here, close price) attached to each timestamp slot has moved.
    assert [b.close for b in shuffled] != [b.close for b in bars]
    assert {b.close for b in shuffled} == {b.close for b in bars}


def test_shuffle_bars_within_day_sorts_out_of_order_input():
    """Even if the caller hands bars in non-ascending order, the shuffle must
    still emit a proper ascending, one-per-minute ts grid."""
    ordered = [bar("SPY", session_open(DAY) + timedelta(minutes=i), 100, 101, 99, 100, 100) for i in range(10)]
    scrambled_input = list(reversed(ordered))
    shuffled = shuffle_bars_within_day(scrambled_input, seed=2)
    ts_out = [b.ts for b in shuffled]
    assert ts_out == sorted(ts_out)
    assert ts_out == [b.ts for b in ordered]


def test_shuffle_bars_within_day_aggregator_stays_populated():
    """Regression guard for the sweep-v2 bug: the shuffled control must not
    starve build_feature_snapshots' N-minute aggregator. Real bug looked
    like bars_3m/5m/15m and opening_range being empty on EVERY snapshot."""
    from bot.scalp.features import build_feature_snapshots

    bars = [
        bar(
            "SPY",
            session_open(DAY) + timedelta(minutes=i),
            100 + (i % 7) * 0.1,
            100.5 + (i % 7) * 0.1,
            99.5 + (i % 7) * 0.1,
            100 + (i % 5) * 0.1,
            1000 + i,
        )
        for i in range(60)
    ]
    shuffled = shuffle_bars_within_day(bars, seed=3)
    snaps = build_feature_snapshots(shuffled, [])
    assert any(s.bars_3m for s in snaps)
    assert any(s.bars_5m for s in snaps)
    assert any(s.opening_range is not None for s in snaps)


def test_shuffle_is_deterministic_given_seed():
    bars = [
        bar("SPY", session_open(DAY) + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100 + i, 100 + i)
        for i in range(30)
    ]
    a = shuffle_bars_within_day(bars, seed=5)
    b = shuffle_bars_within_day(bars, seed=5)
    assert [x.ts for x in a] == [x.ts for x in b]
    assert [x.close for x in a] == [x.close for x in b]


def test_delay_decisions_shifts_timestamp_forward():
    bars = [bar("SPY", session_open(DAY) + timedelta(minutes=i), 100, 101, 99, 100, 100) for i in range(30)]
    decisions = [_decision(bars[10].ts, Recommendation.long)]
    delayed = delay_decisions(decisions, n_bars=3, day_bars_1m=bars)
    assert len(delayed) == 1
    assert delayed[0].decision_ts == bars[13].ts
    assert delayed[0].recommendation == Recommendation.long


def test_delay_decisions_drops_wait_and_out_of_range():
    bars = [bar("SPY", session_open(DAY) + timedelta(minutes=i), 100, 101, 99, 100, 100) for i in range(30)]
    decisions = [
        _decision(bars[5].ts, Recommendation.wait),
        _decision(bars[28].ts, Recommendation.long),  # 28 + 3 = 31 >= 30 -> dropped
    ]
    delayed = delay_decisions(decisions, n_bars=3, day_bars_1m=bars)
    assert delayed == []


def test_random_decisions_matched_frequency_preserves_mix():
    bars = [bar("SPY", session_open(DAY) + timedelta(minutes=i), 100, 101, 99, 100, 100) for i in range(30)]
    decisions = [
        _decision(bars[0].ts, Recommendation.long),
        _decision(bars[1].ts, Recommendation.long),
        _decision(bars[2].ts, Recommendation.short),
        _decision(bars[3].ts, Recommendation.wait),
    ]
    randomized = random_decisions_matched_frequency(decisions, seed=3)

    orig_counts = {r: sum(1 for d in decisions if d.recommendation == r) for r in Recommendation}
    new_counts = {r: sum(1 for d in randomized if d.recommendation == r) for r in Recommendation}
    assert orig_counts == new_counts
    # Timestamps must be untouched - only the label assignment is randomized.
    assert [d.decision_ts for d in randomized] == [d.decision_ts for d in decisions]


def test_random_decisions_matched_frequency_deterministic_given_seed():
    bars = [bar("SPY", session_open(DAY) + timedelta(minutes=i), 100, 101, 99, 100, 100) for i in range(10)]
    decisions = [_decision(bars[i].ts, Recommendation.long if i % 2 == 0 else Recommendation.short) for i in range(10)]
    a = random_decisions_matched_frequency(decisions, seed=42)
    b = random_decisions_matched_frequency(decisions, seed=42)
    assert [d.recommendation for d in a] == [d.recommendation for d in b]


def test_run_negative_controls_end_to_end_shapes_reports():
    """Smoke test: full pipeline (features -> rules -> controls) on synthetic
    bars produces a report for all 4 arms without raising, with each control's
    trade count derived sensibly from the real run."""
    sessions = make_multi_day_sessions("SPY", DAY, 5, seed=11)
    cfg = RuleConfig(min_confirmations=1)  # loosen so *some* trades occur in this small fixture

    sorted_days = sorted(sessions)
    decisions_by_day = {}
    prior_sessions_by_day = {}
    for idx, d in enumerate(sorted_days):
        prior_days = [sessions[pd_] for pd_ in sorted_days[:idx]]
        prior_sessions_by_day[d] = prior_days
        snaps = build_feature_snapshots(sessions[d], prior_days)
        decisions_by_day[d] = decide_all(snaps, cfg)

    report = run_negative_controls(sessions, prior_sessions_by_day, decisions_by_day, cfg)

    assert report.real.trade_count >= 0
    # Delayed-signal control never has MORE trades than the real run (it can only drop some).
    assert report.delayed_signal.trade_count <= report.real.trade_count
    # Random-matched control preserves the real run's total decision count exactly.
    assert report.random_matched.trade_count == report.real.trade_count
    # Shuffled-bars control runs its own independent pipeline; just assert it produced a report.
    assert report.shuffled_bars.trade_count >= 0


def test_shuffled_control_produces_trades_and_does_not_reproduce_real_expectancy():
    """Regression test for the sweep-v2 bug: the old shuffle (full-Bar-object
    permutation) starved build_feature_snapshots' aggregator and produced
    ZERO trades in every config -- a silent, vacuous control. On a fixture
    with a real trending signal (so the real rubric reliably fires trades),
    the shuffled control must (a) actually produce trades of its own, proving
    the aggregator is not starved, and (b) not reproduce the real run's
    expectancy, proving the price-path signal was genuinely destroyed rather
    than passed through unchanged."""
    from bot.scalp.synthetic import make_session_bars

    cfg = RuleConfig(min_confirmations=1)  # loosen so trades reliably fire in a small fixture

    n_days = 15
    sessions: dict = {}
    d = DAY
    count = 0
    while count < n_days:
        if d.weekday() < 5:
            # Strong, consistent trend so the structure/VWAP/breakout
            # confirmations reliably line up on the REAL (unshuffled) path.
            sessions[d] = make_session_bars(
                "SPY", d, seed=100 + count, drift_bps_per_min=3.0, vol_bps=4.0
            )
            count += 1
        d = d + timedelta(days=1)

    sorted_days = sorted(sessions)
    decisions_by_day = {}
    prior_sessions_by_day = {}
    for idx, day in enumerate(sorted_days):
        prior_days = [sessions[pd_] for pd_ in sorted_days[:idx]]
        prior_sessions_by_day[day] = prior_days
        snaps = build_feature_snapshots(sessions[day], prior_days)
        decisions_by_day[day] = decide_all(snaps, cfg)

    report = run_negative_controls(sessions, prior_sessions_by_day, decisions_by_day, cfg)

    assert report.real.trade_count > 0, "fixture must produce real trades for this test to be meaningful"
    assert report.shuffled_bars.trade_count > 0, (
        "shuffled-bars control produced 0 trades -- the aggregator-starvation bug is back"
    )

    real_exp = report.real.per_horizon[HORIZONS_MIN[0]].expectancy_bps
    shuffled_exp = report.shuffled_bars.per_horizon[HORIZONS_MIN[0]].expectancy_bps
    assert real_exp is not None and shuffled_exp is not None
    assert shuffled_exp != real_exp
