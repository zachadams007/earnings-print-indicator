"""Forward-return evaluation tests: entry fill, horizon returns, MFE/MAE, folds, drawdown."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from bot.scalp.evaluate import (
    build_trade_outcomes,
    chronological_folds,
    evaluate_decisions,
    max_drawdown_from_returns,
)
from bot.scalp.features import TimeWindow
from bot.scalp.rules import Decision, Recommendation
from tests.scalp.helpers import bar, session_open

DAY = date(2026, 6, 1)


def _mk_bars(closes: list[float], volume: int = 1000):
    start = session_open(DAY)
    bars = []
    prev = closes[0]
    for i, c in enumerate(closes):
        o = prev
        h = max(o, c) + 1.0
        l = min(o, c) - 1.0
        bars.append(bar("SPY", start + timedelta(minutes=i), o, h, l, c, volume))
        prev = c
    return bars


def _decision(ts, rec, window=TimeWindow.main1, confirmations=3):
    return Decision(
        decision_ts=ts,
        recommendation=rec,
        confirmations_met=confirmations,
        long_confirmations=frozenset(),
        short_confirmations=frozenset(),
        vetoes=(),
        window=window,
    )


def test_entry_price_is_decision_bar_open_not_earlier():
    closes = [100.0] * 20
    bars = _mk_bars(closes)
    decisions = [_decision(bars[5].ts, Recommendation.long)]
    outcomes = build_trade_outcomes(bars, decisions, horizons=(5,))
    assert len(outcomes) == 1
    assert outcomes[0].entry_price == bars[5].open
    assert outcomes[0].entry_idx == 5


def test_wait_decisions_produce_no_trade_outcome():
    closes = [100.0] * 20
    bars = _mk_bars(closes)
    decisions = [_decision(bars[5].ts, Recommendation.wait)]
    outcomes = build_trade_outcomes(bars, decisions, horizons=(5,))
    assert outcomes == []


def test_long_forward_return_positive_on_uptrend():
    # Flat then a jump at bar 10 -> entry at bar 5's open (~100), exit at bar 5+5=10's close (~110).
    closes = [100.0] * 10 + [110.0] * 10
    bars = _mk_bars(closes)
    decisions = [_decision(bars[5].ts, Recommendation.long)]
    outcomes = build_trade_outcomes(bars, decisions, horizons=(5,))
    ret = outcomes[0].horizon_returns[5]
    assert ret is not None
    assert ret > 0


def test_short_forward_return_sign_flips_relative_to_long():
    closes = [100.0] * 10 + [110.0] * 10
    bars = _mk_bars(closes)
    long_outcome = build_trade_outcomes(bars, [_decision(bars[5].ts, Recommendation.long)], horizons=(5,))[0]
    short_outcome = build_trade_outcomes(bars, [_decision(bars[5].ts, Recommendation.short)], horizons=(5,))[0]
    assert short_outcome.horizon_returns[5] == pytest.approx(-long_outcome.horizon_returns[5])


def test_horizon_beyond_session_end_is_none():
    closes = [100.0] * 10
    bars = _mk_bars(closes)
    decisions = [_decision(bars[8].ts, Recommendation.long)]
    outcomes = build_trade_outcomes(bars, decisions, horizons=(5, 60))
    assert outcomes[0].horizon_returns[5] is None   # 8+5=13 >= len(bars)=10
    assert outcomes[0].horizon_returns[60] is None


def test_mfe_mae_long_direction():
    # entry at bar 5 open (100). Path afterward dips to 90 then rallies to 120.
    closes = [100.0] * 6 + [90.0, 95.0, 105.0, 120.0, 115.0]
    bars = _mk_bars(closes)
    decisions = [_decision(bars[5].ts, Recommendation.long)]
    outcomes = build_trade_outcomes(bars, decisions, horizons=(5,))
    o = outcomes[0]
    entry = o.entry_price
    assert o.horizon_mfe[5] == pytest.approx((max(b.high for b in bars[5:11]) - entry) / entry)
    assert o.horizon_mae[5] == pytest.approx((min(b.low for b in bars[5:11]) - entry) / entry)


def test_chronological_folds_are_contiguous_and_ordered():
    dates = [DAY + timedelta(days=i) for i in range(12)]
    folds = chronological_folds(dates, fold_size_days=5)
    assert folds[0] == dates[0:5]
    assert folds[1] == dates[5:10]
    assert folds[2] == dates[10:12]
    # No random shuffling: every fold's dates stay in chronological order.
    for ds in folds.values():
        assert ds == sorted(ds)


def test_max_drawdown_from_returns():
    # +10%, -20%, +5% -> equity 1.10 -> 0.88 -> 0.924; peak 1.10 -> dd = (0.88-1.10)/1.10
    returns = [0.10, -0.20, 0.05]
    dd = max_drawdown_from_returns(returns)
    assert dd == pytest.approx((0.88 - 1.10) / 1.10)


def test_max_drawdown_empty_is_none():
    assert max_drawdown_from_returns([]) is None


def test_evaluate_decisions_multi_day_aggregation():
    day1 = DAY
    day2 = DAY + timedelta(days=1)
    closes = [100.0] * 10 + [105.0] * 10
    bars1 = _mk_bars(closes)
    bars2 = [bar("SPY", session_open(day2) + timedelta(minutes=i), b.open, b.high, b.low, b.close, b.volume) for i, b in enumerate(bars1)]

    decisions1 = [_decision(bars1[5].ts, Recommendation.long), _decision(bars1[7].ts, Recommendation.wait)]
    decisions2 = [_decision(bars2[5].ts, Recommendation.short)]

    report = evaluate_decisions(
        {day1: bars1, day2: bars2},
        {day1: decisions1, day2: decisions2},
        horizons=(5,),
        fold_size_days=1,
        default_horizon_min=5,
    )
    assert report.trade_count == 2
    assert report.wait_count == 1
    assert 5 in report.per_horizon
    assert report.per_horizon[5].n == 2
    assert TimeWindow.main1.value in report.by_window
    assert len(report.by_fold) == 2  # one fold per day at fold_size_days=1
