"""Forward-return evaluation for Stage A decisions.

No-leakage discipline: entry is the OPEN of the decision bar itself - the
first bar not yet fully closed as of decision_ts (see features.py module
docstring for why that is the correct "next bar after T"), never earlier.
Forward returns are measured from that entry price to the CLOSE of the bar
``horizon_minutes`` later.

Resolved ambiguity - "max drawdown of simulated trade sequence": the spec
asks for one drawdown number without specifying which horizon defines a
closed trade. We use the SHORTEST configured horizon (5m by default) as the
default single trade-close assumption for the equity curve, while every
horizon is still reported independently for hit-rate/expectancy/MFE/MAE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import mean
from typing import Optional, Sequence

from ..data.models import Bar
from .features import TimeWindow, to_et
from .rules import Decision, Recommendation

HORIZONS_MIN: tuple[int, ...] = (5, 10, 15, 30, 60)


@dataclass
class TradeOutcome:
    decision_ts: object
    direction: Recommendation
    window: TimeWindow
    confirmations_met: int
    entry_price: float
    entry_idx: int
    trade_date: date
    horizon_returns: dict[int, Optional[float]] = field(default_factory=dict)
    horizon_mfe: dict[int, Optional[float]] = field(default_factory=dict)
    horizon_mae: dict[int, Optional[float]] = field(default_factory=dict)


@dataclass
class HorizonStats:
    n: int
    hit_rate: Optional[float]
    expectancy_bps: Optional[float]
    avg_mfe_bps: Optional[float]
    avg_mae_bps: Optional[float]


@dataclass
class EvalReport:
    trade_count: int
    wait_count: int
    per_horizon: dict[int, HorizonStats]
    by_window: dict[str, dict[int, HorizonStats]]
    by_fold: dict[int, dict[int, HorizonStats]]
    max_drawdown_pct: Optional[float]
    default_horizon_min: int


def build_trade_outcomes(
    day_bars_1m: Sequence[Bar],
    decisions: Sequence[Decision],
    *,
    horizons: Sequence[int] = HORIZONS_MIN,
) -> list[TradeOutcome]:
    """Pair each non-Wait decision for one day with its entry fill and forward outcomes.

    Each ``Decision.decision_ts`` must match some ``day_bars_1m[i].ts`` exactly
    (as produced by ``rules.decide_all`` over ``features.build_feature_snapshots``
    for the SAME day_bars_1m). Entry price = that bar's OPEN (no earlier than
    the decision bar itself).
    """
    outcomes: list[TradeOutcome] = []
    n = len(day_bars_1m)
    ts_to_idx = {to_et(b.ts): i for i, b in enumerate(day_bars_1m)}

    for d in decisions:
        if d.recommendation == Recommendation.wait:
            continue
        i = ts_to_idx.get(to_et(d.decision_ts))
        if i is None:
            continue

        entry_bar = day_bars_1m[i]
        entry_price = entry_bar.open
        direction_sign = 1.0 if d.recommendation == Recommendation.long else -1.0

        horizon_returns: dict[int, Optional[float]] = {}
        horizon_mfe: dict[int, Optional[float]] = {}
        horizon_mae: dict[int, Optional[float]] = {}

        for h in horizons:
            exit_idx = i + h
            if exit_idx >= n:
                horizon_returns[h] = None
                horizon_mfe[h] = None
                horizon_mae[h] = None
                continue

            exit_bar = day_bars_1m[exit_idx]
            horizon_returns[h] = direction_sign * (exit_bar.close - entry_price) / entry_price

            favorable: list[float] = []
            adverse: list[float] = []
            for pb in day_bars_1m[i: exit_idx + 1]:
                fav_price = pb.high if direction_sign > 0 else pb.low
                adv_price = pb.low if direction_sign > 0 else pb.high
                favorable.append(direction_sign * (fav_price - entry_price) / entry_price)
                adverse.append(direction_sign * (adv_price - entry_price) / entry_price)
            horizon_mfe[h] = max(favorable) if favorable else None
            horizon_mae[h] = min(adverse) if adverse else None

        outcomes.append(
            TradeOutcome(
                decision_ts=d.decision_ts,
                direction=d.recommendation,
                window=d.window,
                confirmations_met=d.confirmations_met,
                entry_price=entry_price,
                entry_idx=i,
                trade_date=to_et(d.decision_ts).date(),
                horizon_returns=horizon_returns,
                horizon_mfe=horizon_mfe,
                horizon_mae=horizon_mae,
            )
        )
    return outcomes


def _stats_for(outcomes: Sequence[TradeOutcome], horizon: int) -> HorizonStats:
    rets = [o.horizon_returns.get(horizon) for o in outcomes]
    rets = [r for r in rets if r is not None]
    if not rets:
        return HorizonStats(n=0, hit_rate=None, expectancy_bps=None, avg_mfe_bps=None, avg_mae_bps=None)

    mfes = [o.horizon_mfe.get(horizon) for o in outcomes if o.horizon_mfe.get(horizon) is not None]
    maes = [o.horizon_mae.get(horizon) for o in outcomes if o.horizon_mae.get(horizon) is not None]

    hits = sum(1 for r in rets if r > 0)
    return HorizonStats(
        n=len(rets),
        hit_rate=hits / len(rets),
        expectancy_bps=mean(rets) * 10_000.0,
        avg_mfe_bps=(mean(mfes) * 10_000.0) if mfes else None,
        avg_mae_bps=(mean(maes) * 10_000.0) if maes else None,
    )


def _group_by_window(outcomes: Sequence[TradeOutcome]) -> dict[str, list[TradeOutcome]]:
    groups: dict[str, list[TradeOutcome]] = {}
    for o in outcomes:
        groups.setdefault(o.window.value, []).append(o)
    return groups


def chronological_folds(trade_dates: Sequence[date], fold_size_days: int = 5) -> dict[int, list[date]]:
    """Contiguous, non-overlapping blocks of trading days in chronological order.

    Blocked chronological day/week folds per spec - no random row splits.
    Default block size of 5 trading days approximates a calendar week.
    """
    dates = sorted(set(trade_dates))
    folds: dict[int, list[date]] = {}
    for i in range(0, len(dates), max(1, fold_size_days)):
        folds[i // fold_size_days] = dates[i: i + fold_size_days]
    return folds


def _group_by_fold(
    outcomes: Sequence[TradeOutcome], folds: dict[int, list[date]]
) -> dict[int, list[TradeOutcome]]:
    date_to_fold: dict[date, int] = {}
    for fid, ds in folds.items():
        for d in ds:
            date_to_fold[d] = fid
    groups: dict[int, list[TradeOutcome]] = {}
    for o in outcomes:
        fid = date_to_fold.get(o.trade_date)
        if fid is None:
            continue
        groups.setdefault(fid, []).append(o)
    return groups


def max_drawdown_from_returns(returns: Sequence[float]) -> Optional[float]:
    """Max drawdown (negative fraction, e.g. -0.05 = [n]%) of a sequential trade equity curve."""
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        max_dd = min(max_dd, (equity - peak) / peak)
    return max_dd


def evaluate_decisions(
    bars_by_day: dict[date, Sequence[Bar]],
    decisions_by_day: dict[date, Sequence[Decision]],
    *,
    horizons: Sequence[int] = HORIZONS_MIN,
    fold_size_days: int = 5,
    default_horizon_min: int = 5,
) -> EvalReport:
    """Evaluate decisions across many trading days: hit rate, expectancy, MFE/MAE,
    splits by time window and by blocked chronological day folds, plus max
    drawdown of the sequential trade equity curve at ``default_horizon_min``.
    """
    all_outcomes: list[TradeOutcome] = []
    wait_count = 0

    for day in sorted(bars_by_day):
        day_bars = bars_by_day[day]
        day_decisions = decisions_by_day.get(day, [])
        wait_count += sum(1 for d in day_decisions if d.recommendation == Recommendation.wait)
        all_outcomes.extend(build_trade_outcomes(day_bars, day_decisions, horizons=horizons))

    all_outcomes.sort(key=lambda o: o.decision_ts)

    per_horizon = {h: _stats_for(all_outcomes, h) for h in horizons}

    by_window: dict[str, dict[int, HorizonStats]] = {}
    for wname, group in _group_by_window(all_outcomes).items():
        by_window[wname] = {h: _stats_for(group, h) for h in horizons}

    trade_dates = [o.trade_date for o in all_outcomes]
    folds = chronological_folds(trade_dates, fold_size_days)
    by_fold: dict[int, dict[int, HorizonStats]] = {}
    for fid, group in _group_by_fold(all_outcomes, folds).items():
        by_fold[fid] = {h: _stats_for(group, h) for h in horizons}

    default_returns = [
        r for r in (o.horizon_returns.get(default_horizon_min) for o in all_outcomes) if r is not None
    ]
    max_dd = max_drawdown_from_returns(default_returns)

    return EvalReport(
        trade_count=len(all_outcomes),
        wait_count=wait_count,
        per_horizon=per_horizon,
        by_window=by_window,
        by_fold=by_fold,
        max_drawdown_pct=max_dd,
        default_horizon_min=default_horizon_min,
    )
