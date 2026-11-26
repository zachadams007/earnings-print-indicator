"""R-multiple trade evaluation for the Stage A sweep v2 (real exits, long/
short split) - see rules_v02.py and exits.py.

ADDITIVE: does not touch evaluate.py; reuses its ``HorizonStats``/
``chronological_folds``/``_stats_for`` building blocks so the fixed-horizon
comparability column (item 3's "keep one fixed-horizon column for
comparability with v1") is computed identically to v1, not reinvented.

No-leakage: entry is the OPEN of the decision bar itself (same convention as
evaluate.py); the stop/target levels are fixed at entry from the already-
completed TRIGGER bar (snapshot.bars_5m[-1]) / ATR(14 on 5m-agg) - see
exits.py's module docstring for why the yet-unclosed fill bar itself is
never used for the stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from statistics import mean
from typing import Optional, Sequence

from ..data.models import Bar
from .evaluate import chronological_folds
from .exits import ExitConfig, atr_from_agg_bars, compute_stop_and_target, simulate_exit
from .features import FeatureSnapshot, TimeWindow, to_et
from .rules import Decision, Recommendation

DEFAULT_FIXED_HORIZON_MIN = 15


@dataclass
class TradeOutcomeV2:
    decision_ts: datetime
    direction: Recommendation
    window: TimeWindow
    entry_price: float
    entry_idx: int
    trade_date: date
    r_multiple: float
    exit_reason: str
    hold_minutes: int
    fixed_horizon_return: Optional[float]  # comparability column, same horizon convention as v1


@dataclass
class DirectionStats:
    n: int
    win_rate: Optional[float]  # fraction of trades with r_multiple > 0
    r_expectancy: Optional[float]  # mean r_multiple
    avg_hold_minutes: Optional[float]


def _direction_stats(outcomes: Sequence[TradeOutcomeV2]) -> DirectionStats:
    if not outcomes:
        return DirectionStats(n=0, win_rate=None, r_expectancy=None, avg_hold_minutes=None)
    rs = [o.r_multiple for o in outcomes]
    hits = sum(1 for r in rs if r > 0)
    return DirectionStats(
        n=len(outcomes),
        win_rate=hits / len(outcomes),
        r_expectancy=mean(rs),
        avg_hold_minutes=mean(o.hold_minutes for o in outcomes),
    )


@dataclass
class EvalReportV2:
    trade_count: int
    wait_count: int
    overall: DirectionStats
    long: DirectionStats
    short: DirectionStats
    by_fold: dict[int, DirectionStats]
    fixed_horizon_expectancy_bps: Optional[float]
    trades_per_day: Optional[float]
    exit_reason_counts: dict[str, int]


def build_trade_outcomes_v2(
    day_bars_1m: Sequence[Bar],
    decisions: Sequence[Decision],
    snapshots_by_ts: dict[datetime, FeatureSnapshot],
    exit_cfg: ExitConfig,
    *,
    fixed_horizon_min: int = DEFAULT_FIXED_HORIZON_MIN,
) -> list[TradeOutcomeV2]:
    """Pair each non-Wait decision with its entry fill, simulated R-multiple
    exit, and (for comparability) a fixed-horizon forward return.

    ``snapshots_by_ts`` keys on ``to_et(snapshot.decision_ts)`` so this same
    function serves the real run and every negative control (delayed/random/
    shuffled) - a delayed decision's shifted timestamp is looked up against
    the snapshot for THAT later instant (the setup a late-executing trader
    would actually be looking at), not a stale snapshot from the original
    decision time.
    """
    outcomes: list[TradeOutcomeV2] = []
    n = len(day_bars_1m)
    ts_to_idx = {to_et(b.ts): i for i, b in enumerate(day_bars_1m)}

    for d in decisions:
        if d.recommendation == Recommendation.wait:
            continue
        i = ts_to_idx.get(to_et(d.decision_ts))
        if i is None:
            continue
        snap = snapshots_by_ts.get(to_et(d.decision_ts))
        if snap is None or not snap.bars_5m:
            continue  # no trigger bar available -- cannot size a stop, skip rather than fabricate one

        entry_bar = day_bars_1m[i]
        entry_price = entry_bar.open
        direction_sign = 1.0 if d.recommendation == Recommendation.long else -1.0

        trigger_bar = snap.bars_5m[-1]
        atr5 = atr_from_agg_bars(snap.bars_5m, window=14)
        levels = compute_stop_and_target(
            direction_sign, entry_price, trigger_bar.high, trigger_bar.low, atr5, exit_cfg
        )
        if levels is None:
            continue
        stop_price, target_price = levels

        exit_result = simulate_exit(day_bars_1m, i, direction_sign, entry_price, stop_price, target_price, exit_cfg)
        if exit_result is None:
            continue

        fixed_ret: Optional[float] = None
        exit_idx_fixed = i + fixed_horizon_min
        if exit_idx_fixed < n:
            fixed_ret = direction_sign * (day_bars_1m[exit_idx_fixed].close - entry_price) / entry_price

        outcomes.append(
            TradeOutcomeV2(
                decision_ts=d.decision_ts,
                direction=d.recommendation,
                window=d.window,
                entry_price=entry_price,
                entry_idx=i,
                trade_date=to_et(d.decision_ts).date(),
                r_multiple=exit_result.r_multiple,
                exit_reason=exit_result.reason,
                hold_minutes=exit_result.hold_minutes,
                fixed_horizon_return=fixed_ret,
            )
        )
    return outcomes


def evaluate_decisions_v2(
    bars_by_day: dict[date, Sequence[Bar]],
    snapshots_by_day: dict[date, Sequence[FeatureSnapshot]],
    decisions_by_day: dict[date, Sequence[Decision]],
    exit_cfg: ExitConfig,
    *,
    fold_size_days: int = 5,
    fixed_horizon_min: int = DEFAULT_FIXED_HORIZON_MIN,
) -> EvalReportV2:
    all_outcomes: list[TradeOutcomeV2] = []
    wait_count = 0

    for day in sorted(bars_by_day):
        day_bars = bars_by_day[day]
        day_decisions = decisions_by_day.get(day, [])
        wait_count += sum(1 for d in day_decisions if d.recommendation == Recommendation.wait)
        snaps = snapshots_by_day.get(day, [])
        snaps_by_ts = {to_et(s.decision_ts): s for s in snaps}
        all_outcomes.extend(
            build_trade_outcomes_v2(day_bars, day_decisions, snaps_by_ts, exit_cfg, fixed_horizon_min=fixed_horizon_min)
        )

    all_outcomes.sort(key=lambda o: o.decision_ts)

    overall = _direction_stats(all_outcomes)
    longs = [o for o in all_outcomes if o.direction == Recommendation.long]
    shorts = [o for o in all_outcomes if o.direction == Recommendation.short]
    long_stats = _direction_stats(longs)
    short_stats = _direction_stats(shorts)

    trade_dates = [o.trade_date for o in all_outcomes]
    folds = chronological_folds(trade_dates, fold_size_days)
    date_to_fold: dict[date, int] = {}
    for fid, ds in folds.items():
        for d in ds:
            date_to_fold[d] = fid
    by_fold_outcomes: dict[int, list[TradeOutcomeV2]] = {}
    for o in all_outcomes:
        fid = date_to_fold.get(o.trade_date)
        if fid is None:
            continue
        by_fold_outcomes.setdefault(fid, []).append(o)
    by_fold = {fid: _direction_stats(group) for fid, group in by_fold_outcomes.items()}

    fixed_rets = [o.fixed_horizon_return for o in all_outcomes if o.fixed_horizon_return is not None]
    fixed_horizon_expectancy_bps = (mean(fixed_rets) * 10_000.0) if fixed_rets else None

    n_days = len(bars_by_day) or None
    trades_per_day = (len(all_outcomes) / n_days) if n_days else None

    exit_reason_counts: dict[str, int] = {}
    for o in all_outcomes:
        exit_reason_counts[o.exit_reason] = exit_reason_counts.get(o.exit_reason, 0) + 1

    return EvalReportV2(
        trade_count=len(all_outcomes),
        wait_count=wait_count,
        overall=overall,
        long=long_stats,
        short=short_stats,
        by_fold=by_fold,
        fixed_horizon_expectancy_bps=fixed_horizon_expectancy_bps,
        trades_per_day=trades_per_day,
        exit_reason_counts=exit_reason_counts,
    )
