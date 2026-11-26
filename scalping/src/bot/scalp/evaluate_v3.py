"""R-multiple + bps trade evaluation for the Stage A sweep v3 (rules_v03.py /
exits.py's ExitConfigV3) -- the owner's HIGH-WIN-RATE hunt (2026-07-09).

ADDITIVE: does not touch evaluate.py or evaluate_v2.py. Reuses evaluate_v2's
``DirectionStats`` / ``TradeOutcomeV2`` / ``EvalReportV2`` / ``_direction_stats``
building blocks directly (duck-typed -- ``_direction_stats`` only reads
``.r_multiple``/``.hold_minutes`` off whatever outcome objects it is given) and
``evaluate.chronological_folds`` for fold assignment, same as v2.

The ONE new piece is exit-LEVEL computation: ``build_trade_outcomes_v3``
dispatches to ``exits.compute_stop_and_target_v3`` (threading through
``snapshot.vwap`` and a computed swing level) when given an ``ExitConfigV3``,
or falls back to the existing ``exits.compute_stop_and_target`` for a plain
``ExitConfig`` -- so this same evaluator serves BOTH the two brand-new v0.3
exit archetypes (vwap_half / prior_swing_capped) AND the two v0.3 arms that
reuse plain R-multiple exits (band-fade scratch-quick, midline-retest-hold),
without duplicating the plain-exit branch's logic.

NEW reporting fields this module adds beyond ``EvalReportV2`` (the owner's explicit
ask: "expectancy net@1bp-per-side in bps" per config):
  - ``gross_expectancy_bps``: mean per-trade price return in bps, UNCOSTED.
    Computed exactly (not approximated) as ``r_multiple * risk_bps`` per
    trade, where ``risk_bps = |entry - stop| / entry * 10_000`` -- this equals
    ``direction_sign * (exit - entry) / entry * 10_000`` algebraically (both
    sides divide by risk_price then multiply back by it), so no extra price
    field needs to be threaded through ``ExitResult``.
  - ``net_expectancy_bps_1bp``: ``gross_expectancy_bps - ROUND_TRIP_COST_BPS``
    (1bp per side, 2bps round trip -- the owner's stated cost assumption for this
    sweep). A cost-adjusted MEAN is a linear function of the per-trade mean,
    so subtracting a constant is exact, not an approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean
from typing import Optional, Sequence, Union

from ..data.models import Bar
from .evaluate import chronological_folds
from .evaluate_v2 import DirectionStats, TradeOutcomeV2, _direction_stats
from .exits import (
    ExitConfig,
    ExitConfigV3,
    atr_from_agg_bars,
    compute_stop_and_target,
    compute_stop_and_target_v3,
    simulate_exit,
    swing_level_from_agg_bars,
)
from .features import FeatureSnapshot, TimeWindow, to_et
from .rules import Decision, Recommendation

DEFAULT_FIXED_HORIZON_MIN = 15
COST_BPS_PER_SIDE = 1.0
ROUND_TRIP_COST_BPS = 2.0 * COST_BPS_PER_SIDE

ExitCfgV3 = Union[ExitConfig, ExitConfigV3]


@dataclass
class TradeOutcomeV3:
    decision_ts: datetime
    direction: Recommendation
    window: TimeWindow
    entry_price: float
    entry_idx: int
    trade_date: date
    r_multiple: float
    trade_bps: float  # exact, uncosted price return in bps (direction-adjusted)
    exit_reason: str
    hold_minutes: int
    fixed_horizon_return: Optional[float]


@dataclass
class EvalReportV3:
    trade_count: int
    wait_count: int
    overall: DirectionStats
    long: DirectionStats
    short: DirectionStats
    by_fold: dict[int, DirectionStats]
    fixed_horizon_expectancy_bps: Optional[float]
    trades_per_day: Optional[float]
    exit_reason_counts: dict[str, int]
    gross_expectancy_bps: Optional[float]
    net_expectancy_bps_1bp: Optional[float]


def build_trade_outcomes_v3(
    day_bars_1m: Sequence[Bar],
    decisions: Sequence[Decision],
    snapshots_by_ts: dict[datetime, FeatureSnapshot],
    exit_cfg: ExitCfgV3,
    *,
    fixed_horizon_min: int = DEFAULT_FIXED_HORIZON_MIN,
) -> list[TradeOutcomeV3]:
    outcomes: list[TradeOutcomeV3] = []
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

        if isinstance(exit_cfg, ExitConfigV3):
            swing_level = None
            if exit_cfg.target_kind == "prior_swing_capped":
                swing_level = swing_level_from_agg_bars(snap.bars_5m, exit_cfg.swing_lookback, direction_sign > 0)
            levels = compute_stop_and_target_v3(
                direction_sign, entry_price, trigger_bar.high, trigger_bar.low, atr5, snap.vwap, swing_level, exit_cfg
            )
        else:
            levels = compute_stop_and_target(
                direction_sign, entry_price, trigger_bar.high, trigger_bar.low, atr5, exit_cfg
            )
        if levels is None:
            continue
        stop_price, target_price = levels

        exit_result = simulate_exit(day_bars_1m, i, direction_sign, entry_price, stop_price, target_price, exit_cfg)
        if exit_result is None:
            continue

        risk_bps = abs(entry_price - stop_price) / entry_price * 10_000.0
        trade_bps = exit_result.r_multiple * risk_bps  # exact -- see module docstring

        fixed_ret: Optional[float] = None
        exit_idx_fixed = i + fixed_horizon_min
        if exit_idx_fixed < n:
            fixed_ret = direction_sign * (day_bars_1m[exit_idx_fixed].close - entry_price) / entry_price

        outcomes.append(
            TradeOutcomeV3(
                decision_ts=d.decision_ts,
                direction=d.recommendation,
                window=d.window,
                entry_price=entry_price,
                entry_idx=i,
                trade_date=to_et(d.decision_ts).date(),
                r_multiple=exit_result.r_multiple,
                trade_bps=trade_bps,
                exit_reason=exit_result.reason,
                hold_minutes=exit_result.hold_minutes,
                fixed_horizon_return=fixed_ret,
            )
        )
    return outcomes


def evaluate_decisions_v3(
    bars_by_day: dict[date, Sequence[Bar]],
    snapshots_by_day: dict[date, Sequence[FeatureSnapshot]],
    decisions_by_day: dict[date, Sequence[Decision]],
    exit_cfg: ExitCfgV3,
    *,
    fold_size_days: int = 5,
    fixed_horizon_min: int = DEFAULT_FIXED_HORIZON_MIN,
) -> EvalReportV3:
    all_outcomes: list[TradeOutcomeV3] = []
    wait_count = 0

    for day in sorted(bars_by_day):
        day_bars = bars_by_day[day]
        day_decisions = decisions_by_day.get(day, [])
        wait_count += sum(1 for d in day_decisions if d.recommendation == Recommendation.wait)
        snaps = snapshots_by_day.get(day, [])
        snaps_by_ts = {to_et(s.decision_ts): s for s in snaps}
        all_outcomes.extend(
            build_trade_outcomes_v3(day_bars, day_decisions, snaps_by_ts, exit_cfg, fixed_horizon_min=fixed_horizon_min)
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
    by_fold_outcomes: dict[int, list[TradeOutcomeV3]] = {}
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

    gross_bps = mean(o.trade_bps for o in all_outcomes) if all_outcomes else None
    net_bps = (gross_bps - ROUND_TRIP_COST_BPS) if gross_bps is not None else None

    return EvalReportV3(
        trade_count=len(all_outcomes),
        wait_count=wait_count,
        overall=overall,
        long=long_stats,
        short=short_stats,
        by_fold=by_fold,
        fixed_horizon_expectancy_bps=fixed_horizon_expectancy_bps,
        trades_per_day=trades_per_day,
        exit_reason_counts=exit_reason_counts,
        gross_expectancy_bps=gross_bps,
        net_expectancy_bps_1bp=net_bps,
    )
