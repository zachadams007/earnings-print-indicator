"""Negative controls for Stage A (project rule 4 - keep failed/rejected evidence honest).

Three controls, each designed to destroy or misalign the hypothesized edge:

1. shuffled intraday bars - a day's 1m bar CONTENT (OHLCV) randomly permuted
   across that day's own timestamp grid, then the whole feature+rule
   pipeline is rerun on the shuffled path (destroys VWAP/ORB/structure
   signal entirely - see ``shuffle_bars_within_day`` docstring for why this
   is NOT a naive full-Bar-object shuffle).
2. delayed signal - the REAL decisions are kept, but shifted N bars later in
   execution time on the real (unshuffled) bars (tests whether the edge is a
   timing artifact rather than a real advance signal).
3. random Long/Short/Wait at matched frequency - same decision timestamps and
   the same overall Long/Short/Wait mix as the real run, but labels randomly
   reassigned (tests whether the rubric beats chance at its own base rate).

If the real rubric's expectancy is not clearly better than these controls,
that is evidence against a genuine edge, not a bug in the harness - see
project rule 4 and research notes conventions elsewhere
in this repo.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Callable, Sequence

from ..data.models import Bar
from .evaluate import EvalReport, HORIZONS_MIN, evaluate_decisions
from .features import FeatureSnapshot, build_feature_snapshots
from .rules import Decision, Recommendation, RuleConfig, decide_all

DecideFn = Callable[[list[FeatureSnapshot], Any], list[Decision]]


def shuffle_bars_within_day(bars_1m: Sequence[Bar], seed: int = 0) -> list[Bar]:
    """Shuffle a day's bar CONTENT across its own timestamp grid - NOT a
    shuffle of the ``Bar`` objects (and therefore their ``ts`` fields)
    themselves.

    Bug this replaces (found by the sweep-v2 lane, 2026-07-09): the original
    implementation did ``rng.shuffle(list(bars_1m))``, permuting whole ``Bar``
    objects including ``ts``. That produces a day with the same 393 minutes'
    worth of bars in a RANDOM chronological order. ``build_feature_snapshots``
    (features.py) feeds bars one at a time into ``_AggBucketBuilder``, an
    O(1)-per-bar incremental N-minute bucket accumulator that is only correct
    when bars arrive in ascending ``ts`` order (its own docstring says so).
    With shuffled order, the builder's ``bucket_start`` jumps around
    non-monotonically, so it flushes (drops) an almost-always-incomplete
    in-progress bucket on nearly every bar. Empirically confirmed on real
    SPY data (2024-01-09): ``bars_3m``/``bars_5m``/``bars_15m`` and
    ``opening_range`` went from populated on ~97-99% of decision snapshots
    (real run) to populated on ZERO snapshots (shuffled run) - every
    confirmation category that reads those fields
    (structure_15m/setup_trigger_5m/execution_timing) can then never fire,
    so every decision degrades to Wait. Across a 21-day dev slice this took
    the shuffled control from a comparable trade count to exactly 0 in
    every config, in both sweep v1 and v2 - the negative control silently
    never ran, and every past "passed the shuffled control" claim was
    vacuous.

    A valid null must destroy the price PATH (the sequence of opens/highs/
    lows/closes/volumes a strategy actually reacts to) while preserving
    every STRUCTURAL invariant the downstream pipeline assumes: one bar per
    minute, ascending timestamps, an intact session grid. So this shuffles
    bar CONTENT only: the day's `ts` values are kept in their original
    ascending sequence, and the OHLCV payloads are permuted across that
    fixed sequence of time slots. The aggregator sees exactly the same
    well-formed, monotonic, one-bar-per-minute session it would on a real
    day (all buckets still complete normally) - but the open/high/low/
    close/volume "observed" at any given minute is now randomized, so any
    edge that depends on the actual sequential price path (VWAP drift,
    HH/HL structure, breakout timing, impulse candles, etc.) is destroyed.
    This makes the control a fair race against the real run: same feature
    machinery, same trade-count ballpark, only the price path is randomized.
    """
    ordered = sorted(bars_1m, key=lambda b: b.ts)
    ts_sequence = [b.ts for b in ordered]
    rng = random.Random(seed)
    shuffled_content = list(ordered)
    rng.shuffle(shuffled_content)
    return [content.model_copy(update={"ts": ts}) for ts, content in zip(ts_sequence, shuffled_content)]


def delay_decisions(
    decisions: Sequence[Decision], n_bars: int, day_bars_1m: Sequence[Bar]
) -> list[Decision]:
    """Shift each non-Wait decision's execution timestamp forward by ``n_bars`` 1m bars.

    Looks up the shifted timestamp on the SAME day's bar list so the delayed
    decision still lands on a real bar; drops decisions that would shift past
    the session's last bar.
    """
    ts_list = [b.ts for b in day_bars_1m]
    ts_index = {ts: i for i, ts in enumerate(ts_list)}

    delayed: list[Decision] = []
    for d in decisions:
        if d.recommendation == Recommendation.wait:
            continue
        i = ts_index.get(d.decision_ts)
        if i is None:
            continue
        new_i = i + n_bars
        if new_i >= len(ts_list):
            continue
        delayed.append(replace(d, decision_ts=ts_list[new_i]))
    return delayed


def random_decisions_matched_frequency(decisions: Sequence[Decision], seed: int = 0) -> list[Decision]:
    """Same decision timestamps and the same Long/Short/Wait mix, randomly reassigned."""
    rng = random.Random(seed)
    labels = [d.recommendation for d in decisions]
    rng.shuffle(labels)
    return [replace(d, recommendation=label) for d, label in zip(decisions, labels)]


@dataclass
class ControlsReport:
    real: EvalReport
    shuffled_bars: EvalReport
    delayed_signal: EvalReport
    random_matched: EvalReport


def run_negative_controls(
    bars_by_day: dict[date, list[Bar]],
    prior_sessions_by_day: dict[date, list[list[Bar]]],
    real_decisions_by_day: dict[date, list[Decision]],
    cfg: RuleConfig,
    *,
    horizons: Sequence[int] = HORIZONS_MIN,
    delay_bars: int = 3,
    seed: int = 7,
    decide_fn: DecideFn = decide_all,
) -> ControlsReport:
    """Build and evaluate all three negative controls against the real run.

    ``decide_fn`` defaults to the Stage A placeholder rubric's ``decide_all``
    for full backward compatibility; pass ``bot.scalp.rules_v01.decide_core_gate_all``
    (with a matching ``CoreGateConfig`` as ``cfg``) to run controls against the
    v0.1 core-gate decision model instead - see rules_v01.py.
    """
    real_report = evaluate_decisions(bars_by_day, real_decisions_by_day, horizons=horizons)

    # 1. Shuffled bars - rebuild features + decisions on the permuted path.
    shuffled_bars_by_day: dict[date, list[Bar]] = {
        d: shuffle_bars_within_day(b, seed=seed) for d, b in bars_by_day.items()
    }
    shuffled_decisions_by_day: dict[date, list[Decision]] = {}
    for d, day_bars in shuffled_bars_by_day.items():
        snaps = build_feature_snapshots(day_bars, prior_sessions_by_day.get(d, []))
        shuffled_decisions_by_day[d] = decide_fn(snaps, cfg)
    shuffled_report = evaluate_decisions(shuffled_bars_by_day, shuffled_decisions_by_day, horizons=horizons)

    # 2. Delayed signal - same real decisions, executed late on the real bars.
    delayed_decisions_by_day: dict[date, list[Decision]] = {
        d: delay_decisions(real_decisions_by_day.get(d, []), delay_bars, bars_by_day[d])
        for d in bars_by_day
    }
    delayed_report = evaluate_decisions(bars_by_day, delayed_decisions_by_day, horizons=horizons)

    # 3. Random labels at matched frequency, same timestamps, same real bars.
    random_decisions_by_day: dict[date, list[Decision]] = {
        d: random_decisions_matched_frequency(real_decisions_by_day.get(d, []), seed=seed + i)
        for i, d in enumerate(sorted(bars_by_day))
    }
    random_report = evaluate_decisions(bars_by_day, random_decisions_by_day, horizons=horizons)

    return ControlsReport(
        real=real_report,
        shuffled_bars=shuffled_report,
        delayed_signal=delayed_report,
        random_matched=random_report,
    )
