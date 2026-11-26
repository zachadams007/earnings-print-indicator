"""SWEEP v5 "HARVEST" -- practitioner strategy harvest confirmation testing
(docs/STRATEGY-HARVEST-2026-07-09.md, 2026-07-09 night).

Five PRE-REGISTERED strategy specs (read the doc before touching any rule
below -- this module implements those exact rules; only the doc's own named
parameter ranges are swept):

  1. ib_breakout             -- Initial-Balance day-type breakout, volume-
                                 confirmed, IB-width gated.
  2. gap_continuation        -- ATR-normalized gap continuation.
  3. opening_drive_reversal  -- 10:00 AM opening-drive reversal fade.
  4. nr7_breakout            -- NR7 / IDNR7 contraction -> next-day breakout.
  5. turtle_soup             -- rolling N-bar false-breakout fade.

DATA CONVENTION (matches noise_boundary.py, NOT features.py): ``bar.ts`` is
read as a CLOSE-stamp directly (no -1-minute shift). This module builds on
top of ``bot.scalp.noise_boundary``'s ``DaySession``/``Trade``/``side_stats``/
``fold_consistency``/``yearly_expectancy``/``random_matched_control`` machinery
rather than duplicating it -- session construction (RTH filter, VWAP-by-time)
is identical to that module and importing it keeps ``build_day_session``'s
RTH-window semantics ("bar ts in (9:30, 16:00] ET") as the single source of
truth for what counts as a completed 1-minute candle at a given clock time.

POINT-IN-TIME DISCIPLINE (every strategy below follows this SAME pattern):
  - A trigger is detected off a bar's CLOSE (a fully-formed candle).
  - The resulting order is FILLED at the very NEXT 1-minute bar's OPEN --
    never the trigger bar's own close/high/low. This is the same convention
    ``exits.py``'s ``simulate_exit``/``compute_stop_and_target`` uses
    elsewhere in ``bot.scalp``.
  - Structural reference levels (IB high/low, prior-day OHLC, NR7 range,
    rolling N-bar extreme) are always computed from bars strictly BEFORE the
    trigger bar (or, for cross-day levels like NR7/ATR, strictly before the
    trigger DAY) -- see each ``build_intents_*`` function's own comments.

Every strategy is expressed as a two-stage pipeline:
  1. ``build_intents_*`` scans (point-in-time) and emits ``TradeIntent``
     objects -- the STRUCTURAL decision (direction, stop, target, time-stop)
     is fixed at trigger time; nothing about it depends on what happens after
     the fill bar.
  2. ``_materialize`` (real) / ``_materialize_delayed`` (delayed-signal
     control) walk a ``TradeIntent`` forward bar-by-bar via ``_walk_generic``
     (same conservative "stop checked before target" both-hit tie-break as
     ``exits.simulate_exit``) to produce a ``noise_boundary.Trade``.

This split is what makes the delayed-signal negative control cheap and
faithful: the SAME structural levels (stop/target were fixed using only
pre-trigger information) are re-walked from an entry N bars later -- exactly
``bot.scalp.controls.delay_decisions``'s "same decision, executed late"
semantics, generalized to a stop/target-carrying intent instead of a bare
Decision.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, time
from enum import Enum
from statistics import mean, median
from typing import Literal, Optional, Sequence

from ..data.models import Bar
from .features import DailyOHLC
from .features import atr as daily_atr
from .noise_boundary import (
    DaySession,
    Trade,
    build_day_session,
    to_et,
)

__all__ = [
    "IBConfig",
    "GapConfig",
    "DriveConfig",
    "NR7Config",
    "TurtleSoupConfig",
    "StrategyKind",
    "TradeIntent",
    "WalkResult",
    "build_sessions",
    "build_intents",
    "build_intents_ib",
    "build_intents_gap",
    "build_intents_drive",
    "build_intents_nr7",
    "build_intents_turtle_soup",
    "evaluate_config",
]


# ── shared primitives ────────────────────────────────────────────────────


def build_sessions(symbol: str, bars_by_day: dict) -> dict:
    """One ``DaySession`` per calendar day with usable RTH bars (see
    ``noise_boundary.build_day_session``'s own RTH-filter docstring)."""
    out: dict = {}
    for d, bars in bars_by_day.items():
        s = build_day_session(symbol, d, bars)
        if s is not None:
            out[d] = s
    return out


def _day_bars_list(session: DaySession) -> list:
    return [session.by_time[t] for t in session.ordered_times]


def _daily_ohlc_from_session(session: DaySession) -> DailyOHLC:
    """RTH-only daily OHLC derived from one ``DaySession`` -- deliberately
    NOT ``features.daily_ohlc_from_1m`` (that function aggregates whatever
    bars it is given with no RTH filter; ``scalp.db`` also carries extended-
    hours rows, per ``noise_boundary.build_day_session``'s own filter, so an
    unfiltered daily OHLC would leak pre/post-market extremes into ATR/NR7/
    gap reference levels)."""
    bars = list(session.by_time.values())
    return DailyOHLC(
        trade_date=session.day,
        open=session.open_price,
        high=max(b.high for b in bars),
        low=min(b.low for b in bars),
        close=session.close_price,
    )


@dataclass(frozen=True)
class TradeIntent:
    """A fully-specified, point-in-time trade setup, fixed at trigger time.

    ``day_bars`` is the trigger day's own bar list (0-indexed); ``entry_idx``
    is the PIT-safe fill index (the bar strictly AFTER the trigger bar --
    never the trigger bar itself). ``max_idx`` is the last index in
    ``day_bars`` the trade may be held to (a time-stop mark or end-of-day).
    """

    symbol: str
    day: date
    day_bars: list  # list[Bar], 0-indexed for this trading day
    entry_idx: int
    direction: Literal["long", "short"]
    stop_price: float
    target_price: Optional[float]
    max_idx: int


@dataclass(frozen=True)
class WalkResult:
    exit_price: float
    exit_reason: Literal["stop", "target", "time_stop", "eod_close"]
    exit_idx: int
    hold_minutes: int


def _walk_generic(
    day_bars: Sequence,
    entry_idx: int,
    direction: Literal["long", "short"],
    entry_price: float,
    stop_price: float,
    target_price: Optional[float],
    max_idx: int,
) -> WalkResult:
    """Walk forward from ``entry_idx`` (inclusive -- the fill bar's own
    high/low ARE checked, matching ``exits.simulate_exit``'s convention:
    once filled at that bar's open, the rest of that same bar's range is
    fair game for the same-bar stop/target check). Stop is checked BEFORE
    target on any bar where both levels fall inside that bar's [low, high]
    -- the same conservative both-hit tie-break ``exits.py`` documents and
    uses throughout ``bot.scalp``.
    """
    n = len(day_bars)
    max_idx = min(max_idx, n - 1)
    entry_ts = day_bars[entry_idx].ts
    for j in range(entry_idx, max_idx + 1):
        b = day_bars[j]
        hit_stop = (b.low <= stop_price) if direction == "long" else (b.high >= stop_price)
        hit_target = target_price is not None and (
            (b.high >= target_price) if direction == "long" else (b.low <= target_price)
        )
        if hit_stop:  # checked first -- conservative both-hit tie-break
            hold = max(0, int((b.ts - entry_ts).total_seconds() // 60))
            return WalkResult(stop_price, "stop", j, hold)
        if hit_target:
            hold = max(0, int((b.ts - entry_ts).total_seconds() // 60))
            return WalkResult(target_price, "target", j, hold)
    exit_bar = day_bars[max_idx]
    reason: Literal["time_stop", "eod_close"] = "eod_close" if max_idx == n - 1 else "time_stop"
    hold = max(0, int((exit_bar.ts - entry_ts).total_seconds() // 60))
    return WalkResult(exit_bar.close, reason, max_idx, hold)


def _make_trade(it: TradeIntent, entry_idx: int, entry_price: float, wr: WalkResult) -> Trade:
    return Trade(
        symbol=it.symbol,
        day=it.day,
        direction=it.direction,
        entry_mark=to_et(it.day_bars[entry_idx].ts).time(),
        entry_price=entry_price,
        entry_stop_price=it.stop_price,
        exit_mark=to_et(it.day_bars[wr.exit_idx].ts).time(),
        exit_price=wr.exit_price,
        exit_reason=wr.exit_reason,
        hold_minutes=wr.hold_minutes,
    )


def _materialize(intents: Sequence[TradeIntent]) -> list[Trade]:
    out: list[Trade] = []
    for it in intents:
        entry_price = it.day_bars[it.entry_idx].open
        wr = _walk_generic(it.day_bars, it.entry_idx, it.direction, entry_price, it.stop_price, it.target_price, it.max_idx)
        out.append(_make_trade(it, it.entry_idx, entry_price, wr))
    return out


def _materialize_delayed(intents: Sequence[TradeIntent], delay_bars: int) -> list[Trade]:
    """Delayed-signal negative control: SAME structural levels (stop/target
    were fixed at trigger time using only pre-trigger information), but the
    entry fill is pushed ``delay_bars`` bars later. Mirrors
    ``bot.scalp.controls.delay_decisions``'s "same decision, executed late"
    semantics. Skips a trade if the delayed entry would fall on/after the
    time-stop, or if the (unchanged) stop price no longer makes sense on the
    other side of the new, later entry price (degenerate -- would fabricate
    negative or zero initial risk).
    """
    out: list[Trade] = []
    for it in intents:
        new_idx = it.entry_idx + delay_bars
        if new_idx > it.max_idx or new_idx >= len(it.day_bars):
            continue
        entry_price = it.day_bars[new_idx].open
        if it.direction == "long" and it.stop_price >= entry_price:
            continue
        if it.direction == "short" and it.stop_price <= entry_price:
            continue
        wr = _walk_generic(it.day_bars, new_idx, it.direction, entry_price, it.stop_price, it.target_price, it.max_idx)
        out.append(_make_trade(it, new_idx, entry_price, wr))
    return out


# ── 1. Initial-Balance day-type breakout ────────────────────────────────


@dataclass(frozen=True)
class IBConfig:
    """docs/STRATEGY-HARVEST-2026-07-09.md item 1. ``target_mult`` in {1.0,
    2.0} and ``day_type_gate`` on/off (ablation of the doc's own claimed
    mechanism -- "regime-gates the breakout population") are this module's
    only swept axes; every other number (1.5x wide-IB skip, 1.5x volume
    confirm, 20-day lookback, 50%-back-inside stop) is fixed per the spec.
    """

    wide_ib_skip_mult: float = 1.5
    vol_confirm_mult: float = 1.5
    target_mult: float = 1.0
    day_type_gate: bool = True
    lookback_days: int = 20
    ib_end: time = time(10, 30)
    label: str = "ib_breakout"


def _compute_ib(session: DaySession, ib_end: time) -> Optional[tuple]:
    """(high, low, width, median_volume) over bars in (9:30, ib_end] ET.
    None if the window isn't (near) fully populated (data gap / early close)
    -- an under-populated IB would understate its own width and silently
    bias the wide-day gate.
    """
    times = [t for t in session.ordered_times if t <= ib_end]
    if len(times) < 55:  # need close to the full 60-bar (9:30-10:30) window
        return None
    bars = [session.by_time[t] for t in times]
    hi = max(b.high for b in bars)
    lo = min(b.low for b in bars)
    width = hi - lo
    med_vol = median(b.volume for b in bars)
    return hi, lo, width, med_vol


def build_intents_ib(
    symbol: str, sessions: dict, eval_start: date, eval_end: date, cfg: IBConfig
) -> list[TradeIntent]:
    sorted_days = sorted(sessions)
    ib_by_day = {d: _compute_ib(sessions[d], cfg.ib_end) for d in sorted_days}
    intents: list[TradeIntent] = []

    for idx, d in enumerate(sorted_days):
        if d < eval_start or d > eval_end:
            continue
        ib = ib_by_day[d]
        if ib is None:
            continue
        hi, lo, width, med_vol = ib
        if width <= 0:
            continue

        if cfg.day_type_gate:
            # Trailing 20-day avg IB width -- strictly PRIOR days only
            # (sorted_days[idx-lookback:idx] never includes d itself).
            prior = [ib_by_day[sorted_days[j]] for j in range(max(0, idx - cfg.lookback_days), idx)]
            prior_widths = [p[2] for p in prior if p is not None]
            if len(prior_widths) < cfg.lookback_days:
                continue  # insufficient IB history for the gate -- skip rather than guess
            avg_width = mean(prior_widths)
            if avg_width > 0 and width > cfg.wide_ib_skip_mult * avg_width:
                continue  # wide-IB day -- skipped per the spec's day-type gate

        session = sessions[d]
        day_bars = _day_bars_list(session)
        times = session.ordered_times
        last_idx = len(day_bars) - 1

        for i, t in enumerate(times):
            if t <= cfg.ib_end:
                continue  # IB itself hasn't closed yet -- no entries inside it
            b = day_bars[i]
            direction: Optional[str] = None
            if b.close > hi and b.volume >= cfg.vol_confirm_mult * med_vol:
                direction = "long"
            elif b.close < lo and b.volume >= cfg.vol_confirm_mult * med_vol:
                direction = "short"
            if direction is None:
                continue

            fill_idx = i + 1  # PIT: fill at the NEXT bar's open, never bar i's own price
            if fill_idx > last_idx:
                break  # no bar left today to fill on -- can't fabricate a fill
            entry_price = day_bars[fill_idx].open
            mid = (hi + lo) / 2.0  # "50% back inside the range" -- the IB midpoint
            if direction == "long":
                if mid >= entry_price:
                    break  # degenerate (stop already through/at entry) -- skip
                target = hi + cfg.target_mult * width
            else:
                if mid <= entry_price:
                    break
                target = lo - cfg.target_mult * width

            intents.append(TradeIntent(symbol, d, day_bars, fill_idx, direction, mid, target, last_idx))
            break  # one trade per day for this family
    return intents


# ── 2. ATR-normalized gap continuation ──────────────────────────────────


@dataclass(frozen=True)
class GapConfig:
    """docs item 2. ``atr_mult`` in {0.7, 1.2} is the doc's own named
    sweep axis (the two "arms" the doc explicitly asks for)."""

    atr_mult: float = 0.7
    atr_window: int = 14
    open_range_end: time = time(9, 35)
    label: str = "gap_continuation"


def _classify_gap(today_open: float, prior_close: float, atr14: float, atr_mult: float) -> Optional[str]:
    """Pure gap classifier -- isolated for unit testing "gap classification
    at the open" independent of any session/loader plumbing. Returns
    "long"/"short" only if the gap magnitude clears ``atr_mult * atr14``;
    None (no trade) otherwise."""
    gap = today_open - prior_close
    if abs(gap) < atr_mult * atr14:
        return None
    return "long" if gap > 0 else "short"


def build_intents_gap(
    symbol: str, sessions: dict, eval_start: date, eval_end: date, cfg: GapConfig
) -> list[TradeIntent]:
    sorted_days = sorted(sessions)
    daily = {d: _daily_ohlc_from_session(sessions[d]) for d in sorted_days}
    daily_sorted = [daily[d] for d in sorted_days]
    intents: list[TradeIntent] = []

    for idx, d in enumerate(sorted_days):
        if d < eval_start or d > eval_end:
            continue
        if idx == 0:
            continue  # no prior close to gap from
        atr14 = daily_atr(daily_sorted[:idx], window=cfg.atr_window)  # strictly prior days
        if atr14 is None or atr14 <= 0:
            continue
        prior_close = daily_sorted[idx - 1].close
        prior_range = daily_sorted[idx - 1].high - daily_sorted[idx - 1].low

        session = sessions[d]
        today_open = session.open_price
        direction = _classify_gap(today_open, prior_close, atr14, cfg.atr_mult)
        if direction is None:
            continue

        times = session.ordered_times
        day_bars = _day_bars_list(session)
        last_idx = len(day_bars) - 1
        open_range_times = [t for t in times if t <= cfg.open_range_end]
        if len(open_range_times) < 4:
            continue  # opening range not (near) fully formed -- data gap
        or_bars = [session.by_time[t] for t in open_range_times]
        or_high = max(b.high for b in or_bars)
        or_low = min(b.low for b in or_bars)

        for i, t in enumerate(times):
            if t <= cfg.open_range_end:
                continue
            b = day_bars[i]
            triggered = (direction == "long" and b.close > or_high) or (
                direction == "short" and b.close < or_low
            )
            if not triggered:
                continue

            fill_idx = i + 1
            if fill_idx > last_idx:
                break
            entry_price = day_bars[fill_idx].open
            stop = today_open  # "full gap giveback (open price)" -- per the spec, literally the session open
            if direction == "long" and stop >= entry_price:
                break
            if direction == "short" and stop <= entry_price:
                break
            sign = 1.0 if direction == "long" else -1.0
            target = today_open + sign * prior_range  # prior-day range projected from the open

            intents.append(TradeIntent(symbol, d, day_bars, fill_idx, direction, stop, target, last_idx))
            break  # one trade per day
    return intents


# ── 3. 10:00 AM opening-drive reversal ──────────────────────────────────


@dataclass(frozen=True)
class DriveConfig:
    """docs item 3. Three named threshold arms: pct=0.4%, pct=0.5%,
    atr=0.7x prior ATR(14)."""

    threshold_kind: Literal["pct", "atr"] = "pct"
    threshold_pct: float = 0.004
    threshold_atr_mult: float = 0.7
    atr_window: int = 14
    drive_end: time = time(10, 0)
    confirm_end: time = time(10, 15)
    time_stop: time = time(11, 0)
    label: str = "opening_drive_reversal"


def _confirmed_no_new_extreme(confirm_bars: Sequence, extreme: float, drive_up: bool) -> bool:
    """True iff none of ``confirm_bars`` prints a NEW extreme beyond
    ``extreme`` (the 9:30-10:00 high for an up-drive / low for a down-drive)
    -- isolated for direct unit testing of the "no new extreme prints
    10:00-10:15" confirmation gate."""
    for b in confirm_bars:
        if drive_up and b.high > extreme:
            return False
        if not drive_up and b.low < extreme:
            return False
    return True


def build_intents_drive(
    symbol: str, sessions: dict, eval_start: date, eval_end: date, cfg: DriveConfig
) -> list[TradeIntent]:
    sorted_days = sorted(sessions)
    daily = {d: _daily_ohlc_from_session(sessions[d]) for d in sorted_days}
    daily_sorted = [daily[d] for d in sorted_days]
    intents: list[TradeIntent] = []

    for idx, d in enumerate(sorted_days):
        if d < eval_start or d > eval_end:
            continue
        session = sessions[d]
        times = session.ordered_times
        day_bars = _day_bars_list(session)
        last_idx = len(day_bars) - 1

        pre10_times = [t for t in times if t <= cfg.drive_end]
        if not pre10_times or pre10_times[-1] != cfg.drive_end:
            continue  # no bar closing exactly at 10:00 -- data gap, skip
        bar_10 = session.by_time[cfg.drive_end]
        open_930 = session.open_price
        drive_move = bar_10.close - open_930

        if cfg.threshold_kind == "pct":
            threshold_move = cfg.threshold_pct * open_930
        else:
            atr14 = daily_atr(daily_sorted[:idx], window=cfg.atr_window)  # strictly prior days
            if atr14 is None or atr14 <= 0:
                continue
            threshold_move = cfg.threshold_atr_mult * atr14

        if abs(drive_move) < threshold_move:
            continue  # drive too small -- no setup today

        drive_up = drive_move > 0
        pre10_bars = [session.by_time[t] for t in pre10_times]
        extreme = max(b.high for b in pre10_bars) if drive_up else min(b.low for b in pre10_bars)

        confirm_times = [t for t in times if cfg.drive_end < t <= cfg.confirm_end]
        if not confirm_times:
            continue
        confirm_bars = [session.by_time[t] for t in confirm_times]
        if not _confirmed_no_new_extreme(confirm_bars, extreme, drive_up):
            continue  # a new extreme printed -- drive is still extending, no fade

        confirm_mark = confirm_times[-1]
        confirm_idx = times.index(confirm_mark)
        fill_idx = confirm_idx + 1
        if fill_idx > last_idx:
            continue

        direction = "short" if drive_up else "long"
        entry_price = day_bars[fill_idx].open
        stop = extreme
        if direction == "short" and stop <= entry_price:
            continue
        if direction == "long" and stop >= entry_price:
            continue

        # Target = VWAP/open (the spec's stated magnets); fall back to the
        # open if VWAP is already on the wrong side of entry, and to no
        # target at all (stop + time-stop only) if even the open is
        # unusable -- never fabricate a target on the losing side of entry.
        target = session.vwap_by_time.get(confirm_mark)
        if target is None or (direction == "short" and target >= entry_price) or (
            direction == "long" and target <= entry_price
        ):
            if direction == "short" and open_930 < entry_price:
                target = open_930
            elif direction == "long" and open_930 > entry_price:
                target = open_930
            else:
                target = None

        time_stop_times = [t for t in times if t <= cfg.time_stop]
        if not time_stop_times:
            continue
        max_idx = times.index(time_stop_times[-1])
        if max_idx < fill_idx:
            continue

        intents.append(TradeIntent(symbol, d, day_bars, fill_idx, direction, stop, target, max_idx))
    return intents


# ── 4. NR7 / IDNR7 contraction -> next-day breakout ─────────────────────


@dataclass(frozen=True)
class NR7Config:
    """docs item 4. ``require_inside_day=False`` -> NR7 (narrowest-of-7);
    ``True`` -> IDNR7 (also must be an inside day, Crabel's stricter
    variant). ``target_mult`` in {1.0, 2.0}: the spec states the STOP
    ("opposite side of the NR7 range") but not a target -- this module picks
    a range-projection target (same convention as the IB strategy's 1x/2x
    projection) as a documented, reasonable completion, swept exactly like
    that other family's target_mult so the choice is visible in the ledger
    rather than silently baked in.
    """

    require_inside_day: bool = False
    target_mult: float = 1.0
    lookback_days: int = 7
    label: str = "nr7_breakout"


def _is_nr7_day(daily_sorted: Sequence[DailyOHLC], prior_idx: int, lookback_days: int, require_inside_day: bool) -> bool:
    """True iff ``daily_sorted[prior_idx]`` (day D-1, relative to the
    trigger day D at ``prior_idx + 1``) is NR7 (optionally IDNR7). Reads
    ONLY ``daily_sorted[: prior_idx + 1]`` -- isolated specifically so a
    test can assert "flagged only from completed prior days" by mutating
    ``daily_sorted[prior_idx + 1:]`` (day D and later) and checking the
    flag is unaffected.
    """
    if prior_idx - lookback_days + 1 < 0:
        return False
    window = daily_sorted[prior_idx - lookback_days + 1 : prior_idx + 1]
    if len(window) < lookback_days:
        return False
    prior_day = daily_sorted[prior_idx]
    prior_range = prior_day.high - prior_day.low
    ranges = [w.high - w.low for w in window]
    if prior_range != min(ranges):
        return False
    if require_inside_day:
        if prior_idx - 1 < 0:
            return False
        prior2 = daily_sorted[prior_idx - 1]
        if not (prior_day.high <= prior2.high and prior_day.low >= prior2.low):
            return False
    return True


def build_intents_nr7(
    symbol: str, sessions: dict, eval_start: date, eval_end: date, cfg: NR7Config
) -> list[TradeIntent]:
    sorted_days = sorted(sessions)
    daily = {d: _daily_ohlc_from_session(sessions[d]) for d in sorted_days}
    daily_sorted = [daily[d] for d in sorted_days]
    intents: list[TradeIntent] = []

    for idx, d in enumerate(sorted_days):
        if d < eval_start or d > eval_end:
            continue
        if idx < 1:
            continue
        prior_idx = idx - 1  # D-1
        if not _is_nr7_day(daily_sorted, prior_idx, cfg.lookback_days, cfg.require_inside_day):
            continue
        prior_day = daily_sorted[prior_idx]
        prior_range = prior_day.high - prior_day.low
        if prior_range <= 0:
            continue

        session = sessions[d]
        times = session.ordered_times
        day_bars = _day_bars_list(session)
        last_idx = len(day_bars) - 1

        for i, t in enumerate(times):
            b = day_bars[i]
            direction: Optional[str] = None
            if b.close > prior_day.high:
                direction = "long"
            elif b.close < prior_day.low:
                direction = "short"
            if direction is None:
                continue

            fill_idx = i + 1
            if fill_idx > last_idx:
                break
            entry_price = day_bars[fill_idx].open
            if direction == "long":
                stop = prior_day.low
                if stop >= entry_price:
                    break
                target = prior_day.high + cfg.target_mult * prior_range
            else:
                stop = prior_day.high
                if stop <= entry_price:
                    break
                target = prior_day.low - cfg.target_mult * prior_range

            intents.append(TradeIntent(symbol, d, day_bars, fill_idx, direction, stop, target, last_idx))
            break  # one trade per day
    return intents


# ── 5. Turtle Soup rolling N-bar false-breakout fade ────────────────────


@dataclass(frozen=True)
class TurtleSoupConfig:
    """docs item 5. ``n_bars`` in {195 (half-day), 390 (full-day)} 1-minute
    bars is the doc's own named sweep axis; ``reclaim_within_bars`` in
    {1, 2} is the doc's stated reclaim window. ``min_separation_bars=3`` and
    ``min_rr=2.0`` ("min 1:2") are fixed per the spec.
    """

    n_bars: int = 195
    min_separation_bars: int = 3
    reclaim_within_bars: int = 2
    min_rr: float = 2.0
    label: str = "turtle_soup"


def _rolling_extreme_before(values: Sequence[float], n: int, take_max: bool) -> list:
    """out[i] = (extreme_value, extreme_idx) over values[i-n : i] (the n
    values strictly BEFORE i) once i >= n, else None. O(len(values)) via a
    monotonic deque -- the standard "sliding window extremum" construction.
    Strictly-before-i semantics (never includes values[i] itself) is what
    lets ``build_intents_turtle_soup`` detect "bar i sets a NEW N-bar
    extreme" as a simple ``values[i] > out[i][0]`` (or ``<`` for lows)
    comparison.
    """
    dq: deque = deque()
    out: list = [None] * len(values)
    for i in range(len(values)):
        if i >= 1:
            j = i - 1
            v = values[j]
            if take_max:
                while dq and values[dq[-1]] <= v:
                    dq.pop()
            else:
                while dq and values[dq[-1]] >= v:
                    dq.pop()
            dq.append(j)
        while dq and dq[0] < i - n:
            dq.popleft()
        if i >= n and dq:
            out[i] = (values[dq[0]], dq[0])
    return out


def build_intents_turtle_soup(
    symbol: str, sessions: dict, eval_start: date, eval_end: date, cfg: TurtleSoupConfig
) -> list[TradeIntent]:
    sorted_days = sorted(sessions)
    merged_bars: list = []
    day_of_idx: list = []
    day_start_idx: dict = {}
    day_end_idx: dict = {}
    for d in sorted_days:
        bars = _day_bars_list(sessions[d])
        start = len(merged_bars)
        merged_bars.extend(bars)
        end = len(merged_bars) - 1
        day_start_idx[d] = start
        day_end_idx[d] = end
        day_of_idx.extend([d] * len(bars))

    highs = [b.high for b in merged_bars]
    lows = [b.low for b in merged_bars]
    closes = [b.close for b in merged_bars]
    roll_high = _rolling_extreme_before(highs, cfg.n_bars, take_max=True)
    roll_low = _rolling_extreme_before(lows, cfg.n_bars, take_max=False)

    intents: list[TradeIntent] = []
    busy_until = -1  # merged-array index; no new signal candidates <= this

    for i in range(len(merged_bars)):
        d_i = day_of_idx[i]
        if d_i < eval_start or d_i > eval_end:
            continue
        if i <= busy_until:
            continue

        rh = roll_high[i]
        rl = roll_low[i]
        candidate: Optional[tuple] = None  # (direction, old_level, new_extreme, range_mid)
        if rh is not None and rl is not None and highs[i] > rh[0] and (i - rh[1]) >= cfg.min_separation_bars:
            candidate = ("short", rh[0], highs[i], (rh[0] + rl[0]) / 2.0)
        elif rl is not None and rh is not None and lows[i] < rl[0] and (i - rl[1]) >= cfg.min_separation_bars:
            candidate = ("long", rl[0], lows[i], (rh[0] + rl[0]) / 2.0)
        if candidate is None:
            continue
        direction, old_level, new_extreme, range_mid = candidate

        day_end = day_end_idx[d_i]  # reclaim + hold never crosses a day boundary
        reclaim_j: Optional[int] = None
        for j in range(i + 1, min(i + cfg.reclaim_within_bars, day_end) + 1):
            if direction == "short" and closes[j] < old_level:
                reclaim_j = j
                break
            if direction == "long" and closes[j] > old_level:
                reclaim_j = j
                break
        if reclaim_j is None:
            continue  # no reclaim within the window -- not a confirmed false breakout

        fill_idx = reclaim_j + 1
        if fill_idx > day_end:
            continue
        entry_price = merged_bars[fill_idx].open
        stop = new_extreme  # "beyond the extreme" -- the bar that set the false breakout
        if direction == "short" and stop <= entry_price:
            continue
        if direction == "long" and stop >= entry_price:
            continue
        risk = abs(entry_price - stop)
        reward = abs(range_mid - entry_price)
        if risk <= 0 or reward < cfg.min_rr * risk:
            continue  # "min 1:2" -- the range-midpoint target doesn't clear the floor, skip

        day_start = day_start_idx[d_i]
        day_bars_local = merged_bars[day_start : day_end + 1]
        intents.append(
            TradeIntent(
                symbol, d_i, day_bars_local, fill_idx - day_start, direction, stop, range_mid, day_end - day_start
            )
        )
        busy_until = day_end  # one trade per day for this family too
    return intents


# ── strategy registry + orchestration helper ────────────────────────────


class StrategyKind(str, Enum):
    ib_breakout = "ib_breakout"
    gap_continuation = "gap_continuation"
    opening_drive_reversal = "opening_drive_reversal"
    nr7_breakout = "nr7_breakout"
    turtle_soup = "turtle_soup"


def build_intents(symbol: str, sessions: dict, eval_start: date, eval_end: date, kind: StrategyKind, cfg) -> list[TradeIntent]:
    if kind == StrategyKind.ib_breakout:
        return build_intents_ib(symbol, sessions, eval_start, eval_end, cfg)
    if kind == StrategyKind.gap_continuation:
        return build_intents_gap(symbol, sessions, eval_start, eval_end, cfg)
    if kind == StrategyKind.opening_drive_reversal:
        return build_intents_drive(symbol, sessions, eval_start, eval_end, cfg)
    if kind == StrategyKind.nr7_breakout:
        return build_intents_nr7(symbol, sessions, eval_start, eval_end, cfg)
    if kind == StrategyKind.turtle_soup:
        return build_intents_turtle_soup(symbol, sessions, eval_start, eval_end, cfg)
    raise ValueError(f"unknown StrategyKind {kind!r}")


def evaluate_config(
    symbol: str,
    bars_by_day: dict,
    eval_start: date,
    eval_end: date,
    kind: StrategyKind,
    cfg,
    *,
    delay_bars: int = 3,
    seed: int = 7,
    n_repeats: int = 20,
) -> dict:
    """Real + delayed-signal + random-matched trades for one (symbol, config).

    ``bars_by_day`` must already include enough calendar-day buffer before
    ``eval_start`` for the strategy's own lookback (ATR/IB-history/NR7/
    rolling-N) -- callers (the sweep script) load a buffer identically to
    every other script in this repo.
    """
    sessions = build_sessions(symbol, bars_by_day)
    intents = build_intents(symbol, sessions, eval_start, eval_end, kind, cfg)
    real = _materialize(intents)
    delayed = _materialize_delayed(intents, delay_bars)
    from .noise_boundary import random_matched_control  # local import: keeps this module's public surface flat

    random_ctrl = random_matched_control(symbol, bars_by_day, real, seed=seed, n_repeats=n_repeats) if real else []
    n_eval_days = len({d for d in sessions if eval_start <= d <= eval_end})
    return {
        "sessions": sessions,
        "intents": intents,
        "real": real,
        "delayed_signal": delayed,
        "random_matched": random_ctrl,
        "n_eval_days": n_eval_days,
    }
