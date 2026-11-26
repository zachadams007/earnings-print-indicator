"""Zarattini / Aziz / Barbon "Beat the Market" intraday-momentum noise-
boundary strategy - independent replication.

Own-file lane: this module does NOT import from (or get imported by) any
other ``bot.scalp.*`` module, because a parallel background lane is editing
``bot/scalp/{features,rules,rules_v01,controls,evaluate,synthetic}.py``
concurrently in a sibling worktree. Any resemblance in naming/shape to that
lane's code is coincidental convergent design, not a shared dependency.

======================================================================
AUDITED SPEC (read this before touching the math below)
======================================================================
Source: the authors' own hosted PDF, retrieved 2026-07-09,
https://concretumgroup.com/wp-content/uploads/2026/02/Beat-the-Market.pdf
= "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF
(SPY)", Carlo Zarattini, Andrew Aziz, Andrea Barbon. SSRN 4824172 / Swiss
Finance Institute Research Paper No. 24-97. First version May 10 2024,
this version September 22 2025. (Cross-checked against the QuantConnect
forum replication thread #17091 and the Maroy 2025 follow-up, which agree
on every formula below.)

1. NOISE AREA (paper Section 3, verbatim formulas).
   For trading day t and clock time HH:MM (America/New_York):

     move_{t-i, HH:MM} = | Close_{t-i,HH:MM} / Open_{t-i,9:30} - 1 |,   i = 1..14

     sigma_{t,HH:MM} = (1/14) * sum_{i=1..14} move_{t-i,HH:MM}
                       (trailing 14 PRIOR trading sessions, same HH:MM,
                       point-in-time: only sessions strictly before t)

     UpperBound_{t,HH:MM} = max(Open_{t,9:30}, Close_{t-1,16:00}) * (1 + VM * sigma_{t,HH:MM})
     LowerBound_{t,HH:MM} = min(Open_{t,9:30}, Close_{t-1,16:00}) * (1 - VM * sigma_{t,HH:MM})

   VM ("Volatility Multiplier") = 1.0 in the paper's base spec; the paper's
   own robustness section (Section 4.7) sweeps VM as a free parameter
   ("a VM below 1 reduces the required movement ... a multiplier above 1
   demands a larger movement"). We sweep VM in {0.5, 1.0, 1.5} below.

2. ENTRY (Section 3). Decisions happen ONLY at half-hour clock marks -
   "trading is restricted to semi-hourly intervals, specifically at HH:00
   and HH:30" - i.e. the grid 10:00, 10:30, 11:00, ..., 15:30 (9:30 itself
   is a non-event: move-from-open is 0 there by construction, so it can
   never trigger a breakout). If flat at mark M:
     Close_{t,M} > UpperBound_{t,M}  -> go LONG
     Close_{t,M} < LowerBound_{t,M}  -> go SHORT

3. EXIT / TRAILING STOP (Section 3, Figure 5, Table 2). Checked ONLY at the
   same half-hour marks ("Stop losses can also be triggered only on
   semi-hourly intervals"). The paper's own narrative walks through three
   variants; we implement and sweep all three as ``stop_mode``:

     "opposite_band" - the un-refined base model (Table 1, Sharpe [n]).
       Long stop = LowerBound_{t,M}; Short stop = UpperBound_{t,M}. A
       breach of the OPPOSITE band, by construction, is itself a fresh
       breakout in the other direction, so the paper reverses immediately:
       "the existing position is closed, and a new one is initiated in the
       opposite direction" (Section 3).

     "current_band" - the intermediate step shown only as an illustration
       in Figure 5(a) (same-side band as a tighter stop, no VWAP yet).
       Long stop = UpperBound_{t,M}; Short stop = LowerBound_{t,M}. NOT a
       separately-tabulated result in the paper (Table 2 jumps straight to
       current-band-plus-VWAP) - included here as our own interpolating
       robustness variant, labeled as such throughout.

     "current_band_vwap" - the paper's FEATURED / final stop (Table 2-3,
       Sharpe [n] before sizing, [n] after): "Once a position is
       established, as soon as the price crosses either the current band
       or the VWAP, the position is closed."
         Long TrailingStop_{t,M}  = max(UpperBound_{t,M}, VWAP_{t,M})
         Short TrailingStop_{t,M} = min(LowerBound_{t,M}, VWAP_{t,M})
       VWAP is the cumulative SESSION volume-weighted average price,
       "computed only using market hours data" (paper footnote 2).

   All open positions are force-flattened at the 16:00 close regardless of
   stop state ("Positions are unwound either at market Close or if there is
   a crossover to the opposite boundary of the Noise Area").

4. POSITION SIZING (Section 3, "final refinement" -> Table 3, the paper's
   headline [n]%/[n]%/Sharpe [n] result):
     Shares_t = AUM_{t-1} * min(4, sigma_target / sigma_SPY,t) / Open_{t,9:30}
     sigma_target = 2% daily; sigma_SPY,t = sample stdev (Bessel-corrected,
     n-1=13) of the trailing 14 daily returns; leverage capped at 4x.
   SCOPE DECISION: we report trade-level basis-point returns throughout
   (gross/net bps, hit rate, R-expectancy) rather than a compounded
   dollar/leverage equity curve. bps returns are invariant to the number of
   shares traded, so the vol-target sizing rule above does not change any
   number in this report - it would only be needed to reproduce the
   paper's own $ Sharpe/CAGR/MDD, which is out of scope for a cost-
   sensitivity replication. Flagged here so nobody mistakes this report's
   Sharpe-less bps stats for a claim about the paper's dollar-equity curve.

5. COSTS (Section 3): the paper charges $0.0035/share commission (IBKR
   entry-level) + $0.001/share slippage. Per this task's own spec we
   instead sweep {0.5, 1.0, 2.0} bps of the underlying price PER SIDE (a
   different, coarser-but-standardized cost convention) - applied twice
   per trade (entry + exit).

======================================================================
DATA CONVENTION (read before touching the loader)
======================================================================
scalp.db stores bar ``ts`` as the bar's CLOSE instant (empirically
confirmed and documented in the parallel lane's
scripts/run_scalp_stage_a.py module docstring: the first RTH row of a
session lands at 09:31:00 ET, i.e. it is the close of the 9:30-9:31 bar).

That sibling lane's loader SHIFTS ts back by 1 minute to restore an
OPEN-stamp convention, because the rest of ``bot.scalp`` was built assuming
bar.ts == bar-open. This module DELIBERATELY DOES NOT apply that shift. The
paper's own notation, Close_{t,HH:MM}, already means "the price known at
clock time HH:MM" - which is *exactly* what a close-stamped bar.ts gives
us for free. Applying the sibling lane's -1-minute shift here would just
introduce an extra off-by-one-minute translation this module has no use
for. Every place below that reads "the price at HH:MM" means:
``bar.close`` for the bar whose ``ts`` (converted to America/New_York) is
HH:MM exactly. This is a deliberate, audited choice, not an oversight -
flagged for anyone diffing this file against the sibling stage-A loader.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from statistics import mean, median
from typing import Literal, Optional, Sequence
from zoneinfo import ZoneInfo

from ..data.models import Bar

ET = ZoneInfo("America/New_York")

SESSION_OPEN_ET = time(9, 30)
SESSION_CLOSE_ET = time(16, 0)
FIRST_BAR_CLOSE_ET = time(9, 31)  # close-stamp of the 9:30-9:31 opening bar

# Full half-hour decision grid per the paper: HH:00 and HH:30, 10:00..15:30.
# 9:30 excluded (move-from-open == 0 there, never a signal); 16:00 reserved
# exclusively for the mandatory flat-at-close, never a fresh-entry mark.
HALF_HOUR_MARKS: tuple[time, ...] = tuple(
    time(h, m) for h in range(10, 16) for m in (0, 30)
)  # (10:00, 10:30, 11:00, ..., 15:00, 15:30) -- 12 marks

# Coarser hourly grid (robustness variant, NOT in the paper): subset of
# HALF_HOUR_MARKS by construction, so both grids share one moves cache.
HOURLY_MARKS: tuple[time, ...] = tuple(time(h, 0) for h in range(10, 16))  # 6 marks

LOOKBACK_DAYS_DEFAULT = 14
VOL_TARGET_DEFAULT = 0.02
LEVERAGE_CAP_DEFAULT = 4.0

StopMode = Literal["opposite_band", "current_band", "current_band_vwap"]
Direction = Literal["long", "short"]

# ── execution-lag fill models (confirmation-gauntlet JOB 2) ──────────────
# "signal_close" is the baseline convention documented above (fill at the
# decision mark's own close, zero latency). The other two model a one-bar
# execution delay: the order is still DECIDED at the mark using that mark's
# own close/vwap (see ``_signal_snapshot`` in ``simulate_day`` -- the
# decision logic is completely untouched by ``fill_mode``), but FILLED one
# real 1-minute bar later, either at that next bar's open ("next_open") or
# its close ("plus_1min_close", i.e. "the close one minute after the
# signal"). ``DaySession.by_time``/``ordered_times`` hold EVERY RTH 1-minute
# bar (not just the half-hour/hourly decision marks), so "the next bar" here
# is the next real 1m bar in the session -- independent of, and finer than,
# the coarser decision grid.
FillMode = Literal["signal_close", "next_open", "plus_1min_close"]


def to_et(ts: datetime) -> datetime:
    return ts.astimezone(ET)


@dataclass(frozen=True)
class NoiseBoundaryConfig:
    """One sweepable variant. Defaults match the paper's featured spec
    (Table 3): VM=1.0, current-band+VWAP trailing stop, full half-hour grid.
    """

    vm: float = 1.0
    stop_mode: StopMode = "current_band_vwap"
    marks: tuple[time, ...] = HALF_HOUR_MARKS
    lookback_days: int = LOOKBACK_DAYS_DEFAULT
    label: str = "vm1.0_currband_vwap_halfhour"


@dataclass
class DaySession:
    """One RTH-filtered trading day, indexed by ET clock time (close-stamp)."""

    day: date
    open_price: float
    close_price: float
    close_mark: time
    by_time: dict[time, Bar]
    vwap_by_time: dict[time, float]
    ordered_times: list[time]


@dataclass
class Trade:
    symbol: str
    day: date
    direction: Direction
    entry_mark: time
    entry_price: float
    entry_stop_price: float
    exit_mark: time
    exit_price: float
    # "target" / "time_stop" added for the SWEEP v4 Hypothesis B scalp-exit
    # variants (see ``simulate_day_scalp_exit`` below) -- additive widening of
    # the Literal, no existing caller's exit_reason value changes.
    exit_reason: Literal["stop", "eod_close", "target", "time_stop"]
    hold_minutes: int

    @property
    def gross_ret(self) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return sign * (self.exit_price / self.entry_price - 1.0)

    @property
    def gross_bps(self) -> float:
        return self.gross_ret * 10_000.0

    @property
    def initial_risk_bps(self) -> Optional[float]:
        risk = abs(self.entry_price - self.entry_stop_price) / self.entry_price
        return risk * 10_000.0 if risk > 0 else None

    @property
    def r_multiple(self) -> Optional[float]:
        risk = self.initial_risk_bps
        return self.gross_bps / risk if risk else None

    def net_bps(self, slippage_bps_per_side: float) -> float:
        return self.gross_bps - 2.0 * slippage_bps_per_side


# ── session construction ─────────────────────────────────────────────────


def build_day_session(symbol: str, day: date, bars: Sequence[Bar]) -> Optional[DaySession]:
    """RTH-filter + ET-clock-index one day's 1m bars. Returns None if the
    day has no usable bar in [09:31, 16:00] ET (see module docstring for why
    ``ts`` is read as a close-stamp with no shift).
    """
    rth = sorted(
        (b for b in bars if SESSION_OPEN_ET < to_et(b.ts).time() <= SESSION_CLOSE_ET),
        key=lambda b: b.ts,
    )
    if not rth:
        return None

    open_price = rth[0].open
    close_price = rth[-1].close
    close_mark = to_et(rth[-1].ts).time()

    by_time: dict[time, Bar] = {}
    vwap_by_time: dict[time, float] = {}
    cum_pv = 0.0
    cum_vol = 0.0
    for b in rth:
        t = to_et(b.ts).time()
        by_time[t] = b
        px = b.vwap if b.vwap is not None else b.close
        cum_pv += px * b.volume
        cum_vol += b.volume
        vwap_by_time[t] = (cum_pv / cum_vol) if cum_vol > 0 else b.close

    return DaySession(
        day=day,
        open_price=open_price,
        close_price=close_price,
        close_mark=close_mark,
        by_time=by_time,
        vwap_by_time=vwap_by_time,
        ordered_times=sorted(by_time),
    )


def day_move_from_open(session: DaySession, mark: time) -> Optional[float]:
    """abs(Close_{day,mark} / Open_{day,9:30} - 1), or None if that mark's
    bar is missing (data gap) for this session.
    """
    bar_ = session.by_time.get(mark)
    if bar_ is None:
        return None
    return abs(bar_.close / session.open_price - 1.0)


# ── noise-area bands ─────────────────────────────────────────────────────


def build_noise_bands(
    history: Sequence[DaySession],
    target: DaySession,
    cfg: NoiseBoundaryConfig,
    *,
    min_history_frac: float = 0.8,
) -> Optional[dict[time, tuple[float, float]]]:
    """Compute {mark: (lower, upper)} for ``target`` using ``history`` (must
    be exactly the ``cfg.lookback_days`` trading sessions immediately
    preceding ``target``, chronologically -- caller's job to slice
    correctly so this stays point-in-time). Returns None if there is not
    even one usable mark (e.g. history too short).
    """
    if len(history) < cfg.lookback_days:
        return None

    prior_close = history[-1].close_price
    ref_hi = max(target.open_price, prior_close)
    ref_lo = min(target.open_price, prior_close)

    bands: dict[time, tuple[float, float]] = {}
    min_obs = max(1, int(cfg.lookback_days * min_history_frac))
    for mark in cfg.marks:
        moves = [m for m in (day_move_from_open(h, mark) for h in history) if m is not None]
        if len(moves) < min_obs:
            continue
        sigma = mean(moves)
        upper = ref_hi * (1.0 + cfg.vm * sigma)
        lower = ref_lo * (1.0 - cfg.vm * sigma)
        bands[mark] = (lower, upper)
    return bands or None


# ── single-day simulation ────────────────────────────────────────────────


def _stop_price(
    direction: Direction, lower: float, upper: float, vwap: Optional[float], stop_mode: StopMode
) -> float:
    if stop_mode == "opposite_band":
        return lower if direction == "long" else upper
    if stop_mode == "current_band":
        return upper if direction == "long" else lower
    # current_band_vwap
    vwap_ = vwap if vwap is not None else (upper if direction == "long" else lower)
    return max(upper, vwap_) if direction == "long" else min(lower, vwap_)


def _lagged_fill_price(session: DaySession, mark: time, fill_mode: FillMode) -> float:
    """Fill price for an order decided at ``mark``, per ``fill_mode``.

    "signal_close" (baseline, zero latency): the mark's own close -- exactly
    the pre-existing convention, unchanged.

    "next_open" / "plus_1min_close": the very next REAL 1-minute bar in the
    session (looked up via ``session.ordered_times``, which holds every RTH
    1m bar -- NOT the coarser half-hour/hourly decision grid), its open or
    close respectively. This never reaches into the signal bar's own
    high/low/close as a look-ahead proxy -- it only ever reads a genuinely
    later bar's own open/close, or falls through to (b) below.

    If ``mark`` is the session's last available bar (e.g. the flat-at-close
    mark) there IS no next bar to lag into, so this falls back to ``mark``'s
    own close -- you cannot lag execution past the end of the trading
    session. Deliberate, not a bug: it keeps the lag models from ever
    reaching into a later trading day.
    """
    if fill_mode == "signal_close":
        return session.by_time[mark].close
    times = session.ordered_times
    idx = times.index(mark)
    if idx + 1 >= len(times):
        return session.by_time[mark].close  # session ended -- can't lag further
    next_bar = session.by_time[times[idx + 1]]
    return next_bar.open if fill_mode == "next_open" else next_bar.close


def simulate_day(
    symbol: str,
    session: DaySession,
    bands: dict[time, tuple[float, float]],
    cfg: NoiseBoundaryConfig,
    *,
    signal_lag_marks: int = 0,
    fill_mode: FillMode = "signal_close",
) -> list[Trade]:
    """Run one day's decision marks in order. ``signal_lag_marks`` > 0 is the
    delayed-signal negative control: the breakout/stop CHECK uses price/vwap
    from ``signal_lag_marks`` marks ago, while any resulting fill still
    executes at the current mark's price (a "you'd have acted on stale
    information" leakage probe -- see run_noise_boundary.py).

    ``fill_mode`` is orthogonal to ``signal_lag_marks``: it only changes
    WHERE the resulting order is FILLED (see ``_lagged_fill_price``), never
    the price used to DECIDE whether/what to trade (``sig_price`` below,
    from ``_signal_snapshot``, is untouched by ``fill_mode``).
    """
    marks_in_day = [m for m in cfg.marks if m in session.by_time and m in bands]
    trades: list[Trade] = []

    direction: Optional[Direction] = None
    entry_mark: Optional[time] = None
    entry_price = 0.0
    entry_stop = 0.0

    def _signal_snapshot(idx: int) -> tuple[float, float, float, Optional[float]]:
        """(price, lower, upper, vwap) evaluated ``signal_lag_marks`` marks back."""
        sig_idx = max(0, idx - signal_lag_marks)
        sig_mark = marks_in_day[sig_idx]
        lower, upper = bands[sig_mark]
        return session.by_time[sig_mark].close, lower, upper, session.vwap_by_time.get(sig_mark)

    for idx, mark in enumerate(marks_in_day):
        fill_price = _lagged_fill_price(session, mark, fill_mode)
        sig_price, lower, upper, vwap = _signal_snapshot(idx)

        if direction is None:
            if sig_price > upper:
                direction, entry_mark, entry_price = "long", mark, fill_price
                entry_stop = _stop_price("long", lower, upper, vwap, cfg.stop_mode)
            elif sig_price < lower:
                direction, entry_mark, entry_price = "short", mark, fill_price
                entry_stop = _stop_price("short", lower, upper, vwap, cfg.stop_mode)
            continue

        stop = _stop_price(direction, lower, upper, vwap, cfg.stop_mode)
        breached = (direction == "long" and sig_price <= stop) or (
            direction == "short" and sig_price >= stop
        )
        if not breached:
            continue

        hold = _minutes_between(entry_mark, mark)
        trades.append(
            Trade(
                symbol=symbol,
                day=session.day,
                direction=direction,
                entry_mark=entry_mark,
                entry_price=entry_price,
                entry_stop_price=entry_stop,
                exit_mark=mark,
                exit_price=fill_price,
                exit_reason="stop",
                hold_minutes=hold,
            )
        )
        direction = None

        # Same-mark reversal iff the stop breach is ALSO a fresh opposite
        # breakout (always true for stop_mode="opposite_band"; may or may
        # not hold for the tighter current-band/VWAP stops).
        if sig_price > upper:
            direction, entry_mark, entry_price = "long", mark, fill_price
            entry_stop = _stop_price("long", lower, upper, vwap, cfg.stop_mode)
        elif sig_price < lower:
            direction, entry_mark, entry_price = "short", mark, fill_price
            entry_stop = _stop_price("short", lower, upper, vwap, cfg.stop_mode)

    if direction is not None:
        hold = _minutes_between(entry_mark, session.close_mark)
        trades.append(
            Trade(
                symbol=symbol,
                day=session.day,
                direction=direction,
                entry_mark=entry_mark,
                entry_price=entry_price,
                entry_stop_price=entry_stop,
                exit_mark=session.close_mark,
                exit_price=session.close_price,
                exit_reason="eod_close",
                hold_minutes=hold,
            )
        )

    return trades


def _minutes_between(t0: time, t1: time) -> int:
    d0 = datetime.combine(date.today(), t0)
    d1 = datetime.combine(date.today(), t1)
    return int((d1 - d0).total_seconds() // 60)


# ── SWEEP v4 Hypothesis B: scalp-compatible exit variants ────────────────
#
# ADDITIVE section: ``simulate_day``/``run_variant`` above are UNCHANGED, and
# remain the exact baseline ("opposite-band trail") this hypothesis compares
# against -- the noise-boundary gauntlet (scripts/run_noise_boundary_gauntlet.py)
# already established the QQQ survivor as {VM=1.0, VM=1.25} x opposite_band x
# half-hour grid, holding an average ~4.5h (per research notes
# result.md) -- not the owner's scalping style. Question: is there a
# scalp-compatible (sub-90-min) version of the SAME entry rule (breakout
# through the noise band at a half-hour mark), and what does shortening the
# hold cost in bps?
#
# Entries stay on the half-hour decision grid (unchanged mechanics: a
# breakout through the band at a mark, exactly as in ``simulate_day``).
# What changes is EXIT granularity and rule:
#   - "time_limit": force-flatten N minutes (30/60/90) after entry, OR the
#     opposite-band stop, whichever comes first -- checked at REAL 1-minute
#     bar resolution (``session.ordered_times``), not just the half-hour
#     grid, so hold-time measurement here is exact, not rounded to the
#     nearest mark.
#   - "first_target": exit at a fixed profit target = entry_price +/-
#     ``target_width_mult`` * (the noise band's own width AT THE ENTRY MARK,
#     i.e. upper - lower then), OR the opposite-band stop, whichever comes
#     first -- also checked at 1-minute resolution.
# Both variants check the stop BEFORE the target/time-limit on any bar where
# more than one condition is met simultaneously -- same conservative
# both-hit tie-break convention as ``exits.py``'s ``simulate_exit``.
#
# FREQUENCY toggle (the "one re-entry after a stop-out" question):
# ``allow_one_reentry_after_stop=False`` (default) caps a day at ONE trade --
# once the first trade of the day closes (by any reason), the strategy sits
# flat for the rest of the session. This default deliberately differs from
# the baseline ``simulate_day``, which (under stop_mode="opposite_band")
# ALWAYS reverses into a fresh trade on every stop-out by construction (a
# breach of the opposite band's own stop level is, by definition, a fresh
# breakout the other way -- see that function's docstring) and so never
# really goes "flat" intraday. These new exit variants introduce genuine
# flat periods (a target/time-limit exit does NOT necessarily coincide with
# an opposite-band breach), so "how many trades per day, and does a second
# shot help" becomes a real, answerable frequency question for the first
# time. Setting ``allow_one_reentry_after_stop=True`` allows exactly one
# additional entry, but ONLY if the first trade of the day closed via
# "stop" specifically (not "target"/"time_stop") -- i.e. "you got stopped
# out once, do you take one more shot."

ScalpExitKind = Literal["time_limit", "first_target"]


@dataclass(frozen=True)
class ScalpExitConfig:
    """One Hypothesis-B exit variant. ``kind="time_limit"`` requires
    ``time_limit_minutes``; ``kind="first_target"`` requires
    ``target_width_mult``. See module-docstring section above for the exact
    mechanics and the re-entry toggle's semantics."""

    kind: ScalpExitKind
    time_limit_minutes: Optional[int] = None
    target_width_mult: Optional[float] = None
    allow_one_reentry_after_stop: bool = False
    label: str = "scalp_exit"


def simulate_day_scalp_exit(
    symbol: str,
    session: DaySession,
    bands: dict[time, tuple[float, float]],
    cfg: NoiseBoundaryConfig,
    exit_cfg: ScalpExitConfig,
    *,
    signal_lag_marks: int = 0,
    fill_mode: FillMode = "signal_close",
) -> list[Trade]:
    """One day's decisions under a Hypothesis-B scalp-exit variant. Entries
    fire only at ``cfg.marks`` (unchanged mechanics vs ``simulate_day``);
    exits are checked at every real 1-minute bar in
    ``session.ordered_times`` for exact hold-time measurement. See the
    module-docstring section above for the full design (stop-before-target
    tie-break, default one-trade-per-day cap, re-entry toggle semantics).

    ``fill_mode`` changes execution only, never the breakout decision:
    ``next_open`` fills at the next real 1m bar open and may be stopped or
    targeted inside that bar; ``plus_1min_close`` fills at that bar's close,
    so exit evaluation starts with the following bar.  This distinction is
    required to avoid using a fill bar's pre-fill high/low.

    ``signal_lag_marks`` (delayed-signal negative control, mirrors
    ``simulate_day``'s parameter of the same name): > 0 means the ENTRY
    breakout check at mark ``t`` uses the price/bounds from
    ``signal_lag_marks`` marks EARLIER in the day's own half-hour sequence
    instead of ``t``'s own -- "would you still have taken this trade on
    stale information." The exit side (stop/target/time-limit, all checked
    on real 1-minute bars) is untouched by this parameter -- once in a
    trade, the position is managed on live data exactly as in the
    zero-lag case; only the decision to ENTER can be checked late.
    """
    if exit_cfg.kind == "time_limit" and exit_cfg.time_limit_minutes is None:
        raise ValueError("ScalpExitConfig(kind='time_limit') requires time_limit_minutes")
    if exit_cfg.kind == "first_target" and exit_cfg.target_width_mult is None:
        raise ValueError("ScalpExitConfig(kind='first_target') requires target_width_mult")

    marks_in_day = [m for m in cfg.marks if m in session.by_time and m in bands]
    marks_index = {m: i for i, m in enumerate(marks_in_day)}
    times = session.ordered_times
    trades: list[Trade] = []

    direction: Optional[Direction] = None
    entry_mark: Optional[time] = None
    entry_price = 0.0
    entry_stop = 0.0
    entry_target: Optional[float] = None
    first_exit_index = 0

    def _entry_fill(decision_mark: time) -> tuple[float, time, int]:
        """Return price, actual fill clock, and first bar eligible for exits."""
        idx = times.index(decision_mark)
        if fill_mode == "signal_close" or idx + 1 >= len(times):
            return session.by_time[decision_mark].close, decision_mark, idx + 1
        next_mark = times[idx + 1]
        next_bar = session.by_time[next_mark]
        if fill_mode == "next_open":
            # The next bar opens at the decision bar's close instant.  Store
            # the decision clock as the fill clock, but allow this next bar's
            # post-open path to hit the stop/target.
            return next_bar.open, decision_mark, idx + 1
        # A close fill occurs only after the next bar's high/low already
        # happened, so that bar cannot be used for an exit.
        return next_bar.close, next_mark, idx + 2

    def _eligible_for_entry() -> bool:
        if not trades:
            return True
        if exit_cfg.allow_one_reentry_after_stop and len(trades) == 1 and trades[-1].exit_reason == "stop":
            return True
        return False

    for time_index, t in enumerate(times):
        bar = session.by_time[t]

        if direction is None:
            if t in marks_index and _eligible_for_entry():
                sig_idx = max(0, marks_index[t] - signal_lag_marks)
                sig_mark = marks_in_day[sig_idx]
                lower, upper = bands[sig_mark]
                sig_price = session.by_time[sig_mark].close
                fill_price, fill_mark, eligible_exit_index = _entry_fill(t)
                width = upper - lower
                if sig_price > upper:
                    direction = "long"
                    entry_stop = lower
                    entry_target = fill_price + exit_cfg.target_width_mult * width if exit_cfg.kind == "first_target" else None
                elif sig_price < lower:
                    direction = "short"
                    entry_stop = upper
                    entry_target = fill_price - exit_cfg.target_width_mult * width if exit_cfg.kind == "first_target" else None
                if direction is not None:
                    entry_mark = fill_mark
                    entry_price = fill_price
                    first_exit_index = eligible_exit_index
            continue

        if time_index < first_exit_index:
            continue

        # In position and at/after the first bar whose path occurs after the
        # selected fill instant.
        elapsed = _minutes_between(entry_mark, t)
        hit_stop = (bar.low <= entry_stop) if direction == "long" else (bar.high >= entry_stop)
        if hit_stop:  # checked first -- conservative both-hit tie-break
            trades.append(
                Trade(
                    symbol=symbol, day=session.day, direction=direction, entry_mark=entry_mark,
                    entry_price=entry_price, entry_stop_price=entry_stop, exit_mark=t,
                    exit_price=entry_stop, exit_reason="stop", hold_minutes=elapsed,
                )
            )
            direction = None
            continue

        if exit_cfg.kind == "first_target":
            hit_target = (bar.high >= entry_target) if direction == "long" else (bar.low <= entry_target)
            if hit_target:
                trades.append(
                    Trade(
                        symbol=symbol, day=session.day, direction=direction, entry_mark=entry_mark,
                        entry_price=entry_price, entry_stop_price=entry_stop, exit_mark=t,
                        exit_price=entry_target, exit_reason="target", hold_minutes=elapsed,
                    )
                )
                direction = None
                continue

        if exit_cfg.kind == "time_limit" and elapsed >= exit_cfg.time_limit_minutes:
            trades.append(
                Trade(
                    symbol=symbol, day=session.day, direction=direction, entry_mark=entry_mark,
                    entry_price=entry_price, entry_stop_price=entry_stop, exit_mark=t,
                    exit_price=bar.close, exit_reason="time_stop", hold_minutes=elapsed,
                )
            )
            direction = None
            continue

    if direction is not None:
        hold = _minutes_between(entry_mark, session.close_mark)
        trades.append(
            Trade(
                symbol=symbol, day=session.day, direction=direction, entry_mark=entry_mark,
                entry_price=entry_price, entry_stop_price=entry_stop, exit_mark=session.close_mark,
                exit_price=session.close_price, exit_reason="eod_close", hold_minutes=hold,
            )
        )

    return trades


def run_variant_scalp_exit(
    symbol: str,
    bars_by_day: dict[date, Sequence[Bar]],
    cfg: NoiseBoundaryConfig,
    exit_cfg: ScalpExitConfig,
    *,
    eval_start: Optional[date] = None,
    eval_end: Optional[date] = None,
    signal_lag_marks: int = 0,
    fill_mode: FillMode = "signal_close",
) -> list[Trade]:
    """Multi-day runner for the Hypothesis-B scalp-exit variants -- same
    point-in-time noise-band construction as ``run_variant`` (each eval day's
    band history is exactly the ``cfg.lookback_days`` sessions strictly
    before it), just dispatching to ``simulate_day_scalp_exit`` instead of
    ``simulate_day`` for the per-day simulation. ``signal_lag_marks`` is
    forwarded unchanged -- see ``simulate_day_scalp_exit``'s docstring.
    """
    sessions: dict[date, DaySession] = {}
    for d, bars in bars_by_day.items():
        s = build_day_session(symbol, d, bars)
        if s is not None:
            sessions[d] = s
    sorted_days = sorted(sessions)

    trades: list[Trade] = []
    for idx, d in enumerate(sorted_days):
        if eval_start is not None and d < eval_start:
            continue
        if eval_end is not None and d > eval_end:
            continue
        if idx < cfg.lookback_days:
            continue
        history = [sessions[sorted_days[j]] for j in range(idx - cfg.lookback_days, idx)]
        bands = build_noise_bands(history, sessions[d], cfg)
        if bands is None:
            continue
        trades.extend(
            simulate_day_scalp_exit(
                symbol,
                sessions[d],
                bands,
                cfg,
                exit_cfg,
                signal_lag_marks=signal_lag_marks,
                fill_mode=fill_mode,
            )
        )
    return trades


# ── multi-day runner ─────────────────────────────────────────────────────


def run_variant(
    symbol: str,
    bars_by_day: dict[date, Sequence[Bar]],
    cfg: NoiseBoundaryConfig,
    *,
    eval_start: Optional[date] = None,
    eval_end: Optional[date] = None,
    signal_lag_marks: int = 0,
    fill_mode: FillMode = "signal_close",
) -> list[Trade]:
    """Point-in-time loop: for each eval day, the noise-area history window
    is exactly the ``cfg.lookback_days`` sessions strictly BEFORE it in
    ``bars_by_day`` (chronological index, never a future session).

    ``fill_mode`` (JOB 2, execution-lag sensitivity): forwarded unchanged to
    ``simulate_day`` for every eval day -- see ``_lagged_fill_price``.
    """
    sessions: dict[date, DaySession] = {}
    for d, bars in bars_by_day.items():
        s = build_day_session(symbol, d, bars)
        if s is not None:
            sessions[d] = s
    sorted_days = sorted(sessions)

    trades: list[Trade] = []
    for idx, d in enumerate(sorted_days):
        if eval_start is not None and d < eval_start:
            continue
        if eval_end is not None and d > eval_end:
            continue
        if idx < cfg.lookback_days:
            continue
        history = [sessions[sorted_days[j]] for j in range(idx - cfg.lookback_days, idx)]
        bands = build_noise_bands(history, sessions[d], cfg)
        if bands is None:
            continue
        trades.extend(
            simulate_day(
                symbol, sessions[d], bands, cfg,
                signal_lag_marks=signal_lag_marks, fill_mode=fill_mode,
            )
        )
    return trades


# ── negative control: fixed shuffled-bars (ts-grid preserved) ────────────
#
# Own-file-lane COPY of the ts-grid-preserving payload shuffle in
# ``bot.scalp.controls.shuffle_bars_within_day`` (see that module's
# docstring for the full rationale, including the bug it replaced: a naive
# whole-Bar shuffle scrambles ``ts`` itself and breaks anything that assumes
# ascending-order bars). Copied rather than imported -- this module's own
# docstring above already commits to never importing from (or being
# imported by) any other ``bot.scalp.*`` module, since ``controls.py`` may
# be under concurrent edit in a sibling worktree; ``scripts/run_noise_boundary.py``
# makes the same "copy the pattern, don't import it" choice for its loader,
# for the same reason.


def shuffle_bar_content_within_day(bars: Sequence[Bar], seed: int) -> list[Bar]:
    """Keep a day's ``ts`` grid exactly as-is (ascending, one bar per real
    timestamp) and permute only the OHLCV payload across that fixed sequence
    of time slots. Downstream code (``build_day_session``'s RTH filter + VWAP
    accumulation) sees a structurally normal, monotonic session -- only the
    open/high/low/close/volume "observed" at any given minute is randomized,
    destroying any edge that depends on the actual sequential price path
    (the noise-boundary bands, VWAP drift, breakout timing) while leaving
    every structural invariant this module's loaders assume intact.
    """
    ordered = sorted(bars, key=lambda b: b.ts)
    ts_sequence = [b.ts for b in ordered]
    rng = random.Random(seed)
    shuffled_content = list(ordered)
    rng.shuffle(shuffled_content)
    return [content.model_copy(update={"ts": ts}) for ts, content in zip(ts_sequence, shuffled_content)]


def shuffle_bars_by_day(
    bars_by_day: dict[date, Sequence[Bar]], seed: int
) -> dict[date, list[Bar]]:
    """Apply ``shuffle_bar_content_within_day`` to every day independently,
    deriving a distinct sub-seed per day (so different days don't all share
    one identical permutation pattern) while the whole multi-day shuffle
    stays fully reproducible from a single ``seed``. Avoids Python's built-in
    ``hash()`` for the per-day sub-seed (str/bytes/datetime hashing is
    randomized per-process via ``PYTHONHASHSEED`` unless disabled) so the
    same ``seed`` reproduces byte-identical results across runs/machines.
    """
    out: dict[date, list[Bar]] = {}
    for d, bars in bars_by_day.items():
        day_seed = (seed * 1_000_003 + d.toordinal()) & 0xFFFFFFFF
        out[d] = shuffle_bar_content_within_day(bars, seed=day_seed)
    return out


# ── negative control: random-matched ─────────────────────────────────────


def random_matched_control(
    symbol: str,
    bars_by_day: dict[date, Sequence[Bar]],
    real_trades: Sequence[Trade],
    *,
    seed: int = 0,
    n_repeats: int = 20,
) -> list[Trade]:
    """Frequency/holding-time-matched randomization: for each real trade,
    draw a RANDOM (day, entry mark) pair from the days that actually have
    sessions, hold for that SAME real trade's ``hold_minutes`` (clipped to
    the session close), with a random 50/50 direction. Repeated
    ``n_repeats`` times and pooled, so the single-draw noise of "which random
    marks got picked" washes out. This answers: "does trading this often,
    for this long, with NO signal at all, look like the real result?"
    """
    rng = random.Random(seed)
    sessions: dict[date, DaySession] = {}
    for d, bars in bars_by_day.items():
        s = build_day_session(symbol, d, bars)
        if s is not None:
            sessions[d] = s
    usable_days = [d for d, s in sessions.items() if len(s.ordered_times) >= 2]
    if not usable_days or not real_trades:
        return []

    synthetic: list[Trade] = []
    for _ in range(n_repeats):
        for rt in real_trades:
            d = rng.choice(usable_days)
            s = sessions[d]
            times = s.ordered_times
            entry_idx = rng.randrange(0, len(times))
            entry_mark = times[entry_idx]
            entry_price = s.by_time[entry_mark].close
            target_exit_dt = datetime.combine(date.today(), entry_mark) + timedelta(
                minutes=rt.hold_minutes
            )
            exit_mark = target_exit_dt.time()
            # Clip to the session's actual close if the matched hold runs
            # past it, exactly like the real strategy's flat-at-close rule.
            if exit_mark > s.close_mark or exit_mark <= entry_mark:
                exit_mark = s.close_mark
                exit_price = s.close_price
            else:
                # nearest available bar at/after the target exit time
                candidates = [t for t in times if t >= exit_mark]
                exit_mark = candidates[0] if candidates else s.close_mark
                exit_price = s.by_time.get(exit_mark, Bar(
                    symbol=symbol, ts=datetime.now(ET), timeframe="1m",
                    open=s.close_price, high=s.close_price, low=s.close_price,
                    close=s.close_price, volume=0, source="synthetic",
                    ingested_ts=datetime.now(ET),
                )).close
            direction: Direction = "long" if rng.random() < 0.5 else "short"
            synthetic.append(
                Trade(
                    symbol=symbol,
                    day=d,
                    direction=direction,
                    entry_mark=entry_mark,
                    entry_price=entry_price,
                    entry_stop_price=entry_price,  # undefined for a control; R not reported
                    exit_mark=exit_mark,
                    exit_price=exit_price,
                    exit_reason="eod_close",
                    hold_minutes=_minutes_between(entry_mark, exit_mark),
                )
            )
    return synthetic


# ── summary statistics ───────────────────────────────────────────────────


@dataclass
class SideStats:
    n: int
    trades_per_day: Optional[float]
    win_rate: Optional[float]
    avg_hold_minutes: Optional[float]
    gross_expectancy_bps: Optional[float]
    r_expectancy: Optional[float]
    net_expectancy_bps: dict[float, Optional[float]] = field(default_factory=dict)


def side_stats(trades: Sequence[Trade], n_days: int, slippage_grid: Sequence[float]) -> SideStats:
    if not trades:
        return SideStats(0, None, None, None, None, None, {s: None for s in slippage_grid})
    n = len(trades)
    wins = sum(1 for t in trades if t.gross_bps > 0)
    r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]
    return SideStats(
        n=n,
        trades_per_day=(n / n_days) if n_days else None,
        win_rate=wins / n,
        avg_hold_minutes=mean(t.hold_minutes for t in trades),
        gross_expectancy_bps=mean(t.gross_bps for t in trades),
        r_expectancy=mean(r_vals) if r_vals else None,
        net_expectancy_bps={s: mean(t.net_bps(s) for t in trades) for s in slippage_grid},
    )


def split_by_direction(trades: Sequence[Trade]) -> tuple[list[Trade], list[Trade]]:
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    return longs, shorts


def weekly_folds(trades: Sequence[Trade]) -> dict[str, list[Trade]]:
    """Blocked weekly folds keyed by ISO (year, week) of the trade's day."""
    folds: dict[str, list[Trade]] = {}
    for t in trades:
        y, w, _ = t.day.isocalendar()
        folds.setdefault(f"{y}-W{w:02d}", []).append(t)
    return folds


def yearly_expectancy(trades: Sequence[Trade], slippage_bps: float = 1.0) -> dict[int, dict[str, float]]:
    by_year: dict[int, list[Trade]] = {}
    for t in trades:
        by_year.setdefault(t.day.year, []).append(t)
    out: dict[int, dict[str, float]] = {}
    for y, ts in sorted(by_year.items()):
        out[y] = {
            "n": len(ts),
            "gross_bps": mean(t.gross_bps for t in ts),
            "net_bps": mean(t.net_bps(slippage_bps) for t in ts),
        }
    return out


def fold_consistency(trades: Sequence[Trade], slippage_bps: float = 1.0) -> dict[str, float]:
    """Summary of the per-(ISO)-week net-expectancy distribution: fraction of
    weeks with a positive mean, plus the median/IQR of weekly means. Weeks
    with zero trades are excluded (nothing to be "consistent" about).
    """
    folds = weekly_folds(trades)
    means = [mean(t.net_bps(slippage_bps) for t in ts) for ts in folds.values() if ts]
    if not means:
        return {"n_folds": 0, "pct_positive": float("nan"), "median_bps": float("nan"), "iqr_bps": float("nan")}
    means_sorted = sorted(means)
    n = len(means_sorted)
    q1 = means_sorted[int(0.25 * (n - 1))]
    q3 = means_sorted[int(0.75 * (n - 1))]
    return {
        "n_folds": n,
        "pct_positive": sum(1 for m in means if m > 0) / n,
        "median_bps": median(means),
        "iqr_bps": q3 - q1,
    }


# ── realized-vol regime split (robustness lane) ──────────────────────────


def daily_realized_vol_series(
    sessions_sorted: Sequence[DaySession], window: int = 20
) -> dict[date, Optional[float]]:
    """Trailing ``window``-day realized vol (stdev of close-to-close daily
    returns) for every session in ``sessions_sorted``, point-in-time: the
    value attached to day ``d`` only uses the ``window`` daily returns
    STRICTLY BEFORE ``d`` (i.e. returns computed from the ``window+1``
    closes immediately preceding ``d``'s own close). ``sessions_sorted`` must
    already be in chronological order (caller's job, same convention as
    ``run_variant``'s ``history`` slicing). Days with insufficient prior
    history map to ``None`` rather than being silently dropped, so callers
    can tell "no signal yet" from "zero vol".
    """
    from statistics import stdev

    closes = [s.close_price for s in sessions_sorted]
    out: dict[date, Optional[float]] = {}
    for idx, s in enumerate(sessions_sorted):
        if idx <= window:  # need `window` returns => `window+1` prior closes
            out[s.day] = None
            continue
        prior_closes = closes[idx - window - 1 : idx]  # window+1 closes, all strictly before s
        rets = [prior_closes[i] / prior_closes[i - 1] - 1.0 for i in range(1, len(prior_closes))]
        out[s.day] = stdev(rets) if len(rets) >= 2 else None
    return out


def vol_regime_split(
    vol_series: dict[date, Optional[float]]
) -> tuple[Optional[float], dict[date, str]]:
    """Median-split ``vol_series`` (dropping ``None``s) into "high"/"low" tags
    per day. This is a diagnostic (ex-post categorization of trades already
    taken), not a trading rule, so the median is computed once over the
    whole series rather than expanding point-in-time -- using the full-
    sample median to LABEL a day's regime does not leak information into any
    entry/exit decision (the strategy itself never sees this label).
    Returns ``(None, {})`` if fewer than 2 non-None observations exist.
    """
    vals = sorted(v for v in vol_series.values() if v is not None)
    if len(vals) < 2:
        return None, {}
    n = len(vals)
    med = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    tags = {d: ("high" if v >= med else "low") for d, v in vol_series.items() if v is not None}
    return med, tags


def split_trades_by_regime(
    trades: Sequence[Trade], regime_tags: dict[date, str]
) -> dict[str, list[Trade]]:
    """Bucket ``trades`` into {"high": [...], "low": [...]} by each trade's
    ``day`` tag. Trades on an untagged day (e.g. insufficient vol history)
    are dropped from both buckets -- they simply don't contribute evidence
    to the regime-split question.
    """
    out: dict[str, list[Trade]] = {"high": [], "low": []}
    for t in trades:
        tag = regime_tags.get(t.day)
        if tag in out:
            out[tag].append(t)
    return out
