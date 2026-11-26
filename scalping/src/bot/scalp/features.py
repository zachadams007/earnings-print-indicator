"""Point-in-time feature derivation for the Stage A scalping backtest.

HARD RULE (project rule 5 - point-in-time discipline):
    A 1-minute bar with open timestamp ``ts`` closes at ``ts + 1 minute``.
    A feature computed "as of" decision time ``T`` may only use bars whose
    CLOSE timestamp is ``<= T``.

Resolved convention (documented ambiguity)
-------------------------------------------
We walk decision points at each 1-minute bar's own OPEN timestamp. If
``day_bars_1m[i].ts == T``, the set of bars whose close is ``<= T`` is
exactly ``day_bars_1m[:i]`` (bar ``i-1`` closes exactly at ``T``; bar ``i``
itself closes at ``T + 1m`` and is therefore excluded). All feature
functions in this module receive an already-"completed" bar slice from the
caller and never look past it - see ``build_feature_snapshots`` below, which
is the only place that performs the ``[:i]`` slicing. Entry execution
("no earlier than the next bar after T") is bar ``i``'s OPEN price, handled
in ``evaluate.py``, not here.

Timestamps: bars may arrive UTC (as ingested from Alpaca) or already
ET-aware. Every date/time-of-day comparison in this module goes through
``to_et()`` first; naive timestamps are assumed to already be ET (matches
this repo's existing test convention, e.g. tests/backtest/test_pit_integrity.py).

Real-data note: ``data/sqlite/scalp.db`` stores ``ts`` as the bar's CLOSE
instant, not its open (the ingest script shifted Alpaca's native open-stamps
+1 minute). This module assumes OPEN-stamped ``ts`` throughout, matching
every synthetic/test fixture. scripts/run_scalp_stage_a.py's real-data loader
converts scalp.db rows back to open-stamped ``ts`` (subtracts 1 minute)
before anything in bot.scalp ever sees them - see that script's module
docstring. Do not feed scalp.db rows into this module without that shift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

from ..data.models import Bar

ET = ZoneInfo("America/New_York")

SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

# Time-window buckets per docs/SCALPING-COPILOT-DATA-FEASIBILITY-2026-07-08.md
# section 5 build spec: "open 9:30-9:45, main1 9:45-11:30, lunch, main2
# 1:30-3:30, close". The lunch/close boundaries are not spelled out
# numerically there; resolved using the avoid-windows in
# docs/SCALPING-COPILOT-CURSOR-RESEARCH-2026-07-08.md section (locked product
# constraints: avoid 9:30-9:45, 11:30-1:30, 3:30-4:00) so lunch = 11:30-13:30
# and close = 15:30-16:00.
_OPEN_END = time(9, 45)
_MAIN1_END = time(11, 30)
_LUNCH_END = time(13, 30)
_MAIN2_END = time(15, 30)


class TimeWindow(str, Enum):
    open = "open"
    main1 = "main1"
    lunch = "lunch"
    main2 = "main2"
    close = "close"
    outside_rth = "outside_rth"


class StructureState(str, Enum):
    higher_highs_lows = "hh_hl"
    lower_highs_lows = "lh_ll"
    mixed = "mixed"
    unknown = "unknown"


class DayType(str, Enum):
    """Point-in-time day-type classifier (v0.1 decision model).

    Definition (docs/COPILOT-V0-SYSTEM-PROMPT-2026-07-09.md decision model
    + docs/VWAP-TOUCH-STUDY-2026-07-09.md): at decision time T, look at the
    share of post-10:00 ET minutes (bars whose close <= T) spent on one side
    of session VWAP. Before 10:30 ET there simply isn't enough of that
    window closed yet to classify -> ``undecided``. From 10:30 ET onward:
    >=80% of post-10:00 minutes on one side -> a trend day in that
    direction; otherwise -> rotation. This mirrors the empirical one-sided
    vs rotation split in the VWAP touch study (~50/50 split, [n]% avg
    dominant-side share on one-sided days).
    """

    trend_up = "trend_up"
    trend_down = "trend_down"
    rotation = "rotation"
    undecided = "undecided"


# Day-type classification window boundaries (ET).
_DAY_TYPE_LOOKBACK_START = time(10, 0)
_DAY_TYPE_DECISION_FLOOR = time(10, 30)
_DAY_TYPE_ONE_SIDED_SHARE = 0.8


def to_et(ts: datetime) -> datetime:
    """Normalize a bar timestamp to ET. Naive timestamps are assumed ET already."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=ET)
    return ts.astimezone(ET)


def classify_time_window(ts: datetime) -> TimeWindow:
    t = to_et(ts).time()
    if t < SESSION_OPEN or t >= SESSION_CLOSE:
        return TimeWindow.outside_rth
    if t < _OPEN_END:
        return TimeWindow.open
    if t < _MAIN1_END:
        return TimeWindow.main1
    if t < _LUNCH_END:
        return TimeWindow.lunch
    if t < _MAIN2_END:
        return TimeWindow.main2
    return TimeWindow.close


# ── derived N-minute bars ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AggBar:
    """An N-minute bar derived strictly from a run of completed 1m bars."""

    ts_open: datetime
    ts_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    n_minutes: int


def derive_agg_bars(bars_1m: Sequence[Bar], minutes: int) -> list[AggBar]:
    """Aggregate 1m bars into N-minute buckets aligned to the session open (9:30).

    Only FULLY POPULATED buckets are emitted - a trailing bucket with fewer
    than ``minutes`` constituent bars is dropped (it has not closed yet).
    Callers enforce no-leakage by only ever passing an already-PIT-filtered
    ``bars_1m`` slice (see module docstring); this function performs no time
    filtering of its own beyond bucket alignment.
    """
    if not bars_1m:
        return []

    day = to_et(bars_1m[0].ts).date()
    session_open_dt = datetime.combine(day, SESSION_OPEN, tzinfo=ET)

    buckets: dict[datetime, list[Bar]] = {}
    for b in bars_1m:
        ts = to_et(b.ts)
        elapsed_min = (ts - session_open_dt).total_seconds() / 60.0
        bucket_idx = int(elapsed_min // minutes)
        bucket_start = session_open_dt + timedelta(minutes=bucket_idx * minutes)
        buckets.setdefault(bucket_start, []).append(b)

    out: list[AggBar] = []
    for bucket_start in sorted(buckets):
        group = sorted(buckets[bucket_start], key=lambda x: x.ts)
        if len(group) < minutes:
            continue  # incomplete bucket (usually the trailing partial one) - not closed
        out.append(
            AggBar(
                ts_open=bucket_start,
                ts_close=bucket_start + timedelta(minutes=minutes),
                open=group[0].open,
                high=max(x.high for x in group),
                low=min(x.low for x in group),
                close=group[-1].close,
                volume=sum(x.volume for x in group),
                n_minutes=len(group),
            )
        )
    return out


def structure_state(agg_bars: Sequence[AggBar]) -> StructureState:
    """HH/HL vs LH/LL vs mixed over the last 3 completed bars of a given timeframe."""
    if len(agg_bars) < 3:
        return StructureState.unknown
    a, b, c = agg_bars[-3], agg_bars[-2], agg_bars[-1]
    higher_highs = a.high < b.high < c.high
    higher_lows = a.low < b.low < c.low
    lower_highs = a.high > b.high > c.high
    lower_lows = a.low > b.low > c.low
    if higher_highs and higher_lows:
        return StructureState.higher_highs_lows
    if lower_highs and lower_lows:
        return StructureState.lower_highs_lows
    return StructureState.mixed


# ── session VWAP ─────────────────────────────────────────────────────────────


def session_vwap_series(bars_1m: Sequence[Bar]) -> list[float]:
    """Cumulative session VWAP aligned 1:1 with ``bars_1m``; resets at each RTH day.

    ``result[i]`` is the VWAP through (and including) ``bars_1m[i]``'s close,
    using typical price ``(high+low+close)/3`` weighted by volume.
    """
    result: list[float] = []
    cum_pv = 0.0
    cum_vol = 0.0
    current_day: Optional[date] = None
    for b in bars_1m:
        day = to_et(b.ts).date()
        if day != current_day:
            cum_pv = 0.0
            cum_vol = 0.0
            current_day = day
        tp = (b.high + b.low + b.close) / 3.0
        cum_pv += tp * b.volume
        cum_vol += b.volume
        result.append(cum_pv / cum_vol if cum_vol > 0 else b.close)
    return result


def sample_vwap_at_agg_closes(bars_1m: Sequence[Bar], agg_bars: Sequence[AggBar]) -> list[Optional[float]]:
    """Session VWAP value sampled at the close of each completed N-minute bar."""
    vwap_series = session_vwap_series(bars_1m)
    ts_to_vwap = {to_et(b.ts): v for b, v in zip(bars_1m, vwap_series)}
    samples: list[Optional[float]] = []
    for agg in agg_bars:
        last_minute_ts = agg.ts_close - timedelta(minutes=1)
        samples.append(ts_to_vwap.get(last_minute_ts))
    return samples


def vwap_slope(samples: Sequence[Optional[float]], lookback: int = 1) -> Optional[float]:
    """Simple difference slope between the last two VWAP samples ``lookback`` apart."""
    if len(samples) <= lookback:
        return None
    latest = samples[-1]
    prior = samples[-1 - lookback]
    if latest is None or prior is None:
        return None
    return latest - prior


def session_vwap_sigma_series(bars_1m: Sequence[Bar]) -> list[float]:
    """Cumulative session VWAP standard deviation, aligned 1:1 with ``bars_1m``.

    Volume-weighted sigma of typical price around the (also cumulative,
    volume-weighted) session VWAP - the TradingView-style band construction
    used in docs/VWAP-TOUCH-STUDY-2026-07-09.md section 5. ``result[i]`` is
    the sigma through (and including) ``bars_1m[i]``'s close; resets at each
    RTH day exactly like ``session_vwap_series``. Deliberately a separate
    cumulative pass (not folded into ``session_vwap_series``) to avoid
    touching that already-tested function's signature/behavior.
    """
    result: list[float] = []
    cum_pv = 0.0
    cum_pv2 = 0.0
    cum_vol = 0.0
    current_day: Optional[date] = None
    for b in bars_1m:
        day = to_et(b.ts).date()
        if day != current_day:
            cum_pv = 0.0
            cum_pv2 = 0.0
            cum_vol = 0.0
            current_day = day
        tp = (b.high + b.low + b.close) / 3.0
        cum_pv += tp * b.volume
        cum_pv2 += tp * tp * b.volume
        cum_vol += b.volume
        if cum_vol > 0:
            vwap = cum_pv / cum_vol
            variance = max(cum_pv2 / cum_vol - vwap * vwap, 0.0)
            sigma = variance ** 0.5
        else:
            sigma = 0.0
        result.append(sigma)
    return result


def ema_series(values: Sequence[float], period: int) -> list[float]:
    """Standard exponential moving average, seeded with the first value.

    Used for EMA9/EMA20 on the 5m and 3m aggregated bars (v0.1 decision
    model trigger families). Seeding with the first value rather than a
    simple-average warmup keeps this well-defined for the short agg-bar
    histories typical early in a session; acceptable for a trigger-level
    reference line (not used as a precision technical-analysis EMA).
    """
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out


# ── opening range ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpeningRange:
    high: float
    low: float
    width: float
    width_vs_atr: Optional[float]


def opening_range(
    completed_today_bars: Sequence[Bar],
    minutes: int = 15,
    atr_14: Optional[float] = None,
) -> Optional[OpeningRange]:
    """First ``minutes`` RTH bars' high/low, only once fully formed.

    Returns None until ``minutes`` bars from session open are present in
    ``completed_today_bars`` (i.e. the range has not closed yet).
    """
    if not completed_today_bars:
        return None
    day = to_et(completed_today_bars[0].ts).date()
    session_open_dt = datetime.combine(day, SESSION_OPEN, tzinfo=ET)
    window_end = session_open_dt + timedelta(minutes=minutes)

    window_bars = [b for b in completed_today_bars if session_open_dt <= to_et(b.ts) < window_end]
    if len(window_bars) < minutes:
        return None

    high = max(b.high for b in window_bars)
    low = min(b.low for b in window_bars)
    width = high - low
    width_vs_atr = (width / atr_14) if atr_14 and atr_14 > 0 else None
    return OpeningRange(high=high, low=low, width=width, width_vs_atr=width_vs_atr)


# ── daily OHLC / ATR / prior-day levels ──────────────────────────────────────


@dataclass(frozen=True)
class DailyOHLC:
    trade_date: date
    open: float
    high: float
    low: float
    close: float


def daily_ohlc_from_1m(bars_1m: Sequence[Bar]) -> list[DailyOHLC]:
    """Collapse a (possibly multi-day) 1m bar series into one OHLC row per ET calendar day."""
    by_day: dict[date, list[Bar]] = {}
    for b in bars_1m:
        by_day.setdefault(to_et(b.ts).date(), []).append(b)

    out: list[DailyOHLC] = []
    for day in sorted(by_day):
        group = sorted(by_day[day], key=lambda x: x.ts)
        out.append(
            DailyOHLC(
                trade_date=day,
                open=group[0].open,
                high=max(x.high for x in group),
                low=min(x.low for x in group),
                close=group[-1].close,
            )
        )
    return out


def atr(daily_bars: Sequence[DailyOHLC], window: int = 14) -> Optional[float]:
    """Simple (non-Wilder) ATR over the trailing ``window`` sessions.

    Requires ``window + 1`` daily bars (a prior close is needed for the
    first true-range value). Returns None if insufficient history.
    """
    if len(daily_bars) < window + 1:
        return None
    trs = []
    for i in range(1, len(daily_bars)):
        prior_close = daily_bars[i - 1].close
        d = daily_bars[i]
        tr = max(d.high - d.low, abs(d.high - prior_close), abs(d.low - prior_close))
        trs.append(tr)
    last_window = trs[-window:]
    return sum(last_window) / len(last_window)


def prior_day_levels(daily_bars: Sequence[DailyOHLC], trade_date: date) -> Optional[DailyOHLC]:
    """Most recent daily OHLC strictly before ``trade_date``."""
    prior = [d for d in daily_bars if d.trade_date < trade_date]
    return prior[-1] if prior else None


# ── RVOL for time-of-day ─────────────────────────────────────────────────────


def cumulative_volume_by_minute_index(bars_1m_day: Sequence[Bar]) -> list[int]:
    """Cumulative RTH volume at each minute-of-session index (0-based from session open)."""
    cum = []
    total = 0
    for b in bars_1m_day:
        total += b.volume
        cum.append(total)
    return cum


def rvol_for_time_of_day(
    current_day_bars_completed: Sequence[Bar],
    prior_sessions: Sequence[Sequence[Bar]],
    lookback_sessions: int = 20,
) -> Optional[float]:
    """RVOL-TOD = today's cumulative volume so far / trailing-N-session average
    cumulative volume through the SAME minute-of-session index.

    Documented Stage A simplification: alignment is by raw bar position
    (minute-of-session index), which assumes each prior session is a full,
    regular RTH day (no early closes / halts). Early-close sessions would
    misalign; acceptable for this scaffold, worth revisiting before Stage A
    results are trusted.
    """
    if not current_day_bars_completed:
        return None
    idx = len(current_day_bars_completed) - 1
    today_cum = cumulative_volume_by_minute_index(current_day_bars_completed)[-1]

    relevant = list(prior_sessions)[-lookback_sessions:]
    hist_values = []
    for session in relevant:
        if len(session) <= idx:
            continue
        cum = cumulative_volume_by_minute_index(session[: idx + 1])
        hist_values.append(cum[-1])

    if not hist_values:
        return None
    avg_hist = sum(hist_values) / len(hist_values)
    if avg_hist <= 0:
        return None
    return today_cum / avg_hist


# ── per-decision feature snapshot ────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureSnapshot:
    decision_ts: datetime
    idx: int  # index into the day's 1m bar list this snapshot decides for
    window: TimeWindow
    last_close: float  # last COMPLETED 1m bar's close (reference price at decision time)
    vwap: Optional[float]
    vwap_slope_5m: Optional[float]
    rvol_tod: Optional[float]
    structure_15m: StructureState
    opening_range: Optional[OpeningRange]
    prior_day: Optional[DailyOHLC]
    atr_14: Optional[float]
    bars_3m: list[AggBar] = field(default_factory=list)
    bars_5m: list[AggBar] = field(default_factory=list)
    bars_15m: list[AggBar] = field(default_factory=list)
    # v0.1 decision model additions (default so pre-existing FeatureSnapshot
    # construction sites - tests included - keep working unchanged).
    day_type: DayType = DayType.undecided
    vwap_sigma: Optional[float] = None
    ema9_5m: Optional[float] = None
    ema20_5m: Optional[float] = None
    ema9_3m: Optional[float] = None
    # v0.2 (sweep v2) addition: completed 1-minute bars wrapped as AggBar
    # (n_minutes=1) - same shape/PIT guarantee as bars_3m/5m/15m - so
    # bot.scalp.rules_v02's band-reversal trigger can confirm on a 1m close
    # without needing a separately-threaded raw Bar list. Default empty list
    # keeps every pre-existing FeatureSnapshot construction site (tests
    # included) working unchanged.
    bars_1m: list[AggBar] = field(default_factory=list)


@dataclass
class _AggBucketBuilder:
    """Incremental, time-based, gap-safe N-minute bucket accumulator.

    Feeding bars one at a time (in ascending ts order) via ``observe`` and
    reading ``.completed`` after each bar produces EXACTLY the same list, in
    the same order, as calling ``derive_agg_bars(bars_seen_so_far, minutes)``
    fresh after every new bar - but in O(1) amortized work per bar instead of
    O(n) per call. This is what makes ``build_feature_snapshots`` O(n) per
    day instead of O(n^2); see performance note in that function's docstring.
    Still fully time-based (uses each bar's own ts to compute its bucket),
    so it stays correct across data gaps/halts, unlike an index-only shortcut.
    """

    minutes: int
    session_open_dt: datetime
    completed: list[AggBar] = field(default_factory=list)
    _bucket_start: Optional[datetime] = field(default=None, repr=False)
    _group: list[Bar] = field(default_factory=list, repr=False)

    def observe(self, b: Bar) -> None:
        ts = to_et(b.ts)
        elapsed_min = (ts - self.session_open_dt).total_seconds() / 60.0
        bucket_idx = int(elapsed_min // self.minutes)
        bucket_start = self.session_open_dt + timedelta(minutes=bucket_idx * self.minutes)

        if self._bucket_start is not None and bucket_start != self._bucket_start:
            self._flush()
        self._bucket_start = bucket_start
        self._group.append(b)
        if len(self._group) == self.minutes:
            self._flush()

    def _flush(self) -> None:
        # Only emit a FULLY populated bucket (mirrors derive_agg_bars dropping
        # any bucket with fewer than `minutes` bars) - whether it's the
        # trailing partial bucket or a mid-sequence one orphaned by a gap.
        if len(self._group) == self.minutes:
            group = self._group
            self.completed.append(
                AggBar(
                    ts_open=self._bucket_start,
                    ts_close=self._bucket_start + timedelta(minutes=self.minutes),
                    open=group[0].open,
                    high=max(x.high for x in group),
                    low=min(x.low for x in group),
                    close=group[-1].close,
                    volume=sum(x.volume for x in group),
                    n_minutes=len(group),
                )
            )
        self._group = []


def _rvol_from_cumulative(
    idx: int,
    today_cum_full: Sequence[int],
    prior_cum_arrays: Sequence[Sequence[int]],
) -> Optional[float]:
    """Same math as ``rvol_for_time_of_day``, off arrays precomputed ONCE
    per session (see build_feature_snapshots) rather than resummed on every
    call. Safe because entry k of a cumulative-volume array depends only on
    entries 0..k of that SAME array - indexing a precomputed full-session
    array is mathematically identical to recomputing on a truncated prefix.
    """
    if idx < 0 or idx >= len(today_cum_full):
        return None
    today_cum = today_cum_full[idx]
    hist_values = [arr[idx] for arr in prior_cum_arrays if len(arr) > idx]
    if not hist_values:
        return None
    avg_hist = sum(hist_values) / len(hist_values)
    if avg_hist <= 0:
        return None
    return today_cum / avg_hist


def build_feature_snapshots(
    day_bars_1m: Sequence[Bar],
    prior_sessions_1m: Sequence[Sequence[Bar]],
    *,
    opening_range_minutes: int = 15,
    atr_window: int = 14,
    rvol_lookback: int = 20,
    min_warmup_bars: int = 1,
) -> list[FeatureSnapshot]:
    """Build one FeatureSnapshot per decision point in a single trading day.

    Decision point ``i`` (``min_warmup_bars <= i < len(day_bars_1m)``) uses
    only ``day_bars_1m[:i]`` - see module docstring for why that is exactly
    "bars whose close <= decision_ts". ``prior_sessions_1m`` supplies prior
    trading days (sorted ascending or not - sorted internally) for ATR(14),
    prior-day high/low/close, and RVOL-TOD history; none of it is ever mixed
    with same-day future bars.

    Performance note: an earlier version of this function called
    ``derive_agg_bars`` / ``session_vwap_series`` / ``rvol_for_time_of_day``
    fresh on the growing ``day_bars_1m[:i]`` prefix at every decision point -
    correct, but O(n) work per decision means O(n^2) per day (~390^2 for a
    full RTH session), which is too slow once real multi-month SPY/QQQ
    history lands. This version computes each per-session array ONCE
    (``session_vwap_series``, ``cumulative_volume_by_minute_index``) and
    feeds bars into ``_AggBucketBuilder`` incrementally, giving identical
    results in O(n) per day. Every shortcut here only ever reads entries
    that depend on strictly-prior positions in the SAME array/builder state
     - no future bar's data can leak into an earlier snapshot.
    """
    if not day_bars_1m:
        return []

    daily_history: list[DailyOHLC] = []
    for session in prior_sessions_1m:
        daily_history.extend(daily_ohlc_from_1m(session))
    daily_history.sort(key=lambda d: d.trade_date)

    today = to_et(day_bars_1m[0].ts).date()
    atr14 = atr(daily_history, window=atr_window)
    prior_day = prior_day_levels(daily_history, today)

    sorted_prior_sessions = sorted(prior_sessions_1m, key=lambda s: to_et(s[0].ts).date() if s else date.min)
    recent_prior_sessions = sorted_prior_sessions[-rvol_lookback:] if rvol_lookback > 0 else []

    # Precompute once per day - each is a single O(n) pass whose i-th entry
    # depends only on entries 0..i of the SAME bar sequence (see docstring).
    vwap_full = session_vwap_series(day_bars_1m)
    vwap_sigma_full = session_vwap_sigma_series(day_bars_1m)
    ts_to_vwap_full = {to_et(b.ts): v for b, v in zip(day_bars_1m, vwap_full)}
    today_cum_full = cumulative_volume_by_minute_index(day_bars_1m)
    prior_cum_arrays = [cumulative_volume_by_minute_index(s) for s in recent_prior_sessions]

    # Day-type precompute: running cumulative counts of post-10:00 ET minutes
    # spent above/below that same minute's own VWAP. cum_*[j] depends only on
    # bars 0..j of day_bars_1m (same no-leakage guarantee as today_cum_full).
    post10_above_cum: list[int] = []
    post10_below_cum: list[int] = []
    post10_total_cum: list[int] = []
    _run_above = _run_below = _run_total = 0
    for _j, _b in enumerate(day_bars_1m):
        if to_et(_b.ts).time() >= _DAY_TYPE_LOOKBACK_START:
            if _b.close > vwap_full[_j]:
                _run_above += 1
            elif _b.close < vwap_full[_j]:
                _run_below += 1
            _run_total += 1
        post10_above_cum.append(_run_above)
        post10_below_cum.append(_run_below)
        post10_total_cum.append(_run_total)

    session_open_dt = datetime.combine(today, SESSION_OPEN, tzinfo=ET)
    builder1 = _AggBucketBuilder(1, session_open_dt)
    builder3 = _AggBucketBuilder(3, session_open_dt)
    builder5 = _AggBucketBuilder(5, session_open_dt)
    builder15 = _AggBucketBuilder(15, session_open_dt)

    snapshots: list[FeatureSnapshot] = []
    observed = 0
    for i in range(min_warmup_bars, len(day_bars_1m)):
        # Feed the builders any bars newly completed since the last decision.
        while observed < i:
            b = day_bars_1m[observed]
            builder1.observe(b)
            builder3.observe(b)
            builder5.observe(b)
            builder15.observe(b)
            observed += 1

        decision_ts = to_et(day_bars_1m[i].ts)
        window = classify_time_window(decision_ts)

        vwap = vwap_full[i - 1]

        # .completed is mutated in place as later bars are observed, so we
        # MUST copy here - storing a live reference would let future
        # buckets silently "appear" in an already-built snapshot.
        agg1 = list(builder1.completed)
        agg3 = list(builder3.completed)
        agg5 = list(builder5.completed)
        agg15 = list(builder15.completed)

        slope5: Optional[float] = None
        if len(agg5) >= 2:
            v_latest = ts_to_vwap_full.get(agg5[-1].ts_close - timedelta(minutes=1))
            v_prior = ts_to_vwap_full.get(agg5[-2].ts_close - timedelta(minutes=1))
            if v_latest is not None and v_prior is not None:
                slope5 = v_latest - v_prior

        rvol = _rvol_from_cumulative(i - 1, today_cum_full, prior_cum_arrays)
        structure = structure_state(agg15)

        vwap_sigma = vwap_sigma_full[i - 1]

        # Day-type classification (see DayType docstring): before 10:30 ET
        # there's an insurmountable warmup floor regardless of share; from
        # 10:30 on, read the post-10:00 above/below shares off the ONE
        # cumulative pass computed above (index i-1 = last completed bar).
        if decision_ts.time() < _DAY_TYPE_DECISION_FLOOR:
            day_type = DayType.undecided
        else:
            total10 = post10_total_cum[i - 1]
            if total10 == 0:
                day_type = DayType.undecided
            else:
                share_above = post10_above_cum[i - 1] / total10
                share_below = post10_below_cum[i - 1] / total10
                if share_above >= _DAY_TYPE_ONE_SIDED_SHARE:
                    day_type = DayType.trend_up
                elif share_below >= _DAY_TYPE_ONE_SIDED_SHARE:
                    day_type = DayType.trend_down
                else:
                    day_type = DayType.rotation

        # EMA9/EMA20 (5m) and EMA9 (3m) recomputed fresh off the already-
        # materialized agg3/agg5 lists (same cost class as the agg3/agg5
        # copies just above - small, bounded by minutes-per-session/N).
        ema9_5m = ema_series([ab.close for ab in agg5], 9)[-1] if agg5 else None
        ema20_5m = ema_series([ab.close for ab in agg5], 20)[-1] if agg5 else None
        ema9_3m = ema_series([ab.close for ab in agg3], 9)[-1] if agg3 else None

        # Opening range only ever depends on the first `opening_range_minutes`
        # bars of the session; passing a small bounded prefix (instead of the
        # full growing `day_bars_1m[:i]`) is equivalent once i has reached
        # that window, and correctly None before it - opening_range() itself
        # still filters by timestamp, this just bounds the slice size.
        cap = min(i, opening_range_minutes + 5)
        orange = opening_range(day_bars_1m[:cap], opening_range_minutes, atr14)

        snapshots.append(
            FeatureSnapshot(
                decision_ts=decision_ts,
                idx=i,
                window=window,
                last_close=day_bars_1m[i - 1].close,
                vwap=vwap,
                vwap_slope_5m=slope5,
                rvol_tod=rvol,
                structure_15m=structure,
                opening_range=orange,
                prior_day=prior_day,
                atr_14=atr14,
                bars_3m=agg3,
                bars_5m=agg5,
                day_type=day_type,
                vwap_sigma=vwap_sigma,
                ema9_5m=ema9_5m,
                ema20_5m=ema20_5m,
                ema9_3m=ema9_3m,
                bars_1m=agg1,
                bars_15m=agg15,
            )
        )
    return snapshots
