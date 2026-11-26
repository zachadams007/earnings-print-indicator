"""Real R-multiple exit engine for the Stage A sweep v2 (rules_v02.py).

Sweep v1 measured pure fixed-horizon forward returns (5/15/30/60m), which
hides how a trade would actually be managed. This module adds a real
stop/target/time-stop exit simulated bar-by-bar on 1m data, so v2 can report
R-multiple expectancy, win rate, and average hold time instead.

CONSERVATIVE INTRABAR ORDERING (explicit assumption, stated once here and
enforced in ``simulate_exit``): OHLC bars do not tell us whether the stop or
the target was touched FIRST within a single 1m bar's [low, high] range. If
both levels fall inside the same bar, this module always resolves it as a
STOP-OUT, never a target hit. That is the conservative (pessimistic) choice
deliberately made to avoid overstating any edge - see project rule 4 / the
"keep failed/rejected evidence honest" convention this repo already applies
in controls.py.

Resolved ambiguity - "entry bar's adverse extreme": the initial stop is
computed from the last COMPLETED trigger bar (the 5m bar the decision model
fired on, i.e. ``snapshot.bars_5m[-1]``), not the yet-unclosed 1-minute FILL
bar the trade actually enters on. Using the fill bar's own high/low as a stop
level would require knowing that bar's range before it closes - a point-in-
time violation given this repo's convention (features.py docstring, project rules
#5) that entry is the fill bar's OPEN and nothing about that bar is known
until it closes. The already-closed trigger bar's extreme is available at
decision time and is the standard "structure that proves the idea wrong"
framing used in docs/COPILOT-V0-SYSTEM-PROMPT-2026-07-09.md's EXITS section.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from ..data.models import Bar
from .features import AggBar

StopType = Literal["entry_extreme", "atr_k"]


@dataclass(frozen=True)
class ExitConfig:
    stop_type: StopType
    atr_k: float = 1.0  # only used when stop_type == "atr_k"
    target_r: float = 1.5  # target expressed as a multiple of the initial risk (R)
    time_stop_minutes: int = 30


@dataclass(frozen=True)
class ExitResult:
    exit_price: float
    r_multiple: float
    reason: str  # "stop" | "target" | "time_stop" | "eod"
    hold_minutes: int


def atr_from_agg_bars(bars: Sequence[AggBar], window: int = 14) -> Optional[float]:
    """Simple (non-Wilder) ATR over the trailing ``window`` completed agg bars.

    Mirrors ``features.atr()``'s daily-bar version but for intraday agg bars
    (e.g. the 5m bars in ``FeatureSnapshot.bars_5m``). Requires ``window + 1``
    bars (a prior close is needed for the first true-range value); returns
    None otherwise. PIT-safe as long as ``bars`` is itself a point-in-time
    list (true of every ``FeatureSnapshot.bars_*`` list in this package) -
    this function performs no time filtering of its own.
    """
    if len(bars) < window + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(bars)):
        prior_close = bars[i - 1].close
        b = bars[i]
        tr = max(b.high - b.low, abs(b.high - prior_close), abs(b.low - prior_close))
        trs.append(tr)
    last_window = trs[-window:]
    return sum(last_window) / len(last_window)


def compute_stop_and_target(
    direction_sign: float,
    entry_price: float,
    trigger_bar_high: float,
    trigger_bar_low: float,
    atr_5m: Optional[float],
    cfg: ExitConfig,
) -> Optional[tuple[float, float]]:
    """Stop/target price levels fixed at entry (never revisited during the trade).

    Returns None for a degenerate trade (e.g. ATR unavailable for the
    atr_k stop, or the trigger bar's extreme is already on the wrong side of
    entry_price so risk <= 0) - callers should skip the trade rather than
    fabricate a level.
    """
    if cfg.stop_type == "entry_extreme":
        stop = trigger_bar_low if direction_sign > 0 else trigger_bar_high
    elif cfg.stop_type == "atr_k":
        if atr_5m is None or atr_5m <= 0:
            return None
        stop = entry_price - direction_sign * cfg.atr_k * atr_5m
    else:
        raise ValueError(f"unknown stop_type {cfg.stop_type!r}")

    risk = direction_sign * (entry_price - stop)
    if risk <= 0:
        return None
    target = entry_price + direction_sign * cfg.target_r * risk
    return stop, target


def simulate_exit(
    day_bars_1m: Sequence[Bar],
    entry_idx: int,
    direction_sign: float,
    entry_price: float,
    stop_price: float,
    target_price: float,
    cfg: ExitConfig,
) -> Optional[ExitResult]:
    """Walk forward bar-by-bar from the entry fill, checking stop/target intrabar.

    Scans ``day_bars_1m[entry_idx : entry_idx + time_stop_minutes]`` (never
    past the end of the day's bar list, which makes end-of-day-flat automatic
    since ``day_bars_1m`` is always a single session's bars in this harness).
    Checks the stop BEFORE the target on every bar - see module docstring for
    why: this is exactly the conservative both-hit tie-break, expressed as
    code order rather than a separate branch.
    """
    n = len(day_bars_1m)
    if entry_idx >= n:
        return None
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return None

    max_idx = min(n - 1, entry_idx + cfg.time_stop_minutes)
    for j in range(entry_idx, max_idx + 1):
        b = day_bars_1m[j]
        if direction_sign > 0:
            hit_stop = b.low <= stop_price
            hit_target = b.high >= target_price
        else:
            hit_stop = b.high >= stop_price
            hit_target = b.low <= target_price

        if hit_stop:  # checked first -- conservative both-hit tie-break
            return ExitResult(
                exit_price=stop_price,
                r_multiple=direction_sign * (stop_price - entry_price) / risk,
                reason="stop",
                hold_minutes=j - entry_idx + 1,
            )
        if hit_target:
            return ExitResult(
                exit_price=target_price,
                r_multiple=direction_sign * (target_price - entry_price) / risk,
                reason="target",
                hold_minutes=j - entry_idx + 1,
            )

    exit_bar = day_bars_1m[max_idx]
    exit_price = exit_bar.close
    reason = "eod" if max_idx == n - 1 else "time_stop"
    return ExitResult(
        exit_price=exit_price,
        r_multiple=direction_sign * (exit_price - entry_price) / risk,
        reason=reason,
        hold_minutes=max_idx - entry_idx + 1,
    )


# ── v0.3 exit archetypes (the owner's high-win-rate hunt, 2026-07-09) ────────────
#
# ADDITIVE section: ``ExitConfig``/``compute_stop_and_target``/``simulate_exit``
# above are UNCHANGED. High-win-rate mechanics need two stop/target shapes that
# a plain R-multiple-off-a-fixed-risk-unit config cannot express:
#
# The remaining two v0.3 arms (2sigma band-fade "scratch-quick": target 0.5R
# fixed / stop 1R; midline-retest-hold: target [n]R / stop 1R) need NO new
# exit code at all -- ``ExitConfig(stop_type="entry_extreme", target_r=...)``
# already expresses them exactly, since "stop 1R" is tautologically true of
# any entry_extreme config (the stop IS the risk unit by definition).

StopTypeV3 = Literal["entry_extreme", "atr_k", "band_extreme_buffer"]
TargetKindV3 = Literal["r_multiple", "vwap_half", "prior_swing_capped"]


@dataclass(frozen=True)
class ExitConfigV3:
    """Exit templates for the v0.3 high-win-rate sweep (rules_v03.py /
    sweep v3). Parallels ``ExitConfig`` -- does not modify
    it. ``simulate_exit`` only ever reads ``cfg.time_stop_minutes``, which this
    dataclass also carries, so an ``ExitConfigV3`` duck-types straight into
    that function unchanged; no changes to ``simulate_exit`` were needed.

    See the module-docstring section above for exactly which
    (stop_type, target_kind) pairs are meaningful:
      - ("band_extreme_buffer", "vwap_half")   -- band-fade-to-nearer-target
      - ("entry_extreme", "prior_swing_capped") -- trend-pullback-hold
      - ("entry_extreme" | "atr_k", "r_multiple") -- falls back to the exact
        same formula as ``compute_stop_and_target`` (kept for completeness /
        so this dataclass can stand in for ``ExitConfig`` if convenient).
    """

    stop_type: StopTypeV3
    atr_k: float = 1.0  # only used when stop_type == "atr_k"
    stop_mult: float = 1.0  # multiplier on target_distance; only used by "band_extreme_buffer"
    target_kind: TargetKindV3 = "r_multiple"
    target_r: float = 1.5  # only used when target_kind == "r_multiple"
    swing_lookback: int = 10  # completed 5m bars scanned for "prior_swing_capped"'s swing level
    time_stop_minutes: int = 20


def swing_level_from_agg_bars(bars_5m: Sequence[AggBar], lookback: int, is_long: bool) -> Optional[float]:
    """Most extreme high (long) / low (short) over the ``lookback`` COMPLETED
    5m bars strictly BEFORE the current (pullback/trigger) bar -- the "prior
    swing" a trend-pullback-hold's target aims back at. Excludes the trigger
    bar itself (mirrors ``_avg_volume``'s ``bars[:-1][-lookback:]`` pattern in
    rules_v02.py, for the same reason: the trigger/pullback bar is not part of
    the swing it is pulling back from). None if there is no prior history."""
    history = bars_5m[:-1][-lookback:] if len(bars_5m) > 1 else []
    if not history:
        return None
    return max(b.high for b in history) if is_long else min(b.low for b in history)


def compute_stop_and_target_v3(
    direction_sign: float,
    entry_price: float,
    trigger_bar_high: float,
    trigger_bar_low: float,
    atr_5m: Optional[float],
    vwap: Optional[float],
    swing_level: Optional[float],
    cfg: ExitConfigV3,
) -> Optional[tuple[float, float]]:
    """v0.3 stop/target computation. Returns None for a degenerate trade
    (unavailable inputs, or a non-positive risk/target distance) -- callers
    skip the trade rather than fabricate a level, same convention as
    ``compute_stop_and_target``.
    """
    if cfg.target_kind == "vwap_half":
        if cfg.stop_type != "band_extreme_buffer":
            raise ValueError("target_kind='vwap_half' must pair with stop_type='band_extreme_buffer'")
        if vwap is None:
            return None
        target_distance = 0.5 * abs(vwap - entry_price)
        if target_distance <= 0:
            return None
        target = entry_price + direction_sign * target_distance
        trigger_extreme = trigger_bar_low if direction_sign > 0 else trigger_bar_high
        stop = trigger_extreme - direction_sign * cfg.stop_mult * target_distance
        risk = direction_sign * (entry_price - stop)
        if risk <= 0:
            return None
        return stop, target

    # Every other combination needs a risk unit from the stop first -- same
    # two stop formulas as compute_stop_and_target (band_extreme_buffer is
    # only meaningful with target_kind="vwap_half", handled above).
    if cfg.stop_type == "entry_extreme":
        stop = trigger_bar_low if direction_sign > 0 else trigger_bar_high
    elif cfg.stop_type == "atr_k":
        if atr_5m is None or atr_5m <= 0:
            return None
        stop = entry_price - direction_sign * cfg.atr_k * atr_5m
    else:
        raise ValueError(f"stop_type {cfg.stop_type!r} requires target_kind='vwap_half'")

    risk = direction_sign * (entry_price - stop)
    if risk <= 0:
        return None

    if cfg.target_kind == "r_multiple":
        target = entry_price + direction_sign * cfg.target_r * risk
    elif cfg.target_kind == "prior_swing_capped":
        raw_distance = direction_sign * (swing_level - entry_price) if swing_level is not None else None
        # No usable/favorable swing -> fall back to exactly 1R (the stated cap).
        capped_distance = risk if raw_distance is None or raw_distance <= 0 else min(raw_distance, risk)
        target = entry_price + direction_sign * capped_distance
    else:
        raise ValueError(f"unknown target_kind {cfg.target_kind!r}")

    return stop, target
