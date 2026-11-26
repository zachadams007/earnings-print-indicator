"""Decision model v0.1 - the real CORE GATE + TRIGGER FAMILIES + BOOSTERS rubric.

Implements docs/COPILOT-V0-SYSTEM-PROMPT-2026-07-09.md's "DECISION MODEL
v0.1" section on top of the Stage A no-leakage harness, informed by the
empirical base rates in docs/VWAP-TOUCH-STUDY-2026-07-09.md.

This module is ADDITIVE - it does not touch ``rules.py``'s placeholder
"3-of-5 confirmation" rubric (``decide`` / ``RuleConfig`` / the
``CONFIRMATION_CATEGORIES`` shape), which stays exactly as-is so the existing
tests/scalp/test_rules.py suite keeps passing unchanged. ``decide_core_gate``
below produces the SAME ``Decision`` type ``rules.decide`` does, so it slots
directly into the existing ``evaluate.py`` / ``controls.py`` pipeline with no
changes needed there beyond controls.py's new optional ``decide_fn`` param.

CORE GATE (all three must agree, or the answer is WAIT):
  1. CONTEXT (15m): HH/HL for longs, LH/LL for shorts (``structure_15m``).
  2. BIAS (5m VWAP): the last COMPLETED 5m candle closes on the biased side
     of VWAP, with a non-adverse VWAP slope.
  3. TRIGGER (5m): at least one ENABLED trigger family fires in that
     direction - see the five ``_trig_*`` functions below, one per family
     named in the spec:
       (a) trend_pullback - trend-day pullback holds at EMA9/20 or
                                       a VWAP +-1sigma/+-2sigma band.
       (b) rotation_fade - rotation-day +-2sigma band fade toward VWAP.
       (c) vwap_midline - VWAP mid-line reclaim/reject.
       (d) orb - opening-range break.
       (e) failed_break_reversal - failed-breakout reversal at the prior
                                       day's high/low.
     (a) is day-type-gated to the matching trend direction; (b) is gated to
     rotation days; (c)/(d)/(e) fire on any day type (matches the spec's
     "ANY DAY" framing for ORB/failed-break, and the mid-line's "tradable on
     any day, weaker on rotation" framing - that day-type-dependent WEAKNESS
     is left to the booster sweep, not hard-gated, so the sweep can measure
     whether it actually matters).

BOOSTERS (sweep-controlled; they gate entries in this backtest instance
rather than only nudging a continuous confidence score, since Stage A
measures binary Long/Short/Wait outcomes, not a 0-10 confidence scale):
  - RVOL booster: if ``rvol_booster_threshold`` is set, require
    ``rvol_tod >= threshold`` in addition to the gate.
  - Confirming-candle booster: if ``confirming_candle_required``, require the
    last completed 3m bar to close in the trade direction on the correct
    side of EMA9(3m) (proxy for the spec's "clean 3m confirming candle on
    the right side of EMA 9"; Boom Hunter Pro is not implemented anywhere in
    this repo - same resolved-ambiguity treatment as rules.py's placeholder
    execution-timing category).

HARD VETOES -> forced WAIT: RVOL below the hard floor (independent of the
booster, a lower bar), outside the configured trade window (only enforced
if ``trade_window_filter_enabled``), and a macro-blackout flag (always False
in Stage A - no news feed - kept for interface parity with ``RuleConfig``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .features import DayType, FeatureSnapshot, TimeWindow
from .rules import Decision, Recommendation


class TriggerFamily(str, Enum):
    trend_pullback = "trend_pullback"
    rotation_fade = "rotation_fade"
    vwap_midline = "vwap_midline"
    orb = "orb"
    failed_break_reversal = "failed_break_reversal"


ALL_TRIGGER_FAMILIES: frozenset[TriggerFamily] = frozenset(TriggerFamily)


def _default_trade_windows() -> frozenset[TimeWindow]:
    return frozenset({TimeWindow.main1, TimeWindow.main2})


def _default_trigger_families() -> frozenset[TriggerFamily]:
    return frozenset(ALL_TRIGGER_FAMILIES)


@dataclass
class CoreGateConfig:
    trigger_families: frozenset[TriggerFamily] = field(default_factory=_default_trigger_families)
    rvol_booster_threshold: Optional[float] = None  # None = booster off
    confirming_candle_required: bool = False
    trade_window_filter_enabled: bool = True
    trade_windows: frozenset[TimeWindow] = field(default_factory=_default_trade_windows)
    rvol_veto_threshold: float = 1.0
    macro_blackout: bool = False


# ── CORE GATE: context + bias ───────────────────────────────────────────────


def _core_context(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """CORE GATE #1 - 15m structure. Long needs HH/HL, Short needs LH/LL."""
    from .features import StructureState

    s = snapshot.structure_15m
    return (s == StructureState.higher_highs_lows, s == StructureState.lower_highs_lows)


def _core_bias(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """CORE GATE #2 - 5m VWAP bias: last COMPLETED 5m candle holds the side,
    with a non-adverse VWAP slope ("VWAP flat-to-rising" for longs, mirror
    for shorts)."""
    if not snapshot.bars_5m or snapshot.vwap is None:
        return (False, False)
    last5 = snapshot.bars_5m[-1]
    slope = snapshot.vwap_slope_5m
    non_falling = slope is None or slope >= 0
    non_rising = slope is None or slope <= 0
    return (last5.close > snapshot.vwap and non_falling, last5.close < snapshot.vwap and non_rising)


# ── TRIGGER FAMILIES ─────────────────────────────────────────────────────────


def _band(snapshot: FeatureSnapshot, n_sigma: float) -> Optional[float]:
    if snapshot.vwap is None or snapshot.vwap_sigma is None:
        return None
    return snapshot.vwap + n_sigma * snapshot.vwap_sigma


def _trig_trend_pullback(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """(a) Trend-day pullback HOLDS at EMA9/20 or a VWAP +-1sigma/+-2sigma band.

    "Holds" = the last completed 5m bar's range touched the level but closed
    back on the trend side of it (a rejection, not a break-through).
    """
    if not snapshot.bars_5m:
        return (False, False)
    last = snapshot.bars_5m[-1]

    long_ok = False
    if snapshot.day_type == DayType.trend_up:
        levels = [
            lvl
            for lvl in (snapshot.ema9_5m, snapshot.ema20_5m, _band(snapshot, -1.0), _band(snapshot, -2.0))
            if lvl is not None
        ]
        long_ok = any(last.low <= lvl <= last.high and last.close > lvl for lvl in levels)

    short_ok = False
    if snapshot.day_type == DayType.trend_down:
        levels = [
            lvl
            for lvl in (snapshot.ema9_5m, snapshot.ema20_5m, _band(snapshot, 1.0), _band(snapshot, 2.0))
            if lvl is not None
        ]
        short_ok = any(last.low <= lvl <= last.high and last.close < lvl for lvl in levels)

    return (long_ok, short_ok)


def _trig_rotation_fade(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """(b) Rotation-day +-2sigma band fade toward VWAP - the strongest single
    result in the VWAP touch study (single-digit/low-teen break rates)."""
    if snapshot.day_type != DayType.rotation or not snapshot.bars_5m:
        return (False, False)
    lower2 = _band(snapshot, -2.0)
    upper2 = _band(snapshot, 2.0)
    if lower2 is None or upper2 is None:
        return (False, False)
    last = snapshot.bars_5m[-1]
    long_ok = last.low <= lower2 and last.close > lower2
    short_ok = last.high >= upper2 and last.close < upper2
    return (long_ok, short_ok)


def _trig_vwap_midline(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """(c) VWAP mid-line reclaim (long) / reject (short) - prior completed 5m
    bar closed on one side, the latest completed 5m bar closes on the other."""
    if len(snapshot.bars_5m) < 2 or snapshot.vwap is None:
        return (False, False)
    prev, last = snapshot.bars_5m[-2], snapshot.bars_5m[-1]
    long_ok = prev.close < snapshot.vwap and last.close > snapshot.vwap
    short_ok = prev.close > snapshot.vwap and last.close < snapshot.vwap
    return (long_ok, short_ok)


def _trig_orb(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """(d) Opening-range break - any day."""
    orange = snapshot.opening_range
    if orange is None or not snapshot.bars_5m:
        return (False, False)
    last_close = snapshot.bars_5m[-1].close
    return (last_close > orange.high, last_close < orange.low)


def _trig_failed_break_reversal(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """(e) Failed-breakout reversal at a prior-day level - any day. Long =
    failed breakdown at prior-day low (poke below, close back above); Short =
    failed breakout at prior-day high (poke above, close back below)."""
    if snapshot.prior_day is None or not snapshot.bars_5m:
        return (False, False)
    last = snapshot.bars_5m[-1]
    long_ok = last.low < snapshot.prior_day.low and last.close > snapshot.prior_day.low
    short_ok = last.high > snapshot.prior_day.high and last.close < snapshot.prior_day.high
    return (long_ok, short_ok)


_TRIGGER_FNS = {
    TriggerFamily.trend_pullback: _trig_trend_pullback,
    TriggerFamily.rotation_fade: _trig_rotation_fade,
    TriggerFamily.vwap_midline: _trig_vwap_midline,
    TriggerFamily.orb: _trig_orb,
    TriggerFamily.failed_break_reversal: _trig_failed_break_reversal,
}


def _fire_trigger_families(
    snapshot: FeatureSnapshot, cfg: CoreGateConfig
) -> tuple[frozenset[str], frozenset[str]]:
    long_fired: set[str] = set()
    short_fired: set[str] = set()
    for fam in cfg.trigger_families:
        fn = _TRIGGER_FNS.get(fam)
        if fn is None:
            continue
        long_ok, short_ok = fn(snapshot)
        if long_ok:
            long_fired.add(fam.value)
        if short_ok:
            short_fired.add(fam.value)
    return (frozenset(long_fired), frozenset(short_fired))


# ── BOOSTERS ─────────────────────────────────────────────────────────────────


def _confirming_candle(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """Booster: clean 3m confirming candle on the right side of EMA9(3m)."""
    bars3 = snapshot.bars_3m
    if not bars3 or snapshot.ema9_3m is None:
        return (False, False)
    last = bars3[-1]
    long_ok = last.close > last.open and last.close > snapshot.ema9_3m
    short_ok = last.close < last.open and last.close < snapshot.ema9_3m
    return (long_ok, short_ok)


# ── decision entry point ─────────────────────────────────────────────────────


def decide_core_gate(snapshot: FeatureSnapshot, cfg: CoreGateConfig) -> Decision:
    context_long, context_short = _core_context(snapshot)
    bias_long, bias_short = _core_bias(snapshot)
    trig_long_fams, trig_short_fams = _fire_trigger_families(snapshot, cfg)

    vetoes: list[str] = []
    if cfg.trade_window_filter_enabled and snapshot.window not in cfg.trade_windows:
        vetoes.append("outside_trade_window")
    if snapshot.rvol_tod is not None and snapshot.rvol_tod < cfg.rvol_veto_threshold:
        vetoes.append("rvol_below_veto_floor")
    if cfg.macro_blackout:
        vetoes.append("macro_blackout")
    has_veto = bool(vetoes)

    booster_long_ok = True
    booster_short_ok = True
    if cfg.rvol_booster_threshold is not None:
        ok = snapshot.rvol_tod is not None and snapshot.rvol_tod >= cfg.rvol_booster_threshold
        booster_long_ok = booster_long_ok and ok
        booster_short_ok = booster_short_ok and ok
    if cfg.confirming_candle_required:
        cand_long, cand_short = _confirming_candle(snapshot)
        booster_long_ok = booster_long_ok and cand_long
        booster_short_ok = booster_short_ok and cand_short

    long_gate = (
        not has_veto
        and context_long
        and bias_long
        and bool(trig_long_fams)
        and booster_long_ok
    )
    short_gate = (
        not has_veto
        and context_short
        and bias_short
        and bool(trig_short_fams)
        and booster_short_ok
    )

    if long_gate and not short_gate:
        rec = Recommendation.long
        fired = trig_long_fams
    elif short_gate and not long_gate:
        rec = Recommendation.short
        fired = trig_short_fams
    else:
        rec = Recommendation.wait
        fired = frozenset()

    return Decision(
        decision_ts=snapshot.decision_ts,
        recommendation=rec,
        confirmations_met=len(fired),
        long_confirmations=trig_long_fams if rec == Recommendation.long else frozenset(),
        short_confirmations=trig_short_fams if rec == Recommendation.short else frozenset(),
        vetoes=tuple(vetoes),
        window=snapshot.window,
    )


def decide_core_gate_all(snapshots: list[FeatureSnapshot], cfg: CoreGateConfig) -> list[Decision]:
    return [decide_core_gate(s, cfg) for s in snapshots]
