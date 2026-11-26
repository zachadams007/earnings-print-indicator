"""Stage A v0 decision rule: parameterized "3-of-5 confirmation" confluence rubric.

Implements the confirmation model from
docs/SCALPING-COPILOT-CURSOR-RESEARCH-2026-07-08.md section 3.1:
each of 5 independent categories counts as at most one confirmation; Long or
Short requires >= min_confirmations categories AND zero hard vetoes.

This is scaffolding for iteration, NOT a validated strategy - every
threshold lives on RuleConfig so it can be swept/tuned later without editing
this module.

Resolved ambiguity - category 5 "execution timing (3m/1m)": the spec's
Boom Hunter Pro confirmation (Ehlers EOT/COG oscillator) is not implemented
anywhere in this repo yet. Rather than fabricate an indicator, this module
uses an explicit, clearly-labeled placeholder proxy (`impulse candle`: the
last completed 3m bar's body as a fraction of ATR(14)) so the 5-category
shape is testable today. Swap `_confirm_execution_timing` for a real Boom
Hunter port later without touching the rest of the rubric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .features import FeatureSnapshot, StructureState, TimeWindow


class Recommendation(str, Enum):
    long = "long"
    short = "short"
    wait = "wait"


CONFIRMATION_CATEGORIES: tuple[str, ...] = (
    "structure_15m",
    "vwap_bias_5m",
    "setup_trigger_5m",
    "volume_rvol",
    "execution_timing",
)


def _default_trade_windows() -> frozenset[TimeWindow]:
    # Matches the locked avoid-windows in the lane C research doc: trade only
    # 9:45-11:30 and 1:30-3:30; avoid open/lunch/close.
    return frozenset({TimeWindow.main1, TimeWindow.main2})


def _default_required_categories() -> frozenset[str]:
    return frozenset(CONFIRMATION_CATEGORIES)


@dataclass
class RuleConfig:
    rvol_confirm_threshold: float = 1.5
    rvol_veto_threshold: float = 1.0
    min_confirmations: int = 3
    trade_windows: frozenset[TimeWindow] = field(default_factory=_default_trade_windows)
    required_categories: frozenset[str] = field(default_factory=_default_required_categories)
    impulse_candle_atr_frac: float = 0.15
    veto_on_structure_conflict: bool = True
    macro_blackout: bool = False  # Stage A has no macro/news feed; caller may inject True later


@dataclass
class Decision:
    decision_ts: datetime
    recommendation: Recommendation
    confirmations_met: int
    long_confirmations: frozenset[str]
    short_confirmations: frozenset[str]
    vetoes: tuple[str, ...]
    window: TimeWindow


def _confirm_structure(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """Category 1: 15m structure. Long needs HH/HL, Short needs LH/LL."""
    s = snapshot.structure_15m
    return (s == StructureState.higher_highs_lows, s == StructureState.lower_highs_lows)


def _confirm_vwap_bias(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """Category 2: VWAP bias (5m). Close on the biased side + non-adverse slope."""
    if snapshot.vwap is None:
        return (False, False)
    above = snapshot.last_close > snapshot.vwap
    below = snapshot.last_close < snapshot.vwap
    slope = snapshot.vwap_slope_5m
    non_falling = slope is None or slope >= 0
    non_rising = slope is None or slope <= 0
    return (above and non_falling, below and non_rising)


def _confirm_setup_trigger(snapshot: FeatureSnapshot) -> tuple[bool, bool]:
    """Category 3: setup trigger (5m) - opening-range breakout proxy."""
    orange = snapshot.opening_range
    if orange is None:
        return (False, False)
    return (snapshot.last_close > orange.high, snapshot.last_close < orange.low)


def _confirm_volume(snapshot: FeatureSnapshot, cfg: RuleConfig) -> tuple[bool, bool]:
    """Category 4: RVOL-TOD participation filter (direction-agnostic)."""
    r = snapshot.rvol_tod
    ok = r is not None and r >= cfg.rvol_confirm_threshold
    return (ok, ok)


def _confirm_execution_timing(snapshot: FeatureSnapshot, cfg: RuleConfig) -> tuple[bool, bool]:
    """Category 5: execution timing (3m/1m) - impulse-candle placeholder for Boom Hunter."""
    bars3 = snapshot.bars_3m
    if not bars3 or not snapshot.atr_14:
        return (False, False)
    last = bars3[-1]
    body = last.close - last.open
    threshold = cfg.impulse_candle_atr_frac * snapshot.atr_14
    return (body >= threshold, -body >= threshold)


def decide(snapshot: FeatureSnapshot, cfg: RuleConfig) -> Decision:
    checks = {
        "structure_15m": _confirm_structure(snapshot),
        "vwap_bias_5m": _confirm_vwap_bias(snapshot),
        "setup_trigger_5m": _confirm_setup_trigger(snapshot),
        "volume_rvol": _confirm_volume(snapshot, cfg),
        "execution_timing": _confirm_execution_timing(snapshot, cfg),
    }

    long_hits = {name for name, (l, _s) in checks.items() if l and name in cfg.required_categories}
    short_hits = {name for name, (_l, s) in checks.items() if s and name in cfg.required_categories}

    vetoes: list[str] = []
    if snapshot.rvol_tod is not None and snapshot.rvol_tod < cfg.rvol_veto_threshold:
        vetoes.append("rvol_below_veto_floor")

    if cfg.veto_on_structure_conflict:
        struct_long, struct_short = checks["structure_15m"]
        vwap_long, vwap_short = checks["vwap_bias_5m"]
        if (struct_long and vwap_short) or (struct_short and vwap_long):
            vetoes.append("structure_vwap_conflict")

    if snapshot.window not in cfg.trade_windows:
        vetoes.append("outside_trade_window")

    if cfg.macro_blackout:
        vetoes.append("macro_blackout")

    has_veto = len(vetoes) > 0

    if not has_veto and len(long_hits) >= cfg.min_confirmations and len(long_hits) > len(short_hits):
        rec = Recommendation.long
        confirmations = len(long_hits)
    elif not has_veto and len(short_hits) >= cfg.min_confirmations and len(short_hits) > len(long_hits):
        rec = Recommendation.short
        confirmations = len(short_hits)
    else:
        rec = Recommendation.wait
        confirmations = max(len(long_hits), len(short_hits))

    return Decision(
        decision_ts=snapshot.decision_ts,
        recommendation=rec,
        confirmations_met=confirmations,
        long_confirmations=frozenset(long_hits),
        short_confirmations=frozenset(short_hits),
        vetoes=tuple(vetoes),
        window=snapshot.window,
    )


def decide_all(snapshots: list[FeatureSnapshot], cfg: RuleConfig) -> list[Decision]:
    return [decide(s, cfg) for s in snapshots]
