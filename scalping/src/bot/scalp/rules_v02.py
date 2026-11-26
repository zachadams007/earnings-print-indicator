"""Decision model v0.2 (sweep v2) - fixes the two known blind spots in
rules_v01's CORE GATE, per docs/COPILOT-V0-SYSTEM-PROMPT-2026-07-09.md's
v0.2 revision and the sweep-v1 result (data/scalp_sweep_trial_ledger.jsonl,
research notes).

This module is ADDITIVE - it does not touch ``rules.py`` or ``rules_v01.py``,
which stay exactly as-is so their existing test suites keep passing
unchanged. It reuses ``rules_v01``'s ``_core_context`` / ``_core_bias`` /
``_band`` / ``_trig_vwap_midline`` helpers and rules.py's ``Decision`` /
``Recommendation`` types.

TWO FAMILIES:

1. ``band_reversal`` - a reversal AGAINST the current VWAP side at/near a
   +-1sigma/+-2sigma band: the confirming bar (1m or 3m, per
   ``band_reversal_confirm_tf``) pierces the band level then closes back
   inside it, on ABOVE-AVERAGE volume for that timeframe. THIS FAMILY
   DELIBERATELY BYPASSES rules_v01's CORE CONTEXT (15m structure) AND CORE
   BIAS (VWAP-side agreement) gates. That bypass is the actual fix for
   v1's spec bug: a band-reversal trade is, by definition, a bet AGAINST
   the currently dominant side, so requiring "structure/bias agrees with
   the trade direction" - which is what v1's core gate does for every
   family, ``rotation_fade`` included - can only ever block it. v1's
   ``rotation_fade`` never actually fired for this reason (confirmed by the
   day-type gate ALSO restricting it to rotation days only, roughly halving
   opportunity on top of the bias-gate block) - this family removes both
   restrictions, keeping only the hard vetoes (RVOL floor / trade window /
   macro blackout) that apply to every family.

2. ``vwap_midline_retest`` vs ``vwap_midline_cross`` - same CORE GATE
   (context + bias) as v1, but two competing entry-timing rules for the
   trigger itself:
     - ``vwap_midline_cross`` is v1's ``_trig_vwap_midline`` UNCHANGED
       (enters ON the bar that crosses VWAP) - kept as-is so v2 can compare
       directly against it (v1 measured [n]bps median on cross-bar entries).
     - ``vwap_midline_retest`` never enters on the crossing bar itself; it
       waits for a LATER bar that (a) is not the crossing bar, (b) every bar
       since the cross has HELD the reclaimed side, and (c) touches/
       approaches VWAP and closes back on the reclaimed side (a retest-hold).

Both families still respect the hard vetoes (RVOL floor, trade window,
macro blackout) from ``CoreGateConfigV2``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Optional, Sequence

from .features import AggBar, FeatureSnapshot, TimeWindow
from .rules import Decision, Recommendation
from .rules_v01 import _band, _core_bias, _core_context, _trig_vwap_midline


class TriggerFamilyV2(str, Enum):
    band_reversal = "band_reversal"
    vwap_midline_retest = "vwap_midline_retest"
    vwap_midline_cross = "vwap_midline_cross"


def _default_trade_windows() -> frozenset[TimeWindow]:
    return frozenset({TimeWindow.main1, TimeWindow.main2})


@dataclass
class CoreGateConfigV2:
    family: TriggerFamilyV2
    # band_reversal knobs (ignored by the midline families):
    band_reversal_use_1sigma: bool = False  # False = 2sigma only, True = 1sigma+2sigma
    band_reversal_confirm_tf: str = "3m"  # "1m" or "3m" - which agg-bar list confirms the reversal
    band_reversal_volume_lookback: int = 20
    # hard vetoes - same shape as rules_v01.CoreGateConfig, applied to every family
    trade_window_filter_enabled: bool = True
    trade_windows: frozenset[TimeWindow] = field(default_factory=_default_trade_windows)
    rvol_veto_threshold: float = 1.0
    macro_blackout: bool = False


# ── band-reversal trigger ────────────────────────────────────────────────────


def _avg_volume(bars: Sequence[AggBar], lookback: int = 20) -> Optional[float]:
    """Mean volume of the completed bars STRICTLY BEFORE the last one (the
    confirmation bar is excluded from its own reference average)."""
    history = bars[:-1][-lookback:] if len(bars) > 1 else []
    if not history:
        return None
    return mean(b.volume for b in history)


def _trig_band_reversal(snapshot: FeatureSnapshot, cfg: CoreGateConfigV2) -> tuple[bool, bool]:
    """Reversal bar at/through a +-1sigma/+-2sigma band, closing back inside
    it on above-average volume - fires regardless of day_type (unlike v1's
    rotation_fade) and regardless of core context/bias (see module docstring).
    """
    bars = snapshot.bars_1m if cfg.band_reversal_confirm_tf == "1m" else snapshot.bars_3m
    if len(bars) < 2 or snapshot.vwap is None or snapshot.vwap_sigma is None:
        return (False, False)

    last = bars[-1]
    avg_vol = _avg_volume(bars, cfg.band_reversal_volume_lookback)
    if avg_vol is None or last.volume < avg_vol:
        return (False, False)

    long_levels = [_band(snapshot, -2.0)]
    short_levels = [_band(snapshot, 2.0)]
    if cfg.band_reversal_use_1sigma:
        long_levels.append(_band(snapshot, -1.0))
        short_levels.append(_band(snapshot, 1.0))

    long_ok = any(lvl is not None and last.low <= lvl and last.close > lvl for lvl in long_levels)
    short_ok = any(lvl is not None and last.high >= lvl and last.close < lvl for lvl in short_levels)
    return (long_ok, short_ok)


# ── retest-entry midline trigger ─────────────────────────────────────────────


def _held_since(bars: Sequence[AggBar], cross_idx: int, current_idx: int, vwap: float, is_long: bool) -> bool:
    """True if every bar strictly between the crossing bar and the current
    (latest) bar keeps closing on the reclaimed side (no re-cross back out)."""
    for j in range(cross_idx + 1, current_idx):
        c = bars[j].close
        if is_long and c <= vwap:
            return False
        if not is_long and c >= vwap:
            return False
    return True


def _trig_vwap_midline_retest(snapshot: FeatureSnapshot, lookback: int = 10) -> tuple[bool, bool]:
    """Retest-hold entry: never fires on the crossing bar itself.

    Searches for a cross among bars STRICTLY BEFORE the current (last) bar -
    the loop below never lets the candidate crossing bar equal the current
    bar's index, which is what structurally guarantees "no entry on the
    cross bar" rather than relying on an extra runtime check. The current
    bar only qualifies as an entry if, since that earlier cross, every bar
    has held the reclaimed side AND the current bar's range touches/
    approaches VWAP while closing back on the reclaimed side (the retest).
    """
    bars = snapshot.bars_5m
    if snapshot.vwap is None or len(bars) < 3:
        return (False, False)
    vwap = snapshot.vwap
    current_idx = len(bars) - 1
    last = bars[current_idx]
    win = min(current_idx, lookback)

    def _search(is_long: bool) -> bool:
        for k in range(1, win + 1):
            cross_idx = current_idx - k
            if cross_idx < 1:
                break
            cross_bar = bars[cross_idx]
            prev_bar = bars[cross_idx - 1]
            crossed = (
                (prev_bar.close < vwap and cross_bar.close > vwap)
                if is_long
                else (prev_bar.close > vwap and cross_bar.close < vwap)
            )
            if not crossed:
                continue
            if not _held_since(bars, cross_idx, current_idx, vwap, is_long):
                continue
            touches = last.low <= vwap <= last.high
            holds_close = last.close > vwap if is_long else last.close < vwap
            if touches and holds_close:
                return True
        return False

    return (_search(True), _search(False))


# ── decision entry point ─────────────────────────────────────────────────────


def decide_core_gate_v2(snapshot: FeatureSnapshot, cfg: CoreGateConfigV2) -> Decision:
    vetoes: list[str] = []
    if cfg.trade_window_filter_enabled and snapshot.window not in cfg.trade_windows:
        vetoes.append("outside_trade_window")
    if snapshot.rvol_tod is not None and snapshot.rvol_tod < cfg.rvol_veto_threshold:
        vetoes.append("rvol_below_veto_floor")
    if cfg.macro_blackout:
        vetoes.append("macro_blackout")
    has_veto = bool(vetoes)

    if cfg.family == TriggerFamilyV2.band_reversal:
        long_ok, short_ok = _trig_band_reversal(snapshot, cfg)
        fam_name = "band_reversal"
    elif cfg.family == TriggerFamilyV2.vwap_midline_retest:
        context_long, context_short = _core_context(snapshot)
        bias_long, bias_short = _core_bias(snapshot)
        trig_long, trig_short = _trig_vwap_midline_retest(snapshot)
        long_ok = context_long and bias_long and trig_long
        short_ok = context_short and bias_short and trig_short
        fam_name = "vwap_midline_retest"
    elif cfg.family == TriggerFamilyV2.vwap_midline_cross:
        context_long, context_short = _core_context(snapshot)
        bias_long, bias_short = _core_bias(snapshot)
        trig_long, trig_short = _trig_vwap_midline(snapshot)
        long_ok = context_long and bias_long and trig_long
        short_ok = context_short and bias_short and trig_short
        fam_name = "vwap_midline_cross"
    else:
        raise ValueError(f"unknown TriggerFamilyV2 {cfg.family!r}")

    long_gate = not has_veto and long_ok
    short_gate = not has_veto and short_ok

    if long_gate and not short_gate:
        rec = Recommendation.long
        fired = frozenset({fam_name})
    elif short_gate and not long_gate:
        rec = Recommendation.short
        fired = frozenset({fam_name})
    else:
        rec = Recommendation.wait
        fired = frozenset()

    return Decision(
        decision_ts=snapshot.decision_ts,
        recommendation=rec,
        confirmations_met=len(fired),
        long_confirmations=fired if rec == Recommendation.long else frozenset(),
        short_confirmations=fired if rec == Recommendation.short else frozenset(),
        vetoes=tuple(vetoes),
        window=snapshot.window,
    )


def decide_core_gate_v2_all(snapshots: list[FeatureSnapshot], cfg: CoreGateConfigV2) -> list[Decision]:
    return [decide_core_gate_v2(s, cfg) for s in snapshots]
