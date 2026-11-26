"""Decision model v0.3 (sweep v3) -- the owner's HIGH-WIN-RATE hunt (2026-07-09):
find rule configs with win% >= [n] (stretch [n]) and still positive net
expectancy after cost, R:R down to ~1:1 or below acceptable if win rate earns
it. Sweep v3 (not published) has the full design
and the four arms this sweep tests.

ADDITIVE -- does not touch rules.py / rules_v01.py / rules_v02.py. Reuses
their helpers directly rather than reimplementing:
  - rules_v01._core_context / _core_bias           (15m structure + 5m VWAP bias)
  - rules_v01._trig_trend_pullback                  (EMA9/20 + band pullback-hold,
                                                      already day-type-gated to
                                                      trend_up/trend_down internally)
  - rules_v02._trig_band_reversal                   (band-piercing reversal bar,
                                                      bypasses context/bias/day_type
                                                      by design -- see rules_v02's
                                                      module docstring)
  - rules_v02._trig_vwap_midline_retest             (retest-hold entry, never
                                                      fires on the crossing bar)

Only ONE genuinely new trigger family is added here: ``trend_pullback_hold``,
which wires v01's existing ``_trig_trend_pullback`` through the v02-style
context+bias+trigger core-gate composition (v01's own ``decide_core_gate``
already does this, but its ``CoreGateConfig`` bundles ALL 5 v0.1 trigger
families together with v0.1's booster shape; this module gives
trend_pullback_hold its own single-family v0.2-style config so it can be
swept independently, with its own RVOL-veto-as-filter-toggle, matching how
band_reversal/vwap_midline_retest are swept in rules_v02.py).

The other three high-win-rate arms this sweep runs (2sigma band-fade-to-
half-vwap "fade-to-nearer-target", 2sigma band-fade "scratch-quick", and
midline-retest-hold at a shorter target) reuse band_reversal /
vwap_midline_retest UNCHANGED for the decision side -- only their EXIT
templates differ (see exits.py's ExitConfigV3 / compute_stop_and_target_v3),
so no additional decision-model code is needed for those three; they are
still routed through ``decide_core_gate_v3`` below (its band_reversal /
vwap_midline_retest branches are thin pass-throughs to the v02 functions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .features import FeatureSnapshot, TimeWindow
from .rules import Decision, Recommendation
from .rules_v01 import _core_bias, _core_context, _trig_trend_pullback
from .rules_v02 import _trig_band_reversal, _trig_vwap_midline_retest


class TriggerFamilyV3(str, Enum):
    band_reversal = "band_reversal"  # reused from rules_v02 unchanged
    vwap_midline_retest = "vwap_midline_retest"  # reused from rules_v02 unchanged
    trend_pullback_hold = "trend_pullback_hold"  # NEW: v01's trigger, v02-style gate


def _default_trade_windows() -> frozenset[TimeWindow]:
    return frozenset({TimeWindow.main1, TimeWindow.main2})


@dataclass
class CoreGateConfigV3:
    family: TriggerFamilyV3
    # band_reversal knobs (ignored by the other families) -- same shape as
    # rules_v02.CoreGateConfigV2 so _trig_band_reversal (which only reads
    # these three attributes by name) works unchanged when passed this config.
    band_reversal_use_1sigma: bool = False
    band_reversal_confirm_tf: str = "3m"
    band_reversal_volume_lookback: int = 20
    # hard vetoes -- same shape as rules_v01/rules_v02, applied to every family.
    # "RVOL filter on/off" (one of this sweep's swept arms) is implemented by
    # varying rvol_veto_threshold: 1.0 = filter on (the v01/v02 default),
    # 0.0 = filter off (rvol_tod is never negative, so a 0.0 floor never vetoes).
    trade_window_filter_enabled: bool = True
    trade_windows: frozenset[TimeWindow] = field(default_factory=_default_trade_windows)
    rvol_veto_threshold: float = 1.0
    macro_blackout: bool = False


def decide_core_gate_v3(snapshot: FeatureSnapshot, cfg: CoreGateConfigV3) -> Decision:
    vetoes: list[str] = []
    if cfg.trade_window_filter_enabled and snapshot.window not in cfg.trade_windows:
        vetoes.append("outside_trade_window")
    if snapshot.rvol_tod is not None and snapshot.rvol_tod < cfg.rvol_veto_threshold:
        vetoes.append("rvol_below_veto_floor")
    if cfg.macro_blackout:
        vetoes.append("macro_blackout")
    has_veto = bool(vetoes)

    if cfg.family == TriggerFamilyV3.band_reversal:
        long_ok, short_ok = _trig_band_reversal(snapshot, cfg)  # duck-types on the 3 band_reversal_* attrs
        fam_name = "band_reversal"
    elif cfg.family == TriggerFamilyV3.vwap_midline_retest:
        context_long, context_short = _core_context(snapshot)
        bias_long, bias_short = _core_bias(snapshot)
        trig_long, trig_short = _trig_vwap_midline_retest(snapshot)
        long_ok = context_long and bias_long and trig_long
        short_ok = context_short and bias_short and trig_short
        fam_name = "vwap_midline_retest"
    elif cfg.family == TriggerFamilyV3.trend_pullback_hold:
        context_long, context_short = _core_context(snapshot)
        bias_long, bias_short = _core_bias(snapshot)
        trig_long, trig_short = _trig_trend_pullback(snapshot)
        long_ok = context_long and bias_long and trig_long
        short_ok = context_short and bias_short and trig_short
        fam_name = "trend_pullback_hold"
    else:
        raise ValueError(f"unknown TriggerFamilyV3 {cfg.family!r}")

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


def decide_core_gate_v3_all(snapshots: list[FeatureSnapshot], cfg: CoreGateConfigV3) -> list[Decision]:
    return [decide_core_gate_v3(s, cfg) for s in snapshots]
