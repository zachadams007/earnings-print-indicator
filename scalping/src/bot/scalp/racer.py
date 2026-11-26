"""Canonical point-in-time state contract for the Lane A RACER experiment.

RACER is one adaptive state machine for SPY, QQQ, and IWM.  It turns already
completed chart observations into a typed, serializable state record; it does
not place orders, inspect TradingView, or synthesize unavailable 30-second
history.  Candidate geometries are descriptive transitions inside this shared
machine, never independently parameterized strategies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import numpy as np


RACER_STATE_VERSION = "racer.v0.1"
Direction = Literal["long", "short", "neutral", "wait"]
Geometry = Literal[
    "failed_level_retest",
    "first_trend_pullback",
    "compression_expansion_retest",
    "vwap_reclaim_reject",
    "eod_continuation",
    "eod_profit_taking_reversal",
]
EventReactionState = Literal[
    "event_neutral_unknown",
    "known_result_unprovenanced",
    "known_result_and_reaction",
]


@dataclass(frozen=True)
class StateProvenance:
    """What was actually observable at the decision close."""

    provider_eligible: bool
    completed_5m: bool
    completed_3m: bool
    completed_1m_trigger: bool
    cross_index_available: bool
    cross_index_observation_count: int
    event_result_timestamped: bool
    historical_30s_available: bool


@dataclass(frozen=True)
class TransitionInput:
    """Immutable inputs to the one shared, completed-bar transition evaluator."""

    symbol: str
    decision_timestamp: object
    bar_close_timestamp: object
    transition_geometry: Geometry
    candidate_direction: Literal["long", "short"]
    provider_eligible: bool
    completed_5m: bool
    completed_3m: bool
    completed_1m_trigger: bool
    frozen_risk_valid: bool
    minute_index: int
    ema_gap_bps: float
    close_vwap_bps: float
    rsi3: float
    rvol3: float
    day_return_bps: float
    cross_alignment: float | None = None
    cross_index_observation_count: int = 0
    event_reaction_state: EventReactionState = "event_neutral_unknown"
    event_direction: Direction = "neutral"
    event_result_timestamped: bool = False
    historical_30s_available: bool = False


@dataclass(frozen=True)
class RacerState:
    """Result of one deterministic data -> prior -> confirmation transition."""

    state_version: str
    symbol: str
    decision_timestamp: object
    auction_day_state: str
    prior_state: str
    prior_direction: Direction
    transition_geometry: Geometry
    cross_index_state: str
    event_reaction_state: EventReactionState
    transition_reason: str
    invalidation: str | None
    action: Literal["confirmable", "pending_cross_index", "wait"]
    execution_30s_state: Literal["historical_unavailable", "prospective_only"]
    provenance: StateProvenance

    def as_record(self) -> dict[str, object]:
        """Flatten the immutable contract into cache/report-safe scalar columns."""
        record = asdict(self)
        provenance = record.pop("provenance")
        record["state_decision_timestamp"] = record.pop("decision_timestamp")
        record["state_symbol"] = record.pop("symbol")
        for name, value in provenance.items():
            record[f"provenance_{name}"] = value
        record["historical_30s_available"] = provenance["historical_30s_available"]
        return record


def _finite(value: float | None) -> bool:
    return value is not None and bool(np.isfinite(float(value)))


def _auction_day_state(minute_index: int) -> str:
    """Label opportunity context without ever applying a clock-only veto."""
    if minute_index <= 60:
        return "opening_auction"
    if minute_index >= 360:
        return "late_day"
    return "continuous_auction"


def _prior_direction(values: TransitionInput) -> Direction:
    """Use observable day/tape context as a prior, never as an entry trigger."""
    if values.ema_gap_bps > 0.0 and values.close_vwap_bps > 0.0:
        return "long"
    if values.ema_gap_bps < 0.0 and values.close_vwap_bps < 0.0:
        return "short"
    return "neutral"


def _cross_index_state(cross_alignment: float | None, observation_count: int) -> str:
    if observation_count != 3 or not _finite(cross_alignment):
        return "unavailable"
    # At least two of SPY/QQQ/IWM oppose the candidate-side tape.  This is the
    # same numerical rule for all three symbols and is evaluated at the close.
    if float(cross_alignment) <= -(1.0 / 3.0):
        return "conflict"
    if float(cross_alignment) > 0.0:
        return "aligned"
    return "neutral"


def evaluate_transition(values: TransitionInput) -> RacerState:
    """Evaluate the locked RACER sequence using no observation after the close.

    A timestamped known event may inform the directional *prior*, but can never
    bypass the completed 5m/3m/1m confirmation layers.  Historical 30-second
    data are explicitly unavailable and therefore remain prospective-only.
    """
    completed_bar = values.bar_close_timestamp == values.decision_timestamp
    completed_5m = bool(values.completed_5m and completed_bar)
    completed_3m = bool(values.completed_3m and completed_bar)
    completed_1m = bool(values.completed_1m_trigger and completed_bar)
    cross_state = _cross_index_state(
        values.cross_alignment, values.cross_index_observation_count
    )
    provenance = StateProvenance(
        provider_eligible=bool(values.provider_eligible),
        completed_5m=completed_5m,
        completed_3m=completed_3m,
        completed_1m_trigger=completed_1m,
        cross_index_available=cross_state != "unavailable",
        cross_index_observation_count=int(values.cross_index_observation_count),
        event_result_timestamped=bool(values.event_result_timestamped),
        # The locked corpus contains no historical 30s series.  Do not allow a
        # caller to turn a missing source into a retrospective trigger flag.
        historical_30s_available=False,
    )

    prior = _prior_direction(values)
    if (
        values.event_reaction_state == "known_result_and_reaction"
        and values.event_result_timestamped
        and values.event_direction in ("long", "short")
    ):
        prior = values.event_direction

    invalidation: str | None = None
    action: Literal["confirmable", "pending_cross_index", "wait"] = "confirmable"
    if not values.provider_eligible:
        invalidation, action = "provider_ineligible", "wait"
    elif not completed_bar:
        invalidation, action = "decision_not_completed_bar", "wait"
    elif not completed_5m:
        invalidation, action = "completed_5m_thesis_unavailable", "wait"
    elif not completed_3m or not (_finite(values.rsi3) and _finite(values.rvol3)):
        invalidation, action = "completed_3m_qualification_unavailable", "wait"
    elif not completed_1m:
        invalidation, action = "completed_1m_trigger_missing", "wait"
    elif not values.frozen_risk_valid:
        invalidation, action = "frozen_risk_invalid", "wait"
    elif cross_state == "unavailable":
        # Candidate generation is local-first, but the canonical evaluator is
        # never actionable until the exact same-close SPY/QQQ/IWM read arrives.
        invalidation, action = "cross_index_pending", "pending_cross_index"
    elif cross_state == "conflict":
        invalidation, action, prior = "cross_index_conflict", "wait", "neutral"

    stage_names = "data/event gate -> auction/day -> directional prior -> geometry -> 5m -> 3m -> 1m"
    reason = (
        f"{values.transition_geometry}: {stage_names} completed"
        if action == "confirmable"
        else f"{values.transition_geometry}: {invalidation}"
    )
    return RacerState(
        state_version=RACER_STATE_VERSION,
        symbol=values.symbol,
        decision_timestamp=values.decision_timestamp,
        auction_day_state=_auction_day_state(values.minute_index),
        prior_state="event_neutral" if values.event_reaction_state == "event_neutral_unknown" else "known_reaction",
        prior_direction=prior,
        transition_geometry=values.transition_geometry,
        cross_index_state=cross_state,
        event_reaction_state=values.event_reaction_state,
        transition_reason=reason,
        invalidation=invalidation,
        action=action,
        execution_30s_state="historical_unavailable",
        provenance=provenance,
    )


def reconcile_cross_index_state(
    record: Mapping[str, object],
    cross_alignment: float | None,
    observation_count: int,
) -> dict[str, object]:
    """Apply the point-in-time peer read without changing a local transition.

    Cross-index values are only available after all three same-close frames are
    aligned.  This pure reconciliation preserves the local completed-bar state
    and converts a material peer conflict to a neutral WAIT; it never invents a
    new trigger or risk plan.
    """
    updated = dict(record)
    state = _cross_index_state(cross_alignment, observation_count)
    updated["cross_index_state"] = state
    updated["provenance_cross_index_available"] = state != "unavailable"
    updated["provenance_cross_index_observation_count"] = int(observation_count)
    if state == "unavailable" and updated.get("action") in {
        "confirmable",
        "pending_cross_index",
    }:
        updated["invalidation"] = "cross_index_unavailable"
        updated["action"] = "wait"
        updated["transition_reason"] = (
            f"{updated['transition_geometry']}: cross_index_unavailable"
        )
    elif state == "conflict":
        updated["prior_direction"] = "neutral"
        updated["invalidation"] = "cross_index_conflict"
        updated["action"] = "wait"
        updated["transition_reason"] = (
            f"{updated['transition_geometry']}: cross_index_conflict"
        )
    elif updated.get("action") == "pending_cross_index":
        updated["invalidation"] = None
        updated["action"] = "confirmable"
        updated["transition_reason"] = (
            f"{updated['transition_geometry']}: data/event gate -> auction/day -> "
            "directional prior -> geometry -> 5m -> 3m -> 1m completed"
        )
    return updated
