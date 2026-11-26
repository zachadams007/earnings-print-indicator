from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from bot.scalp.copilot_v1 import collapse_simultaneous_candidates
from bot.scalp.racer import TransitionInput, evaluate_transition, reconcile_cross_index_state


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_copilot_v1_confirmation_lab",
    ROOT / "scripts" / "run_copilot_v1_confirmation_lab.py",
)
assert SPEC is not None and SPEC.loader is not None
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


def _input(*, symbol: str = "SPY", minute_index: int = 90) -> TransitionInput:
    ts = pd.Timestamp("2025-01-02 11:00", tz="America/New_York")
    return TransitionInput(
        symbol=symbol,
        decision_timestamp=ts,
        bar_close_timestamp=ts,
        transition_geometry="first_trend_pullback",
        candidate_direction="long",
        provider_eligible=True,
        completed_5m=True,
        completed_3m=True,
        completed_1m_trigger=True,
        frozen_risk_valid=True,
        minute_index=minute_index,
        ema_gap_bps=4.0,
        close_vwap_bps=3.0,
        rsi3=58.0,
        rvol3=1.2,
        day_return_bps=25.0,
        cross_alignment=1.0,
        cross_index_observation_count=3,
    )


def test_transition_requires_a_completed_decision_bar_and_completed_trigger():
    complete = evaluate_transition(_input())
    partial = evaluate_transition(
        replace(_input(), bar_close_timestamp=pd.Timestamp("2025-01-02 11:01", tz="America/New_York"))
    )

    assert complete.action == "confirmable"
    assert complete.provenance.completed_1m_trigger
    assert partial.action == "wait"
    assert partial.invalidation == "decision_not_completed_bar"


def test_spy_qqq_iwm_use_identical_state_logic():
    states = [evaluate_transition(_input(symbol=symbol)).as_record() for symbol in LAB.SYMBOLS]
    comparable = []
    for state in states:
        state.pop("state_symbol")
        comparable.append(state)

    assert comparable[0] == comparable[1] == comparable[2]


def test_timestamped_event_prior_cannot_bypass_scalp_confirmation():
    state = evaluate_transition(
        replace(
            _input(),
            event_reaction_state="known_result_and_reaction",
            event_direction="long",
            event_result_timestamped=True,
            completed_1m_trigger=False,
        )
    )

    assert state.prior_direction == "long"
    assert state.action == "wait"
    assert state.invalidation == "completed_1m_trigger_missing"


def test_historical_30_second_input_cannot_be_synthesized_as_available():
    state = evaluate_transition(replace(_input(), historical_30s_available=True))

    assert not state.provenance.historical_30s_available
    assert state.execution_30s_state == "historical_unavailable"


def test_clock_labels_do_not_suppress_an_otherwise_identical_transition():
    opening = evaluate_transition(_input(minute_index=5))
    close = evaluate_transition(_input(minute_index=380))

    assert opening.action == close.action == "confirmable"
    assert opening.auction_day_state == "opening_auction"
    assert close.auction_day_state == "late_day"


def test_cross_index_conflict_forces_neutral_wait():
    state = evaluate_transition(
        replace(_input(), cross_alignment=-1.0, cross_index_observation_count=3)
    )

    assert state.cross_index_state == "conflict"
    assert state.prior_direction == "neutral"
    assert state.action == "wait"
    assert state.invalidation == "cross_index_conflict"


def test_partial_or_missing_cross_index_evidence_fails_closed():
    local = evaluate_transition(_input()).as_record()

    partial = reconcile_cross_index_state(local, -1.0, 1)
    missing = reconcile_cross_index_state(local, None, 0)

    for state in (partial, missing):
        assert state["cross_index_state"] == "unavailable"
        assert state["action"] == "wait"
        assert state["invalidation"] == "cross_index_unavailable"
        assert not state["provenance_cross_index_available"]


def test_canonical_evaluator_is_pending_until_all_three_indices_are_present():
    pending = evaluate_transition(
        replace(_input(), cross_alignment=None, cross_index_observation_count=0)
    )

    assert pending.action == "pending_cross_index"
    assert pending.invalidation == "cross_index_pending"

    completed = reconcile_cross_index_state(pending.as_record(), 1.0, 3)
    assert completed["action"] == "confirmable"
    assert completed["invalidation"] is None


def test_same_side_collapse_preserves_one_typed_geometry_and_audits_the_rest():
    ts = pd.Timestamp("2025-01-02 11:00", tz="America/New_York")
    rows = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "session": ts.date(),
                "decision_ts": ts,
                "candidate_side": "long",
                "trigger_quality": 2.0,
                "family": "first_trend_pullback",
                "transition_geometry": "first_trend_pullback",
                "transition_reason": "first_trend_pullback: completed",
                "level_id": "ema20",
            },
            {
                "symbol": "SPY",
                "session": ts.date(),
                "decision_ts": ts,
                "candidate_side": "long",
                "trigger_quality": 1.0,
                "family": "vwap_reclaim_reject",
                "transition_geometry": "vwap_reclaim_reject",
                "transition_reason": "vwap_reclaim_reject: completed",
                "level_id": "vwap",
            },
        ]
    )

    collapsed = collapse_simultaneous_candidates(rows).iloc[0]

    assert collapsed["family"] == "first_trend_pullback"
    assert collapsed["transition_geometry"] == "first_trend_pullback"
    assert collapsed["transition_reason"].startswith("first_trend_pullback")
    assert collapsed["observed_geometries"] == "first_trend_pullback+vwap_reclaim_reject"


def test_one_correlated_open_plan_is_selected_across_etfs():
    ts = pd.Timestamp("2025-01-02 11:00", tz="America/New_York")
    rows = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "decision_ts": ts,
                "entry_ts": ts,
                "plan_hold_minutes": 15,
                "realized_eligible": True,
                "accepted_raw": True,
                "quality_score": 8.0,
                "expected_net_bps": 5.0,
                "candidate_direction_score": 8.0,
            },
            {
                "symbol": "QQQ",
                "decision_ts": ts,
                "entry_ts": ts,
                "plan_hold_minutes": 15,
                "realized_eligible": True,
                "accepted_raw": True,
                "quality_score": 7.5,
                "expected_net_bps": 6.0,
                "candidate_direction_score": 7.5,
            },
        ]
    )

    selected = LAB._portfolio_select(rows)

    assert selected["symbol"].tolist() == ["SPY"]


def test_racer_cache_fingerprint_and_day_weights_cover_the_state_contract():
    fingerprint = LAB._strategy_fingerprint(racer=True)
    normalized_paths = {path.replace("\\", "/") for path in fingerprint["files"]}
    assert normalized_paths == {"src/bot/scalp/copilot_v1.py", "src/bot/scalp/racer.py"}
    assert fingerprint["candidate_contract_version"] == LAB.RACER_CANDIDATE_CONTRACT_VERSION
    assert fingerprint["racer_state_version"]

    weighted = LAB._add_racer_cross_state_and_day_weights(
        pd.DataFrame(
            {
                "session": [pd.Timestamp("2025-01-02").date()] * 3,
                "candidate_side": ["long", "long", "short"],
            }
        )
    )
    assert weighted["racer_overlap_flag"].all()
    # One session's weights sum to 1.0 up to summation error (n=7 or 49 rows
    # would not land on exactly 1.0), so compare with a tolerance.
    assert math.isclose(weighted["racer_fit_weight"].sum(), 1.0, abs_tol=1e-9)

    retained = pd.DataFrame(
        {
            "session": [pd.Timestamp("2025-01-02").date()] * 2
            + [pd.Timestamp("2025-01-03").date()],
        }
    )
    fit_weight = LAB._fit_weight(retained)
    retained["fit_weight"] = fit_weight
    assert retained.groupby("session")["fit_weight"].sum().tolist() == [1.0, 1.0]


def test_each_racer_variant_has_at_most_one_interaction_and_no_hidden_subbot_categories():
    banned = {"symbol", "family", "time_bucket", "level_id"}
    for variant in LAB.RACER_SEQUENTIAL_VARIANTS:
        numeric, categorical = LAB._feature_names(variant)
        assert len(LAB._racer_variant_interactions(variant)) <= 1
        assert not categorical
        assert not (set(numeric) & banned)
    assert LAB._racer_variant_interactions(LAB.RACER_EVENT_NEUTRAL_CORE) == ()


def test_racer_reports_are_isolated_and_core_is_always_the_comparison_baseline():
    assert LAB.RACER_REPORT_JSON != LAB.REPORT_JSON
    assert LAB.RACER_REPORT_MD != LAB.REPORT_MD
    assert LAB.RACER_CACHE != LAB.CACHE

    summary = {
        "n": 100,
        "win_rate_net_1bp": 0.55,
        "net_bps_0_5bp": 1.0,
        "net_bps_1bp": 0.5,
        "delayed_net_bps_1bp": 0.1,
        "net_bps_2bp": -0.5,
    }
    analyses = {}
    for variant in LAB.RACER_SEQUENTIAL_VARIANTS:
        analyses[f"logistic__{variant}"] = {
            "accepted_portfolio": dict(summary),
            "by_year": {},
            "by_symbol": {},
            "by_family": {},
            "by_auction_day_state": {},
        }
    comparisons = LAB._racer_incremental_comparisons(analyses)
    core_id = f"logistic__{LAB.RACER_EVENT_NEUTRAL_CORE}"
    assert comparisons
    assert {item["baseline"] for item in comparisons.values()} == {core_id}
    assert all(not item["live_promotion_allowed"] for item in comparisons.values())


def test_every_racer_variant_fits_through_the_weighted_numeric_path():
    sessions = list(pd.date_range("2017-01-01", periods=600, freq="D"))
    sessions += list(pd.date_range("2019-01-01", periods=120, freq="D"))
    sessions += list(pd.date_range("2020-01-01", periods=20, freq="D"))
    n = len(sessions)
    rows = pd.DataFrame(
        {
            "session": [stamp.date() for stamp in sessions],
            "year": [stamp.year for stamp in sessions],
            "action": "confirmable",
            "realized_eligible": True,
            "future_up": [index % 2 for index in range(n)],
            "candidate_net_win_1bp": [(index // 2) % 2 for index in range(n)],
            "candidate_net_bps_1bp": [float((index % 7) - 3) for index in range(n)],
            "candidate_side": ["long" if index % 2 else "short" for index in range(n)],
            "racer_candidate_side_sign": [1.0 if index % 2 else -1.0 for index in range(n)],
            "racer_prior_sign": [1.0 if index % 3 else -1.0 for index in range(n)],
            "cross_alignment": [float((index % 3) - 1) for index in range(n)],
            "rvol3": [0.8 + (index % 5) * 0.2 for index in range(n)],
        }
    )
    for feature_index, feature in enumerate(LAB.NUMERIC_FEATURES):
        if feature not in rows:
            rows[feature] = [float((index + feature_index) % 11) for index in range(n)]

    for variant in LAB.RACER_SEQUENTIAL_VARIANTS:
        scored = LAB.walk_forward(rows, "logistic", variant)
        assert not scored.empty
        assert set(scored["feature_variant"]) == {variant}
