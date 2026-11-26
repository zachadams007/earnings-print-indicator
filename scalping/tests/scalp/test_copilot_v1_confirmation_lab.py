from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_copilot_v1_confirmation_lab",
    ROOT / "scripts" / "run_copilot_v1_confirmation_lab.py",
)
assert SPEC is not None and SPEC.loader is not None
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


def test_cross_symbol_mask_owns_a_writable_array_under_copy_on_write():
    ts = pd.Timestamp("2025-01-02 10:00", tz="America/New_York")
    events = pd.DataFrame(
        [{"decision_ts": ts, "candidate_side": "long", "symbol": "SPY"}]
    )
    peers = {
        symbol: pd.DataFrame(
            {
                "ts": [ts],
                "ema_gap_bps": [1.0],
                "close_vwap_bps": [2.0],
                "rsi3": [55.0],
                "research_eligible": [symbol != "QQQ"],
            }
        )
        for symbol in LAB.SYMBOLS
    }

    enriched = LAB._add_cross_symbol_features(events, peers)

    assert enriched.loc[0, "SPY_ema_gap_bps"] == 1.0
    assert np.isnan(enriched.loc[0, "QQQ_ema_gap_bps"])
    assert enriched.loc[0, "cross_alignment"] == 1.0


def test_realized_rows_excludes_unfilled_candidates_from_model_population():
    rows = pd.DataFrame(
        [
            {"opportunity_id": "filled", "realized_eligible": True},
            {"opportunity_id": "gap", "realized_eligible": False},
        ]
    )

    realized = LAB._realized_rows(rows)

    assert realized["opportunity_id"].tolist() == ["filled"]


def test_no_time_ablation_removes_only_the_clock_bucket():
    full_numeric, full_categorical = LAB._feature_names("full")
    no_time_numeric, no_time_categorical = LAB._feature_names("no_time")

    assert no_time_numeric == full_numeric
    assert "time_bucket" in full_categorical
    assert "time_bucket" not in no_time_categorical
    assert set(no_time_categorical) == set(full_categorical) - {"time_bucket"}


def test_score_shuffle_control_uses_positions_after_filtered_sparse_index(monkeypatch):
    rows = pd.DataFrame(
        [
            {
                "outer_year": 2025,
                "time_bucket": "close",
                "candidate_direction_score": 7.5,
                "quality_score": 7.5,
                "expected_net_bps": 3.0,
                "candidate_gross_bps": 4.0,
            },
            {
                "outer_year": 2025,
                "time_bucket": "close",
                "candidate_direction_score": 6.5,
                "quality_score": 6.5,
                "expected_net_bps": 1.0,
                "candidate_gross_bps": -1.0,
            },
        ],
        index=[4, 9],
    )
    monkeypatch.setattr(
        LAB,
        "_portfolio_select",
        lambda randomized: randomized.loc[randomized["accepted_raw"]].copy(),
    )

    result = LAB._score_shuffle_control(rows, seeds=3)

    assert result["seeds"] == 3
    assert result["mean_net_bps"] is not None


def test_candidate_cache_fingerprint_excludes_analysis_script():
    fingerprint = LAB._strategy_fingerprint()

    normalized_paths = {path.replace("\\", "/") for path in fingerprint["files"]}
    assert normalized_paths == {"src/bot/scalp/copilot_v1.py"}
    assert fingerprint["candidate_pipeline_sha256"]
    assert fingerprint["candidate_contract_version"] == LAB.CANDIDATE_CONTRACT_VERSION


def test_legacy_cache_is_rejected_when_candidate_contract_changes():
    current = LAB._strategy_fingerprint()
    legacy_sha, migration = next(iter(LAB.LEGACY_CACHE_MIGRATIONS.items()))
    legacy = {
        "sha256": legacy_sha,
        "files": {
            "scripts\\run_copilot_v1_confirmation_lab.py": migration["lab_file_sha256"],
            "src\\bot\\scalp\\copilot_v1.py": migration["copilot_file_sha256"],
        },
        **{key: current[key] for key in ("symbols", "outer_years", "primary_years", "outcome")},
    }

    # RACER state/provenance is now part of the candidate pipeline, so a cache
    # emitted before that schema cannot be silently reused.
    assert not LAB._legacy_candidate_fingerprint_matches(legacy, current)
    legacy["files"]["src\\bot\\scalp\\copilot_v1.py"] = "changed-candidate-code"
    assert not LAB._legacy_candidate_fingerprint_matches(legacy, current)


def test_frame_fingerprint_covers_schema_index_and_values():
    original = pd.DataFrame({"value": [1.0, 2.0], "label": ["a", "b"]}, index=[4, 9])
    changed_value = original.copy()
    changed_value.loc[9, "value"] = 3.0
    changed_index = original.reset_index(drop=True)

    fingerprint = LAB._frame_fingerprint(original)

    assert fingerprint == LAB._frame_fingerprint(original.copy())
    assert fingerprint != LAB._frame_fingerprint(changed_value)
    assert fingerprint != LAB._frame_fingerprint(changed_index)


def test_run_provenance_binds_scored_rows_and_analysis_payload():
    metadata = {
        "db_fingerprint": {"sha256": "db"},
        "strategy_fingerprint": {"sha256": "candidate-code"},
        "events_sha256": "candidate-rows",
    }
    scored = pd.DataFrame({"score": [5.0]})
    analysis = {"accepted": 0}

    original = LAB._run_provenance(metadata, scored, analysis)
    changed_scored = LAB._run_provenance(metadata, pd.DataFrame({"score": [7.0]}), analysis)
    changed_analysis = LAB._run_provenance(metadata, scored, {"accepted": 1})

    assert original["scored_sha256"] != changed_scored["scored_sha256"]
    assert original["analysis_payload_sha256"] != changed_analysis["analysis_payload_sha256"]
