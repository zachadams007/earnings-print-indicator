from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_copilot_v1_variant_worker",
    ROOT / "scripts" / "run_copilot_v1_variant_worker.py",
)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def test_merge_rejects_stale_candidate_provenance(tmp_path: Path, monkeypatch):
    scored = pd.DataFrame(
        {"model_kind": ["logistic"], "feature_variant": ["full"], "value": [1.0]}
    )
    metadata = {
        "db_fingerprint": {"sha256": "db-current"},
        "strategy_fingerprint": {"sha256": "candidate-current"},
        "events_sha256": "events-current",
    }
    fake_lab = SimpleNamespace(
        DEFAULT_DB=tmp_path / "unused.db",
        load_or_build_candidates=lambda *_: (pd.DataFrame(), metadata),
        _run_provenance=lambda metadata, scored, analysis: {
            "db_sha256": "db-current",
            "candidate_strategy_sha256": "candidate-current",
            "candidate_events_sha256": "events-current",
            "analysis_fingerprint": {"sha256": "analysis-current"},
            "scored_sha256": f"frame-{len(scored)}",
            "scored_rows": len(scored),
            "analysis_payload_sha256": "analysis-payload-current",
        },
    )
    monkeypatch.setattr(WORKER, "_load_lab", lambda: fake_lab)
    payload = {
        "run_id": "logistic__full",
        "variant": "full",
        "bootstrap_repeats": 500,
        "analysis": {},
        "scored": scored,
        "provenance": {
            "db_sha256": "db-current",
            "candidate_strategy_sha256": "candidate-stale",
            "candidate_events_sha256": "events-current",
            "analysis_fingerprint": {"sha256": "analysis-current"},
            "scored_sha256": "frame-1",
            "scored_rows": 1,
        },
    }
    output = tmp_path / "logistic__full.pkl"
    with output.open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(RuntimeError, match="candidate_strategy_sha256 provenance mismatch"):
        WORKER.merge_variants(tmp_path / "unused.pkl", tmp_path, 500)
