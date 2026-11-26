"""Nested walk-forward lab for the chart-visible Lane A confirmation proxy.

This script answers a narrow question: among deterministic setup candidates a
human can identify on the canonical 5m/3m/1m/30s layout, can a point-in-time
model selectively confirm enough candidates to clear the locked promotion
gate?  It does not reconstruct the owner's unlogged discretion and does not model
options or 30-second execution.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bot.scalp.copilot_v1 import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_chart_features,
    collapse_simultaneous_candidates,
    complementary_scores,
    generate_candidate_events,
)
from bot.scalp.racer import RACER_STATE_VERSION, reconcile_cross_index_state  # noqa: E402

DEFAULT_DB = Path("data/sqlite/scalp_sip.db")
CACHE = ROOT / "data" / "cache" / "copilot_v1_candidate_events.pkl.gz"
REPORT_JSON = ROOT / "data" / "reports" / "copilot_v1_confirmation_lab.json"
REPORT_MD = ROOT / "data" / "reports" / "copilot_v1_confirmation_lab.md"
RACER_CACHE = ROOT / "data" / "cache" / "racer_v0_candidate_events.pkl.gz"
RACER_REPORT_JSON = ROOT / "data" / "reports" / "racer_v0_sequential_ablation.json"
RACER_REPORT_MD = ROOT / "data" / "reports" / "racer_v0_sequential_ablation.md"
SYMBOLS = ("SPY", "QQQ", "IWM")
OUTER_YEARS = tuple(range(2020, 2027))
PRIMARY_FULL_YEARS = tuple(range(2020, 2026))
ModelKind = Literal["logistic", "shallow_gbt"]
CANDIDATE_CONTRACT_VERSION = "2026-07-16.1"
RACER_CANDIDATE_CONTRACT_VERSION = "2026-07-18.racer-v0.1"
RACER_EVENT_NEUTRAL_CORE = "racer_event_neutral_core"
# This is a pre-registered, ordered ablation list.  Every challenger starts
# from the same event-neutral core and adds exactly one named interaction; no
# run may select a best row as the new primary model.
RACER_SEQUENTIAL_VARIANTS = (
    RACER_EVENT_NEUTRAL_CORE,
    "racer_prior_x_cross_alignment",
    "racer_prior_x_rvol",
)
RACER_INTERACTION_BY_VARIANT = {
    RACER_EVENT_NEUTRAL_CORE: None,
    "racer_prior_x_cross_alignment": "racer_prior_x_cross_alignment",
    "racer_prior_x_rvol": "racer_prior_x_rvol",
}
LEGACY_CACHE_MIGRATIONS = {
    "fd8dfdd5e65a4c7fd6a568b9ada36142be55b7f9db1eae91969c09fc9a2a9b46": {
        "lab_file_sha256": "cfccbba8bf3107118930d6c47c161968af5b98f33f19d402fd576e0737354234",
        "copilot_file_sha256": "1862d9d3ebadf3e5679513b95d25aa9b9f450eba0d7cbdc701b11fdc4e2bd34b",
        "candidate_pipeline_sha256": "35d0b4a7b44f55835251d07c8bcea35e083b84686b14887070f1149c14cbcc90",
    }
}


def _strategy_fingerprint(*, racer: bool = False) -> dict:
    # Candidate-cache provenance must not change when reporting, controls, or
    # model analysis changes. Hash the imported candidate implementation plus
    # only the local functions that construct/cache candidate rows.
    files = [ROOT / "src" / "bot" / "scalp" / "copilot_v1.py"]
    if racer:
        files.append(ROOT / "src" / "bot" / "scalp" / "racer.py")
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    pipeline_functions = (
        "_read_symbol",
        "_add_cross_symbol_features",
        "_add_racer_cross_state_and_day_weights",
        "_build_candidate_events",
    )
    pipeline_source = "\n\n".join(inspect.getsource(globals()[name]) for name in pipeline_functions)
    payload = {
        "files": hashes,
        "candidate_pipeline_sha256": hashlib.sha256(pipeline_source.encode()).hexdigest(),
        "candidate_contract_version": (
            RACER_CANDIDATE_CONTRACT_VERSION if racer else CANDIDATE_CONTRACT_VERSION
        ),
        "racer_state_version": RACER_STATE_VERSION if racer else None,
        "symbols": SYMBOLS,
        "outer_years": OUTER_YEARS,
        "primary_years": PRIMARY_FULL_YEARS,
        "outcome": "next-open; structural stop; half at [n]R; half at 2R; 15m time; same-bar stop-first",
    }
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def _legacy_candidate_fingerprint_matches(cached: dict, current: dict) -> bool:
    """Accept only an explicitly reconciled legacy candidate-cache build."""
    if not cached or cached.get("candidate_pipeline_sha256") is not None:
        return False
    cached_files = {
        str(path).replace("\\", "/"): digest
        for path, digest in cached.get("files", {}).items()
    }
    current_files = {
        str(path).replace("\\", "/"): digest
        for path, digest in current.get("files", {}).items()
    }
    candidate_path = "src/bot/scalp/copilot_v1.py"
    migration = LEGACY_CACHE_MIGRATIONS.get(cached.get("sha256"))
    if migration is None:
        return False
    shared_contract = ("symbols", "outer_years", "primary_years", "outcome")
    return (
        cached_files.get("scripts/run_copilot_v1_confirmation_lab.py")
        == migration["lab_file_sha256"]
        and cached_files.get(candidate_path) == migration["copilot_file_sha256"]
        and current_files.get(candidate_path) == migration["copilot_file_sha256"]
        and current.get("candidate_pipeline_sha256")
        == migration["candidate_pipeline_sha256"]
        and all(cached.get(key) == current.get(key) for key in shared_contract)
    )


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash frame schema, index, and values for cache/artifact provenance."""
    hasher = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    hasher.update(json.dumps(schema, separators=(",", ":")).encode())
    row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    hasher.update(row_hashes.to_numpy(dtype=np.uint64, copy=False).tobytes())
    return hasher.hexdigest()


def _analysis_fingerprint() -> dict:
    """Hash only model/scoring/control code, excluding report presentation."""
    function_names = (
        "_feature_names",
        "_racer_variant_interactions",
        "_materialize_racer_features",
        "_racer_eligible_rows",
        "_fit_weight",
        "_preprocessor",
        "_classifier",
        "_regressor",
        "_platt_fit",
        "_platt_predict",
        "_realized_rows",
        "walk_forward",
        "_portfolio_select",
        "_summary",
        "_threshold_curve",
        "_score_shuffle_control",
        "_cluster_bootstrap",
        "_matched_random_control",
        "_calibration_table",
        "_calibration_monotonicity",
        "_score_distribution",
        "_group_summaries",
        "_profit_concentration",
        "analyze_run",
    )
    source = "\n\n".join(inspect.getsource(globals()[name]) for name in function_names)
    payload = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "outer_years": OUTER_YEARS,
        "primary_years": PRIMARY_FULL_YEARS,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def _json_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _run_provenance(metadata: dict, scored: pd.DataFrame, analysis: dict) -> dict:
    """Bind one reported run to data, code, scored rows, and analysis values."""
    return {
        "db_sha256": metadata["db_fingerprint"]["sha256"],
        "candidate_strategy_sha256": metadata["strategy_fingerprint"]["sha256"],
        "candidate_events_sha256": metadata["events_sha256"],
        "analysis_fingerprint": _analysis_fingerprint(),
        "scored_sha256": _frame_fingerprint(scored),
        "scored_rows": len(scored),
        "analysis_payload_sha256": _json_fingerprint(analysis),
    }


def _read_symbol(db_path: Path, symbol: str) -> pd.DataFrame:
    """Read one symbol in chunks and retain only premarket/RTH observations."""
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    query = (
        "SELECT ts,open,high,low,close,volume FROM bars "
        "WHERE symbol=? AND timeframe='1m' AND source='alpaca_sip' AND ts>=? ORDER BY ts"
    )
    kept: list[pd.DataFrame] = []
    for chunk in pd.read_sql_query(
        query,
        con,
        params=(symbol, "2016-01-01T00:00:00+00:00"),
        chunksize=200_000,
    ):
        stamp = pd.to_datetime(chunk["ts"], utc=True).dt.tz_convert("America/New_York")
        clock = stamp.dt.hour * 60 + stamp.dt.minute
        useful = (clock > 4 * 60) & (clock <= 16 * 60)
        if useful.any():
            part = chunk.loc[useful].copy()
            part["ts"] = stamp.loc[useful]
            kept.append(part)
    con.close()
    if not kept:
        raise RuntimeError(f"no SIP bars loaded for {symbol}")
    return pd.concat(kept, ignore_index=True)


def _db_fingerprint(db_path: Path) -> dict:
    stat = db_path.stat()
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT symbol,COUNT(*),MIN(ts),MAX(ts) FROM bars "
        "WHERE timeframe='1m' AND source='alpaca_sip' GROUP BY symbol ORDER BY symbol"
    ).fetchall()
    sources = [row[0] for row in con.execute("SELECT DISTINCT source FROM bars ORDER BY source")]
    con.close()
    payload = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "rows": rows, "sources": sources}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()
    return payload


def _add_cross_symbol_features(events: pd.DataFrame, peers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = events.copy()
    timestamps = pd.DatetimeIndex(out["decision_ts"])
    for symbol, peer in peers.items():
        indexed = peer.drop_duplicates("ts").set_index("ts")
        aligned = indexed.reindex(timestamps)
        eligible = aligned["research_eligible"].fillna(False).to_numpy(dtype=bool, copy=True)
        for column in ("ema_gap_bps", "close_vwap_bps", "rsi3"):
            # Pandas may expose a read-only view here (notably with copy-on-write
            # enabled). We intentionally mask provider-gap peer observations,
            # so request a writable owner rather than mutating the frame view.
            values = aligned[column].to_numpy(dtype=float, copy=True)
            values[~eligible] = np.nan
            out[f"{symbol}_{column}"] = values
    side = np.where(out["candidate_side"].eq("long"), 1.0, -1.0)
    ema_signs = np.column_stack(
        [np.sign(pd.to_numeric(out[f"{symbol}_ema_gap_bps"], errors="coerce")) for symbol in SYMBOLS]
    )
    vwap_signs = np.column_stack(
        [np.sign(pd.to_numeric(out[f"{symbol}_close_vwap_bps"], errors="coerce")) for symbol in SYMBOLS]
    )
    rsi_signs = np.column_stack(
        [np.sign(pd.to_numeric(out[f"{symbol}_rsi3"], errors="coerce") - 50.0) for symbol in SYMBOLS]
    )
    def available_mean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = np.isfinite(values).sum(axis=1).astype(int)
        totals = np.nansum(values, axis=1)
        means = np.full(len(values), np.nan, dtype=float)
        np.divide(totals, counts, out=means, where=counts > 0)
        return means, counts

    ema_mean, ema_count = available_mean(ema_signs)
    vwap_mean, _ = available_mean(vwap_signs)
    rsi_mean, _ = available_mean(rsi_signs)
    out["cross_index_observation_count"] = ema_count
    out["cross_alignment"] = ema_mean * side
    out["cross_vwap_alignment"] = vwap_mean * side
    out["cross_rsi_alignment"] = rsi_mean * side
    return _add_racer_cross_state_and_day_weights(out)


def _add_racer_cross_state_and_day_weights(events: pd.DataFrame) -> pd.DataFrame:
    """Finish immutable RACER records once same-close peer data are aligned.

    All rows from one RTH date share one fitting mass.  This prevents a dense
    burst of overlapping intraday episodes (including correlated ETFs) from
    counting as many independent days, while retaining the raw overlap fields
    for the report and candidate-ledger audit.
    """
    out = events.copy()
    if out.empty:
        return out
    required = {"cross_alignment", "transition_geometry", "state_version"}
    if required.issubset(out.columns):
        reconciled = [
            reconcile_cross_index_state(record, alignment, observation_count)
            for record, alignment, observation_count in zip(
                out.to_dict("records"),
                out["cross_alignment"],
                out["cross_index_observation_count"],
                strict=True,
            )
        ]
        state_columns = pd.DataFrame(reconciled, index=out.index)
        for column in (
            "prior_direction",
            "cross_index_state",
            "transition_reason",
            "invalidation",
            "action",
            "provenance_cross_index_available",
        ):
            if column in state_columns:
                out[column] = state_columns[column]
    direction_sign = out.get("prior_direction", pd.Series("neutral", index=out.index)).map(
        {"long": 1.0, "short": -1.0, "neutral": 0.0, "wait": 0.0}
    )
    out["racer_prior_sign"] = direction_sign.fillna(0.0).astype(float)
    out["racer_event_reaction_available"] = out.get(
        "provenance_event_result_timestamped", pd.Series(False, index=out.index)
    ).fillna(False).astype(float)
    if "session" in out:
        day_sizes = out.groupby("session", sort=False)["session"].transform("size").astype(int)
    else:
        day_sizes = pd.Series(1, index=out.index, dtype=int)
    out["racer_day_candidate_count"] = day_sizes
    out["racer_overlap_flag"] = day_sizes.gt(1)
    out["racer_fit_weight"] = 1.0 / day_sizes.astype(float)
    return out


def _build_candidate_events(db_path: Path) -> pd.DataFrame:
    """Build the immutable point-in-time candidate rows from consolidated bars."""
    events_by_symbol: list[pd.DataFrame] = []
    peer_frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        print(f"[{symbol}] loading SIP bars", flush=True)
        raw = _read_symbol(db_path, symbol)
        print(f"[{symbol}] building chart-parity features from {len(raw):,} useful bars", flush=True)
        features = build_chart_features(raw, symbol)
        del raw
        peer_frames[symbol] = features[
            ["ts", "ema_gap_bps", "close_vwap_bps", "rsi3", "research_eligible"]
        ].copy()
        generated = generate_candidate_events(features)
        print(f"[{symbol}] {len(generated):,} setup episodes", flush=True)
        events_by_symbol.append(generated)
        del features
    events = pd.concat(events_by_symbol, ignore_index=True)
    events = collapse_simultaneous_candidates(events)
    events = _add_cross_symbol_features(events, peer_frames)
    events["year"] = pd.to_datetime(events["session"]).dt.year
    events = events.loc[events["year"].between(2016, 2026)].copy()
    events = events.sort_values(["decision_ts", "symbol"]).reset_index(drop=True)
    return events


def build_candidate_cache(
    db_path: Path, cache_path: Path, *, racer: bool = False
) -> tuple[pd.DataFrame, dict]:
    fingerprint = _db_fingerprint(db_path)
    strategy_fingerprint = _strategy_fingerprint(racer=racer)
    events = _build_candidate_events(db_path)
    fill_status = events.get("fill_status", pd.Series("filled", index=events.index))
    realized = _realized_rows(events)
    metadata = {
        "db": str(db_path),
        "db_fingerprint": fingerprint,
        "strategy_fingerprint": strategy_fingerprint,
        "events_sha256": _frame_fingerprint(events),
        "n_candidates": len(events),
        "n_realized_candidates": len(realized),
        "n_gap_invalidated_unfilled": int(fill_status.eq("gap_invalidated_unfilled").sum()),
        "by_fill_status": fill_status.value_counts(dropna=False).sort_index().to_dict(),
        "by_symbol": events["symbol"].value_counts().sort_index().to_dict(),
        "by_family": events["family"].value_counts().sort_index().to_dict(),
        "racer_overlap": {
            "flagged_candidates": int(
                events.get("racer_overlap_flag", pd.Series(False, index=events.index))
                .fillna(False)
                .sum()
            ),
            "session_days": int(events["session"].nunique()) if "session" in events else 0,
            "fit_weight_total": float(
                pd.to_numeric(
                    events.get("racer_fit_weight", pd.Series(dtype=float)), errors="coerce"
                ).sum()
            ),
            "cross_conflict_wait_candidates": int(
                events.get("invalidation", pd.Series(index=events.index, dtype=object))
                .eq("cross_index_conflict")
                .sum()
            ),
        },
        "built_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"metadata": metadata, "events": events}, cache_path, compression="gzip")
    return events, metadata


def load_or_build_candidates(
    db_path: Path, cache_path: Path, rebuild: bool, *, racer: bool = False
) -> tuple[pd.DataFrame, dict]:
    current = _db_fingerprint(db_path)
    current_strategy = _strategy_fingerprint(racer=racer)
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path, compression="gzip")
        metadata = cached.get("metadata", {})
        events = cached.get("events", pd.DataFrame())
        cached_strategy = metadata.get("strategy_fingerprint", {})
        data_matches = metadata.get("db_fingerprint", {}).get("sha256") == current["sha256"]
        strategy_matches = cached_strategy.get("sha256") == current_strategy["sha256"]
        legacy_matches = (not racer) and _legacy_candidate_fingerprint_matches(cached_strategy, current_strategy)
        actual_events_sha256 = _frame_fingerprint(events)
        content_matches = metadata.get("events_sha256") == actual_events_sha256
        if legacy_matches:
            content_matches = True
        if data_matches and content_matches and (strategy_matches or legacy_matches):
            print(f"using verified candidate cache: {cache_path}", flush=True)
            if legacy_matches:
                migrated_metadata = dict(metadata)
                migrated_metadata["legacy_strategy_fingerprint"] = cached_strategy
                migrated_metadata["strategy_fingerprint"] = current_strategy
                migrated_metadata["events_sha256"] = actual_events_sha256
                temporary = cache_path.with_name(cache_path.name + ".tmp")
                pd.to_pickle(
                    {"metadata": migrated_metadata, "events": events},
                    temporary,
                    compression="gzip",
                )
                temporary.replace(cache_path)
                print("atomically migrated exact allowlisted legacy candidate cache", flush=True)
                return events, migrated_metadata
            return events, metadata
        print("candidate cache data/code/contract fingerprint is stale; rebuilding", flush=True)
    return build_candidate_cache(db_path, cache_path, racer=racer)


def _feature_names(variant: str) -> tuple[list[str], list[str]]:
    if variant in RACER_SEQUENTIAL_VARIANTS:
        # No free symbol, family, level, or clock categorical values are
        # allowed here.  Geometry remains on the immutable candidate record for
        # audit only; all three ETFs use this one normalized numeric contract.
        numeric = list(NUMERIC_FEATURES) + (
            ["racer_candidate_side_sign", "racer_prior_sign"]
        )
        interaction = RACER_INTERACTION_BY_VARIANT[variant]
        if interaction is not None:
            numeric.append(interaction)
        return numeric, []
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    if variant == "no_rsi":
        numeric = [name for name in numeric if name not in {"rsi3", "rsi3_delta", "cross_rsi_alignment"}]
    elif variant == "no_rvol":
        numeric = [name for name in numeric if name != "rvol3"]
    elif variant == "no_time":
        categorical = [name for name in categorical if name != "time_bucket"]
    elif variant != "full":
        raise ValueError(f"unknown feature variant: {variant}")
    return numeric, categorical


def _racer_variant_interactions(variant: str) -> tuple[str, ...]:
    """Return the one predeclared interaction (if any) for a fixed challenger."""
    if variant not in RACER_INTERACTION_BY_VARIANT:
        raise ValueError(f"unknown RACER variant: {variant}")
    interaction = RACER_INTERACTION_BY_VARIANT[variant]
    return () if interaction is None else (interaction,)


def _materialize_racer_features(events: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Create only the fixed, one-at-a-time RACER interaction columns."""
    if variant not in RACER_SEQUENTIAL_VARIANTS:
        return events
    out = events.copy()
    def numeric_column(name: str) -> pd.Series:
        return pd.to_numeric(
            out.get(name, pd.Series(0.0, index=out.index)), errors="coerce"
        ).fillna(0.0)

    prior = numeric_column("racer_prior_sign")
    cross = numeric_column("cross_alignment")
    rvol = numeric_column("rvol3")
    out["racer_prior_x_cross_alignment"] = prior * cross
    # Same-slot RVOL is already point-in-time normalized by copilot_v1.  The
    # interaction is deliberately one fixed product rather than a search over
    # bins or time/symbol-specific thresholds.
    out["racer_prior_x_rvol"] = prior * np.clip(rvol - 1.0, -5.0, 5.0)
    return out


def _racer_eligible_rows(events: pd.DataFrame) -> pd.DataFrame:
    """Keep conflict/invalid states visible in reports but out of RACER fits."""
    if "action" not in events.columns:
        return events.copy()
    return events.loc[events["action"].eq("confirmable")].copy()


def _fit_weight(rows: pd.DataFrame) -> np.ndarray:
    if "session" not in rows:
        return np.ones(len(rows), dtype=float)
    # Normalize after every eligibility/year/fill filter. Each retained
    # session contributes total fitting/calibration mass exactly one.
    counts = rows.groupby("session", sort=False)["session"].transform("size")
    return (1.0 / counts.astype(float)).to_numpy(dtype=float)


def _racer_numeric_matrices(
    base: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    base_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the complete learned numeric transform with day-balanced mass."""
    # Constant imputation learns no location statistic; missingness indicators
    # stay explicit. The only learned transform is the scale, and it receives
    # the same normalized session weights as the downstream estimators.
    imputer = SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)
    base_imputed = imputer.fit_transform(base[feature_columns])
    calibration_imputed = imputer.transform(calibration[feature_columns])
    test_imputed = imputer.transform(test[feature_columns])
    scaler = StandardScaler()
    scaler.fit(base_imputed, sample_weight=base_weight)
    return (
        scaler.transform(base_imputed),
        scaler.transform(calibration_imputed),
        scaler.transform(test_imputed),
    )


def _preprocessor(numeric: list[str], categorical: list[str], *, dense: bool) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )


def _classifier(kind: ModelKind, numeric: list[str], categorical: list[str]) -> Pipeline:
    if kind == "logistic":
        estimator = LogisticRegression(C=0.25, max_iter=1_500, solver="lbfgs")
        dense = False
    else:
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=150,
            l2_regularization=10.0,
            random_state=20260715,
        )
        dense = True
    return Pipeline([("features", _preprocessor(numeric, categorical, dense=dense)), ("model", estimator)])


def _regressor(kind: ModelKind, numeric: list[str], categorical: list[str]) -> Pipeline:
    if kind == "logistic":
        estimator = Ridge(alpha=20.0)
        dense = False
    else:
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=150,
            l2_regularization=10.0,
            loss="absolute_error",
            random_state=20260715,
        )
        dense = True
    return Pipeline([("features", _preprocessor(numeric, categorical, dense=dense)), ("model", estimator)])


def _platt_fit(
    raw_probability: np.ndarray, outcome: np.ndarray, sample_weight: np.ndarray | None = None
) -> LogisticRegression | None:
    if len(raw_probability) < 100 or len(np.unique(outcome)) < 2:
        return None
    clipped = np.clip(raw_probability, 1e-5, 1 - 1e-5)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return LogisticRegression(C=1_000.0, solver="lbfgs").fit(
        logit, outcome, sample_weight=sample_weight
    )


def _platt_predict(calibrator: LogisticRegression | None, raw_probability: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.clip(raw_probability, 0.0, 1.0)
    clipped = np.clip(raw_probability, 1e-5, 1 - 1e-5)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def _realized_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return only candidates with an observed, valid next-open fill."""
    if "realized_eligible" not in rows.columns:
        return rows.copy()
    mask = rows["realized_eligible"].fillna(False).astype(bool)
    return rows.loc[mask].copy()


def walk_forward(events: pd.DataFrame, kind: ModelKind, variant: str) -> pd.DataFrame:
    """Fit through Y-2, Platt-calibrate on Y-1, and touch Y only after freezing."""
    racer = variant in RACER_SEQUENTIAL_VARIANTS
    events = _materialize_racer_features(events, variant)
    numeric, categorical = _feature_names(variant)
    feature_columns = numeric + categorical
    eligible = _racer_eligible_rows(events) if racer else events
    realized = _realized_rows(eligible)
    outputs: list[pd.DataFrame] = []
    for test_year in OUTER_YEARS:
        base = realized.loc[realized["year"] <= test_year - 2].copy()
        calibration = realized.loc[realized["year"] == test_year - 1].copy()
        # Score every point-in-time test candidate. Future gap-invalidated rows
        # stay visible to coverage accounting but never supply model labels or
        # realized performance.
        test = eligible.loc[eligible["year"] == test_year].copy()
        if len(base) < 500 or len(calibration) < 100 or test.empty:
            continue
        print(
            f"{kind}/{variant}: fit <= {test_year - 2}, calibrate {test_year - 1}, test {test_year} "
            f"({len(base):,}/{len(calibration):,}/{len(test):,})",
            flush=True,
        )

        base_weight = _fit_weight(base) if racer else None
        calibration_weight = _fit_weight(calibration) if racer else None
        if racer:
            base_x, calibration_x, test_x = _racer_numeric_matrices(
                base, calibration, test, feature_columns, base_weight
            )
            direction = LogisticRegression(C=0.25, max_iter=1_500, solver="lbfgs")
            direction.fit(base_x, base["future_up"].astype(int), sample_weight=base_weight)
            direction_cal_raw = direction.predict_proba(calibration_x)[:, 1]
            direction_test_raw = direction.predict_proba(test_x)[:, 1]
        else:
            direction = _classifier(kind, numeric, categorical)
            direction.fit(base[feature_columns], base["future_up"].astype(int))
            direction_cal_raw = direction.predict_proba(calibration[feature_columns])[:, 1]
            direction_test_raw = direction.predict_proba(test[feature_columns])[:, 1]
        direction_platt = _platt_fit(
            direction_cal_raw,
            calibration["future_up"].to_numpy(dtype=int),
            calibration_weight,
        )
        probability_up = _platt_predict(direction_platt, direction_test_raw)

        if racer:
            quality = LogisticRegression(C=0.25, max_iter=1_500, solver="lbfgs")
            quality.fit(
                base_x,
                base["candidate_net_win_1bp"].astype(int),
                sample_weight=base_weight,
            )
            quality_cal_raw = quality.predict_proba(calibration_x)[:, 1]
            quality_test_raw = quality.predict_proba(test_x)[:, 1]
        else:
            quality = _classifier(kind, numeric, categorical)
            quality.fit(base[feature_columns], base["candidate_net_win_1bp"].astype(int))
            quality_cal_raw = quality.predict_proba(calibration[feature_columns])[:, 1]
            quality_test_raw = quality.predict_proba(test[feature_columns])[:, 1]
        quality_platt = _platt_fit(
            quality_cal_raw,
            calibration["candidate_net_win_1bp"].to_numpy(dtype=int),
            calibration_weight,
        )
        probability_quality = _platt_predict(quality_platt, quality_test_raw)

        if racer:
            expectancy = Ridge(alpha=20.0)
            expectancy.fit(
                base_x,
                base["candidate_net_bps_1bp"].astype(float),
                sample_weight=base_weight,
            )
            cal_expectancy = expectancy.predict(calibration_x)
            test_expectancy = expectancy.predict(test_x)
        else:
            expectancy = _regressor(kind, numeric, categorical)
            expectancy.fit(base[feature_columns], base["candidate_net_bps_1bp"].astype(float))
            cal_expectancy = expectancy.predict(calibration[feature_columns])
            test_expectancy = expectancy.predict(test[feature_columns])
        # A single held-out location correction keeps the bps estimate honest
        # without fitting an unconstrained nonlinear mapping on one year.
        location_bias = float(
            np.median(calibration["candidate_net_bps_1bp"].to_numpy(dtype=float) - cal_expectancy)
        )
        expected_net_bps = test_expectancy + location_bias

        scored = test.copy()
        scored["outer_year"] = test_year
        scored["model_kind"] = kind
        scored["feature_variant"] = variant
        scored["probability_up"] = probability_up
        scores = np.array([complementary_scores(p) for p in probability_up])
        scored["long_score"] = scores[:, 0]
        scored["short_score"] = scores[:, 1]
        scored["quality_score"] = probability_quality * 10.0
        scored["candidate_direction_score"] = np.where(
            scored["candidate_side"].eq("long"), scored["long_score"], scored["short_score"]
        )
        scored["expected_net_bps"] = expected_net_bps
        scored["accepted_raw"] = (
            (scored["candidate_direction_score"] >= 7.0)
            & (scored["quality_score"] >= 7.0)
        )
        outputs.append(scored)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _portfolio_select(scored: pd.DataFrame) -> pd.DataFrame:
    """Pick one simultaneous ETF candidate and allow only one open plan."""
    accepted = _realized_rows(scored)
    accepted = accepted.loc[accepted["accepted_raw"]].copy()
    if accepted.empty:
        return accepted
    accepted = accepted.sort_values(
        ["decision_ts", "quality_score", "expected_net_bps", "candidate_direction_score"],
        ascending=[True, False, False, False],
    )
    accepted = accepted.groupby("decision_ts", sort=True, as_index=False).head(1)
    kept = []
    busy_until = None
    for idx, row in accepted.sort_values("entry_ts").iterrows():
        entry = pd.Timestamp(row["entry_ts"])
        if busy_until is not None and entry < busy_until:
            continue
        kept.append(idx)
        busy_until = entry + pd.Timedelta(minutes=int(row["plan_hold_minutes"]))
    return accepted.loc[kept].copy().reset_index(drop=True)


def _summary(rows: pd.DataFrame) -> dict:
    if not rows.empty and "candidate_gross_bps" in rows.columns:
        realized = pd.to_numeric(rows["candidate_gross_bps"], errors="coerce").notna()
        rows = rows.loc[realized].copy()
    if rows.empty:
        return {
            "n": 0,
            "days": 0,
            "win_rate_net_1bp": None,
            "net_bps_0_5bp": None,
            "net_bps_1bp": None,
            "delayed_net_bps_1bp": None,
            "net_bps_2bp": None,
            "gross_bps": None,
            "profit_factor_net_1bp": None,
            "median_outcome_r": None,
            "p05_net_bps": None,
            "p95_net_bps": None,
            "median_mfe_bps": None,
            "median_mae_bps": None,
        }
    gross = pd.to_numeric(rows["candidate_gross_bps"], errors="coerce")
    delayed = pd.to_numeric(rows["delayed_candidate_gross_bps"], errors="coerce")
    net = gross - 2.0
    positive = float(net.clip(lower=0.0).sum())
    negative = float(-net.clip(upper=0.0).sum())
    def numeric_or_nan(name: str) -> pd.Series:
        if name not in rows:
            return pd.Series(np.nan, index=rows.index, dtype=float)
        return pd.to_numeric(rows[name], errors="coerce")

    outcome_r = numeric_or_nan("candidate_outcome_r")
    mfe = numeric_or_nan("candidate_mfe_bps")
    mae = numeric_or_nan("candidate_mae_bps")
    return {
        "n": int(len(rows)),
        "days": int(rows["session"].nunique()),
        "win_rate_net_1bp": float((gross > 2.0).mean()),
        "net_bps_0_5bp": float((gross - 1.0).mean()),
        "net_bps_1bp": float((gross - 2.0).mean()),
        "delayed_net_bps_1bp": float((delayed - 2.0).mean()),
        "net_bps_2bp": float((gross - 4.0).mean()),
        "gross_bps": float(gross.mean()),
        "profit_factor_net_1bp": positive / negative if negative > 0.0 else None,
        "median_outcome_r": float(outcome_r.median()) if outcome_r.notna().any() else None,
        "p05_net_bps": float(net.quantile(0.05)),
        "p95_net_bps": float(net.quantile(0.95)),
        "median_mfe_bps": float(mfe.median()) if mfe.notna().any() else None,
        "median_mae_bps": float(mae.median()) if mae.notna().any() else None,
    }


def _threshold_curve(scored: pd.DataFrame) -> list[dict]:
    """Show the predeclared score neighborhood without choosing a test-year row."""
    rows = []
    for threshold in (6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0):
        candidate = scored.copy()
        candidate["accepted_raw"] = (
            (candidate["candidate_direction_score"] >= threshold)
            & (candidate["quality_score"] >= threshold)
        )
        selected = _portfolio_select(candidate)
        rows.append(
            {
                "threshold": threshold,
                "coverage": float(candidate["accepted_raw"].mean()) if len(candidate) else None,
                **_summary(selected),
            }
        )
    return rows


def _score_shuffle_control(scored: pd.DataFrame, seeds: int = 100) -> dict:
    """Destroy score/candidate alignment but preserve yearly score distributions."""
    if scored.empty:
        return {"seeds": 0, "mean_net_bps": None, "p95_net_bps": None}
    # ``analyze_run`` deliberately filters unfilled candidates before this
    # control.  Keep the shuffle indices positional after that filtering;
    # retained pandas labels can be sparse and are not valid NumPy offsets.
    scored = scored.reset_index(drop=True)
    columns = ["candidate_direction_score", "quality_score", "expected_net_bps"]
    groups = [
        group.index.to_numpy()
        for _, group in scored.groupby(["outer_year", "time_bucket"], sort=False)
    ]
    values = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        randomized = scored.copy()
        bundle = randomized[columns].to_numpy(copy=True)
        for indices in groups:
            bundle[indices] = bundle[rng.permutation(indices)]
        randomized.loc[:, columns] = bundle
        randomized["accepted_raw"] = (
            (randomized["candidate_direction_score"] >= 7.0)
            & (randomized["quality_score"] >= 7.0)
        )
        selected = _portfolio_select(randomized)
        if not selected.empty:
            values.append(float((selected["candidate_gross_bps"] - 2.0).mean()))
    values.sort()
    return {
        "seeds": len(values),
        "mean_net_bps": float(np.mean(values)) if values else None,
        "p95_net_bps": values[int(0.95 * (len(values) - 1))] if values else None,
    }


def _cluster_bootstrap(rows: pd.DataFrame, repeats: int) -> dict:
    if rows.empty or repeats <= 0:
        return {"repeats": 0, "net_bps_ci95": [None, None], "win_rate_ci95": [None, None]}
    clusters = {day: group for day, group in rows.groupby("session", sort=True)}
    days = sorted(clusters)
    rng = random.Random(20260715)
    net_values: list[float] = []
    win_values: list[float] = []
    for _ in range(repeats):
        sampled = pd.concat([clusters[rng.choice(days)] for _j in days], ignore_index=True)
        gross = sampled["candidate_gross_bps"].to_numpy(dtype=float)
        net_values.append(float(np.mean(gross - 2.0)))
        win_values.append(float(np.mean(gross > 2.0)))
    net_values.sort()
    win_values.sort()
    lo = int(0.025 * (repeats - 1))
    hi = int(0.975 * (repeats - 1))
    return {
        "repeats": repeats,
        "net_bps_ci95": [net_values[lo], net_values[hi]],
        "win_rate_ci95": [win_values[lo], win_values[hi]],
    }


def _matched_random_control(scored: pd.DataFrame, selected: pd.DataFrame, seeds: int = 200) -> dict:
    if selected.empty:
        return {"seeds": 0, "mean_net_bps": None, "p95_net_bps": None}
    scored = _realized_rows(scored)
    pools: dict[tuple, np.ndarray] = {}
    keys = ["outer_year", "symbol", "candidate_side", "time_bucket"]
    for key, group in scored.groupby(keys, sort=False):
        pools[key] = group["candidate_net_bps_1bp"].to_numpy(dtype=float)
    results = []
    for seed in range(seeds):
        rng = random.Random(seed)
        values = []
        for _, row in selected.iterrows():
            key = tuple(row[column] for column in keys)
            pool = pools.get(key)
            if pool is not None and len(pool):
                values.append(float(pool[rng.randrange(len(pool))]))
        if values:
            results.append(mean(values))
    results.sort()
    return {
        "seeds": len(results),
        "mean_net_bps": mean(results) if results else None,
        "p95_net_bps": results[int(0.95 * (len(results) - 1))] if results else None,
    }


def _calibration_table(scored: pd.DataFrame, probability_column: str, outcome_column: str) -> list[dict]:
    if scored.empty:
        return []
    work = scored[[probability_column, outcome_column]].dropna().copy()
    work["bin"] = pd.cut(
        work[probability_column],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        labels=False,
    )
    rows = []
    for bin_id, group in work.groupby("bin", observed=True):
        rows.append(
            {
                "bin": int(bin_id),
                "n": int(len(group)),
                "predicted": float(group[probability_column].mean()),
                "actual": float(group[outcome_column].mean()),
            }
        )
    return rows


def _calibration_monotonicity(rows: list[dict]) -> float | None:
    useful = [row for row in rows if row["n"] >= 50]
    if len(useful) < 3:
        return None
    predicted = np.asarray([row["predicted"] for row in useful], dtype=float)
    actual = np.asarray([row["actual"] for row in useful], dtype=float)
    if np.std(predicted) == 0 or np.std(actual) == 0:
        return 0.0
    return float(np.corrcoef(predicted, actual)[0, 1])


def _score_distribution(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {}
    return {
        column: {
            "min": float(scored[column].min()),
            "p50": float(scored[column].median()),
            "p90": float(scored[column].quantile(0.9)),
            "p99": float(scored[column].quantile(0.99)),
            "max": float(scored[column].max()),
        }
        for column in ("long_score", "short_score", "quality_score", "candidate_direction_score")
    }


def _group_summaries(rows: pd.DataFrame, column: str) -> dict:
    return {str(key): _summary(group) for key, group in rows.groupby(column, sort=True)}


def _profit_concentration(rows: pd.DataFrame, column: str) -> float | None:
    if rows.empty:
        return None
    totals = rows.assign(net=rows["candidate_gross_bps"] - 2.0).groupby(column)["net"].sum()
    positive_total = float(totals.clip(lower=0).sum())
    if positive_total <= 0:
        return None
    return float(totals.clip(lower=0).max() / positive_total)


def analyze_run(scored: pd.DataFrame, bootstrap_repeats: int) -> dict:
    full = scored.loc[scored["outer_year"].isin(PRIMARY_FULL_YEARS)].copy()
    realized_full = _realized_rows(full)
    raw = realized_full.loc[realized_full["accepted_raw"]].copy()
    portfolio = _portfolio_select(realized_full)
    vetoed = realized_full.loc[~realized_full["accepted_raw"]].copy()
    primary = _summary(portfolio)
    bootstrap = _cluster_bootstrap(portfolio, bootstrap_repeats)
    random_control = _matched_random_control(realized_full, portfolio)
    score_shuffle = _score_shuffle_control(realized_full)
    threshold_curve = _threshold_curve(full)
    by_symbol = _group_summaries(portfolio, "symbol")
    by_year = _group_summaries(portfolio, "outer_year")
    by_family = _group_summaries(portfolio, "family")
    by_auction_day_state = (
        _group_summaries(portfolio, "auction_day_state")
        if "auction_day_state" in portfolio
        else {}
    )
    profitable_symbols = sum(
        1 for value in by_symbol.values() if value["n"] >= 1 and (value["net_bps_1bp"] or -999) > 0
    )
    profitable_years = sum(
        1 for value in by_year.values() if value["n"] >= 1 and (value["net_bps_1bp"] or -999) > 0
    )
    profitable_families = sum(
        1 for value in by_family.values() if value["n"] >= 10 and (value["net_bps_1bp"] or -999) > 0
    )
    symbols_with_30 = sum(1 for value in by_symbol.values() if value["n"] >= 30)
    baseline = _summary(realized_full)
    ci_low = bootstrap["net_bps_ci95"][0]
    win_ci_low = bootstrap["win_rate_ci95"][0]
    direction_calibration = _calibration_table(realized_full, "probability_up", "future_up")
    quality_calibration = _calibration_table(
        realized_full.assign(probability_quality=realized_full["quality_score"] / 10.0),
        "probability_quality",
        "candidate_net_win_1bp",
    )
    direction_monotonicity = _calibration_monotonicity(direction_calibration)
    quality_monotonicity = _calibration_monotonicity(quality_calibration)
    neighbor_6_5 = next(row for row in threshold_curve if row["threshold"] == 6.5)
    neighbor_7_5 = next(row for row in threshold_curve if row["threshold"] == 7.5)
    promotion_checks = {
        "at_least_100_trades": primary["n"] >= 100,
        "at_least_75_days": primary["days"] >= 75,
        "two_symbols_with_30": symbols_with_30 >= 2,
        "win_rate_above_60pct": (primary["win_rate_net_1bp"] or 0) > 0.60,
        "win_rate_ci_lower_above_60pct": win_ci_low is not None and win_ci_low > 0.60,
        "bootstrap_expectancy_lower_bound_positive": ci_low is not None and ci_low > 0,
        "delayed_expectancy_positive": (primary["delayed_net_bps_1bp"] or -999) > 0,
        "two_bp_per_side_nonnegative": (primary["net_bps_2bp"] or -999) >= 0,
        "two_symbols_positive": profitable_symbols >= 2,
        "two_setup_families_positive": profitable_families >= 2,
        "majority_full_years_positive": profitable_years >= 4,
        "no_symbol_over_half_positive_profit": (_profit_concentration(portfolio, "symbol") or 999) <= 0.50,
        "no_year_over_half_positive_profit": (_profit_concentration(portfolio, "outer_year") or 999) <= 0.50,
        "no_setup_over_half_positive_profit": (_profit_concentration(portfolio, "family") or 999) <= 0.50,
        "beats_all_candidates": (primary["net_bps_1bp"] or -999) > (baseline["net_bps_1bp"] or 999),
        "beats_matched_random_p95": random_control["p95_net_bps"] is not None
        and (primary["net_bps_1bp"] or -999) > random_control["p95_net_bps"],
        "beats_score_shuffle_p95": score_shuffle["p95_net_bps"] is not None
        and (primary["net_bps_1bp"] or -999) > score_shuffle["p95_net_bps"],
        "neighbor_thresholds_positive": (neighbor_6_5["net_bps_1bp"] or -999) > 0
        and (neighbor_7_5["net_bps_1bp"] or -999) > 0,
        "direction_bins_monotonic": direction_monotonicity is not None and direction_monotonicity > 0,
        "quality_bins_monotonic": quality_monotonicity is not None and quality_monotonicity > 0,
    }
    return {
        "all_candidates": baseline,
        "candidate_coverage_audit": {
            "point_in_time_candidates": int(len(full)),
            "realized_candidates": int(len(realized_full)),
            "gap_invalidated_unfilled": int(
                full.get("fill_status", pd.Series(index=full.index, dtype=object))
                .eq("gap_invalidated_unfilled")
                .sum()
            ),
            "realized_fill_rate": float(len(realized_full) / len(full)) if len(full) else None,
            "accepted_point_in_time": int(full["accepted_raw"].sum()) if len(full) else 0,
            "accepted_gap_invalidated_unfilled": int(
                (
                    full["accepted_raw"]
                    & full.get("fill_status", pd.Series(index=full.index, dtype=object)).eq(
                        "gap_invalidated_unfilled"
                    )
                ).sum()
            ) if len(full) else 0,
            "overlap_flagged_candidates": int(
                full.get("racer_overlap_flag", pd.Series(False, index=full.index))
                .fillna(False)
                .sum()
            ),
            "cross_conflict_wait_candidates": int(
                full.get("invalidation", pd.Series(index=full.index, dtype=object))
                .eq("cross_index_conflict")
                .sum()
            ),
        },
        "accepted_before_portfolio_constraint": _summary(raw),
        "accepted_portfolio": primary,
        "vetoed_candidates": _summary(vetoed),
        "acceptance_rate": float(len(raw) / len(full)) if len(full) else None,
        "accepted_trades_per_month": float(primary["n"] / (len(PRIMARY_FULL_YEARS) * 12)),
        "by_symbol": by_symbol,
        "by_year": by_year,
        "by_family": by_family,
        "by_auction_day_state": by_auction_day_state,
        "bootstrap": bootstrap,
        "matched_random": random_control,
        "score_shuffle": score_shuffle,
        "threshold_curve": threshold_curve,
        "score_distribution": _score_distribution(full),
        "direction_calibration": direction_calibration,
        "quality_calibration": quality_calibration,
        "calibration_monotonicity": {
            "direction": direction_monotonicity,
            "quality": quality_monotonicity,
        },
        "profit_concentration": {
            "symbol": _profit_concentration(portfolio, "symbol"),
            "year": _profit_concentration(portfolio, "outer_year"),
            "family": _profit_concentration(portfolio, "family"),
        },
        "promotion_checks": promotion_checks,
        "promoted": all(promotion_checks.values()),
        "partial_2026": _summary(_portfolio_select(scored.loc[scored["outer_year"] == 2026])),
    }


RACER_DELTA_METRICS = (
    "win_rate_net_1bp",
    "net_bps_0_5bp",
    "net_bps_1bp",
    "delayed_net_bps_1bp",
    "net_bps_2bp",
)


def _summary_delta(challenger: dict, core: dict) -> dict:
    delta: dict[str, float | int | None] = {
        "challenger_n": int(challenger.get("n", 0)),
        "core_n": int(core.get("n", 0)),
    }
    for metric in RACER_DELTA_METRICS:
        challenger_value = challenger.get(metric)
        core_value = core.get(metric)
        delta[metric] = (
            None
            if challenger_value is None or core_value is None
            else float(challenger_value - core_value)
        )
    return delta


def _group_deltas(challenger: dict, core: dict) -> dict:
    return {
        str(key): _summary_delta(challenger.get(key, {}), core.get(key, {}))
        for key in sorted(set(challenger) | set(core), key=str)
    }


def _racer_incremental_comparisons(analyses: dict) -> dict:
    """Rank fixed challengers against the immutable core without promotion."""
    core_id = f"logistic__{RACER_EVENT_NEUTRAL_CORE}"
    core = analyses.get(core_id)
    if core is None:
        return {}
    comparisons = {}
    for variant in RACER_SEQUENTIAL_VARIANTS[1:]:
        run_id = f"logistic__{variant}"
        challenger = analyses.get(run_id)
        if challenger is None:
            continue
        overall = _summary_delta(
            challenger["accepted_portfolio"], core["accepted_portfolio"]
        )
        by_year_delta = _group_deltas(challenger["by_year"], core["by_year"])
        by_symbol_delta = _group_deltas(challenger["by_symbol"], core["by_symbol"])
        supported_years = sum(
            1
            for value in by_year_delta.values()
            if value.get("net_bps_1bp") is not None and value["net_bps_1bp"] > 0.0
        )
        supported_symbols = sum(
            1
            for value in by_symbol_delta.values()
            if value.get("net_bps_1bp") is not None and value["net_bps_1bp"] > 0.0
        )
        required = (
            overall.get("net_bps_1bp"),
            overall.get("delayed_net_bps_1bp"),
            overall.get("net_bps_2bp"),
        )
        historically_stable = (
            required[0] is not None
            and required[0] > 0.0
            and required[1] is not None
            and required[1] >= 0.0
            and required[2] is not None
            and required[2] >= 0.0
            and supported_years >= 4
            and supported_symbols >= 2
        )
        comparisons[run_id] = {
            "baseline": core_id,
            "overall_delta": overall,
            "by_year_delta": by_year_delta,
            "by_symbol_delta": by_symbol_delta,
            "positive_lift_years": supported_years,
            "positive_lift_symbols": supported_symbols,
            "by_transition_state_delta": _group_deltas(
                challenger["by_family"], core["by_family"]
            ),
            "by_auction_day_state_delta": _group_deltas(
                challenger.get("by_auction_day_state", {}),
                core.get("by_auction_day_state", {}),
            ),
            "historical_rank_only_verdict": (
                "retain_for_unspent_forward_falsification"
                if historically_stable
                else "reject_interaction_from_forward_queue"
            ),
            "live_promotion_allowed": False,
        }
    return comparisons


def _fmt(value, *, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:+.2f}"


def write_reports(
    metadata: dict,
    events: pd.DataFrame,
    analyses: dict,
    scored_runs: dict,
    *,
    report_json: Path = REPORT_JSON,
    report_md: Path = REPORT_MD,
    primary_id: str = "logistic__full",
    racer: bool = False,
) -> None:
    realized_events = _realized_rows(events)
    gap_invalidated = int(
        events.get("fill_status", pd.Series(index=events.index, dtype=object))
        .eq("gap_invalidated_unfilled")
        .sum()
    )
    deterministic_by_family = {
        str(key): _summary(group.loc[group["year"].isin(PRIMARY_FULL_YEARS)])
        for key, group in realized_events.groupby("family", sort=True)
    }
    run_provenance = {
        run_id: _run_provenance(metadata, scored_runs[run_id], analysis)
        for run_id, analysis in analyses.items()
    }
    racer_comparisons = _racer_incremental_comparisons(analyses) if racer else {}
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "contract": {
            "data": "Alpaca SIP close-stamped 1m bars; provider-gap sessions excluded with 20-session warmout",
            "layout": "5m regime / 3m setup / 1m trigger / 30s prospective execution only",
            "fill": "next real 1m open; an open through the frozen stop is retained as gap-invalidated/unfilled and has no model or realized label",
            "exit": "structural stop; 50% at [n]R; 50% at 2R; 15-minute time exit; same-bar stop-first",
            "primary_cost": "1 bp per side",
            "scores": "Long + Short = 10; candidate direction >= 7; quality >= 7",
            "outer_validation": "fit through Y-2, Platt calibrate Y-1, test Y; full promotion years 2020-2025",
            "racer": (
                "event-neutral core is the canonical primary; each listed challenger adds exactly one predeclared interaction; no best-row selection"
                if racer
                else None
            ),
        },
        "candidate_metadata": metadata,
        "deterministic_by_family_2020_2025": deterministic_by_family,
        "runs": analyses,
        "run_provenance": run_provenance,
        "racer_incremental_comparisons": racer_comparisons,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# RACER v0 Sequential Ablation" if racer else "# Copilot v1 Candidate-Confirmation Lab",
        "",
        "This is a nested walk-forward **historical mechanical proxy** for the owner's chart-first confirmation workflow. It is not evidence of owner-plus-AI uplift; the blind candidate ledger is required for that claim.",
        "",
        "## Locked contract",
        "",
        "- Consolidated Alpaca SIP bars; close-stamped and completed-bar only.",
        "- Exact RTH feature stack: continuous 5m EMA 9/20 and ATR(14), continuous 3m Wilder RSI(14), same-slot prior-20-session median RVOL, session-reset VWAP/sigma.",
        "- Deterministic setup episodes are emitted at the decision close. A next-open gap through the frozen stop remains a point-in-time candidate but is classified gap-invalidated/unfilled and excluded from model labels and realized win/loss. Filled rows use the frozen structural stop, half at [n]R, half at 2R, 15-minute time exit (capped at the close), and conservative same-bar stop-first ordering; 1 bp/side primary cost.",
        "- Long + Short = 10. Candidate-side direction and quality must both be at least 7.0. Expected bps is reported and used only as a tie-break, not a hidden entry gate.",
        "- Fit through Y-2, calibrate on Y-1, test Y. The full promotion window is 2020-2025; 2026 is partial and separate.",
        "",
        f"Point-in-time candidates after simultaneous-label collapse: **{len(events):,}**; realized valid fills: **{len(realized_events):,}**; gap-invalidated/unfilled: **{gap_invalidated:,}**.",
        "",
        "## Model and indicator ablations",
        "",
        "| Model | Inputs | Portfolio N | Win after cost | Net bps | +1m net | 2bp/side net | Coverage | Promoted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if racer:
        overlap = metadata.get("racer_overlap", {})
        lines.extend(
            [
                "- The event-neutral logistic core is canonical. Each following row is a fixed one-interaction challenger, not a best-of-many selection; family, symbol, level, and clock labels are audit-only.",
                "- Same-day overlapping candidates receive total fitting weight of one. Raw counts and overlap flags remain in the candidate cache/report provenance.",
                f"- Candidate overlap audit: {overlap.get('flagged_candidates', 0):,} flagged rows across {overlap.get('session_days', 0):,} session days; aggregate fit weight {overlap.get('fit_weight_total', 0.0):.1f}; cross-conflict WAIT rows {overlap.get('cross_conflict_wait_candidates', 0):,}.",
                "",
            ]
        )
    for run_id, analysis in analyses.items():
        model, variant = run_id.split("__", 1)
        summary = analysis["accepted_portfolio"]
        lines.append(
            f"| {model} | {variant} | {summary['n']} | {_fmt(summary['win_rate_net_1bp'], pct=True)} | "
            f"{_fmt(summary['net_bps_1bp'])} | {_fmt(summary['delayed_net_bps_1bp'])} | "
            f"{_fmt(summary['net_bps_2bp'])} | {analysis['acceptance_rate']:.3%} | "
            f"{'YES' if analysis['promoted'] else 'NO'} |"
        )

    if racer and racer_comparisons:
        lines.extend(
            [
                "",
                "## Fixed challenger lift versus the event-neutral core",
                "",
                "These are spent-history ranking results only. They cannot promote a live overlay.",
                "",
                "| Challenger | Win-rate delta | Net-bps delta | +1m delta | 2bp/side delta | Verdict |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for run_id, comparison in racer_comparisons.items():
            delta = comparison["overall_delta"]
            lines.append(
                f"| {run_id.split('__', 1)[1]} | "
                f"{_fmt(delta['win_rate_net_1bp'], pct=True)} | "
                f"{_fmt(delta['net_bps_1bp'])} | "
                f"{_fmt(delta['delayed_net_bps_1bp'])} | "
                f"{_fmt(delta['net_bps_2bp'])} | "
                f"{comparison['historical_rank_only_verdict']} |"
            )

    primary = analyses.get(primary_id)
    if primary:
        lines.extend(
            [
                "",
                f"## Canonical primary: {primary_id.replace('__', ' / ')}",
                "",
                f"- Accepted before the one-position rule: {primary['accepted_before_portfolio_constraint']['n']}.",
                f"- Accepted after the one-position rule: {primary['accepted_portfolio']['n']} across {primary['accepted_portfolio']['days']} days.",
                f"- Net-win rate: {_fmt(primary['accepted_portfolio']['win_rate_net_1bp'], pct=True)}; net expectancy: {_fmt(primary['accepted_portfolio']['net_bps_1bp'])} bps.",
                f"- Day-cluster [n]% win-rate interval: {_fmt(primary['bootstrap']['win_rate_ci95'][0], pct=True)} to {_fmt(primary['bootstrap']['win_rate_ci95'][1], pct=True)}.",
                f"- Day-cluster 95% expectancy interval: {_fmt(primary['bootstrap']['net_bps_ci95'][0])} to {_fmt(primary['bootstrap']['net_bps_ci95'][1])} bps.",
                f"- Net profit factor: {_fmt(primary['accepted_portfolio']['profit_factor_net_1bp'])}; median outcome: {_fmt(primary['accepted_portfolio']['median_outcome_r'])}R; 5th/95th net tails: {_fmt(primary['accepted_portfolio']['p05_net_bps'])} / {_fmt(primary['accepted_portfolio']['p95_net_bps'])} bps.",
                f"- Median MFE / MAE: {_fmt(primary['accepted_portfolio']['median_mfe_bps'])} / {_fmt(primary['accepted_portfolio']['median_mae_bps'])} bps.",
                f"- Matched-random 95th percentile: {_fmt(primary['matched_random']['p95_net_bps'])} bps.",
                f"- Score-shuffle 95th percentile: {_fmt(primary['score_shuffle']['p95_net_bps'])} bps.",
                f"- Vetoed-candidate result: {_fmt(primary['vetoed_candidates']['win_rate_net_1bp'], pct=True)} wins, {_fmt(primary['vetoed_candidates']['net_bps_1bp'])} net bps.",
                f"- Usable frequency: {primary['accepted_trades_per_month']:.2f} accepted trades/month.",
                f"- Score-spread verdict: **FAIL**. Candidate-direction p99 was {primary['score_distribution']['candidate_direction_score']['p99']:.2f}/10 and the maximum was {primary['score_distribution']['candidate_direction_score']['max']:.2f}/10; only {primary['accepted_before_portfolio_constraint']['n']} of {primary['candidate_coverage_audit']['point_in_time_candidates']:,} point-in-time candidates reached both 7.0 gates.",
                "- The proxy score is a calibrated probability, while the live Long/Short score is explicitly a relative-evidence scale. Because this proxy had unusable spread and non-monotonic direction calibration, it is **not** a validated live scoring overlay.",
                "",
                "### By year",
                "",
                "| Year | N | Win | Net bps | +1m net |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for year, summary in primary["by_year"].items():
            lines.append(
                f"| {year} | {summary['n']} | {_fmt(summary['win_rate_net_1bp'], pct=True)} | "
                f"{_fmt(summary['net_bps_1bp'])} | {_fmt(summary['delayed_net_bps_1bp'])} |"
            )
        lines.extend(
            [
                "",
                "### By symbol",
                "",
                "| Symbol | N | Win | Net bps |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for symbol, summary in primary["by_symbol"].items():
            lines.append(
                f"| {symbol} | {summary['n']} | {_fmt(summary['win_rate_net_1bp'], pct=True)} | {_fmt(summary['net_bps_1bp'])} |"
            )
        lines.extend(
            [
                "",
                "### Score-threshold sensitivity",
                "",
                "| Threshold | N | Coverage | Win | Net bps | +1m net |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in primary["threshold_curve"]:
            lines.append(
                f"| {row['threshold']:.1f} | {row['n']} | {row['coverage']:.3%} | "
                f"{_fmt(row['win_rate_net_1bp'], pct=True)} | {_fmt(row['net_bps_1bp'])} | "
                f"{_fmt(row['delayed_net_bps_1bp'])} |"
            )
        lines.extend(["", "### Promotion checks", ""])
        for check, passed in primary["promotion_checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {check.replace('_', ' ')}")

    lines.extend(
        [
            "",
            "## End-of-day hypotheses (deterministic candidate population)",
            "",
            "| Family | N | Win after cost | Net bps | +1m net |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family in ("eod_continuation", "eod_profit_taking_reversal"):
        summary = deterministic_by_family.get(family, _summary(pd.DataFrame()))
        lines.append(
            f"| {family} | {summary['n']} | {_fmt(summary['win_rate_net_1bp'], pct=True)} | "
            f"{_fmt(summary['net_bps_1bp'])} | {_fmt(summary['delayed_net_bps_1bp'])} |"
        )

    lines.extend(
        [
            "",
            "These results reject automatic close-direction and automatic profit-taking-fade rules. They do **not** justify avoiding the close: the close remains an opportunity regime where continuation, pullback, and failed-extreme reversal must be distinguished from current structure and participation.",
        ]
    )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A run that does not clear every promotion check remains research-only, regardless of its headline win rate. This run failed promotion and must not alter live calls. Historical bars cannot establish whether the copilot improves the owner's own nominations; that requires at least 100 blind, resolved candidate rows with the owner's call timestamped before the AI response.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--models", choices=("logistic", "shallow_gbt", "all"), default="all")
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument(
        "--racer",
        action="store_true",
        help="run the explicit RACER v0 event-neutral sequential ablation and write only racer_v0 reports",
    )
    args = parser.parse_args()

    cache_path = args.cache or (RACER_CACHE if args.racer else CACHE)
    events, metadata = load_or_build_candidates(
        args.db, cache_path, args.rebuild_cache, racer=args.racer
    )
    if events.empty:
        raise RuntimeError("candidate generator produced no events")
    if args.racer:
        if args.models == "shallow_gbt":
            parser.error("RACER v0 keeps logistic/event-neutral core canonical; shallow GBT is not a primary path")
        # Deliberately fixed: the core remains primary even if a later
        # challenger happens to have a more attractive historical row.
        kinds: Iterable[ModelKind] = ("logistic",)
        variants = RACER_SEQUENTIAL_VARIANTS
    else:
        kinds = ("logistic", "shallow_gbt") if args.models == "all" else (args.models,)
        variants = ("full", "no_rsi", "no_rvol", "no_time")
    analyses = {}
    scored_runs = {}
    for kind in kinds:
        for variant in variants:
            run_id = f"{kind}__{variant}"
            scored = walk_forward(events, kind, variant)
            scored_runs[run_id] = scored
            analyses[run_id] = analyze_run(scored, args.bootstrap_repeats)
    report_json = RACER_REPORT_JSON if args.racer else REPORT_JSON
    report_md = RACER_REPORT_MD if args.racer else REPORT_MD
    primary_id = (
        f"logistic__{RACER_EVENT_NEUTRAL_CORE}" if args.racer else "logistic__full"
    )
    write_reports(
        metadata,
        events,
        analyses,
        scored_runs,
        report_json=report_json,
        report_md=report_md,
        primary_id=primary_id,
        racer=args.racer,
    )
    print(report_md.read_text(encoding="utf-8"))
    print(f"JSON: {report_json}")


if __name__ == "__main__":
    main()
