"""Parallel worker/merge helper for frozen Copilot v1 confirmation variants.

Each variant is statistically independent after the candidate cache is frozen.
This helper only parallelizes those runs; it delegates all modeling, analysis,
and report generation to ``run_copilot_v1_confirmation_lab.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB_PATH = ROOT / "scripts" / "run_copilot_v1_confirmation_lab.py"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "cache" / "copilot_v1_parallel"
VARIANTS = ("full", "no_rsi", "no_rvol", "no_time")


def _load_lab():
    spec = importlib.util.spec_from_file_location("copilot_v1_confirmation_lab_worker", LAB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load confirmation lab: {LAB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant_path(output_dir: Path, variant: str) -> Path:
    return output_dir / f"logistic__{variant}.pkl"


def run_variant(variant: str, cache: Path, output_dir: Path, bootstrap_repeats: int) -> None:
    lab = _load_lab()
    events, metadata = lab.load_or_build_candidates(lab.DEFAULT_DB, cache, False)
    scored = lab.walk_forward(events, "logistic", variant)
    analysis = lab.analyze_run(scored, bootstrap_repeats)
    provenance = lab._run_provenance(metadata, scored, analysis)
    payload = {
        "run_id": f"logistic__{variant}",
        "variant": variant,
        "bootstrap_repeats": bootstrap_repeats,
        "provenance": provenance,
        "analysis": analysis,
        "scored": scored,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = _variant_path(output_dir, variant)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    print(output)


def merge_variants(cache: Path, output_dir: Path, bootstrap_repeats: int) -> None:
    lab = _load_lab()
    events, metadata = lab.load_or_build_candidates(lab.DEFAULT_DB, cache, False)
    analyses = {}
    scored_runs = {}
    for variant in VARIANTS:
        path = _variant_path(output_dir, variant)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        run_id = f"logistic__{variant}"
        if payload.get("run_id") != run_id or payload.get("variant") != variant:
            raise RuntimeError(f"variant payload mismatch: {path}")
        if payload.get("bootstrap_repeats") != bootstrap_repeats:
            raise RuntimeError(f"bootstrap repeat mismatch: {path}")
        scored = payload["scored"]
        provenance = payload.get("provenance", {})
        expected_provenance = lab._run_provenance(metadata, scored, payload["analysis"])
        for key, expected in expected_provenance.items():
            if provenance.get(key) != expected:
                raise RuntimeError(f"{key} provenance mismatch: {path}")
        if set(scored["model_kind"].dropna().unique()) != {"logistic"}:
            raise RuntimeError(f"model-kind mismatch in scored rows: {path}")
        if set(scored["feature_variant"].dropna().unique()) != {variant}:
            raise RuntimeError(f"variant mismatch in scored rows: {path}")
        analyses[run_id] = payload["analysis"]
        scored_runs[run_id] = scored
    lab.write_reports(metadata, events, analyses, scored_runs)
    print(lab.REPORT_MD.read_text(encoding="utf-8"))
    print(f"JSON: {lab.REPORT_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--variant", choices=VARIANTS, required=True)
    run_parser.add_argument("--cache", type=Path, default=None)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--bootstrap-repeats", type=int, default=500)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--cache", type=Path, default=None)
    merge_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    merge_parser.add_argument("--bootstrap-repeats", type=int, default=500)

    args = parser.parse_args()
    lab = _load_lab()
    cache = args.cache or lab.CACHE
    if args.command == "run":
        run_variant(args.variant, cache, args.output_dir, args.bootstrap_repeats)
    else:
        merge_variants(cache, args.output_dir, args.bootstrap_repeats)


if __name__ == "__main__":
    main()
