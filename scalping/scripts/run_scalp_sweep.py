"""Stage A v0.1 CORE GATE parameter sweep - DEV WINDOW ONLY, no holdout peeking.

Sweeps the real ``bot.scalp.rules_v01`` CORE GATE decision model (see that
module's docstring for the full spec) over a grid of booster/filter/trigger-
family toggles, on real SPY 1m bars from ``data/sqlite/scalp.db``, restricted
to the DEV WINDOW **2023-07-01 through 2025-06-30 inclusive**.

ABSOLUTE RULE - LOCKED HOLDOUT: 2025-07-01 onward must never be loaded,
never be scored, and never be reported by this script. ``HOLDOUT_START``
below is enforced with a hard assertion right after loading bars; there is
no code path in this script that can request data past ``DEV_END``.

Grid (see module docstring for the exact combinatorics - the literal spec
in the handoff prompt says "~128 configs"; the actual product of the four
described axes is 4 x 2 x 2 x 6 = 96; that is reported honestly rather than
padded to 128, since the trigger-family axis has exactly 6 natural levels
(5 singles + all-combined), not 8):
  - RVOL booster threshold:      off, 1.25, 1.5, 2.0                (4)
  - Confirming-candle requirement: off, on                          (2)
  - Trade-window filter:         off, on                            (2)
  - Trigger families:            each of the 5 alone, all combined  (6)
Horizons 5/15/30/60m are NOT a grid axis - ``evaluate_decisions`` computes
all four from one decision run per config, so they come for free.

Every distinct config is appended to the trial ledger
(``data/scalp_sweep_trial_ledger.jsonl`` by default) via
``bot.backtest.trial_ledger.record_trial`` BEFORE ranking, so the trial
count used for any later multiple-testing correction reflects every config
actually run - no cherry-picking after the fact.

Negative controls (shuffled bars, delayed signal, random-matched-frequency)
are run for the top-5 configs by 15m expectancy_bps, via
``bot.scalp.controls.run_negative_controls(..., decide_fn=decide_core_gate_all)``.

Usage:
    python scripts/run_scalp_sweep.py
    python scripts/run_scalp_sweep.py --symbol SPY --top-n 10
    python scripts/run_scalp_sweep.py --dev-start 2024-01-01 --dev-end 2024-01-31  # smoke test
"""
from __future__ import annotations

import argparse
import itertools
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bot.backtest.trial_ledger import count_trials, record_trial  # noqa: E402
from bot.scalp.controls import run_negative_controls  # noqa: E402
from bot.scalp.evaluate import EvalReport, evaluate_decisions  # noqa: E402
from bot.scalp.features import FeatureSnapshot, build_feature_snapshots  # noqa: E402
from bot.scalp.rules_v01 import CoreGateConfig, TriggerFamily, decide_core_gate_all  # noqa: E402
from run_scalp_stage_a import (  # noqa: E402
    DEFAULT_DB,
    DEFAULT_PRIOR_BUFFER_DAYS,
    MAX_PRIOR_SESSIONS,
    _load_real_bars,
)

HORIZONS: tuple[int, ...] = (5, 15, 30, 60)

# ── locked dev/holdout boundary ──────────────────────────────────────────────
DEV_START = date(2023, 7, 1)
DEV_END = date(2025, 6, 30)
HOLDOUT_START = date(2025, 7, 1)  # never loaded, never scored, never reported here

DEFAULT_LEDGER_PATH = Path("data/scalp_sweep_trial_ledger.jsonl")

RVOL_BOOSTER_GRID: list[float | None] = [None, 1.25, 1.5, 2.0]
CONFIRM_CANDLE_GRID: list[bool] = [False, True]
TRADE_WINDOW_GRID: list[bool] = [False, True]


def _trigger_family_grid() -> list[tuple[str, frozenset[TriggerFamily]]]:
    singles = [(fam.value, frozenset({fam})) for fam in TriggerFamily]
    combined = [("all_combined", frozenset(TriggerFamily))]
    return singles + combined


def build_grid() -> list[tuple[str, CoreGateConfig]]:
    """Every (config_id, CoreGateConfig) combination in the sweep grid."""
    out: list[tuple[str, CoreGateConfig]] = []
    for rvol, candle, win_filter, (fam_label, fams) in itertools.product(
        RVOL_BOOSTER_GRID, CONFIRM_CANDLE_GRID, TRADE_WINDOW_GRID, _trigger_family_grid()
    ):
        rvol_s = "off" if rvol is None else str(rvol)
        config_id = f"rvol{rvol_s}_candle{'on' if candle else 'off'}_winfilt{'on' if win_filter else 'off'}_fam{fam_label}"
        cfg = CoreGateConfig(
            trigger_families=fams,
            rvol_booster_threshold=rvol,
            confirming_candle_required=candle,
            trade_window_filter_enabled=win_filter,
        )
        out.append((config_id, cfg))
    return out


def build_all_snapshots(
    bars_by_day: dict[date, list],
    sorted_days: list[date],
    eval_days: list[date],
) -> tuple[dict[date, list[FeatureSnapshot]], dict[date, list[list]]]:
    """Feature snapshots are decision-model-agnostic - build them ONCE for the
    whole sweep instead of once per config (the expensive part of Stage A)."""
    day_to_idx = {d: i for i, d in enumerate(sorted_days)}
    snaps_by_day: dict[date, list[FeatureSnapshot]] = {}
    prior_sessions_by_day: dict[date, list[list]] = {}
    for d in eval_days:
        idx = day_to_idx[d]
        prior_days = [bars_by_day[pd_] for pd_ in sorted_days[max(0, idx - MAX_PRIOR_SESSIONS):idx]]
        prior_sessions_by_day[d] = prior_days
        snaps_by_day[d] = build_feature_snapshots(bars_by_day[d], prior_days)
    return snaps_by_day, prior_sessions_by_day


def _summarize(report: EvalReport, cfg: CoreGateConfig) -> dict:
    per_horizon = {
        h: {"n": s.n, "hit_rate": s.hit_rate, "expectancy_bps": s.expectancy_bps}
        for h, s in report.per_horizon.items()
    }
    fold_15m = [
        report.by_fold[fid][15].expectancy_bps
        for fid in sorted(report.by_fold)
        if 15 in report.by_fold[fid] and report.by_fold[fid][15].expectancy_bps is not None
    ]
    return {
        "trade_count": report.trade_count,
        "wait_count": report.wait_count,
        "max_drawdown_pct": report.max_drawdown_pct,
        "per_horizon": per_horizon,
        "n_folds": len(report.by_fold),
        "fold_15m_expectancy_bps": fold_15m,
        "rvol_booster_threshold": cfg.rvol_booster_threshold,
        "confirming_candle_required": cfg.confirming_candle_required,
        "trade_window_filter_enabled": cfg.trade_window_filter_enabled,
        "trigger_families": sorted(f.value for f in cfg.trigger_families),
    }


def _rank_key(metrics: dict) -> tuple[bool, float]:
    exp = metrics["per_horizon"].get(15, {}).get("expectancy_bps")
    return (exp is None, -(exp if exp is not None else 0.0))


def _fmt(v, spec: str = ".1f") -> str:
    return format(v, spec) if v is not None else "n/a"


def _print_row(config_id: str, metrics: dict) -> None:
    h15 = metrics["per_horizon"].get(15, {})
    n = h15.get("n", 0)
    hr = _fmt(h15.get("hit_rate") * 100 if h15.get("hit_rate") is not None else None)
    exp = _fmt(h15.get("expectancy_bps"))
    folds = metrics["fold_15m_expectancy_bps"]
    n_pos_folds = sum(1 for f in folds if f > 0)
    fold_str = f"{n_pos_folds}/{len(folds)}" if folds else "n/a"
    print(f"{config_id:<55} n={n:>5} hit={hr:>6}% exp15={exp:>7}bps pos_folds={fold_str:>7}")


def run_sweep(
    symbol: str = "SPY",
    db_path: Path = DEFAULT_DB,
    dev_start: date = DEV_START,
    dev_end: date = DEV_END,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    fold_size_days: int = 5,
    prior_buffer_days: int = DEFAULT_PRIOR_BUFFER_DAYS,
    top_n: int = 10,
    controls_top_k: int = 5,
) -> None:
    assert dev_end <= DEV_END, f"refusing to sweep past the locked dev-window end {DEV_END}"
    assert dev_end < HOLDOUT_START, "refusing to sweep into the locked holdout window (2025-07-01+)"

    bars_by_day = _load_real_bars(db_path, symbol, dev_start, dev_end, prior_buffer_days)
    if bars_by_day is None:
        raise SystemExit(
            f"[sweep] No real {symbol} 1m data found at {db_path} for [{dev_start}, {dev_end}]. "
            "Refusing to run the sweep on synthetic data — the report must be evidence, not a smoke test."
        )
    # Holdout safety net: even though _load_real_bars is bounded by `end`,
    # assert nothing in the loaded set crosses the locked boundary.
    assert all(d < HOLDOUT_START for d in bars_by_day), "loader returned bars past the locked holdout boundary"

    sorted_days = sorted(bars_by_day)
    eval_days = [d for d in sorted_days if dev_start <= d <= dev_end]
    print(
        f"[sweep] symbol={symbol} db={db_path} dev_window=[{dev_start},{dev_end}] "
        f"eval_days={len(eval_days)} (+{len(bars_by_day) - len(eval_days)} buffer-only days for ATR/RVOL context)"
    )

    snaps_by_day, prior_sessions_by_day = build_all_snapshots(bars_by_day, sorted_days, eval_days)
    eval_bars_by_day = {d: bars_by_day[d] for d in eval_days}

    grid = build_grid()
    print(f"[sweep] {len(grid)} distinct configs in the grid "
          f"({len(RVOL_BOOSTER_GRID)} rvol x {len(CONFIRM_CANDLE_GRID)} candle x "
          f"{len(TRADE_WINDOW_GRID)} window-filter x {len(_trigger_family_grid())} trigger-family-sets)")

    window_str = f"{dev_start.isoformat()}/{dev_end.isoformat()}"
    results: list[tuple[str, CoreGateConfig, EvalReport, dict]] = []

    for i, (config_id, cfg) in enumerate(grid, 1):
        decisions_by_day = {d: decide_core_gate_all(snaps_by_day[d], cfg) for d in eval_days}
        report = evaluate_decisions(
            eval_bars_by_day, decisions_by_day, horizons=HORIZONS, fold_size_days=fold_size_days
        )
        metrics = _summarize(report, cfg)
        record_trial(config_id, window=window_str, metrics=metrics, ledger_path=ledger_path)
        results.append((config_id, cfg, report, metrics))
        if i % 10 == 0 or i == len(grid):
            print(f"[sweep] ran {i}/{len(grid)} configs...")

    ranked = sorted(results, key=lambda r: _rank_key(r[3]))

    print(f"\n=== Trial ledger: {count_trials(ledger_path)} total records at {ledger_path} ===")

    print(f"\n=== TOP {top_n} by 15m expectancy_bps ===")
    for config_id, _cfg, _report, metrics in ranked[:top_n]:
        _print_row(config_id, metrics)

    print(f"\n=== BOTTOM 5 by 15m expectancy_bps ===")
    for config_id, _cfg, _report, metrics in ranked[-5:]:
        _print_row(config_id, metrics)

    n_zero_trade = sum(1 for *_r, m in results if m["per_horizon"].get(15, {}).get("n", 0) == 0)
    print(f"\n[sweep] {n_zero_trade}/{len(results)} configs produced zero trades at the 15m horizon.")

    top_k = ranked[:controls_top_k]
    print(f"\n=== Negative controls for the top {len(top_k)} configs (15m horizon expectancy_bps) ===")
    for config_id, cfg, real_report, _metrics in top_k:
        decisions_by_day = {d: decide_core_gate_all(snaps_by_day[d], cfg) for d in eval_days}
        controls = run_negative_controls(
            eval_bars_by_day,
            prior_sessions_by_day,
            decisions_by_day,
            cfg,
            horizons=HORIZONS,
            decide_fn=decide_core_gate_all,
        )

        def _exp15(rep: EvalReport) -> str:
            s = rep.per_horizon.get(15)
            return _fmt(s.expectancy_bps) if s else "n/a"

        print(
            f"{config_id:<55} real={_exp15(controls.real):>7} "
            f"shuffled={_exp15(controls.shuffled_bars):>7} "
            f"delayed={_exp15(controls.delayed_signal):>7} "
            f"random={_exp15(controls.random_matched):>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--dev-start", default=DEV_START.isoformat())
    parser.add_argument("--dev-end", default=DEV_END.isoformat())
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--fold-size-days", type=int, default=5)
    parser.add_argument("--prior-buffer-days", type=int, default=DEFAULT_PRIOR_BUFFER_DAYS)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--controls-top-k", type=int, default=5)
    args = parser.parse_args()

    run_sweep(
        symbol=args.symbol,
        db_path=Path(args.db),
        dev_start=date.fromisoformat(args.dev_start),
        dev_end=date.fromisoformat(args.dev_end),
        ledger_path=Path(args.ledger),
        fold_size_days=args.fold_size_days,
        prior_buffer_days=args.prior_buffer_days,
        top_n=args.top_n,
        controls_top_k=args.controls_top_k,
    )


if __name__ == "__main__":
    main()
