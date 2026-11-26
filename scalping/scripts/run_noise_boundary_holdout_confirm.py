"""Holdout confirmation for preselected noise-boundary candidates.

This script intentionally tests only the shortlist selected from the locked
dev window. It is not a tuning sweep.

Dev window used for selection elsewhere: 2023-07-01 -> 2025-06-30.
Holdout confirmation here: 2025-07-01 -> latest date available in scalp.db.

Usage:
    python scripts/run_noise_boundary_holdout_confirm.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bot.data.store import Store  # noqa: E402
from bot.scalp.noise_boundary import (  # noqa: E402
    HALF_HOUR_MARKS,
    NoiseBoundaryConfig,
    ScalpExitConfig,
    Trade,
    run_variant_scalp_exit,
    side_stats,
    split_by_direction,
)

DEFAULT_DB = Path("data/sqlite/scalp.db")
LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "scalp_zarattini_ledger.jsonl"
RUN_TAG = "v6-nb-holdout-confirm"
SYMBOLS = ("SPY", "QQQ")
HOLDOUT_START = date(2025, 7, 1)
PRIOR_BUFFER_DAYS = 60
SLIPPAGE_GRID_BPS = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class Candidate:
    config_id: str
    vm: float
    exit_cfg: ScalpExitConfig


CANDIDATES = (
    Candidate("v4nb_vm1.0_target_0.5x", 1.0, ScalpExitConfig(kind="first_target", target_width_mult=0.5)),
    Candidate("v4nb_vm1.25_target_0.5x", 1.25, ScalpExitConfig(kind="first_target", target_width_mult=0.5)),
    Candidate(
        "v4nb_vm1.0_time_limit_60_reentry",
        1.0,
        ScalpExitConfig(kind="time_limit", time_limit_minutes=60, allow_one_reentry_after_stop=True),
    ),
)


def _latest_date(db_path: Path, symbol: str) -> date:
    store = Store(db_path)
    dates = store.get_trading_dates(symbol, date(2025, 1, 1), date(2030, 12, 31), "1m")
    if not dates:
        raise RuntimeError(f"no {symbol} 1m bars found in {db_path}")
    return dates[-1]


def _load_bars(db_path: Path, symbol: str, start: date, end: date) -> dict[date, list]:
    if not db_path.exists():
        raise FileNotFoundError(f"scalp.db not found at {db_path}")
    store = Store(db_path)
    buffered_start = start - timedelta(days=PRIOR_BUFFER_DAYS)
    dates = store.get_trading_dates(symbol, buffered_start, end, "1m")
    out = {}
    for d in dates:
        bars = store.get_bars_for_day(symbol, d, "1m")
        if bars:
            out[d] = bars
    return out


def _net_at(trades: list[Trade], bp: float = 1.0):
    if not trades:
        return None
    return mean(t.net_bps(bp) for t in trades)


def _stats(trades: list[Trade], n_days: int) -> dict:
    longs, shorts = split_by_direction(trades)

    def row(ts: list[Trade]) -> dict:
        s = side_stats(ts, n_days, SLIPPAGE_GRID_BPS)
        return {
            "n": s.n,
            "trades_per_day": s.trades_per_day,
            "win_rate": s.win_rate,
            "avg_hold_minutes": s.avg_hold_minutes,
            "gross_expectancy_bps": s.gross_expectancy_bps,
            "r_expectancy": s.r_expectancy,
            "net_expectancy_bps": {str(k): v for k, v in s.net_expectancy_bps.items()},
        }

    return {"overall": row(trades), "long": row(longs), "short": row(shorts)}


def _append(row: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _fmt(v, spec: str = ".2f") -> str:
    return format(v, spec) if v is not None else "n/a"


def run_holdout(db_path: Path = DEFAULT_DB, end: date | None = None) -> list[dict]:
    holdout_end = end or min(_latest_date(db_path, symbol) for symbol in SYMBOLS)
    if holdout_end < HOLDOUT_START:
        raise RuntimeError(f"holdout end {holdout_end} is before holdout start {HOLDOUT_START}")

    print(f"Holdout confirmation: {HOLDOUT_START} -> {holdout_end}")
    bars_by_symbol = {}
    n_days_by_symbol = {}
    for symbol in SYMBOLS:
        print(f"[load] {symbol} ...", end=" ", flush=True)
        bars_by_symbol[symbol] = _load_bars(db_path, symbol, HOLDOUT_START, holdout_end)
        n_days_by_symbol[symbol] = len([d for d in bars_by_symbol[symbol] if HOLDOUT_START <= d <= holdout_end])
        print(f"{len(bars_by_symbol[symbol])} sessions ({n_days_by_symbol[symbol]} holdout)")

    rows: list[dict] = []
    for candidate in CANDIDATES:
        print(f"\n{'=' * 90}\n{candidate.config_id}\n{'=' * 90}")
        combined: list[Trade] = []
        combined_delayed: list[Trade] = []
        for symbol in SYMBOLS:
            cfg = NoiseBoundaryConfig(
                vm=candidate.vm,
                stop_mode="opposite_band",
                marks=HALF_HOUR_MARKS,
                label=f"vm{candidate.vm}_opposite_band_halfhour",
            )
            bars = bars_by_symbol[symbol]
            n_days = n_days_by_symbol[symbol]
            trades = run_variant_scalp_exit(
                symbol, bars, cfg, candidate.exit_cfg, eval_start=HOLDOUT_START, eval_end=holdout_end
            )
            delayed = run_variant_scalp_exit(
                symbol,
                bars,
                cfg,
                candidate.exit_cfg,
                eval_start=HOLDOUT_START,
                eval_end=holdout_end,
                signal_lag_marks=1,
            )
            combined.extend(trades)
            combined_delayed.extend(delayed)
            stats = _stats(trades, n_days)
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "strategy": "zarattini_noise_boundary",
                "run_tag": RUN_TAG,
                "config_id": candidate.config_id,
                "symbol": symbol,
                "vm": candidate.vm,
                "holdout_start": HOLDOUT_START.isoformat(),
                "holdout_end": holdout_end.isoformat(),
                "n_eval_days": n_days,
                **stats,
                "delayed_n": len(delayed),
                "delayed_net_1bp": _net_at(delayed),
            }
            _append(row)
            rows.append(row)
            print(
                f"  {symbol} real={_fmt(_net_at(trades)):>7} (n={len(trades):>4}) "
                f"delayed={_fmt(_net_at(delayed)):>7}"
            )

        combined_row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "strategy": "zarattini_noise_boundary",
            "run_tag": RUN_TAG,
            "config_id": candidate.config_id,
            "symbol": "SPY_QQQ_COMBINED",
            "vm": candidate.vm,
            "holdout_start": HOLDOUT_START.isoformat(),
            "holdout_end": holdout_end.isoformat(),
            "real_n": len(combined),
            "real_net_1bp": _net_at(combined),
            "delayed_n": len(combined_delayed),
            "delayed_net_1bp": _net_at(combined_delayed),
        }
        _append(combined_row)
        rows.append(combined_row)
        print(
            f"  COMBINED real={_fmt(combined_row['real_net_1bp']):>7} "
            f"(n={combined_row['real_n']:>4}) delayed={_fmt(combined_row['delayed_net_1bp']):>7}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--end", default=None, help="Optional YYYY-MM-DD holdout end")
    args = parser.parse_args()
    run_holdout(Path(args.db), date.fromisoformat(args.end) if args.end else None)


if __name__ == "__main__":
    main()

