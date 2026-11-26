"""Zarattini/Aziz/Barbon "Beat the Market" noise-boundary replication runner.

Dev window ONLY: 2023-07-01 -> 2025-06-30 (hard-asserted below; the locked
holdout at/after 2025-07-01 is never loaded by this script). SPY primary,
QQQ secondary. Every variant + control run is appended to
data/scalp_zarattini_ledger.jsonl (own ledger file -- never touches the
parallel lane's data/scalp_sweep_trial_ledger.jsonl).

Reads 1m bars from data/sqlite/scalp.db
via bot.data.store.Store, using the same loader PATTERN as the parallel
lane's scripts/run_scalp_stage_a.py::_load_real_bars (buffer days before
`start` for lookback context, PIT-safe day-by-day fetch) -- copied rather
than imported, since importing that script would couple this lane to files
under concurrent edit elsewhere. See src/bot/scalp/noise_boundary.py's
module docstring for why this copy does NOT apply that sibling loader's
-1-minute ts shift (deliberate: this lane's own convention reads ts as a
close-stamp directly, matching the paper's own Close_{t,HH:MM} notation).

Usage:
    python scripts/run_noise_boundary.py
    python scripts/run_noise_boundary.py --symbol QQQ
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bot.data.store import Store  # noqa: E402
from bot.scalp.noise_boundary import (  # noqa: E402
    HALF_HOUR_MARKS,
    HOURLY_MARKS,
    NoiseBoundaryConfig,
    Trade,
    fold_consistency,
    random_matched_control,
    run_variant,
    side_stats,
    split_by_direction,
    yearly_expectancy,
)

DEFAULT_DB = Path("data/sqlite/scalp.db")
LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "scalp_zarattini_ledger.jsonl"

DEV_START = date(2023, 7, 1)
DEV_END = date(2025, 6, 30)
LOCKED_HOLDOUT_START = date(2025, 7, 1)  # never load on/after this date

PRIOR_BUFFER_DAYS = 60  # calendar days fetched before DEV_START, buffer-only (for the 14-day lookback)
SLIPPAGE_GRID_BPS = (0.5, 1.0, 2.0)
DEFAULT_STOP_MODES = ("opposite_band", "current_band", "current_band_vwap")
DEFAULT_VMS = (0.5, 1.0, 1.5)


def _load_real_bars(db_path: Path, symbol: str, start: date, end: date, prior_buffer_days: int):
    """Copied loader pattern (see module docstring) - NO -1min shift here;
    see src/bot/scalp/noise_boundary.py for why.
    """
    assert end < LOCKED_HOLDOUT_START, (
        f"HARD RULE violated: requested end={end} >= locked holdout start "
        f"{LOCKED_HOLDOUT_START}. Never load the holdout in this lane."
    )
    if not db_path.exists():
        raise FileNotFoundError(f"scalp.db not found at {db_path}")
    store = Store(db_path)
    buffered_start = date(start.year, start.month, start.day)
    from datetime import timedelta
    buffered_start = start - timedelta(days=prior_buffer_days)
    trading_dates = store.get_trading_dates(symbol, buffered_start, end, "1m")
    if not trading_dates:
        raise RuntimeError(f"No {symbol} 1m rows in {db_path} for [{buffered_start}, {end}]")
    bars_by_day = {}
    for d in trading_dates:
        day_bars = store.get_bars_for_day(symbol, d, "1m")
        if day_bars:
            bars_by_day[d] = day_bars
    return bars_by_day


def _variant_grid() -> list[NoiseBoundaryConfig]:
    grid = []
    for vm in DEFAULT_VMS:
        for stop_mode in DEFAULT_STOP_MODES:
            for grid_name, marks in (("halfhour", HALF_HOUR_MARKS), ("hourly", HOURLY_MARKS)):
                grid.append(
                    NoiseBoundaryConfig(
                        vm=vm,
                        stop_mode=stop_mode,
                        marks=marks,
                        label=f"vm{vm}_{stop_mode}_{grid_name}",
                    )
                )
    return grid  # 3 x 3 x 2 = 18 variants


DEFAULT_CFG = NoiseBoundaryConfig()  # vm=1.0, current_band_vwap, half-hour grid -- the paper's own spec


def _is_default(cfg: NoiseBoundaryConfig) -> bool:
    """Compare by field, not by label string -- _variant_grid()'s generated
    labels ("current_band_vwap") don't literally match NoiseBoundaryConfig's
    own default label field ("currband_vwap"), so a naive label== check
    silently never fires. Bit us once already; keep this explicit.
    """
    return cfg.vm == DEFAULT_CFG.vm and cfg.stop_mode == DEFAULT_CFG.stop_mode and cfg.marks == DEFAULT_CFG.marks


def _n_eval_days(bars_by_day: dict, start: date, end: date, lookback_days: int = 14) -> int:
    days = sorted(d for d in bars_by_day if start <= d <= end)
    return len(days)


def _ledger_row(symbol: str, cfg_label: str, trades: list[Trade], n_days: int, extra: dict) -> dict:
    longs, shorts = split_by_direction(trades)
    all_stats = side_stats(trades, n_days, SLIPPAGE_GRID_BPS)
    long_stats = side_stats(longs, n_days, SLIPPAGE_GRID_BPS)
    short_stats = side_stats(shorts, n_days, SLIPPAGE_GRID_BPS)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "strategy": "zarattini_noise_boundary",
        "symbol": symbol,
        "config": cfg_label,
        "n_eval_days": n_days,
        "overall": vars(all_stats),
        "long": vars(long_stats),
        "short": vars(short_stats),
    }
    row.update(extra)
    return row


def _append_ledger(row: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _print_side_table(title: str, trades: list[Trade], n_days: int) -> None:
    longs, shorts = split_by_direction(trades)
    print(f"\n{title}  (n={len(trades)}, {n_days} eval days)")
    header = f"{'side':>8} {'n':>6} {'trd/day':>8} {'win%':>6} {'hold_m':>7} {'gross_bps':>10}"
    header += "".join(f" net@{s}bp".rjust(10) for s in SLIPPAGE_GRID_BPS) + f" {'R_exp':>7}"
    print(header)
    for name, ts in (("ALL", trades), ("LONG", longs), ("SHORT", shorts)):
        s = side_stats(ts, n_days, SLIPPAGE_GRID_BPS)
        if s.n == 0:
            print(f"{name:>8} {'0':>6}")
            continue
        line = (
            f"{name:>8} {s.n:>6} {s.trades_per_day:>8.2f} {s.win_rate*100:>5.1f}% "
            f"{s.avg_hold_minutes:>7.0f} {s.gross_expectancy_bps:>10.2f}"
        )
        for sl in SLIPPAGE_GRID_BPS:
            line += f" {s.net_expectancy_bps[sl]:>9.2f}"
        line += f" {s.r_expectancy:>7.3f}" if s.r_expectancy is not None else f" {'n/a':>7}"
        print(line)


def run_for_symbol(symbol: str, db_path: Path, is_primary: bool) -> None:
    print(f"\n{'='*70}\n{symbol} -- loading real bars from {db_path}\n{'='*70}")
    bars_by_day = _load_real_bars(db_path, symbol, DEV_START, DEV_END, PRIOR_BUFFER_DAYS)
    eval_days_available = sorted(d for d in bars_by_day if DEV_START <= d <= DEV_END)
    print(
        f"[{symbol}] {len(bars_by_day)} total sessions loaded "
        f"({len(eval_days_available)} in dev window {DEV_START}..{DEV_END}, "
        f"{len(bars_by_day) - len(eval_days_available)} buffer-only days for the 14-day lookback)"
    )

    variants = _variant_grid() if is_primary else [DEFAULT_CFG]
    variant_summary_rows = []

    for cfg in variants:
        trades = run_variant(symbol, bars_by_day, cfg, eval_start=DEV_START, eval_end=DEV_END)
        n_days = _n_eval_days(bars_by_day, DEV_START, DEV_END)
        row = _ledger_row(symbol, cfg.label, trades, n_days, {"vm": cfg.vm, "stop_mode": cfg.stop_mode})
        _append_ledger(row)
        s = side_stats(trades, n_days, SLIPPAGE_GRID_BPS)
        variant_summary_rows.append((cfg.label, s))

        if _is_default(cfg):
            # This is the paper's featured spec -- full detail.
            _print_side_table(f"[{symbol}] DEFAULT (paper spec): {cfg.label}", trades, n_days)

            years = yearly_expectancy(trades, slippage_bps=1.0)
            print(f"\n[{symbol}] Expectancy by year (net @ 1.0bp/side):")
            for y, stats in years.items():
                print(f"  {y}: n={stats['n']:>4}  gross={stats['gross_bps']:>7.2f}bps  net={stats['net_bps']:>7.2f}bps")

            fc = fold_consistency(trades, slippage_bps=1.0)
            print(
                f"\n[{symbol}] Blocked weekly-fold consistency (net @ 1.0bp/side): "
                f"{fc['n_folds']} folds, {fc['pct_positive']*100:.0f}% positive-mean weeks, "
                f"median={fc['median_bps']:.2f}bps, IQR={fc['iqr_bps']:.2f}bps"
            )
            _append_ledger(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "strategy": "zarattini_noise_boundary",
                    "symbol": symbol,
                    "config": cfg.label + "__by_year",
                    "by_year": years,
                    "fold_consistency": fc,
                }
            )

            # ── negative controls (default config only) ──
            delayed = run_variant(
                symbol, bars_by_day, cfg, eval_start=DEV_START, eval_end=DEV_END, signal_lag_marks=1
            )
            random_ctrl = random_matched_control(symbol, bars_by_day, trades, seed=42, n_repeats=20)

            print(f"\n[{symbol}] Negative controls (1.0bp/side net expectancy_bps, all-trades):")
            for name, ctrl_trades in (("real", trades), ("delayed_signal(+1 mark)", delayed), ("random_matched(x20)", random_ctrl)):
                cs = side_stats(ctrl_trades, n_days, SLIPPAGE_GRID_BPS)
                val = cs.net_expectancy_bps.get(1.0) if cs.n else None
                print(f"  {name:>26}: n={cs.n:>6}  net_bps={val if val is None else round(val, 2)}")
                _append_ledger(
                    _ledger_row(symbol, f"{cfg.label}__control_{name}", ctrl_trades, n_days, {"control": name})
                )

    if is_primary:
        print(f"\n[{symbol}] Variant sweep summary (net expectancy_bps @ 1.0bp/side, all-trades):")
        print(f"{'config':>34} {'n':>6} {'trd/day':>8} {'net@1bp':>9}")
        for label, s in variant_summary_rows:
            net1 = s.net_expectancy_bps.get(1.0)
            net1_str = f"{net1:>9.2f}" if net1 is not None else f"{'n/a':>9}"
            trd = f"{s.trades_per_day:>8.2f}" if s.trades_per_day is not None else f"{'n/a':>8}"
            print(f"{label:>34} {s.n:>6} {trd} {net1_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None, help="Run only this symbol (default: both, SPY primary + QQQ secondary)")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.symbol:
        run_for_symbol(args.symbol, db_path, is_primary=True)
    else:
        run_for_symbol("SPY", db_path, is_primary=True)
        run_for_symbol("QQQ", db_path, is_primary=False)


if __name__ == "__main__":
    main()
