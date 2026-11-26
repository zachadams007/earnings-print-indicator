"""Stage A underlying-direction backtest runner (SPY/QQQ, no-leakage harness).

Reads 1-minute bars for the given symbol from
data/sqlite/scalp.db (bars table,
timeframe='1m') via bot.data.store.Store - the same Bar schema used
elsewhere in this repo, just a differently-named DB file for this lane. That
DB is being populated by a parallel ingest lane; if it doesn't exist yet, or
has zero rows for the requested symbol, this script falls back to synthetic
fixture bars (bot.scalp.synthetic) and says so loudly. Synthetic-fixture runs
are a PIPELINE SMOKE TEST ONLY - never evidence of a real edge.

CRITICAL DATA CONVENTION: scalp.db stores ``ts`` as the bar-CLOSE instant -
the ingest script that populated it shifted Alpaca's native bar-OPEN stamps
forward by 1 minute (confirmed empirically: the first RTH row for a session
lands at ``09:31:00-05:00``/``14:31:00Z``, one minute after the 9:30 ET open,
with an opening-bar-sized volume spike - i.e. it is the CLOSE stamp of the
9:30-9:31 bar). Every module under bot.scalp (features.py's PIT convention,
all of tests/scalp/*) was built and unit-tested assuming ``bar.ts`` is the
bar's OPEN instant, matching every other bar source in this repo. Rather than
rework the whole PIT-tested pipeline around a close-stamped convention, this
loader converts AT THE BOUNDARY: every bar pulled from scalp.db has 1 minute
subtracted from its ``ts`` before anything else touches it, restoring the
open-stamp convention the rest of bot.scalp assumes. This conversion must
never be skipped or duplicated for scalp.db data.

Usage:
    python scripts/run_scalp_stage_a.py --symbol SPY
    python scripts/run_scalp_stage_a.py --symbol QQQ --db path/to/scalp.db
    python scripts/run_scalp_stage_a.py --symbol SPY --start 2024-01-01 --end 2024-01-31
    python scripts/run_scalp_stage_a.py --symbol SPY --synthetic-days 40
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bot.data.store import Store  # noqa: E402
from bot.scalp.controls import run_negative_controls  # noqa: E402
from bot.scalp.evaluate import HORIZONS_MIN, EvalReport, HorizonStats, evaluate_decisions  # noqa: E402
from bot.scalp.features import build_feature_snapshots  # noqa: E402
from bot.scalp.rules import RuleConfig, decide_all  # noqa: E402
from bot.scalp.synthetic import make_multi_day_sessions  # noqa: E402

DEFAULT_DB = Path("data/sqlite/scalp.db")

# Cap prior-session history passed into feature building per decision day -
# build_feature_snapshots only ever consumes the trailing atr_window (14) /
# rvol_lookback (20) sessions internally, so this bounds memory/compute on
# long real-data lookbacks without changing any result.
MAX_PRIOR_SESSIONS = 30

# Calendar-day buffer fetched BEFORE `start` purely to give the first few
# evaluation days real ATR(14)/RVOL-TOD(20-session) history instead of
# silently degrading to None/NaN at the start of a requested slice. These
# buffer days are never scored/evaluated themselves - see `eval_days` below.
DEFAULT_PRIOR_BUFFER_DAYS = 60


def _load_real_bars(
    db_path: Path,
    symbol: str,
    start: date,
    end: date,
    prior_buffer_days: int = DEFAULT_PRIOR_BUFFER_DAYS,
) -> dict[date, list] | None:
    """Load real 1m bars for ``[start, end]`` plus a calendar buffer of prior
    sessions (context-only, for ATR/RVOL - see `eval_days` in run_stage_a).

    Returns None (triggering the synthetic fallback) if the DB/table is
    missing, or has zero rows for `symbol` anywhere in `[start, end]`.
    """
    if not db_path.exists():
        return None
    store = Store(db_path)
    buffered_start = start - timedelta(days=prior_buffer_days)
    trading_dates = store.get_trading_dates(symbol, buffered_start, end, "1m")
    if not trading_dates:
        return None
    bars_by_day = {}
    for d in trading_dates:
        day_bars = store.get_bars_for_day(symbol, d, "1m")
        if not day_bars:
            continue
        # See module docstring "CRITICAL DATA CONVENTION": scalp.db's ts is
        # the bar's CLOSE instant; subtract 1 minute to restore the OPEN-
        # stamp convention bot.scalp's PIT logic assumes everywhere else.
        bars_by_day[d] = [b.model_copy(update={"ts": b.ts - timedelta(minutes=1)}) for b in day_bars]
    if not any(d >= start for d in bars_by_day):
        return None
    return bars_by_day


def _print_horizon_table(title: str, per_horizon: dict[int, HorizonStats]) -> None:
    print(f"\n{title}")
    print(f"{'horizon(min)':>12} {'n':>6} {'hit_rate':>9} {'exp_bps':>9} {'mfe_bps':>9} {'mae_bps':>9}")
    for h, stats in sorted(per_horizon.items()):
        hr = f"{stats.hit_rate:.1%}" if stats.hit_rate is not None else "n/a"
        exp = f"{stats.expectancy_bps:.1f}" if stats.expectancy_bps is not None else "n/a"
        mfe = f"{stats.avg_mfe_bps:.1f}" if stats.avg_mfe_bps is not None else "n/a"
        mae = f"{stats.avg_mae_bps:.1f}" if stats.avg_mae_bps is not None else "n/a"
        print(f"{h:>12} {stats.n:>6} {hr:>9} {exp:>9} {mfe:>9} {mae:>9}")


def run_stage_a(
    symbol: str,
    db_path: Path,
    synthetic_days: int = 60,
    start: date | None = None,
    end: date | None = None,
    prior_buffer_days: int = DEFAULT_PRIOR_BUFFER_DAYS,
) -> EvalReport:
    end = end or date.today()
    start = start or (end - timedelta(days=180))

    bars_by_day = _load_real_bars(db_path, symbol, start, end, prior_buffer_days)
    using_synthetic = bars_by_day is None

    if using_synthetic:
        print(
            f"[run_scalp_stage_a] No usable 1m {symbol} rows found at {db_path} "
            f"for [{start}, {end}] (missing DB, missing table, or zero rows). "
            f"Falling back to {synthetic_days} days of SYNTHETIC fixture bars - "
            f"this is a pipeline smoke test only, NOT evidence of any edge."
        )
        bars_by_day = make_multi_day_sessions(
            symbol, date.today() - timedelta(days=synthetic_days * 2), synthetic_days
        )
        eval_days = sorted(bars_by_day)
    else:
        eval_days = sorted(d for d in bars_by_day if start <= d <= end)
        n_buffer = len(bars_by_day) - len(eval_days)
        print(
            f"[run_scalp_stage_a] Loaded {len(bars_by_day)} real trading days for {symbol} "
            f"from {db_path} ({len(eval_days)} in [{start}, {end}], {n_buffer} buffer-only "
            f"days kept for ATR/RVOL context)"
        )

    sorted_days = sorted(bars_by_day)
    day_to_idx = {d: i for i, d in enumerate(sorted_days)}
    cfg = RuleConfig()

    decisions_by_day = {}
    prior_sessions_by_day = {}
    for d in eval_days:
        idx = day_to_idx[d]
        prior_days = [bars_by_day[pd_] for pd_ in sorted_days[max(0, idx - MAX_PRIOR_SESSIONS):idx]]
        prior_sessions_by_day[d] = prior_days
        snaps = build_feature_snapshots(bars_by_day[d], prior_days)
        decisions_by_day[d] = decide_all(snaps, cfg)

    eval_bars_by_day = {d: bars_by_day[d] for d in eval_days}
    report = evaluate_decisions(eval_bars_by_day, decisions_by_day, horizons=HORIZONS_MIN)

    print(f"\n=== Stage A report: {symbol} ({'SYNTHETIC' if using_synthetic else 'REAL'} data) ===")
    print(f"Decisions: {report.trade_count} trades, {report.wait_count} waits")
    if report.max_drawdown_pct is not None:
        print(f"Max drawdown (sequential, {report.default_horizon_min}m horizon): {report.max_drawdown_pct:.2%}")
    else:
        print("Max drawdown: n/a (no closed trades at default horizon)")

    _print_horizon_table("Overall", report.per_horizon)
    for wname, per_h in sorted(report.by_window.items()):
        _print_horizon_table(f"Window: {wname}", per_h)
    for fid, per_h in sorted(report.by_fold.items()):
        _print_horizon_table(f"Fold: {fid}", per_h)

    controls = run_negative_controls(eval_bars_by_day, prior_sessions_by_day, decisions_by_day, cfg)
    h0 = HORIZONS_MIN[0]

    def _exp(rep: EvalReport) -> str:
        s = rep.per_horizon.get(h0)
        return f"{s.expectancy_bps:.1f}" if s and s.expectancy_bps is not None else "n/a"

    print(f"\n=== Negative controls ({h0}m horizon expectancy_bps) ===")
    print(f"real={_exp(report)}")
    print(f"shuffled_bars={_exp(controls.shuffled_bars)}")
    print(f"delayed_signal={_exp(controls.delayed_signal)}")
    print(f"random_matched={_exp(controls.random_matched)}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--synthetic-days", type=int, default=60)
    parser.add_argument("--start", default=None, help="ISO date, e.g. 2024-01-01 (default: 180 days before --end)")
    parser.add_argument("--end", default=None, help="ISO date, e.g. 2024-01-31 (default: today)")
    parser.add_argument(
        "--prior-buffer-days", type=int, default=DEFAULT_PRIOR_BUFFER_DAYS,
        help="Calendar days of extra history loaded before --start for ATR/RVOL context only",
    )
    args = parser.parse_args()
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    run_stage_a(args.symbol, Path(args.db), args.synthetic_days, start=start, end=end, prior_buffer_days=args.prior_buffer_days)


if __name__ == "__main__":
    main()
