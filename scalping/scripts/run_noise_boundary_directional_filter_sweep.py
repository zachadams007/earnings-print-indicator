"""Exact directional filter sweep for Lane A noise-boundary candidates.

This is intentionally narrower than a broad optimizer. It answers whether the
current noise-boundary family can become an executable long-only or short-only
playbook after simple, cockpit-readable filters:

- time window
- same-side VWAP confirmation
- max overnight gap size
- relative volume at the decision mark
- max extension past the noise band

The script evaluates both the locked dev window and the post-2025-07-01
holdout, and appends compact rows to data/scalp_zarattini_ledger.jsonl.

Usage:
    python scripts/run_noise_boundary_directional_filter_sweep.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from statistics import mean
from typing import Literal, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bot.data.models import Bar  # noqa: E402
from bot.scalp.noise_boundary import (  # noqa: E402
    HALF_HOUR_MARKS,
    DaySession,
    Direction,
    FillMode,
    NoiseBoundaryConfig,
    ScalpExitConfig,
    Trade,
    build_day_session,
    build_noise_bands,
    side_stats,
)
from run_noise_boundary import (  # noqa: E402
    DEV_END,
    DEV_START,
    LEDGER_PATH,
    PRIOR_BUFFER_DAYS,
    _load_real_bars,
)
from run_noise_boundary_holdout_confirm import (  # noqa: E402
    HOLDOUT_START,
    _latest_date,
    _load_bars,
)

DEFAULT_DB = Path("data/sqlite/scalp.db")
RUN_TAG = "v7b-nb-directional-filter-sweep"
SYMBOLS = ("SPY", "QQQ")
SLIPPAGE_BPS = 1.0
DirectionFilter = Literal["long", "short"]


@dataclass(frozen=True)
class Candidate:
    config_id: str
    vm: float
    exit_cfg: ScalpExitConfig


@dataclass(frozen=True)
class EntryFilter:
    filter_id: str
    start: time = time(10, 0)
    end: time = time(15, 30)
    require_same_side_vwap: bool = False
    max_gap_bps: Optional[float] = None
    min_rvol20: Optional[float] = None
    max_band_extension: Optional[float] = None


@dataclass(frozen=True)
class DayContext:
    session: DaySession
    prior_close: float
    bands: dict[time, tuple[float, float]]


CANDIDATES = (
    Candidate("vm1.0_time45_reentry", 1.0, ScalpExitConfig(kind="time_limit", time_limit_minutes=45, allow_one_reentry_after_stop=True)),
    Candidate("vm1.0_time60_reentry", 1.0, ScalpExitConfig(kind="time_limit", time_limit_minutes=60, allow_one_reentry_after_stop=True)),
    Candidate("vm1.0_time90_reentry", 1.0, ScalpExitConfig(kind="time_limit", time_limit_minutes=90, allow_one_reentry_after_stop=True)),
    Candidate("vm1.25_time60_reentry", 1.25, ScalpExitConfig(kind="time_limit", time_limit_minutes=60, allow_one_reentry_after_stop=True)),
    Candidate("vm1.0_target0.25x", 1.0, ScalpExitConfig(kind="first_target", target_width_mult=0.25)),
    Candidate("vm1.0_target0.5x", 1.0, ScalpExitConfig(kind="first_target", target_width_mult=0.5)),
    Candidate("vm1.25_target0.25x", 1.25, ScalpExitConfig(kind="first_target", target_width_mult=0.25)),
    Candidate("vm1.25_target0.5x", 1.25, ScalpExitConfig(kind="first_target", target_width_mult=0.5)),
)

FILTERS = (
    EntryFilter("all"),
    EntryFilter("no_late", end=time(14, 30)),
    EntryFilter("midday", start=time(10, 30), end=time(14, 30)),
    EntryFilter("same_vwap", require_same_side_vwap=True),
    EntryFilter("gap_lt_100bps", max_gap_bps=100.0),
    EntryFilter("gap_lt_50bps", max_gap_bps=50.0),
    EntryFilter("no_late_same_vwap", end=time(14, 30), require_same_side_vwap=True),
    EntryFilter("no_late_gap_lt_100bps", end=time(14, 30), max_gap_bps=100.0),
    EntryFilter("same_vwap_gap_lt_100bps", require_same_side_vwap=True, max_gap_bps=100.0),
    EntryFilter("no_late_same_vwap_gap_lt_100bps", end=time(14, 30), require_same_side_vwap=True, max_gap_bps=100.0),
    EntryFilter("rvol20_ge_1", min_rvol20=1.0),
    EntryFilter("no_late_rvol20_ge_1", end=time(14, 30), min_rvol20=1.0),
    EntryFilter("extension_lt_10pct", max_band_extension=0.10),
    EntryFilter("extension_lt_25pct", max_band_extension=0.25),
    EntryFilter("no_late_extension_lt_25pct", end=time(14, 30), max_band_extension=0.25),
)


def _minutes_between(start: time, end: time) -> int:
    start_m = start.hour * 60 + start.minute
    end_m = end.hour * 60 + end.minute
    return max(0, end_m - start_m)


def _net_at(trades: Sequence[Trade], bp: float = SLIPPAGE_BPS) -> Optional[float]:
    if not trades:
        return None
    return mean(t.net_bps(bp) for t in trades)


def _fmt(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "n/a"


def _rvol20(session: DaySession, mark: time) -> Optional[float]:
    times = [t for t in session.ordered_times if t <= mark]
    if len(times) < 2:
        return None
    current = session.by_time[mark].volume
    prior = [session.by_time[t].volume for t in times[-21:-1]]
    if not prior:
        return None
    avg = mean(prior)
    return current / avg if avg > 0 else None


def _entry_allowed(
    direction: Direction,
    mark: time,
    sig_price: float,
    lower: float,
    upper: float,
    vwap: Optional[float],
    prior_close: float,
    session: DaySession,
    entry_filter: EntryFilter,
) -> bool:
    if mark < entry_filter.start or mark > entry_filter.end:
        return False
    if entry_filter.require_same_side_vwap and vwap is not None:
        if direction == "long" and sig_price <= vwap:
            return False
        if direction == "short" and sig_price >= vwap:
            return False
    if entry_filter.max_gap_bps is not None:
        gap_bps = abs(session.open_price / prior_close - 1.0) * 10_000.0
        if gap_bps > entry_filter.max_gap_bps:
            return False
    if entry_filter.min_rvol20 is not None:
        rv = _rvol20(session, mark)
        if rv is None or rv < entry_filter.min_rvol20:
            return False
    if entry_filter.max_band_extension is not None:
        width = upper - lower
        if width <= 0:
            return False
        extension = (sig_price - upper) / width if direction == "long" else (lower - sig_price) / width
        if extension > entry_filter.max_band_extension:
            return False
    return True


def simulate_day_filtered(
    symbol: str,
    session: DaySession,
    prior_close: float,
    bands: dict[time, tuple[float, float]],
    cfg: NoiseBoundaryConfig,
    exit_cfg: ScalpExitConfig,
    direction_filter: DirectionFilter,
    entry_filter: EntryFilter,
    *,
    signal_lag_marks: int = 0,
    fill_mode: FillMode = "signal_close",
) -> list[Trade]:
    if exit_cfg.kind == "time_limit" and exit_cfg.time_limit_minutes is None:
        raise ValueError("time_limit exit requires time_limit_minutes")
    if exit_cfg.kind == "first_target" and exit_cfg.target_width_mult is None:
        raise ValueError("first_target exit requires target_width_mult")

    marks_in_day = [m for m in cfg.marks if m in session.by_time and m in bands]
    marks_index = {m: i for i, m in enumerate(marks_in_day)}
    trades: list[Trade] = []

    direction: Optional[Direction] = None
    entry_mark: Optional[time] = None
    entry_price = 0.0
    entry_stop = 0.0
    entry_target: Optional[float] = None
    first_exit_index = 0

    def entry_fill(decision_mark: time) -> tuple[float, time, int]:
        idx = session.ordered_times.index(decision_mark)
        if fill_mode == "signal_close" or idx + 1 >= len(session.ordered_times):
            return session.by_time[decision_mark].close, decision_mark, idx + 1
        next_mark = session.ordered_times[idx + 1]
        next_bar = session.by_time[next_mark]
        if fill_mode == "next_open":
            return next_bar.open, decision_mark, idx + 1
        return next_bar.close, next_mark, idx + 2

    def eligible_for_entry() -> bool:
        if not trades:
            return True
        return bool(exit_cfg.allow_one_reentry_after_stop and len(trades) == 1 and trades[-1].exit_reason == "stop")

    for time_index, t in enumerate(session.ordered_times):
        bar = session.by_time[t]

        if direction is None:
            if t in marks_index and eligible_for_entry():
                sig_idx = max(0, marks_index[t] - signal_lag_marks)
                sig_mark = marks_in_day[sig_idx]
                lower, upper = bands[sig_mark]
                sig_price = session.by_time[sig_mark].close
                signal_direction: Optional[Direction] = None
                if sig_price > upper:
                    signal_direction = "long"
                elif sig_price < lower:
                    signal_direction = "short"

                if signal_direction == direction_filter and _entry_allowed(
                    signal_direction,
                    t,
                    sig_price,
                    lower,
                    upper,
                    session.vwap_by_time.get(sig_mark),
                    prior_close,
                    session,
                    entry_filter,
                ):
                    direction = signal_direction
                    entry_price, entry_mark, first_exit_index = entry_fill(t)
                    width = upper - lower
                    if direction == "long":
                        entry_stop = lower
                        entry_target = (
                            entry_price + exit_cfg.target_width_mult * width
                            if exit_cfg.kind == "first_target"
                            else None
                        )
                    else:
                        entry_stop = upper
                        entry_target = (
                            entry_price - exit_cfg.target_width_mult * width
                            if exit_cfg.kind == "first_target"
                            else None
                        )
            continue

        if time_index < first_exit_index:
            continue

        elapsed = _minutes_between(entry_mark, t)
        hit_stop = (bar.low <= entry_stop) if direction == "long" else (bar.high >= entry_stop)
        if hit_stop:
            trades.append(
                Trade(
                    symbol=symbol,
                    day=session.day,
                    direction=direction,
                    entry_mark=entry_mark,
                    entry_price=entry_price,
                    entry_stop_price=entry_stop,
                    exit_mark=t,
                    exit_price=entry_stop,
                    exit_reason="stop",
                    hold_minutes=elapsed,
                )
            )
            direction = None
            continue

        if exit_cfg.kind == "first_target":
            hit_target = (bar.high >= entry_target) if direction == "long" else (bar.low <= entry_target)
            if hit_target:
                trades.append(
                    Trade(
                        symbol=symbol,
                        day=session.day,
                        direction=direction,
                        entry_mark=entry_mark,
                        entry_price=entry_price,
                        entry_stop_price=entry_stop,
                        exit_mark=t,
                        exit_price=entry_target,
                        exit_reason="target",
                        hold_minutes=elapsed,
                    )
                )
                direction = None
                continue

        if exit_cfg.kind == "time_limit" and elapsed >= exit_cfg.time_limit_minutes:
            trades.append(
                Trade(
                    symbol=symbol,
                    day=session.day,
                    direction=direction,
                    entry_mark=entry_mark,
                    entry_price=entry_price,
                    entry_stop_price=entry_stop,
                    exit_mark=t,
                    exit_price=bar.close,
                    exit_reason="time_stop",
                    hold_minutes=elapsed,
                )
            )
            direction = None
            continue

    if direction is not None:
        hold = _minutes_between(entry_mark, session.close_mark)
        trades.append(
            Trade(
                symbol=symbol,
                day=session.day,
                direction=direction,
                entry_mark=entry_mark,
                entry_price=entry_price,
                entry_stop_price=entry_stop,
                exit_mark=session.close_mark,
                exit_price=session.close_price,
                exit_reason="eod_close",
                hold_minutes=hold,
            )
        )

    return trades


def _sessions(symbol: str, bars_by_day: dict[date, Sequence[Bar]]) -> tuple[dict[date, DaySession], list[date]]:
    out = {}
    for d, bars in bars_by_day.items():
        session = build_day_session(symbol, d, bars)
        if session is not None:
            out[d] = session
    return out, sorted(out)


def build_contexts(
    symbol: str,
    bars_by_day: dict[date, Sequence[Bar]],
    vm: float,
    start: date,
    end: date,
) -> list[DayContext]:
    sessions, sorted_days = _sessions(symbol, bars_by_day)
    cfg = NoiseBoundaryConfig(
        vm=vm,
        stop_mode="opposite_band",
        marks=HALF_HOUR_MARKS,
        label=f"vm{vm}_opposite_band_halfhour",
    )
    contexts: list[DayContext] = []
    for idx, d in enumerate(sorted_days):
        if d < start or d > end or idx < cfg.lookback_days:
            continue
        history = [sessions[sorted_days[j]] for j in range(idx - cfg.lookback_days, idx)]
        bands = build_noise_bands(history, sessions[d], cfg)
        if bands is None:
            continue
        contexts.append(DayContext(session=sessions[d], prior_close=history[-1].close_price, bands=bands))
    return contexts


def run_filtered_contexts(
    symbol: str,
    contexts: Sequence[DayContext],
    candidate: Candidate,
    direction: DirectionFilter,
    entry_filter: EntryFilter,
    *,
    signal_lag_marks: int = 0,
    fill_mode: FillMode = "signal_close",
) -> list[Trade]:
    cfg = NoiseBoundaryConfig(
        vm=candidate.vm,
        stop_mode="opposite_band",
        marks=HALF_HOUR_MARKS,
        label=f"vm{candidate.vm}_opposite_band_halfhour",
    )
    trades: list[Trade] = []
    for ctx in contexts:
        trades.extend(
            simulate_day_filtered(
                symbol,
                ctx.session,
                ctx.prior_close,
                ctx.bands,
                cfg,
                candidate.exit_cfg,
                direction,
                entry_filter,
                signal_lag_marks=signal_lag_marks,
                fill_mode=fill_mode,
            )
        )
    return trades


def _summary(trades: list[Trade], n_days: int) -> dict:
    s = side_stats(trades, n_days, (SLIPPAGE_BPS,))
    return {
        "n": s.n,
        "trades_per_day": s.trades_per_day,
        "win_rate": s.win_rate,
        "avg_hold_minutes": s.avg_hold_minutes,
        "gross_bps": s.gross_expectancy_bps,
        "net_1bp": s.net_expectancy_bps[SLIPPAGE_BPS],
    }


def _append(row: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _load_periods(db_path: Path, holdout_end: date) -> dict[str, tuple[date, date, dict[str, dict[date, list[Bar]]], dict[str, int]]]:
    dev_bars = {symbol: _load_real_bars(db_path, symbol, DEV_START, DEV_END, PRIOR_BUFFER_DAYS) for symbol in SYMBOLS}
    holdout_bars = {symbol: _load_bars(db_path, symbol, HOLDOUT_START, holdout_end) for symbol in SYMBOLS}
    return {
        "dev": (
            DEV_START,
            DEV_END,
            dev_bars,
            {symbol: len([d for d in dev_bars[symbol] if DEV_START <= d <= DEV_END]) for symbol in SYMBOLS},
        ),
        "holdout": (
            HOLDOUT_START,
            holdout_end,
            holdout_bars,
            {symbol: len([d for d in holdout_bars[symbol] if HOLDOUT_START <= d <= holdout_end]) for symbol in SYMBOLS},
        ),
    }


def run_sweep(db_path: Path = DEFAULT_DB, holdout_end: date | None = None) -> list[dict]:
    end = holdout_end or min(_latest_date(db_path, symbol) for symbol in SYMBOLS)
    periods = _load_periods(db_path, end)
    vms = sorted({candidate.vm for candidate in CANDIDATES})
    contexts: dict[tuple[str, str, float], list[DayContext]] = {}
    for period_name, (start, finish, bars_by_symbol, _) in periods.items():
        for symbol in SYMBOLS:
            for vm in vms:
                print(f"[context] {period_name} {symbol} vm={vm} ...", flush=True)
                contexts[(period_name, symbol, vm)] = build_contexts(symbol, bars_by_symbol[symbol], vm, start, finish)

    rows: list[dict] = []
    pass_rows: list[dict] = []

    for candidate in CANDIDATES:
        print(f"\n{'=' * 88}\n{candidate.config_id}\n{'=' * 88}")
        for direction in ("long", "short"):
            for entry_filter in FILTERS:
                row = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "strategy": "zarattini_noise_boundary",
                    "run_tag": RUN_TAG,
                    "config_id": candidate.config_id,
                    "direction": direction,
                    "filter_id": entry_filter.filter_id,
                    "periods": {},
                }
                for period_name, (start, finish, bars_by_symbol, n_days_by_symbol) in periods.items():
                    combined: list[Trade] = []
                    combined_delayed: list[Trade] = []
                    row["periods"][period_name] = {}
                    for symbol in SYMBOLS:
                        ctx = contexts[(period_name, symbol, candidate.vm)]
                        trades = run_filtered_contexts(symbol, ctx, candidate, direction, entry_filter)
                        delayed = run_filtered_contexts(
                            symbol,
                            ctx,
                            candidate,
                            direction,
                            entry_filter,
                            signal_lag_marks=1,
                        )
                        combined.extend(trades)
                        combined_delayed.extend(delayed)
                        row["periods"][period_name][symbol] = {
                            **_summary(trades, n_days_by_symbol[symbol]),
                            "delayed_n": len(delayed),
                            "delayed_net_1bp": _net_at(delayed),
                        }
                    row["periods"][period_name]["SPY_QQQ_COMBINED"] = {
                        "n": len(combined),
                        "net_1bp": _net_at(combined),
                        "delayed_n": len(combined_delayed),
                        "delayed_net_1bp": _net_at(combined_delayed),
                    }
                _append(row)
                rows.append(row)

                dev = row["periods"]["dev"]
                hold = row["periods"]["holdout"]
                if (
                    (dev["SPY"]["net_1bp"] or -999.0) > 0
                    and (dev["QQQ"]["net_1bp"] or -999.0) > 0
                    and (hold["SPY"]["net_1bp"] or -999.0) > 0
                    and (hold["QQQ"]["net_1bp"] or -999.0) > 0
                    and hold["SPY"]["n"] >= 30
                    and hold["QQQ"]["n"] >= 30
                ):
                    pass_rows.append(row)
                    print(
                        f"  PASS {direction:<5} {entry_filter.filter_id:<30} "
                        f"dev SPY/QQQ={_fmt(dev['SPY']['net_1bp'])}/{_fmt(dev['QQQ']['net_1bp'])} "
                        f"hold SPY/QQQ={_fmt(hold['SPY']['net_1bp'])}/{_fmt(hold['QQQ']['net_1bp'])} "
                        f"hold delayed combo={_fmt(hold['SPY_QQQ_COMBINED']['delayed_net_1bp'])}"
                    )

    print(f"\nPass rows: {len(pass_rows)}")
    top = sorted(
        pass_rows,
        key=lambda r: (
            r["periods"]["holdout"]["SPY_QQQ_COMBINED"]["net_1bp"] or -999.0,
            r["periods"]["dev"]["SPY_QQQ_COMBINED"]["net_1bp"] or -999.0,
        ),
        reverse=True,
    )[:15]
    for r in top:
        dev = r["periods"]["dev"]
        hold = r["periods"]["holdout"]
        print(
            f"TOP {r['config_id']} {r['direction']} {r['filter_id']} | "
            f"dev combo={_fmt(dev['SPY_QQQ_COMBINED']['net_1bp'])} "
            f"hold combo={_fmt(hold['SPY_QQQ_COMBINED']['net_1bp'])} "
            f"hold delayed={_fmt(hold['SPY_QQQ_COMBINED']['delayed_net_1bp'])} "
            f"hold n={hold['SPY_QQQ_COMBINED']['n']}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--holdout-end", default=None, help="Optional YYYY-MM-DD holdout end")
    args = parser.parse_args()
    run_sweep(Path(args.db), date.fromisoformat(args.holdout_end) if args.holdout_end else None)


if __name__ == "__main__":
    main()
