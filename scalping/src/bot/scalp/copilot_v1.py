"""Chart-parity features and candidate episodes for the Lane A copilot.

This module is intentionally separate from the legacy ``FeatureSnapshot``
pipeline.  SIP bars are stored at their close timestamp.  Every feature below
uses completed RTH bars only, and every simulated entry occurs at the next
real one-minute open.

The output is research evidence for a human-in-the-loop confirmation system.
It never places an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from bot.scalp.racer import TransitionInput, evaluate_transition

ET = "America/New_York"
Direction = Literal["long", "short"]

# These symbol/sessions remained incomplete after a forced Alpaca SIP refetch.
# The affected day is removed.  Candidate generation is also blacked out for
# the next 20 completed sessions so RVOL and short-memory indicators do not
# silently treat a two-day-old observation as yesterday's value.
PROVIDER_GAP_DATES: dict[str, set[date]] = {
    "SPY": {date(2016, 2, 2), date(2019, 8, 12), date(2021, 5, 5), date(2023, 6, 5)},
    "QQQ": {date(2016, 2, 2), date(2016, 2, 22), date(2018, 5, 2), date(2018, 5, 3)},
    "IWM": {date(2016, 2, 2), date(2019, 8, 12), date(2021, 5, 5), date(2023, 6, 5)},
}

NUMERIC_FEATURES = (
    "ema_gap_bps",
    "ema9_slope_bps",
    "ema20_slope_bps",
    "atr5_bps",
    "ret15_bps",
    "structure3",
    "close_vwap_bps",
    "vwap_slope5_bps",
    "vwap_z",
    "rsi3",
    "rsi3_delta",
    "rvol3",
    "ret1_bps",
    "ret3_bps",
    "body_frac",
    "upper_wick_frac",
    "lower_wick_frac",
    "day_return_bps",
    "range_position",
    "range_atr",
    "level_distance_bps",
    "stop_distance_bps",
    "reward_distance_bps",
    "trigger_quality",
    "cross_alignment",
    "cross_vwap_alignment",
    "cross_rsi_alignment",
)

CATEGORICAL_FEATURES = (
    "symbol",
    "family",
    "candidate_side",
    "time_bucket",
    "level_id",
)


def _wilder_rma(values: Iterable[float], period: int) -> np.ndarray:
    """Wilder moving average with the first value seeded by a simple mean."""
    arr = np.asarray(list(values), dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if period < 2 or len(arr) < period:
        return out
    for end in range(period - 1, len(arr)):
        window = arr[end - period + 1 : end + 1]
        if np.isfinite(window).all():
            out[end] = float(window.mean())
            start = end + 1
            break
    else:
        return out
    for i in range(start, len(arr)):
        value = arr[i]
        if not np.isfinite(value):
            out[i] = out[i - 1]
        else:
            out[i] = (out[i - 1] * (period - 1) + value) / period
    return out


def wilder_rsi(values: Iterable[float], period: int = 14) -> np.ndarray:
    """Continuous Wilder RSI, matching completed-bar TradingView semantics."""
    closes = np.asarray(list(values), dtype=float)
    out = np.full(closes.shape, np.nan, dtype=float)
    if period < 2 or len(closes) <= period:
        return out
    delta = np.diff(closes)
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = float(gain[:period].mean())
    avg_loss = float(loss[:period].mean())

    def _value(g: float, loss_value: float) -> float:
        if loss_value == 0:
            return 100.0 if g > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + g / loss_value)

    out[period] = _value(avg_gain, avg_loss)
    for close_idx in range(period + 1, len(closes)):
        delta_idx = close_idx - 1
        avg_gain = (avg_gain * (period - 1) + gain[delta_idx]) / period
        avg_loss = (avg_loss * (period - 1) + loss[delta_idx]) / period
        out[close_idx] = _value(avg_gain, avg_loss)
    return out


def complementary_scores(probability_up: float) -> tuple[float, float]:
    """Return calibrated Long/Short scores that always sum to exactly ten."""
    p = float(np.clip(probability_up, 0.0, 1.0))
    long_score = 10.0 * p
    return long_score, 10.0 - long_score


def _aggregate_rth(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate close-stamped 1m RTH bars into 3m/5m bars anchored at 09:30."""
    work = frame.copy()
    work["slot"] = ((work["minute_index"] - 1) // minutes).astype(int)
    grouped = (
        work.groupby(["session", "slot"], sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            last_real_ts=("ts", "max"),
            bar_count=("ts", "size"),
        )
        .reset_index()
    )
    # A bar's values become available at its scheduled close, even if a real
    # exchange halt leaves fewer than ``minutes`` constituent observations.
    anchors = pd.to_datetime(grouped["session"].astype(str) + " 09:30:00").dt.tz_localize(ET)
    grouped["ts"] = anchors + pd.to_timedelta((grouped["slot"] + 1) * minutes, unit="m")
    return grouped.sort_values("ts").reset_index(drop=True)


def _research_eligibility(sessions: pd.Series, symbol: str, warmout: int = 20) -> pd.Series:
    unique = sorted(pd.unique(sessions))
    allowed = {day: True for day in unique}
    for gap in PROVIDER_GAP_DATES.get(symbol, set()):
        following = [day for day in unique if day > gap][:warmout]
        for day in following:
            allowed[day] = False
    return sessions.map(allowed).fillna(False).astype(bool)


def build_chart_features(raw: pd.DataFrame, symbol: str, rvol_sessions: int = 20) -> pd.DataFrame:
    """Build the exact completed-bar feature contract from raw SIP rows.

    Required columns are ``ts, open, high, low, close, volume``.  ``ts`` may be
    UTC strings or timezone-aware timestamps and represents bar close time.
    Premarket data are used only to establish premarket high/low; all indicator
    calculations are RTH-only.
    """
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if rvol_sessions < 2:
        raise ValueError("rvol_sessions must be >= 2")

    data = raw.loc[:, list(required)].copy()
    data["ts"] = pd.to_datetime(data["ts"], utc=True).dt.tz_convert(ET)
    data = data.sort_values("ts").drop_duplicates("ts", keep="last")
    data["session"] = data["ts"].dt.date
    clock_minutes = data["ts"].dt.hour * 60 + data["ts"].dt.minute

    pm_mask = (clock_minutes > 4 * 60) & (clock_minutes <= 9 * 60 + 30)
    premarket = (
        data.loc[pm_mask]
        .groupby("session", observed=True)
        .agg(premarket_high=("high", "max"), premarket_low=("low", "min"))
    )

    rth_mask = (clock_minutes > 9 * 60 + 30) & (clock_minutes <= 16 * 60)
    rth = data.loc[rth_mask].copy()
    gaps = PROVIDER_GAP_DATES.get(symbol, set())
    if gaps:
        rth = rth.loc[~rth["session"].isin(gaps)].copy()
    rth["symbol"] = symbol
    rth["minute_index"] = (
        rth["ts"].dt.hour * 60 + rth["ts"].dt.minute - (9 * 60 + 30)
    ).astype(int)
    rth = rth.loc[rth["minute_index"] >= 1].reset_index(drop=True)

    typical = (rth["high"] + rth["low"] + rth["close"]) / 3.0
    rth["pv"] = typical * rth["volume"]
    rth["pv2"] = typical.pow(2) * rth["volume"]
    grouped = rth.groupby("session", sort=False, observed=True)
    cumulative_volume = grouped["volume"].cumsum().replace(0, np.nan)
    rth["vwap"] = grouped["pv"].cumsum() / cumulative_volume
    variance = grouped["pv2"].cumsum() / cumulative_volume - rth["vwap"].pow(2)
    rth["vwap_sigma"] = variance.clip(lower=0).pow(0.5)
    rth["session_high"] = grouped["high"].cummax()
    rth["session_low"] = grouped["low"].cummin()
    rth["day_open"] = grouped["open"].transform("first")
    rth["day_return_bps"] = (rth["close"] / rth["day_open"] - 1.0) * 10_000.0
    width = (rth["session_high"] - rth["session_low"]).replace(0, np.nan)
    rth["range_position"] = (rth["close"] - rth["session_low"]) / width
    rth["close_vwap_bps"] = (rth["close"] / rth["vwap"] - 1.0) * 10_000.0
    rth["vwap_z"] = (rth["close"] - rth["vwap"]) / rth["vwap_sigma"].replace(0, np.nan)
    rth["vwap_slope5_bps"] = (
        rth["vwap"] / grouped["vwap"].shift(5) - 1.0
    ) * 10_000.0
    rth["ret1_bps"] = grouped["close"].pct_change() * 10_000.0
    rth["ret3_bps"] = (rth["close"] / grouped["close"].shift(3) - 1.0) * 10_000.0
    candle_range = (rth["high"] - rth["low"]).replace(0, np.nan)
    rth["body_frac"] = (rth["close"] - rth["open"]).abs() / candle_range
    rth["upper_wick_frac"] = (rth["high"] - rth[["open", "close"]].max(axis=1)) / candle_range
    rth["lower_wick_frac"] = (rth[["open", "close"]].min(axis=1) - rth["low"]) / candle_range

    daily = (
        rth.groupby("session", sort=True, observed=True)
        .agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"))
    )
    daily[["prior_high", "prior_low", "prior_close"]] = daily[
        ["day_high", "day_low", "day_close"]
    ].shift(1)
    rth = rth.join(daily[["prior_high", "prior_low", "prior_close"]], on="session")
    rth = rth.join(premarket, on="session")

    opening = rth["minute_index"].le(15)
    or_levels = (
        rth.loc[opening]
        .groupby("session", observed=True)
        .agg(or_high=("high", "max"), or_low=("low", "min"))
    )
    rth = rth.join(or_levels, on="session")
    rth.loc[rth["minute_index"] < 15, ["or_high", "or_low"]] = np.nan

    agg3 = _aggregate_rth(rth, 3)
    agg3["rsi3"] = wilder_rsi(agg3["close"], 14)
    agg3["rsi3_delta"] = pd.Series(agg3["rsi3"]).diff(1).to_numpy()
    agg3["structure3"] = np.select(
        [
            (agg3["high"] > agg3["high"].shift(1)) & (agg3["low"] > agg3["low"].shift(1)),
            (agg3["high"] < agg3["high"].shift(1)) & (agg3["low"] < agg3["low"].shift(1)),
        ],
        [1.0, -1.0],
        default=0.0,
    )
    agg3["slot_median_volume"] = agg3.groupby("slot", observed=True)["volume"].transform(
        lambda values: values.shift(1).rolling(rvol_sessions, min_periods=rvol_sessions).median()
    )
    agg3["rvol3"] = agg3["volume"] / agg3["slot_median_volume"].replace(0, np.nan)

    agg5 = _aggregate_rth(rth, 5)
    agg5["ema9"] = agg5["close"].ewm(span=9, adjust=False).mean()
    agg5["ema20"] = agg5["close"].ewm(span=20, adjust=False).mean()
    agg5["ema_gap_bps"] = (agg5["ema9"] / agg5["ema20"] - 1.0) * 10_000.0
    agg5["ema9_slope_bps"] = agg5["ema9"].pct_change() * 10_000.0
    agg5["ema20_slope_bps"] = agg5["ema20"].pct_change() * 10_000.0
    agg5["ret15_bps"] = agg5["close"].pct_change(3) * 10_000.0
    previous_close = agg5["close"].shift(1)
    true_range = pd.concat(
        [
            agg5["high"] - agg5["low"],
            (agg5["high"] - previous_close).abs(),
            (agg5["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    agg5["atr5"] = _wilder_rma(true_range, 14)
    agg5["atr5_bps"] = agg5["atr5"] / agg5["close"] * 10_000.0
    agg5["structure5"] = np.select(
        [
            (agg5["high"] > agg5["high"].shift(1)) & (agg5["low"] > agg5["low"].shift(1)),
            (agg5["high"] < agg5["high"].shift(1)) & (agg5["low"] < agg5["low"].shift(1)),
        ],
        [1.0, -1.0],
        default=0.0,
    )

    base = rth.sort_values(["session", "ts"]).reset_index(drop=True)
    base = pd.merge_asof(
        base.sort_values("ts"),
        agg3[["ts", "session", "rsi3", "rsi3_delta", "rvol3", "structure3"]].sort_values("ts"),
        on="ts",
        by="session",
        direction="backward",
        allow_exact_matches=True,
    )
    base = pd.merge_asof(
        base.sort_values("ts"),
        agg5[
            [
                "ts",
                "session",
                "ema9",
                "ema20",
                "ema_gap_bps",
                "ema9_slope_bps",
                "ema20_slope_bps",
                "ret15_bps",
                "atr5",
                "atr5_bps",
                "structure5",
            ]
        ].sort_values("ts"),
        on="ts",
        by="session",
        direction="backward",
        allow_exact_matches=True,
    )
    base["range_atr"] = (base["session_high"] - base["session_low"]) / base["atr5"].replace(0, np.nan)
    base["research_eligible"] = _research_eligibility(base["session"], symbol)
    base["bar_open_ts"] = base["ts"] - pd.Timedelta(minutes=1)
    base["bar_close_ts"] = base["ts"]
    return base.drop(columns=["pv", "pv2"], errors="ignore").sort_values("ts").reset_index(drop=True)


@dataclass
class _SweepState:
    armed_index: int
    extreme: float


@dataclass
class _CompressionState:
    armed_index: int
    low: float
    high: float
    breakout_index: int | None = None
    side: Direction | None = None


def _simulate_frozen_plan(
    session: pd.DataFrame,
    entry_i: int,
    hold: int,
    *,
    side: Direction,
    stop_price: float,
    t1_r: float = 1.5,
    t2_r: float = 2.0,
) -> dict | None:
    """Resolve a predeclared stop/T1/T2/time plan from completed 1m OHLC.

    When a stop and target are both touched in the same one-minute bar, the
    stop is assumed to occur first.  That deliberately conservative convention
    avoids manufacturing an intrabar path the source data cannot prove.
    """
    if entry_i >= len(session) or hold < 1:
        return None
    entry_price = float(session.iloc[entry_i]["open"])
    sign = 1.0 if side == "long" else -1.0
    risk_price = sign * (entry_price - float(stop_price))
    if not np.isfinite(risk_price) or risk_price <= 0.0:
        return None

    target_1 = entry_price + sign * risk_price * float(t1_r)
    target_2 = entry_price + sign * risk_price * float(t2_r)
    window = session.iloc[entry_i : min(entry_i + hold, len(session))]
    if window.empty:
        return None

    if side == "long":
        mfe_bps = (float(window["high"].max()) / entry_price - 1.0) * 10_000.0
        mae_bps = (float(window["low"].min()) / entry_price - 1.0) * 10_000.0
    else:
        mfe_bps = (1.0 - float(window["low"].min()) / entry_price) * 10_000.0
        mae_bps = (1.0 - float(window["high"].max()) / entry_price) * 10_000.0

    first_leg_price: float | None = None
    effective_exit = float(window.iloc[-1]["close"])
    exit_reason = "time"
    exit_row = window.iloc[-1]

    for _, bar in window.iterrows():
        stop_hit = float(bar["low"]) <= stop_price if side == "long" else float(bar["high"]) >= stop_price
        t1_hit = float(bar["high"]) >= target_1 if side == "long" else float(bar["low"]) <= target_1
        t2_hit = float(bar["high"]) >= target_2 if side == "long" else float(bar["low"]) <= target_2

        if stop_hit:
            effective_exit = (
                float(stop_price)
                if first_leg_price is None
                else 0.5 * first_leg_price + 0.5 * float(stop_price)
            )
            exit_reason = "stop" if first_leg_price is None else "t1_then_stop"
            exit_row = bar
            break
        if first_leg_price is None and t1_hit:
            first_leg_price = target_1
        if first_leg_price is not None and t2_hit:
            effective_exit = 0.5 * first_leg_price + 0.5 * target_2
            exit_reason = "t2"
            exit_row = bar
            break
    else:
        if first_leg_price is not None:
            effective_exit = 0.5 * first_leg_price + 0.5 * effective_exit
            exit_reason = "t1_then_time"

    gross_bps = sign * (effective_exit / entry_price - 1.0) * 10_000.0
    risk_bps = risk_price / entry_price * 10_000.0
    return {
        "entry_price": entry_price,
        "target_1_price": float(target_1),
        "target_2_price": float(target_2),
        "effective_exit_price": float(effective_exit),
        "exit_ts": exit_row["bar_close_ts"],
        "exit_reason": exit_reason,
        "gross_bps": float(gross_bps),
        "outcome_r": float(gross_bps / risk_bps),
        "mfe_bps": float(mfe_bps),
        "mae_bps": float(mae_bps),
    }


def _clock_bucket(minute_index: int) -> str:
    if minute_index <= 60:
        return "open"
    if minute_index <= 180:
        return "morning"
    if minute_index <= 300:
        return "lunch"
    if minute_index <= 360:
        return "power_hour"
    return "close"


def _event(
    session: pd.DataFrame,
    i: int,
    *,
    family: str,
    side: Direction,
    level_id: str,
    level_price: float,
    stop_price: float,
    trigger_quality: float,
) -> dict | None:
    """Freeze a point-in-time candidate and its next-open fill classification.

    Candidate existence is decided entirely from the completed decision bar.
    The next bar may invalidate the planned entry by opening through the frozen
    structural stop, but that future observation must not erase the candidate.
    Such rows remain in the coverage audit and carry no realized/model labels.
    """
    if i + 1 >= len(session):
        return None
    row = session.iloc[i]
    remaining = len(session) - (i + 1)
    hold = min(15, remaining)
    if hold < 1:
        return None
    entry_i = i + 1
    decision_price = float(row["close"])
    sign = 1.0 if side == "long" else -1.0
    decision_risk = sign * (decision_price - float(stop_price))
    if not np.isfinite(decision_risk) or decision_risk <= 0.0:
        # Invalid geometry is knowable at the decision close and may therefore
        # suppress the candidate without introducing future information.
        return None

    observed_next_open = float(session.iloc[entry_i]["open"])
    gap_invalidated = (
        not np.isfinite(observed_next_open)
        or sign * (observed_next_open - float(stop_price)) <= 0.0
    )
    plan = None if gap_invalidated else _simulate_frozen_plan(
        session,
        entry_i,
        hold,
        side=side,
        stop_price=float(stop_price),
    )
    realized_eligible = plan is not None
    fill_status = "filled" if realized_eligible else "gap_invalidated_unfilled"

    entry = float(plan["entry_price"]) if plan is not None else np.nan
    future_bps = (
        (float(session.iloc[entry_i + hold - 1]["close"]) / entry - 1.0) * 10_000.0
        if plan is not None
        else np.nan
    )
    gross_bps = float(plan["gross_bps"]) if plan is not None else np.nan

    delayed_gross = np.nan
    if plan is not None and i + 2 < len(session):
        delayed_entry_i = i + 2
        delayed_hold = min(hold, len(session) - delayed_entry_i)
        if delayed_hold >= 5:
            delayed_plan = _simulate_frozen_plan(
                session,
                delayed_entry_i,
                delayed_hold,
                side=side,
                stop_price=float(stop_price),
            )
            if delayed_plan is not None:
                delayed_gross = float(delayed_plan["gross_bps"])

    # Planned geometry is a decision-time feature.  The next open is outcome
    # data and must never leak into a score input.
    stop_distance = (
        abs(decision_price - stop_price) / decision_price * 10_000.0
        if stop_price > 0
        else np.nan
    )
    known_levels = [
        row.get("prior_high"),
        row.get("prior_low"),
        row.get("premarket_high"),
        row.get("premarket_low"),
        row.get("or_high"),
        row.get("or_low"),
    ]
    valid_levels = [float(value) for value in known_levels if pd.notna(value)]
    if side == "long":
        ahead = [value for value in valid_levels if value > decision_price]
        reward = (
            (min(ahead) / decision_price - 1.0) * 10_000.0
            if ahead
            else float(row["atr5_bps"])
        )
    else:
        ahead = [value for value in valid_levels if value < decision_price]
        reward = (
            (1.0 - max(ahead) / decision_price) * 10_000.0
            if ahead
            else float(row["atr5_bps"])
        )

    result = {
        "opportunity_id": f"{row['symbol']}|{row['session']}|{row['bar_close_ts'].isoformat()}|{family}|{side}",
        "symbol": row["symbol"],
        "session": row["session"],
        "decision_ts": row["bar_close_ts"],
        "entry_ts": session.iloc[entry_i]["bar_open_ts"],
        "family": family,
        "candidate_side": side,
        "time_bucket": _clock_bucket(int(row["minute_index"])),
        "level_id": level_id,
        "level_price": float(level_price),
        "plan_hold_minutes": int(hold),
        "fill_status": fill_status,
        "realized_eligible": bool(realized_eligible),
        "observed_next_open_price": observed_next_open,
        "entry_price": entry,
        "stop_price": float(stop_price),
        "target_1_price": plan["target_1_price"] if plan is not None else np.nan,
        "target_2_price": plan["target_2_price"] if plan is not None else np.nan,
        "effective_exit_price": plan["effective_exit_price"] if plan is not None else np.nan,
        "exit_ts": plan["exit_ts"] if plan is not None else pd.NaT,
        "exit_reason": plan["exit_reason"] if plan is not None else fill_status,
        "candidate_outcome_r": plan["outcome_r"] if plan is not None else np.nan,
        "candidate_mfe_bps": plan["mfe_bps"] if plan is not None else np.nan,
        "candidate_mae_bps": plan["mae_bps"] if plan is not None else np.nan,
        "level_distance_bps": abs(float(row["close"]) / level_price - 1.0) * 10_000.0 if level_price else 0.0,
        "stop_distance_bps": stop_distance,
        "reward_distance_bps": reward,
        "trigger_quality": float(trigger_quality),
        "future_up": int(future_bps > 0.0) if np.isfinite(future_bps) else np.nan,
        "future_bps": future_bps,
        "candidate_gross_bps": gross_bps,
        "candidate_net_bps_1bp": gross_bps - 2.0 if np.isfinite(gross_bps) else np.nan,
        "candidate_net_win_1bp": int(gross_bps > 2.0) if np.isfinite(gross_bps) else np.nan,
        "delayed_candidate_gross_bps": delayed_gross,
        "delayed_candidate_net_bps_1bp": delayed_gross - 2.0 if np.isfinite(delayed_gross) else np.nan,
    }
    # Every deterministic setup episode is also a RACER transition.  The
    # state evaluator receives only completed observations from this decision
    # bar; the next open above is outcome data and is intentionally absent.
    completed_5m = all(
        np.isfinite(float(row.get(name, np.nan)))
        for name in ("ema_gap_bps", "ema9_slope_bps", "ema20_slope_bps", "close_vwap_bps")
    )
    completed_3m = all(np.isfinite(float(row.get(name, np.nan))) for name in ("rsi3", "rvol3"))
    racer_state = evaluate_transition(
        TransitionInput(
            symbol=str(row["symbol"]),
            decision_timestamp=row["bar_close_ts"],
            bar_close_timestamp=row["bar_close_ts"],
            transition_geometry=family,  # type: ignore[arg-type]
            candidate_direction=side,
            provider_eligible=bool(row.get("research_eligible", False)),
            completed_5m=completed_5m,
            completed_3m=completed_3m,
            completed_1m_trigger=True,
            frozen_risk_valid=True,
            minute_index=int(row["minute_index"]),
            ema_gap_bps=float(row.get("ema_gap_bps", np.nan)),
            close_vwap_bps=float(row.get("close_vwap_bps", np.nan)),
            rsi3=float(row.get("rsi3", np.nan)),
            rvol3=float(row.get("rvol3", np.nan)),
            day_return_bps=float(row.get("day_return_bps", np.nan)),
            # Historical consensus/reaction timestamps are not available in
            # the SIP corpus.  The first RACER slice is event-neutral by
            # construction; a future known-result prior must be provenance
            # complete and still cannot bypass this completed trigger.
            event_reaction_state="event_neutral_unknown",
            historical_30s_available=False,
        )
    )
    result.update(racer_state.as_record())
    result["racer_candidate_side_sign"] = 1.0 if side == "long" else -1.0
    for name in NUMERIC_FEATURES:
        if name in result:
            continue
        result[name] = row.get(name, np.nan)
    return result


def generate_candidate_events(features: pd.DataFrame) -> pd.DataFrame:
    """Generate one observation per completed chart-visible setup episode."""
    events: list[dict] = []
    for _day, session in features.groupby("session", sort=True, observed=True):
        session = session.sort_values("ts").reset_index(drop=True)
        if session.empty or not bool(session["research_eligible"].iloc[-1]):
            continue
        if session["rvol3"].notna().sum() == 0:
            continue

        eod_active: dict[tuple[str, Direction], bool] = {
            (family, side): False
            for family in ("eod_continuation", "eod_profit_taking_reversal")
            for side in ("long", "short")
        }
        sweep_states: dict[tuple[str, Direction], _SweepState] = {}
        trend_armed: dict[Direction, int | None] = {"long": None, "short": None}
        vwap_cross: dict[Direction, int | None] = {"long": None, "short": None}
        compression: _CompressionState | None = None

        for i in range(2, len(session)):
            row = session.iloc[i]
            prev = session.iloc[i - 1]
            minute = int(row["minute_index"])
            if not np.isfinite(row.get("atr5", np.nan)) or not np.isfinite(row.get("rsi3", np.nan)):
                continue

            # 1. Failed known-level sweep -> later failed retest.
            level_specs = (
                ("PDH", row.get("prior_high"), "short"),
                ("PMH", row.get("premarket_high"), "short"),
                ("ORH", row.get("or_high"), "short"),
                ("PDL", row.get("prior_low"), "long"),
                ("PML", row.get("premarket_low"), "long"),
                ("ORL", row.get("or_low"), "long"),
            )
            for level_id, raw_level, raw_side in level_specs:
                side: Direction = raw_side  # type: ignore[assignment]
                if pd.isna(raw_level) or ("OR" in level_id and minute < 16):
                    continue
                level = float(raw_level)
                key = (level_id, side)
                state = sweep_states.get(key)
                if state is not None:
                    if i - state.armed_index > 6:
                        sweep_states.pop(key, None)
                    elif side == "short" and row["close"] > level * 1.0004:
                        sweep_states.pop(key, None)
                    elif side == "long" and row["close"] < level * 0.9996:
                        sweep_states.pop(key, None)
                    elif i > state.armed_index:
                        if side == "short":
                            trigger = (
                                row["high"] >= level * 0.9997
                                and row["close"] < level
                                and row["close"] < row["open"]
                                and row["close"] < prev["low"]
                            )
                            stop = max(state.extreme, float(row["high"]))
                            quality = max(0.0, (level - float(row["close"])) / float(row["atr5"]))
                        else:
                            trigger = (
                                row["low"] <= level * 1.0003
                                and row["close"] > level
                                and row["close"] > row["open"]
                                and row["close"] > prev["high"]
                            )
                            stop = min(state.extreme, float(row["low"]))
                            quality = max(0.0, (float(row["close"]) - level) / float(row["atr5"]))
                        if trigger:
                            event = _event(
                                session,
                                i,
                                family="failed_level_retest",
                                side=side,
                                level_id=level_id,
                                level_price=level,
                                stop_price=stop,
                                trigger_quality=quality,
                            )
                            if event:
                                events.append(event)
                            sweep_states.pop(key, None)
                            continue
                if side == "short" and row["high"] >= level * 1.0002 and row["close"] < level:
                    sweep_states[key] = _SweepState(i, float(row["high"]))
                elif side == "long" and row["low"] <= level * 0.9998 and row["close"] > level:
                    sweep_states[key] = _SweepState(i, float(row["low"]))

            # 2. First trend pullback and completed 1m resumption.
            long_regime = (
                row["ema_gap_bps"] > 1.0
                and row["ema9_slope_bps"] > 0
                and row["ema20_slope_bps"] >= 0
                and row["close_vwap_bps"] > 0
                and row["ret15_bps"] > 5
            )
            short_regime = (
                row["ema_gap_bps"] < -1.0
                and row["ema9_slope_bps"] < 0
                and row["ema20_slope_bps"] <= 0
                and row["close_vwap_bps"] < 0
                and row["ret15_bps"] < -5
            )
            near_long_support = row["low"] <= max(row["ema9"], row["vwap"]) * 1.0004
            near_short_resistance = row["high"] >= min(row["ema9"], row["vwap"]) * 0.9996
            if long_regime and near_long_support and row["close"] <= prev["close"]:
                trend_armed["long"] = i
            if short_regime and near_short_resistance and row["close"] >= prev["close"]:
                trend_armed["short"] = i
            for side, regime in (("long", long_regime), ("short", short_regime)):
                armed = trend_armed[side]
                if armed is None:
                    continue
                if i - armed > 5 or not regime:
                    trend_armed[side] = None
                    continue
                if i == armed:
                    continue
                if side == "long":
                    trigger = row["close"] > row["open"] and row["close"] > prev["high"] and row["close"] > row["ema9"]
                    stop = float(session.iloc[armed : i + 1]["low"].min())
                else:
                    trigger = row["close"] < row["open"] and row["close"] < prev["low"] and row["close"] < row["ema9"]
                    stop = float(session.iloc[armed : i + 1]["high"].max())
                if trigger:
                    quality = abs(float(row["close"] - prev["close"])) / float(row["atr5"])
                    event = _event(
                        session,
                        i,
                        family="first_trend_pullback",
                        side=side,
                        level_id="EMA9/VWAP",
                        level_price=float(row["ema9"]),
                        stop_price=stop,
                        trigger_quality=quality,
                    )
                    if event:
                        events.append(event)
                    trend_armed[side] = None

            # 3. VWAP cross, then the first held retest/resumption.
            if prev["close"] <= prev["vwap"] * 0.9998 and row["close"] >= row["vwap"] * 1.0002:
                vwap_cross["long"] = i
                vwap_cross["short"] = None
            if prev["close"] >= prev["vwap"] * 1.0002 and row["close"] <= row["vwap"] * 0.9998:
                vwap_cross["short"] = i
                vwap_cross["long"] = None
            for side in ("long", "short"):
                crossed = vwap_cross[side]
                if crossed is None or i == crossed:
                    continue
                if i - crossed > 6:
                    vwap_cross[side] = None
                    continue
                if side == "long":
                    invalid = row["close"] < row["vwap"] * 0.9997
                    trigger = row["low"] <= row["vwap"] * 1.0003 and row["close"] > row["vwap"] and row["close"] > prev["high"]
                    stop = float(session.iloc[crossed : i + 1]["low"].min())
                else:
                    invalid = row["close"] > row["vwap"] * 1.0003
                    trigger = row["high"] >= row["vwap"] * 0.9997 and row["close"] < row["vwap"] and row["close"] < prev["low"]
                    stop = float(session.iloc[crossed : i + 1]["high"].max())
                if invalid:
                    vwap_cross[side] = None
                elif trigger:
                    event = _event(
                        session,
                        i,
                        family="vwap_reclaim_reject",
                        side=side,
                        level_id="VWAP",
                        level_price=float(row["vwap"]),
                        stop_price=stop,
                        trigger_quality=abs(float(row["close"] - row["vwap"])) / float(row["atr5"]),
                    )
                    if event:
                        events.append(event)
                    vwap_cross[side] = None

            # 4. Compression -> expansion -> first held retest.
            if compression is not None:
                if compression.breakout_index is None and i - compression.armed_index <= 12:
                    if row["close"] > compression.high * 1.0002:
                        compression.breakout_index, compression.side = i, "long"
                    elif row["close"] < compression.low * 0.9998:
                        compression.breakout_index, compression.side = i, "short"
                elif compression.breakout_index is not None:
                    if i - compression.breakout_index > 6:
                        compression = None
                    elif i > compression.breakout_index:
                        side = compression.side
                        if side == "long":
                            trigger = row["low"] <= compression.high * 1.0004 and row["close"] > compression.high and row["close"] > prev["high"]
                            invalid = row["close"] < compression.low
                            level = compression.high
                            stop = min(compression.low, float(row["low"]))
                        else:
                            trigger = row["high"] >= compression.low * 0.9996 and row["close"] < compression.low and row["close"] < prev["low"]
                            invalid = row["close"] > compression.high
                            level = compression.low
                            stop = max(compression.high, float(row["high"]))
                        if invalid:
                            compression = None
                        elif trigger and side is not None:
                            event = _event(
                                session,
                                i,
                                family="compression_expansion_retest",
                                side=side,
                                level_id="compression_edge",
                                level_price=float(level),
                                stop_price=float(stop),
                                trigger_quality=abs(float(row["close"] / level - 1.0)) * 10_000.0 / max(float(row["atr5_bps"]), 1e-9),
                            )
                            if event:
                                events.append(event)
                            compression = None
                elif i - compression.armed_index > 12:
                    compression = None
            if compression is None and i >= 10:
                prior_window = session.iloc[i - 10 : i]
                box_high = float(prior_window["high"].max())
                box_low = float(prior_window["low"].min())
                if box_high - box_low <= 0.70 * float(row["atr5"]):
                    compression = _CompressionState(i, box_low, box_high)

            # 5. EOD continuation and the owner's profit-taking reversal hypothesis.
            # Both use a frozen 15:00 -> 15:30 state. Continuation still needs
            # a fresh pullback/resumption; reversal still needs a known-level
            # close-back-inside and completed 3m structure flip.
            if 360 <= minute <= 389:
                last10 = session.iloc[max(0, i - 10) : i + 1]
                at_1500 = session.loc[session["minute_index"].eq(330)]
                at_1530 = session.loc[session["minute_index"].eq(360)]
                if at_1500.empty or at_1530.empty:
                    continue
                move_1500_1530 = (
                    float(at_1530.iloc[0]["close"]) / float(at_1500.iloc[0]["close"]) - 1.0
                ) * 10_000.0
                day_move = float(row["day_return_bps"])
                pullback_long = (
                    float(last10["low"].min()) <= max(float(row["ema9"]), float(row["vwap"])) * 1.0005
                    and row["close"] > row["open"]
                    and row["close"] > prev["high"]
                )
                pullback_short = (
                    float(last10["high"].max()) >= min(float(row["ema9"]), float(row["vwap"])) * 0.9995
                    and row["close"] < row["open"]
                    and row["close"] < prev["low"]
                )
                continuation_signals = {
                    "long": bool(
                        day_move >= 50
                        and move_1500_1530 >= 8
                        and row["close_vwap_bps"] > 0
                        and row["ema_gap_bps"] > 0
                        and pullback_long
                    ),
                    "short": bool(
                        day_move <= -50
                        and move_1500_1530 <= -8
                        and row["close_vwap_bps"] < 0
                        and row["ema_gap_bps"] < 0
                        and pullback_short
                    ),
                }
                for continuation_side, signal_active in continuation_signals.items():
                    side: Direction = continuation_side  # type: ignore[assignment]
                    key = ("eod_continuation", side)
                    if not signal_active:
                        eod_active[key] = False
                        continue
                    if eod_active[key]:
                        continue
                    eod_active[key] = True
                    stop = float(last10["low"].min() if side == "long" else last10["high"].max())
                    event = _event(
                        session,
                        i,
                        family="eod_continuation",
                        side=side,
                        level_id="session_extreme",
                        level_price=float(row["session_high"] if side == "long" else row["session_low"]),
                        stop_price=stop,
                        trigger_quality=abs(move_1500_1530) / max(float(row["atr5_bps"]), 1e-9),
                    )
                    if event:
                        events.append(event)

                upper_levels = [
                    float(value)
                    for value in (row.get("prior_high"), row.get("premarket_high"), row.get("or_high"))
                    if pd.notna(value)
                ]
                lower_levels = [
                    float(value)
                    for value in (row.get("prior_low"), row.get("premarket_low"), row.get("or_low"))
                    if pd.notna(value)
                ]
                rejected_upper = [
                    level
                    for level in upper_levels
                    if float(last10["high"].max()) >= level and float(row["close"]) < level
                ]
                rejected_lower = [
                    level
                    for level in lower_levels
                    if float(last10["low"].min()) <= level and float(row["close"]) > level
                ]
                high_reject = (
                    day_move >= 70
                    and bool(rejected_upper)
                    and row["structure3"] < 0
                    and row["close"] < prev["low"]
                    and row["close"] < row["open"]
                )
                low_reject = (
                    day_move <= -70
                    and bool(rejected_lower)
                    and row["structure3"] > 0
                    and row["close"] > prev["high"]
                    and row["close"] > row["open"]
                )
                reversal_signals = {"short": bool(high_reject), "long": bool(low_reject)}
                for reversal_side, signal_active in reversal_signals.items():
                    reverse_side: Direction = reversal_side  # type: ignore[assignment]
                    key = ("eod_profit_taking_reversal", reverse_side)
                    if not signal_active:
                        eod_active[key] = False
                        continue
                    if eod_active[key]:
                        continue
                    eod_active[key] = True
                    stop = float(last10["high"].max() if reverse_side == "short" else last10["low"].min())
                    level = (
                        min(rejected_upper, key=lambda value: abs(value - float(row["close"])))
                        if reverse_side == "short"
                        else min(rejected_lower, key=lambda value: abs(value - float(row["close"])))
                    )
                    event = _event(
                        session,
                        i,
                        family="eod_profit_taking_reversal",
                        side=reverse_side,
                        level_id="session_extreme",
                        level_price=level,
                        stop_price=stop,
                        trigger_quality=abs(float(row["close"] / prev["close"] - 1.0)) * 10_000.0 / max(float(row["atr5_bps"]), 1e-9),
                    )
                    if event:
                        events.append(event)

    return pd.DataFrame(events)


def collapse_simultaneous_candidates(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-side labels and discard opposite-side timestamp conflicts."""
    if events.empty:
        return events.copy()
    rows: list[pd.Series] = []
    for (_symbol, _ts), group in events.groupby(["symbol", "decision_ts"], sort=True):
        sides = set(group["candidate_side"])
        if len(sides) != 1:
            continue
        chosen = group.sort_values(["trigger_quality", "family"], ascending=[False, True]).iloc[0].copy()
        # The selected row remains the canonical typed transition. Other
        # same-side geometries are provenance, not a synthetic strategy label
        # that can contradict its frozen reason, level, and risk plan.
        chosen["observed_geometries"] = "+".join(sorted(set(group["family"])))
        chosen["observed_level_ids"] = "+".join(sorted(set(group["level_id"])))
        chosen["opportunity_id"] = f"{chosen['symbol']}|{chosen['session']}|{chosen['decision_ts'].isoformat()}|collapsed|{chosen['candidate_side']}"
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)
