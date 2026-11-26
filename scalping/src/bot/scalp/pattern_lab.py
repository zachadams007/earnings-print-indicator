"""Point-in-time helpers for broad intraday-pattern research.

The module deliberately produces research signals, not live orders.  Database
bars are close-stamped: a signal at index ``i`` may use bars through ``i`` and
is filled no earlier than bar ``i + 1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from statistics import mean, median
from typing import Literal, Sequence
from zoneinfo import ZoneInfo

from ..data.models import Bar

ET = ZoneInfo("America/New_York")
Direction = Literal["long", "short"]


@dataclass(frozen=True)
class SessionPoint:
    index: int
    clock: time
    close: float
    vwap: float
    sigma: float
    rsi: float | None
    rsi_prev: float | None
    session_high: float
    session_low: float
    above_vwap_share: float
    below_vwap_share: float
    volume_ratio: float | None


@dataclass(frozen=True)
class ResearchSignal:
    family: str
    config_id: str
    symbol: str
    day: date
    signal_index: int
    signal_time: time
    direction: Direction
    hold_minutes: int
    rsi_variant: str = "none"


@dataclass(frozen=True)
class ResearchTrade:
    family: str
    config_id: str
    symbol: str
    day: date
    signal_time: time
    direction: Direction
    hold_minutes: int
    gross_bps: float
    rsi_variant: str

    def net_bps(self, cost_bps_per_side: float = 1.0) -> float:
        return self.gross_bps - 2.0 * cost_bps_per_side


@dataclass(frozen=True)
class Summary:
    n: int
    win_rate: float | None
    gross_bps: float | None
    net_bps: float | None
    median_bps: float | None


def to_et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=ET)
    return ts.astimezone(ET)


def wilder_rsi(closes: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder RSI using only closes available at each output position."""
    if period < 2:
        raise ValueError("period must be >= 2")
    out: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def value() -> float:
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = value()
    for close_idx in range(period + 1, len(closes)):
        delta_idx = close_idx - 1
        avg_gain = (avg_gain * (period - 1) + gains[delta_idx]) / period
        avg_loss = (avg_loss * (period - 1) + losses[delta_idx]) / period
        out[close_idx] = value()
    return out


def build_session_points(bars: Sequence[Bar], rsi_period: int = 14) -> list[SessionPoint]:
    ordered = sorted(bars, key=lambda b: b.ts)
    closes = [float(b.close) for b in ordered]
    rsis = wilder_rsi(closes, rsi_period)
    points: list[SessionPoint] = []
    pv = pv2 = vol = 0.0
    above = below = 0
    session_high = float("-inf")
    session_low = float("inf")
    volumes: list[float] = []
    for i, b in enumerate(ordered):
        tp = (b.high + b.low + b.close) / 3.0
        pv += tp * b.volume
        pv2 += tp * tp * b.volume
        vol += b.volume
        vwap = pv / vol if vol else b.close
        variance = max(pv2 / vol - vwap * vwap, 0.0) if vol else 0.0
        sigma = variance**0.5
        if b.close > vwap:
            above += 1
        elif b.close < vwap:
            below += 1
        decisive = above + below
        session_high = max(session_high, b.high)
        session_low = min(session_low, b.low)
        volume_ratio = None
        if len(volumes) >= 20 and mean(volumes[-20:]) > 0:
            volume_ratio = b.volume / mean(volumes[-20:])
        volumes.append(float(b.volume))
        points.append(
            SessionPoint(
                index=i,
                clock=to_et(b.ts).time(),
                close=b.close,
                vwap=vwap,
                sigma=sigma,
                rsi=rsis[i],
                rsi_prev=rsis[i - 1] if i else None,
                session_high=session_high,
                session_low=session_low,
                above_vwap_share=(above / decisive) if decisive else 0.5,
                below_vwap_share=(below / decisive) if decisive else 0.5,
                volume_ratio=volume_ratio,
            )
        )
    return points


def rsi_allows(point: SessionPoint, direction: Direction, variant: str) -> bool:
    if variant == "none":
        return True
    if point.rsi is None:
        return False
    if variant.startswith("extreme"):
        threshold = float(variant.removeprefix("extreme"))
        return point.rsi <= 100.0 - threshold if direction == "long" else point.rsi >= threshold
    if variant.startswith("hook"):
        threshold = float(variant.removeprefix("hook"))
        if point.rsi_prev is None:
            return False
        if direction == "long":
            floor = 100.0 - threshold
            return point.rsi_prev <= floor < point.rsi
        return point.rsi_prev >= threshold > point.rsi
    if variant == "momentum50":
        return point.rsi >= 50.0 if direction == "long" else point.rsi <= 50.0
    raise ValueError(f"unknown RSI variant: {variant}")


def evaluate_signal(
    signal: ResearchSignal,
    bars: Sequence[Bar],
    *,
    extra_delay_minutes: int = 0,
) -> ResearchTrade | None:
    ordered = sorted(bars, key=lambda b: b.ts)
    entry_idx = signal.signal_index + 1 + extra_delay_minutes
    if entry_idx >= len(ordered):
        return None
    exit_idx = min(entry_idx + signal.hold_minutes, len(ordered) - 1)
    if exit_idx <= entry_idx:
        return None
    entry = ordered[entry_idx].open
    exit_price = ordered[exit_idx].close
    sign = 1.0 if signal.direction == "long" else -1.0
    gross = sign * (exit_price - entry) / entry * 10_000.0
    return ResearchTrade(
        family=signal.family,
        config_id=signal.config_id,
        symbol=signal.symbol,
        day=signal.day,
        signal_time=signal.signal_time,
        direction=signal.direction,
        hold_minutes=signal.hold_minutes,
        gross_bps=gross,
        rsi_variant=signal.rsi_variant,
    )


def summarize(trades: Sequence[ResearchTrade], cost_bps_per_side: float = 1.0) -> Summary:
    if not trades:
        return Summary(0, None, None, None, None)
    vals = [t.gross_bps for t in trades]
    return Summary(
        n=len(vals),
        win_rate=sum(v > 0 for v in vals) / len(vals),
        gross_bps=mean(vals),
        net_bps=mean(t.net_bps(cost_bps_per_side) for t in trades),
        median_bps=median(vals),
    )
