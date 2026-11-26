"""Exit engine tests: stop/target/time-stop/EOD simulation, and the explicit
conservative "stop wins if both are in the same bar's range" assumption."""
from __future__ import annotations

from datetime import date, timedelta

from bot.scalp.exits import (
    ExitConfig,
    ExitConfigV3,
    atr_from_agg_bars,
    compute_stop_and_target,
    compute_stop_and_target_v3,
    simulate_exit,
    swing_level_from_agg_bars,
)
from bot.scalp.features import AggBar
from tests.scalp.helpers import bar, session_open

DAY = date(2026, 6, 1)


def _bars(closes: list[float], highs=None, lows=None) -> list:
    start = session_open(DAY)
    highs = highs or [c + 0.1 for c in closes]
    lows = lows or [c - 0.1 for c in closes]
    out = []
    prev_close = closes[0]
    for i, c in enumerate(closes):
        out.append(bar("SPY", start + timedelta(minutes=i), prev_close, highs[i], lows[i], c))
        prev_close = c
    return out


def test_target_hit_long():
    cfg = ExitConfig(stop_type="entry_extreme", target_r=1.5, time_stop_minutes=30)
    closes = [100.0, 100.2, 100.5, 101.0, 101.6]  # entry at bar0 open=100.0
    highs = [100.1, 100.3, 100.6, 101.1, 101.7]
    lows = [99.9, 100.1, 100.4, 100.9, 101.5]
    bars = _bars(closes, highs, lows)
    # entry_price = bars[0].open = 100.0; stop = 99.0 (risk=1.0); target(1.5R)=101.5
    result = simulate_exit(bars, entry_idx=0, direction_sign=1.0, entry_price=100.0, stop_price=99.0, target_price=101.5, cfg=cfg)
    assert result is not None
    assert result.reason == "target"
    assert result.r_multiple > 0
    assert abs(result.r_multiple - 1.5) < 1e-9


def test_stop_hit_short():
    cfg = ExitConfig(stop_type="entry_extreme", target_r=1.5, time_stop_minutes=30)
    closes = [100.0, 100.5, 101.2]  # short trade: price rises against us
    highs = [100.6, 101.0, 101.5]
    lows = [99.9, 100.3, 101.0]
    bars = _bars(closes, highs, lows)
    # short entry at 100.0, stop = 101.0 (risk=1.0), target(1.5R) = 98.5
    result = simulate_exit(bars, entry_idx=0, direction_sign=-1.0, entry_price=100.0, stop_price=101.0, target_price=98.5, cfg=cfg)
    assert result is not None
    assert result.reason == "stop"
    assert abs(result.r_multiple - (-1.0)) < 1e-9


def test_conservative_ordering_both_stop_and_target_in_same_bar_counts_as_stop():
    """Explicit documented assumption: if one bar's [low, high] contains BOTH
    the stop and the target, this is scored as a STOP, never a target hit."""
    cfg = ExitConfig(stop_type="entry_extreme", target_r=1.5, time_stop_minutes=30)
    entry_price = 100.0
    stop_price = 99.0
    target_price = 101.5
    # a single wide bar whose range covers both levels
    wild_bar = [bar("SPY", session_open(DAY), 100.0, 102.0, 98.5, 100.5)]
    result = simulate_exit(wild_bar, entry_idx=0, direction_sign=1.0, entry_price=entry_price, stop_price=stop_price, target_price=target_price, cfg=cfg)
    assert result is not None
    assert result.reason == "stop"
    assert abs(result.r_multiple - (-1.0)) < 1e-9


def test_time_stop_exits_at_close_when_neither_hit():
    cfg = ExitConfig(stop_type="entry_extreme", target_r=1.5, time_stop_minutes=2)
    closes = [100.0, 100.1, 100.2, 100.3, 100.4]  # drifts up slowly, never reaches target/stop
    bars = _bars(closes)
    result = simulate_exit(bars, entry_idx=0, direction_sign=1.0, entry_price=100.0, stop_price=95.0, target_price=110.0, cfg=cfg)
    assert result is not None
    assert result.reason == "time_stop"
    assert result.hold_minutes == 3  # entry_idx=0 .. max_idx=2 inclusive


def test_eod_flat_when_day_runs_out_before_time_stop():
    cfg = ExitConfig(stop_type="entry_extreme", target_r=1.5, time_stop_minutes=30)
    closes = [100.0, 100.1, 100.2]  # only 3 bars total, time-stop is 30 -- day ends first
    bars = _bars(closes)
    result = simulate_exit(bars, entry_idx=0, direction_sign=1.0, entry_price=100.0, stop_price=95.0, target_price=110.0, cfg=cfg)
    assert result is not None
    assert result.reason == "eod"
    assert result.hold_minutes == 3


def test_atr_k_stop_uses_atr_from_agg_bars():
    # 15 completed 5m bars with a constant true range of 1.0 -> ATR(14) == 1.0
    agg_bars = []
    start = session_open(DAY)
    price = 100.0
    for i in range(15):
        agg_bars.append(
            AggBar(
                ts_open=start + timedelta(minutes=5 * i),
                ts_close=start + timedelta(minutes=5 * (i + 1)),
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1000,
                n_minutes=5,
            )
        )
    atr5 = atr_from_agg_bars(agg_bars, window=14)
    assert atr5 is not None
    assert abs(atr5 - 1.0) < 1e-9

    levels = compute_stop_and_target(
        direction_sign=1.0, entry_price=100.0, trigger_bar_high=100.5, trigger_bar_low=99.5,
        atr_5m=atr5, cfg=ExitConfig(stop_type="atr_k", atr_k=0.5, target_r=2.0),
    )
    assert levels is not None
    stop, target = levels
    assert abs(stop - 99.5) < 1e-9  # entry - 0.5*ATR(1.0)
    assert abs(target - 101.0) < 1e-9  # entry + 2R, R = 0.5


def test_atr_k_stop_returns_none_when_atr_unavailable():
    levels = compute_stop_and_target(
        direction_sign=1.0, entry_price=100.0, trigger_bar_high=100.5, trigger_bar_low=99.5,
        atr_5m=None, cfg=ExitConfig(stop_type="atr_k", atr_k=0.5, target_r=2.0),
    )
    assert levels is None


def test_entry_extreme_stop_uses_trigger_bar_not_fill_bar():
    levels = compute_stop_and_target(
        direction_sign=1.0, entry_price=100.0, trigger_bar_high=100.8, trigger_bar_low=99.2,
        atr_5m=None, cfg=ExitConfig(stop_type="entry_extreme", target_r=1.5),
    )
    assert levels is not None
    stop, target = levels
    assert stop == 99.2  # the trigger bar's low, for a long
    risk = 100.0 - 99.2
    assert abs(target - (100.0 + 1.5 * risk)) < 1e-9


# ── v0.3 exit archetypes: vwap_half / band_extreme_buffer, prior_swing_capped ─


def test_vwap_half_target_and_band_extreme_buffer_stop_long():
    # entry 98.7, vwap 100.0 -> target_distance = 0.5*1.3 = 0.65 -> target = 99.35
    # trigger bar low (the band extreme) = 98.5; stop_mult=1.0 -> stop = 98.5 - 1.0*0.65 = 97.85
    cfg = ExitConfigV3(stop_type="band_extreme_buffer", stop_mult=1.0, target_kind="vwap_half", time_stop_minutes=20)
    levels = compute_stop_and_target_v3(
        direction_sign=1.0, entry_price=98.7, trigger_bar_high=99.0, trigger_bar_low=98.5,
        atr_5m=None, vwap=100.0, swing_level=None, cfg=cfg,
    )
    assert levels is not None
    stop, target = levels
    assert abs(target - 99.35) < 1e-9
    assert abs(stop - 97.85) < 1e-9
    # stop must be WIDER than the raw band extreme (98.5), never tighter, and
    # the risk (entry-stop) must exceed the target distance (0.65) since
    # stop_mult=1.0 pushes an extra 0.65 beyond the already-adverse extreme.
    risk = 98.7 - stop
    assert risk > (98.7 - 98.5)
    assert risk > (target - 98.7)


def test_vwap_half_wider_stop_mult_widens_the_stop_only():
    base = dict(direction_sign=1.0, entry_price=98.7, trigger_bar_high=99.0, trigger_bar_low=98.5, atr_5m=None, vwap=100.0, swing_level=None)
    tight = compute_stop_and_target_v3(cfg=ExitConfigV3(stop_type="band_extreme_buffer", stop_mult=1.0, target_kind="vwap_half"), **base)
    wide = compute_stop_and_target_v3(cfg=ExitConfigV3(stop_type="band_extreme_buffer", stop_mult=1.5, target_kind="vwap_half"), **base)
    assert tight is not None and wide is not None
    assert wide[0] < tight[0]  # a bigger stop_mult -> stop pushed further away (lower, for a long)
    assert tight[1] == wide[1]  # target is independent of stop_mult


def test_vwap_half_returns_none_when_vwap_missing():
    levels = compute_stop_and_target_v3(
        direction_sign=1.0, entry_price=98.7, trigger_bar_high=99.0, trigger_bar_low=98.5,
        atr_5m=None, vwap=None, swing_level=None,
        cfg=ExitConfigV3(stop_type="band_extreme_buffer", stop_mult=1.0, target_kind="vwap_half"),
    )
    assert levels is None


def test_swing_level_from_agg_bars_excludes_trigger_bar():
    start = session_open(DAY)
    history = [
        AggBar(ts_open=start, ts_close=start + timedelta(minutes=5), open=100, high=101.0, low=99.0, close=100.5, volume=1000, n_minutes=5),
        AggBar(ts_open=start + timedelta(minutes=5), ts_close=start + timedelta(minutes=10), open=100.5, high=103.0, low=100.0, close=102.0, volume=1000, n_minutes=5),
    ]
    trigger = AggBar(ts_open=start + timedelta(minutes=10), ts_close=start + timedelta(minutes=15), open=102.0, high=110.0, low=101.5, close=101.8, volume=1000, n_minutes=5)
    bars = history + [trigger]
    # the trigger bar's own high (110.0) must NOT be picked up as the swing
    swing_long = swing_level_from_agg_bars(bars, lookback=10, is_long=True)
    assert swing_long == 103.0
    swing_short = swing_level_from_agg_bars(bars, lookback=10, is_long=False)
    assert swing_short == 99.0


def test_prior_swing_capped_target_caps_at_1r():
    # risk = entry(100.0) - stop(trigger_low=99.0) = 1.0 -> 1R cap.
    # swing_level way beyond 1R away (104.0) must be capped back to 1R (101.0).
    cfg = ExitConfigV3(stop_type="entry_extreme", target_kind="prior_swing_capped", time_stop_minutes=15)
    levels = compute_stop_and_target_v3(
        direction_sign=1.0, entry_price=100.0, trigger_bar_high=100.5, trigger_bar_low=99.0,
        atr_5m=None, vwap=None, swing_level=104.0, cfg=cfg,
    )
    assert levels is not None
    stop, target = levels
    assert abs(stop - 99.0) < 1e-9
    assert abs(target - 101.0) < 1e-9  # capped at exactly 1R, not 4R


def test_prior_swing_capped_target_uses_swing_when_inside_1r():
    cfg = ExitConfigV3(stop_type="entry_extreme", target_kind="prior_swing_capped", time_stop_minutes=15)
    levels = compute_stop_and_target_v3(
        direction_sign=1.0, entry_price=100.0, trigger_bar_high=100.5, trigger_bar_low=99.0,
        atr_5m=None, vwap=None, swing_level=100.6, cfg=cfg,  # only 0.6 away, inside the 1.0R cap
    )
    assert levels is not None
    _, target = levels
    assert abs(target - 100.6) < 1e-9


def test_prior_swing_capped_falls_back_to_1r_when_swing_unfavorable():
    cfg = ExitConfigV3(stop_type="entry_extreme", target_kind="prior_swing_capped", time_stop_minutes=15)
    levels = compute_stop_and_target_v3(
        direction_sign=1.0, entry_price=100.0, trigger_bar_high=100.5, trigger_bar_low=99.0,
        atr_5m=None, vwap=None, swing_level=98.0, cfg=cfg,  # swing is BEHIND entry -- not a usable target
    )
    assert levels is not None
    stop, target = levels
    assert abs(target - 101.0) < 1e-9  # fell back to exactly 1R


def test_v3_stop_wins_ties_same_as_v2_conservative_ordering():
    """The v0.3 archetypes only change LEVEL computation -- simulate_exit's
    conservative "stop wins if both fall in the same bar's range" tie-break
    is unchanged and untouched, so a v0.3-derived stop/target pair must still
    resolve a same-bar double-hit as a stop, exactly like the v0.1/v0.2 exit
    engine does."""
    cfg = ExitConfigV3(stop_type="band_extreme_buffer", stop_mult=1.0, target_kind="vwap_half", time_stop_minutes=20)
    levels = compute_stop_and_target_v3(
        direction_sign=1.0, entry_price=98.7, trigger_bar_high=99.0, trigger_bar_low=98.5,
        atr_5m=None, vwap=100.0, swing_level=None, cfg=cfg,
    )
    assert levels is not None
    stop_price, target_price = levels
    # a single wide bar whose range covers both the derived stop and target
    wild_bar = [bar("SPY", session_open(DAY), 98.7, target_price + 0.5, stop_price - 0.5, 98.9)]
    result = simulate_exit(wild_bar, entry_idx=0, direction_sign=1.0, entry_price=98.7, stop_price=stop_price, target_price=target_price, cfg=cfg)
    assert result is not None
    assert result.reason == "stop"
    assert result.r_multiple < 0
