"""Tests for the Zarattini/Aziz/Barbon noise-boundary replication (own-file
lane, no dependency on sibling bot.scalp modules under parallel edit).
"""
from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bot.data.models import Bar  # noqa: E402
from bot.scalp.noise_boundary import (  # noqa: E402
    HALF_HOUR_MARKS,
    HOURLY_MARKS,
    NoiseBoundaryConfig,
    ScalpExitConfig,
    Trade,
    build_day_session,
    build_noise_bands,
    daily_realized_vol_series,
    day_move_from_open,
    fold_consistency,
    random_matched_control,
    run_variant,
    run_variant_scalp_exit,
    shuffle_bar_content_within_day,
    shuffle_bars_by_day,
    side_stats,
    simulate_day,
    simulate_day_scalp_exit,
    split_by_direction,
    split_trades_by_regime,
    vol_regime_split,
    weekly_folds,
    yearly_expectancy,
)
from run_noise_boundary_directional_filter_sweep import (  # noqa: E402
    EntryFilter,
    simulate_day_filtered,
)

ET = ZoneInfo("America/New_York")


def _bar(symbol: str, d: date, hh: int, mm: int, o: float, h: float, l: float, c: float, v: int = 1000) -> Bar:
    ts = datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)
    return Bar(
        symbol=symbol, ts=ts, timeframe="1m", open=o, high=h, low=l, close=c,
        volume=v, source="test_fixture", ingested_ts=datetime.now(ET),
    )


# ── grid sanity ───────────────────────────────────────────────────────────


def test_hourly_marks_are_subset_of_half_hour_marks():
    assert set(HOURLY_MARKS) <= set(HALF_HOUR_MARKS)
    assert HALF_HOUR_MARKS[0] == time(10, 0)
    assert HALF_HOUR_MARKS[-1] == time(15, 30)
    assert len(HALF_HOUR_MARKS) == 12


# ── day session / move-from-open ─────────────────────────────────────────


def test_build_day_session_filters_rth_and_computes_open_close():
    d = date(2024, 1, 3)
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),  # session open bar
        _bar("SPY", d, 10, 0, 100.5, 101.0, 100.0, 101.0),
        _bar("SPY", d, 16, 0, 102.0, 102.0, 101.5, 102.0),  # session close bar
        _bar("SPY", d, 16, 5, 999.0, 999.0, 999.0, 999.0),  # after-hours -- must be dropped
    ]
    session = build_day_session("SPY", d, bars)
    assert session is not None
    assert session.open_price == 100.0  # first RTH bar's OPEN
    assert session.close_price == 102.0  # last in-session (<=16:00) bar's CLOSE
    assert time(16, 5) not in session.by_time
    assert time(10, 0) in session.by_time


def test_day_move_from_open():
    d = date(2024, 1, 3)
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 100.5, 101.5, 100.0, 101.0),
    ]
    session = build_day_session("SPY", d, bars)
    assert day_move_from_open(session, time(10, 0)) == pytest.approx(0.01)
    assert day_move_from_open(session, time(11, 0)) is None  # no bar at that mark


# ── noise-area band formula ───────────────────────────────────────────────


def test_build_noise_bands_matches_paper_formula():
    """UpperBound = max(Open_t, PriorClose) * (1 + VM*sigma); LowerBound =
    min(Open_t, PriorClose) * (1 - VM*sigma), sigma = mean move-from-open
    over the trailing lookback at that same clock mark.
    """
    cfg = NoiseBoundaryConfig(vm=1.0, lookback_days=3, marks=(time(10, 0),))

    history = []
    for i, mark_close in enumerate([101.0, 102.0, 100.0]):  # moves: .01, .02, .00 -> mean .01
        d = date(2024, 1, 1 + i)
        bars = [
            _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
            _bar("SPY", d, 10, 0, 100.0, mark_close, 100.0, mark_close),
        ]
        s = build_day_session("SPY", d, bars)
        history.append(s)
    # Override the last history day's close_price to exercise the gap adjustment.
    history[-1].close_price = 99.0

    target_day = date(2024, 1, 10)
    target_bars = [
        _bar("SPY", target_day, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", target_day, 10, 0, 100.0, 100.0, 100.0, 100.0),
    ]
    target = build_day_session("SPY", target_day, target_bars)

    bands = build_noise_bands(history, target, cfg)
    assert bands is not None
    lower, upper = bands[time(10, 0)]
    # ref_hi = max(100, 99) = 100 -> upper = 100 * 1.01 = 101.0
    # ref_lo = min(100, 99) = 99  -> lower = 99  * 0.99 = 98.01
    assert upper == pytest.approx(101.0)
    assert lower == pytest.approx(98.01)


def test_build_noise_bands_returns_none_with_insufficient_history():
    cfg = NoiseBoundaryConfig(vm=1.0, lookback_days=14, marks=(time(10, 0),))
    d = date(2024, 1, 10)
    target = build_day_session("SPY", d, [_bar("SPY", d, 9, 31, 100, 100, 100, 100)])
    assert build_noise_bands([], target, cfg) is None


# ── single-day simulation ─────────────────────────────────────────────────


def test_simulate_day_breakout_long_then_flat_at_close():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0), time(10, 30)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 101.5, 101.0, 101.5),   # breaks upper(101) -> long @101.5
        _bar("SPY", d, 10, 30, 101.8, 102.0, 101.8, 102.0),  # stays above lower(95) -- no stop
        _bar("SPY", d, 10, 31, 103.0, 103.0, 102.9, 103.0),  # "session close" for this toy day
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(10, 30): (95.0, 101.0)}

    trades = simulate_day("SPY", session, bands, cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long"
    assert t.entry_mark == time(10, 0)
    assert t.entry_price == pytest.approx(101.5)
    assert t.exit_reason == "eod_close"
    assert t.exit_price == pytest.approx(103.0)
    assert t.gross_bps == pytest.approx((103.0 / 101.5 - 1.0) * 10_000.0)


def test_simulate_day_opposite_band_stop_reverses_same_mark():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0), time(10, 30)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),   # breaks upper(101) -> long @102
        _bar("SPY", d, 10, 30, 91.0, 91.0, 89.0, 90.0),      # breaches lower(95) -> stop + reverse short @90
        _bar("SPY", d, 10, 31, 85.0, 85.5, 84.5, 85.0),      # forced close
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(10, 30): (95.0, 101.0)}

    trades = simulate_day("SPY", session, bands, cfg)
    assert len(trades) == 2
    first, second = trades
    assert first.direction == "long" and first.exit_reason == "stop"
    assert first.entry_price == pytest.approx(102.0)
    assert first.exit_price == pytest.approx(90.0)
    assert second.direction == "short" and second.exit_reason == "eod_close"
    assert second.entry_price == pytest.approx(90.0)
    assert second.exit_price == pytest.approx(85.0)
    # short profits when price falls: gross_ret = -(85/90 - 1) > 0
    assert second.gross_bps > 0


def test_simulate_day_current_band_vwap_stop_uses_vwap_when_tighter():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="current_band_vwap", marks=(time(10, 0), time(10, 30)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0, v=500),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0, v=3000),  # breakout long @102, heavy volume
        _bar("SPY", d, 10, 30, 102.0, 102.0, 100.5, 100.5, v=500),  # dips; cum VWAP still > upper band
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(10, 30): (95.0, 101.0)}
    vwap_at_1030 = session.vwap_by_time[time(10, 30)]
    assert vwap_at_1030 > 101.0  # cumulative VWAP still above the band (tighter stop than band)

    trades = simulate_day("SPY", session, bands, cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(100.5)  # closed once price <= VWAP, tighter than the 95 band


# ── execution-lag fill models (JOB 2) ─────────────────────────────────────


def test_lagged_fill_price_signal_close_matches_baseline():
    """fill_mode='signal_close' must be byte-identical to the pre-existing
    zero-latency convention: the mark's own close.
    """
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0), time(10, 30)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 999.0, 999.0, 101.5),   # breakout long @101.5 close
        _bar("SPY", d, 10, 1, 500.0, 500.0, 500.0, 500.0),   # decoy next-1m bar -- must be ignored
        _bar("SPY", d, 10, 30, 101.8, 102.0, 101.8, 102.0),
        _bar("SPY", d, 10, 31, 103.0, 103.0, 102.9, 103.0),
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(10, 30): (95.0, 101.0)}

    trades = simulate_day("SPY", session, bands, cfg, fill_mode="signal_close")
    assert len(trades) == 1
    assert trades[0].entry_price == pytest.approx(101.5)  # the mark's own close, not the decoy


def test_next_open_fill_uses_next_real_bar_open_not_signal_bars_own_path():
    """next_open must fill at the very NEXT 1-minute bar's OPEN -- never at
    any field (high/low/close) of the signal bar itself, which would smuggle
    in look-ahead knowledge of the signal bar's own subsequent intra-bar path.
    """
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0), time(10, 30)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        # signal bar @10:00: close breaks upper(101) -> long signal. Its own
        # high/low/close are deliberately distinct from the next bar's open
        # so the assertion below can only pass by reading the NEXT bar.
        _bar("SPY", d, 10, 0, 100.5, 200.0, 50.0, 101.5),
        # the next real 1-minute bar -- this open is the ONLY correct fill price.
        _bar("SPY", d, 10, 1, 111.11, 111.11, 111.11, 300.0),
        _bar("SPY", d, 10, 30, 111.5, 112.0, 111.0, 112.0),
        _bar("SPY", d, 10, 31, 113.0, 113.0, 112.9, 113.0),
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(10, 30): (95.0, 101.0)}

    trades = simulate_day("SPY", session, bands, cfg, fill_mode="next_open")
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == pytest.approx(111.11)  # next bar's OPEN
    assert t.entry_price not in (100.5, 200.0, 50.0, 101.5)  # never the signal bar's own OHLC
    assert t.entry_price != 300.0  # and not the next bar's close either


def test_plus_1min_close_fill_uses_next_real_bar_close():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0), time(10, 30)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 100.5, 200.0, 50.0, 101.5),   # signal bar (long breakout)
        _bar("SPY", d, 10, 1, 111.11, 111.11, 111.11, 222.22),  # next real 1m bar
        _bar("SPY", d, 10, 30, 111.5, 112.0, 111.0, 112.0),
        _bar("SPY", d, 10, 31, 113.0, 113.0, 112.9, 113.0),
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(10, 30): (95.0, 101.0)}

    trades = simulate_day("SPY", session, bands, cfg, fill_mode="plus_1min_close")
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == pytest.approx(222.22)  # next bar's CLOSE
    assert t.entry_price not in (100.5, 200.0, 50.0, 101.5, 111.11)


def test_lag_fill_falls_back_to_own_close_when_no_next_bar_exists():
    """A decision mark that lands on the session's LAST available bar (no
    later bar exists in the day) must fall back to that bar's own close --
    lag models must never reach past the end of the trading session.
    """
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0),))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 100.5, 200.0, 50.0, 101.5),  # last bar of the (toy) session
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0)}

    for mode in ("next_open", "plus_1min_close"):
        trades = simulate_day("SPY", session, bands, cfg, fill_mode=mode)
        assert len(trades) == 1
        assert trades[0].entry_price == pytest.approx(101.5)  # own close, no next bar to lag into


def test_execution_lag_is_orthogonal_to_signal_lag():
    """fill_mode must never change WHICH mark the decision fires on --
    signal_lag_marks (a completely separate control) still governs that.
    Here the signal itself is undelayed (signal_lag_marks=0, the default);
    only the FILL price should move between fill_mode variants, not the
    entry_mark or the fact that a trade fires at all.
    """
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0), time(10, 30)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 100.5, 101.6, 100.0, 101.5),
        _bar("SPY", d, 10, 1, 105.0, 105.0, 105.0, 106.0),
        _bar("SPY", d, 10, 30, 105.5, 106.0, 105.0, 106.0),
        _bar("SPY", d, 10, 31, 107.0, 107.0, 106.9, 107.0),
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(10, 30): (95.0, 101.0)}

    close_trades = simulate_day("SPY", session, bands, cfg, fill_mode="signal_close")
    lag_trades = simulate_day("SPY", session, bands, cfg, fill_mode="next_open")
    assert close_trades[0].entry_mark == lag_trades[0].entry_mark == time(10, 0)
    assert close_trades[0].entry_price != lag_trades[0].entry_price


# ── fixed shuffled-bars control (ts-grid preserved) ───────────────────────


def test_shuffle_bar_content_within_day_preserves_ts_grid_permutes_payload():
    d = date(2024, 1, 10)
    bars = [_bar("SPY", d, 9, 30 + i, 100.0 + i, 100.0 + i, 100.0 + i, 100.0 + i) for i in range(20)]
    shuffled = shuffle_bar_content_within_day(bars, seed=1)

    assert [b.ts for b in shuffled] == [b.ts for b in sorted(bars, key=lambda b: b.ts)]
    # payload IS permuted relative to the original ts-aligned order (not a no-op)
    assert [b.close for b in shuffled] != [b.close for b in sorted(bars, key=lambda b: b.ts)]
    # but it's the SAME multiset of closes -- only reordered, nothing invented/dropped
    assert sorted(b.close for b in shuffled) == sorted(b.close for b in bars)


def test_shuffle_bar_content_within_day_is_deterministic_per_seed():
    d = date(2024, 1, 10)
    bars = [_bar("SPY", d, 9, 30 + i, 100.0 + i, 100.0 + i, 100.0 + i, 100.0 + i) for i in range(20)]
    a = shuffle_bar_content_within_day(bars, seed=7)
    b = shuffle_bar_content_within_day(bars, seed=7)
    c = shuffle_bar_content_within_day(bars, seed=8)
    assert [x.close for x in a] == [x.close for x in b]
    assert [x.close for x in a] != [x.close for x in c]


def test_shuffle_bars_by_day_preserves_each_days_ts_grid_and_varies_by_day():
    days = [date(2024, 1, i) for i in range(1, 4)]
    bars_by_day = {
        d: [_bar("SPY", d, 9, 30 + i, 100.0 + i, 100.0 + i, 100.0 + i, 100.0 + i) for i in range(20)]
        for d in days
    }
    shuffled = shuffle_bars_by_day(bars_by_day, seed=3)
    assert set(shuffled) == set(bars_by_day)
    for d, bars in bars_by_day.items():
        assert [b.ts for b in shuffled[d]] == [b.ts for b in sorted(bars, key=lambda b: b.ts)]
    # different days get different permutations (not one shared pattern reused)
    closes_by_day = {d: [b.close for b in shuffled[d]] for d in days}
    assert len({tuple(v) for v in closes_by_day.values()}) > 1


# ── Trade properties ──────────────────────────────────────────────────────


def test_trade_properties():
    d = date(2024, 1, 1)
    t = Trade(
        symbol="SPY", day=d, direction="long", entry_mark=time(10, 0), entry_price=100.0,
        entry_stop_price=99.0, exit_mark=time(10, 30), exit_price=101.0,
        exit_reason="stop", hold_minutes=30,
    )
    assert t.gross_bps == pytest.approx(100.0)  # (101/100 - 1) * 10_000
    assert t.initial_risk_bps == pytest.approx(100.0)  # (100-99)/100 * 10_000
    assert t.r_multiple == pytest.approx(1.0)
    assert t.net_bps(1.0) == pytest.approx(100.0 - 2.0)


# ── point-in-time discipline across days ──────────────────────────────────


def _make_trend_day(symbol: str, d: date, open_px: float, mark_close: float) -> list[Bar]:
    return [
        _bar(symbol, d, 9, 31, open_px, open_px, open_px, open_px),
        _bar(symbol, d, 10, 0, open_px, max(open_px, mark_close), min(open_px, mark_close), mark_close),
    ]


def test_run_variant_skips_days_with_insufficient_lookback_history():
    cfg = NoiseBoundaryConfig(vm=1.0, lookback_days=3, marks=(time(10, 0),))
    bars_by_day = {}
    days = [date(2024, 1, i) for i in range(1, 8)]  # 7 days
    for i, d in enumerate(days):
        bars_by_day[d] = _make_trend_day("SPY", d, 100.0, 100.0 + i)  # widening moves, no huge breakout

    trades = run_variant("SPY", bars_by_day, cfg)
    # First 3 days (idx 0,1,2) must be skipped for lack of trailing history --
    # confirmed indirectly: no trade can be dated on days[0..2].
    traded_days = {t.day for t in trades}
    assert not (traded_days & set(days[:3]))


def test_run_variant_eval_window_filters_dates():
    cfg = NoiseBoundaryConfig(vm=1.0, lookback_days=1, marks=(time(10, 0),))
    bars_by_day = {}
    days = [date(2024, 1, i) for i in range(1, 6)]
    for i, d in enumerate(days):
        bars_by_day[d] = _make_trend_day("SPY", d, 100.0, 200.0)  # huge breakout every day -> always a trade

    trades = run_variant("SPY", bars_by_day, cfg, eval_start=days[3], eval_end=days[3])
    assert all(t.day == days[3] for t in trades)


# ── aggregate stats ────────────────────────────────────────────────────────


def test_side_stats_and_split_by_direction():
    d = date(2024, 1, 1)
    trades = [
        Trade("SPY", d, "long", time(10, 0), 100.0, 99.0, time(10, 30), 101.0, "stop", 30),
        Trade("SPY", d, "short", time(11, 0), 100.0, 101.0, time(11, 30), 99.0, "stop", 30),
    ]
    longs, shorts = split_by_direction(trades)
    assert len(longs) == 1 and len(shorts) == 1
    s = side_stats(trades, n_days=2, slippage_grid=(1.0,))
    assert s.n == 2
    assert s.win_rate == 1.0  # both trades are winners by construction
    assert s.trades_per_day == pytest.approx(1.0)
    assert s.net_expectancy_bps[1.0] == pytest.approx(s.gross_expectancy_bps - 2.0)


def test_weekly_folds_and_fold_consistency():
    trades = [
        Trade("SPY", date(2024, 1, 1), "long", time(10, 0), 100.0, 99.0, time(10, 30), 101.0, "stop", 30),
        Trade("SPY", date(2024, 1, 2), "long", time(10, 0), 100.0, 99.0, time(10, 30), 99.0, "stop", 30),
        Trade("SPY", date(2024, 1, 15), "long", time(10, 0), 100.0, 99.0, time(10, 30), 101.0, "stop", 30),
    ]
    folds = weekly_folds(trades)
    assert len(folds) == 2  # Jan 1-2 (2024) share an ISO week; Jan 15 is a different one
    fc = fold_consistency(trades, slippage_bps=0.0)
    assert fc["n_folds"] == 2


def test_yearly_expectancy_groups_by_year():
    trades = [
        Trade("SPY", date(2023, 12, 31), "long", time(10, 0), 100.0, 99.0, time(10, 30), 101.0, "stop", 30),
        Trade("SPY", date(2024, 1, 2), "long", time(10, 0), 100.0, 99.0, time(10, 30), 101.0, "stop", 30),
    ]
    by_year = yearly_expectancy(trades)
    assert set(by_year) == {2023, 2024}
    assert by_year[2023]["n"] == 1


# ── negative control ────────────────────────────────────────────────────────


def test_random_matched_control_produces_pooled_trades_from_real_data():
    days = [date(2024, 1, i) for i in range(1, 6)]
    bars_by_day = {d: _make_trend_day("SPY", d, 100.0, 101.0) for d in days}
    real_trades = [
        Trade("SPY", days[2], "long", time(9, 31), 100.0, 99.0, time(10, 0), 101.0, "stop", 29),
    ]
    synthetic = random_matched_control("SPY", bars_by_day, real_trades, seed=1, n_repeats=5)
    assert len(synthetic) == 5  # n_repeats * len(real_trades)
    for t in synthetic:
        assert t.day in bars_by_day
        assert t.direction in ("long", "short")


# ── locked-holdout guard (script-level) ─────────────────────────────────────


def test_run_noise_boundary_refuses_to_load_the_holdout():
    import run_noise_boundary as runner

    with pytest.raises(AssertionError):
        runner._load_real_bars(Path("does_not_matter.db"), "SPY", date(2023, 7, 1), date(2025, 7, 1), 60)


# ── realized-vol regime split ──────────────────────────────────────────────


def test_daily_realized_vol_series_none_until_window_plus_one_closes():
    days = [date(2024, 1, i) for i in range(1, 10)]
    closes = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0]
    sessions = [
        build_day_session("SPY", d, _make_trend_day("SPY", d, c, c)) for d, c in zip(days, closes)
    ]
    series = daily_realized_vol_series(sessions, window=3)
    # idx 0..3 (4 sessions) lack the required window+1=4 prior closes -> None
    for s in sessions[:4]:
        assert series[s.day] is None
    # idx 4 onward has enough trailing closes -> a real (non-negative) stdev
    for s in sessions[4:]:
        assert series[s.day] is not None
        assert series[s.day] >= 0.0


def test_vol_regime_split_median_splits_high_low():
    series = {date(2024, 1, i): float(i) for i in range(1, 7)}  # 1.0..6.0, clean median
    med, tags = vol_regime_split(series)
    assert med == pytest.approx(3.5)
    assert tags[date(2024, 1, 1)] == "low"
    assert tags[date(2024, 1, 6)] == "high"
    assert set(tags.values()) <= {"high", "low"}


def test_vol_regime_split_insufficient_data_returns_empty():
    med, tags = vol_regime_split({date(2024, 1, 1): 0.01})
    assert med is None
    assert tags == {}


def test_split_trades_by_regime_buckets_and_drops_untagged():
    d_hi, d_lo, d_untagged = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)
    tags = {d_hi: "high", d_lo: "low"}
    trades = [
        Trade("SPY", d_hi, "long", time(10, 0), 100.0, 99.0, time(10, 30), 101.0, "stop", 30),
        Trade("SPY", d_lo, "long", time(10, 0), 100.0, 99.0, time(10, 30), 101.0, "stop", 30),
        Trade("SPY", d_untagged, "long", time(10, 0), 100.0, 99.0, time(10, 30), 101.0, "stop", 30),
    ]
    buckets = split_trades_by_regime(trades, tags)
    assert len(buckets["high"]) == 1 and buckets["high"][0].day == d_hi
    assert len(buckets["low"]) == 1 and buckets["low"][0].day == d_lo
    assert sum(len(v) for v in buckets.values()) == 2  # untagged day dropped


# ── SWEEP v4 Hypothesis B: simulate_day_scalp_exit ───────────────────────


def test_simulate_day_scalp_exit_time_limit_flattens_after_n_minutes():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0),))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),   # breaks upper(101) -> long @102
        _bar("SPY", d, 10, 15, 102.5, 103.0, 102.0, 102.8),  # no stop, no time limit yet
        _bar("SPY", d, 10, 30, 103.0, 103.5, 102.5, 103.2),  # 30 min since entry -> time_stop here
        _bar("SPY", d, 10, 31, 105.0, 105.5, 104.5, 105.0),  # must NOT be reached
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0)}
    exit_cfg = ScalpExitConfig(kind="time_limit", time_limit_minutes=30)

    trades = simulate_day_scalp_exit("SPY", session, bands, cfg, exit_cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long"
    assert t.entry_mark == time(10, 0) and t.entry_price == pytest.approx(102.0)
    assert t.exit_reason == "time_stop"
    assert t.exit_mark == time(10, 30)
    assert t.exit_price == pytest.approx(103.2)  # exits at that bar's close
    assert t.hold_minutes == 30


def test_simulate_day_scalp_exit_first_target_exits_at_entry_time_noise_width():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0),))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),   # breaks upper(101) -> long @102
        # noise width at entry = upper(101) - lower(95) = 6.0; 0.5x target = entry + 3.0 = 105.0
        _bar("SPY", d, 10, 5, 103.0, 104.0, 102.5, 103.5),   # not there yet
        _bar("SPY", d, 10, 10, 104.5, 105.5, 104.0, 105.2),  # high(105.5) touches target 105.0 -> exit
        _bar("SPY", d, 10, 30, 200.0, 200.0, 200.0, 200.0),  # must NOT be reached
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0)}
    exit_cfg = ScalpExitConfig(kind="first_target", target_width_mult=0.5)

    trades = simulate_day_scalp_exit("SPY", session, bands, cfg, exit_cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "target"
    assert t.exit_mark == time(10, 10)
    assert t.exit_price == pytest.approx(105.0)  # entry(102) + 0.5 * width(6.0)


def test_simulate_day_scalp_exit_stop_checked_before_target_same_bar():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0),))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),   # breaks upper(101) -> long @102, stop=95
        # target(0.5x*6=3 -> 105.0) AND stop(95) both touched intrabar on the same bar
        _bar("SPY", d, 10, 5, 103.0, 106.0, 90.0, 100.0),
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0)}
    exit_cfg = ScalpExitConfig(kind="first_target", target_width_mult=0.5)

    trades = simulate_day_scalp_exit("SPY", session, bands, cfg, exit_cfg)
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"  # conservative both-hit tie-break: stop wins
    assert trades[0].exit_price == pytest.approx(95.0)


def test_scalp_exit_next_open_can_stop_inside_fill_bar():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0),))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),
        # Entry is this bar's OPEN=103.  Its later low reaches the stop=95.
        _bar("SPY", d, 10, 1, 103.0, 104.0, 94.0, 100.0),
        _bar("SPY", d, 10, 2, 100.0, 101.0, 99.0, 100.5),
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0)}
    exit_cfg = ScalpExitConfig(kind="time_limit", time_limit_minutes=30)

    trades = simulate_day_scalp_exit("SPY", session, bands, cfg, exit_cfg, fill_mode="next_open")
    assert len(trades) == 1
    assert trades[0].entry_price == pytest.approx(103.0)
    assert trades[0].exit_mark == time(10, 1)
    assert trades[0].exit_reason == "stop"


def test_scalp_exit_plus_one_close_ignores_prefill_bar_path():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0),))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),
        # The low=94 occurs BEFORE the close fill=100 and must not stop it.
        _bar("SPY", d, 10, 1, 103.0, 104.0, 94.0, 100.0),
        _bar("SPY", d, 10, 2, 100.0, 101.5, 99.0, 101.0),
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0)}
    exit_cfg = ScalpExitConfig(kind="time_limit", time_limit_minutes=30)

    trades = simulate_day_scalp_exit(
        "SPY", session, bands, cfg, exit_cfg, fill_mode="plus_1min_close"
    )
    assert len(trades) == 1
    assert trades[0].entry_mark == time(10, 1)
    assert trades[0].entry_price == pytest.approx(100.0)
    assert trades[0].exit_reason == "eod_close"
    assert trades[0].exit_price == pytest.approx(101.0)


def test_directional_filter_runner_uses_next_open_fill():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0),))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),
        _bar("SPY", d, 10, 1, 103.0, 104.0, 94.0, 100.0),
        _bar("SPY", d, 10, 2, 100.0, 101.0, 99.0, 100.5),
    ]
    session = build_day_session("SPY", d, bars)
    exit_cfg = ScalpExitConfig(kind="time_limit", time_limit_minutes=30)
    trades = simulate_day_filtered(
        "SPY",
        session,
        100.0,
        {time(10, 0): (95.0, 101.0)},
        cfg,
        exit_cfg,
        "long",
        EntryFilter("all"),
        fill_mode="next_open",
    )
    assert len(trades) == 1
    assert trades[0].entry_price == pytest.approx(103.0)
    assert trades[0].exit_reason == "stop"


def test_simulate_day_scalp_exit_default_caps_at_one_trade_per_day():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0), time(14, 0)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),   # breaks upper -> long @102
        _bar("SPY", d, 10, 5, 103.0, 103.5, 103.0, 103.2),   # target hit (0.5x*6=3 -> 105 not yet)
        _bar("SPY", d, 10, 10, 105.5, 106.0, 105.0, 105.8),  # target(105.0) touched -> exits, flat
        _bar("SPY", d, 14, 0, 110.0, 111.0, 110.0, 111.0),   # a FRESH breakout signal -- must be ignored
        _bar("SPY", d, 14, 1, 111.0, 111.0, 111.0, 111.0),
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(14, 0): (100.0, 105.0)}
    exit_cfg = ScalpExitConfig(kind="first_target", target_width_mult=0.5)

    trades = simulate_day_scalp_exit("SPY", session, bands, cfg, exit_cfg)
    assert len(trades) == 1  # the 14:00 breakout is NOT taken -- one trade/day cap
    assert trades[0].exit_reason == "target"


def test_simulate_day_scalp_exit_reentry_after_stop_allows_second_trade():
    d = date(2024, 1, 10)
    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0), time(14, 0)))
    bars = [
        _bar("SPY", d, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d, 10, 0, 101.0, 102.0, 101.0, 102.0),   # breaks upper -> long @102, stop=95
        _bar("SPY", d, 10, 5, 94.0, 94.5, 93.0, 94.0),        # stop(95) breached -> "stop" exit
        _bar("SPY", d, 14, 0, 110.0, 111.0, 110.0, 111.0),    # a FRESH breakout -- should be taken (reentry)
        _bar("SPY", d, 14, 1, 111.5, 112.0, 111.0, 111.8),    # forced close for this toy day
    ]
    session = build_day_session("SPY", d, bars)
    bands = {time(10, 0): (95.0, 101.0), time(14, 0): (100.0, 105.0)}

    exit_cfg_no_reentry = ScalpExitConfig(kind="first_target", target_width_mult=0.5, allow_one_reentry_after_stop=False)
    trades_no_reentry = simulate_day_scalp_exit("SPY", session, bands, cfg, exit_cfg_no_reentry)
    assert len(trades_no_reentry) == 1
    assert trades_no_reentry[0].exit_reason == "stop"

    exit_cfg_reentry = ScalpExitConfig(kind="first_target", target_width_mult=0.5, allow_one_reentry_after_stop=True)
    trades_reentry = simulate_day_scalp_exit("SPY", session, bands, cfg, exit_cfg_reentry)
    assert len(trades_reentry) == 2
    assert trades_reentry[0].exit_reason == "stop"
    assert trades_reentry[1].direction == "long"
    assert trades_reentry[1].entry_mark == time(14, 0)
    assert trades_reentry[1].entry_price == pytest.approx(111.0)


def test_run_variant_scalp_exit_matches_single_day_and_respects_lookback():
    d0 = date(2024, 1, 2)
    hist_days = []
    for i in range(14):
        hd = date(2024, 1, 2) - __import__("datetime").timedelta(days=14 - i)
        hist_days.append(
            (
                hd,
                [
                    _bar("SPY", hd, 9, 31, 100.0, 100.05, 99.95, 100.0),
                    _bar("SPY", hd, 10, 0, 100.0, 100.05, 99.95, 100.0),
                ],
            )
        )
    target_bars = [
        _bar("SPY", d0, 9, 31, 100.0, 100.0, 100.0, 100.0),
        _bar("SPY", d0, 10, 0, 101.0, 102.0, 101.0, 102.0),
        _bar("SPY", d0, 10, 30, 103.0, 103.5, 102.5, 103.2),
    ]
    bars_by_day = {hd: hb for hd, hb in hist_days}
    bars_by_day[d0] = target_bars

    cfg = NoiseBoundaryConfig(vm=1.0, stop_mode="opposite_band", marks=(time(10, 0),), lookback_days=14)
    exit_cfg = ScalpExitConfig(kind="time_limit", time_limit_minutes=30)

    trades = run_variant_scalp_exit("SPY", bars_by_day, cfg, exit_cfg, eval_start=d0, eval_end=d0)
    assert len(trades) == 1
    assert trades[0].day == d0
    assert trades[0].direction == "long"

    # Fewer than lookback_days of history -> no eval days produce trades.
    short_history = {k: v for k, v in bars_by_day.items() if k >= d0 - __import__("datetime").timedelta(days=5)}
    trades_short = run_variant_scalp_exit("SPY", short_history, cfg, exit_cfg, eval_start=d0, eval_end=d0)
    assert trades_short == []
