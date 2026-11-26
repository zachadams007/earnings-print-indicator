from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from bot.scalp.copilot_v1 import (
    _event,
    _simulate_frozen_plan,
    build_chart_features,
    collapse_simultaneous_candidates,
    complementary_scores,
    generate_candidate_events,
    wilder_rsi,
)

ET = ZoneInfo("America/New_York")


def _raw_sessions(days: int = 25, minutes: int = 30, volume_multiplier_last: float = 1.0):
    rows = []
    start_day = pd.Timestamp("2025-01-02")
    trading_days = pd.bdate_range(start_day, periods=days)
    price = 100.0
    for day_i, day in enumerate(trading_days):
        start = datetime(day.year, day.month, day.day, 9, 31, tzinfo=ET)
        for minute in range(minutes):
            close = price + 0.01
            volume = 1000
            if day_i == days - 1 and minute < 3:
                volume = round(volume * volume_multiplier_last)
            rows.append(
                {
                    "ts": start + timedelta(minutes=minute),
                    "open": price,
                    "high": close + 0.02,
                    "low": price - 0.02,
                    "close": close,
                    "volume": volume,
                }
            )
            price = close
    return pd.DataFrame(rows)


def test_complementary_scores_use_the_whole_ten_point_definition():
    long_score, short_score = complementary_scores(0.73)
    assert long_score == pytest.approx(7.3)
    assert short_score == pytest.approx(2.7)
    assert long_score + short_score == pytest.approx(10.0)


def test_wilder_rsi_is_continuous_and_directional():
    assert wilder_rsi(range(30))[-1] == 100.0
    assert wilder_rsi(range(30, 0, -1))[-1] == 0.0


def test_same_slot_rvol_uses_prior_sessions_not_recent_bars():
    features = build_chart_features(
        _raw_sessions(days=21, minutes=6, volume_multiplier_last=2.0),
        "TEST",
    )
    last_day = features[features["session"] == features["session"].max()]
    at_first_3m_close = last_day[last_day["minute_index"] == 3].iloc[0]
    assert at_first_3m_close["rvol3"] == pytest.approx(2.0)


def test_completed_three_minute_value_does_not_peek_into_partial_bar():
    raw = _raw_sessions(days=25, minutes=30)
    last_day = pd.to_datetime(raw["ts"]).dt.date.max()
    last_indices = raw.index[pd.to_datetime(raw["ts"]).dt.date == last_day]
    raw.loc[last_indices[3:6], "volume"] = 3000
    features = build_chart_features(raw, "TEST")
    last = features[features["session"] == features["session"].max()].reset_index(drop=True)
    # 09:34 and 09:35 still use the 09:33 completed 3m value.
    assert last.loc[3, "minute_index"] == 4
    assert last.loc[4, "minute_index"] == 5
    assert last.loc[3, "rvol3"] == pytest.approx(last.loc[4, "rvol3"])
    # 09:36 is allowed to update.
    assert last.loc[5, "minute_index"] == 6
    assert last.loc[5, "rvol3"] == pytest.approx(3.0)
    assert last.loc[5, "rvol3"] != pytest.approx(last.loc[4, "rvol3"])


def test_future_mutation_cannot_change_earlier_chart_features():
    raw = _raw_sessions(days=25, minutes=30)
    original = build_chart_features(raw, "TEST")
    altered = raw.copy()
    altered.loc[altered.index[-1], ["high", "close", "volume"]] = [500.0, 500.0, 9_000_000]
    changed = build_chart_features(altered, "TEST")
    columns = ["vwap", "rsi3", "rvol3", "ema9", "ema20", "atr5"]
    pd.testing.assert_frame_equal(original.iloc[:-1][columns], changed.iloc[:-1][columns])


def test_event_fills_at_next_open_and_delayed_entry_is_later():
    features = build_chart_features(_raw_sessions(days=25, minutes=30), "TEST")
    session = features[features["session"] == features["session"].max()].reset_index(drop=True)
    event = _event(
        session,
        15,
        family="test",
        side="long",
        level_id="VWAP",
        level_price=float(session.loc[15, "vwap"]),
        stop_price=float(session.loc[15, "low"]),
        trigger_quality=1.0,
    )
    assert event is not None
    assert event["fill_status"] == "filled"
    assert event["realized_eligible"] is True
    assert event["entry_ts"] == session.loc[16, "bar_open_ts"]
    assert event["entry_price"] == session.loc[16, "open"]
    assert event["state_version"].startswith("racer.v0")
    assert event["transition_geometry"] == "test"
    assert event["event_reaction_state"] == "event_neutral_unknown"
    assert event["historical_30s_available"] is False


def test_future_gap_through_stop_keeps_candidate_but_removes_all_labels():
    features = build_chart_features(_raw_sessions(days=25, minutes=30), "TEST")
    session = features[features["session"] == features["session"].max()].reset_index(drop=True)
    decision_i = 15
    stop_price = float(session.loc[decision_i, "low"])
    session.loc[decision_i + 1, "open"] = stop_price - 1.0

    event = _event(
        session,
        decision_i,
        family="test",
        side="long",
        level_id="VWAP",
        level_price=float(session.loc[decision_i, "vwap"]),
        stop_price=stop_price,
        trigger_quality=1.0,
    )

    assert event is not None
    assert event["fill_status"] == "gap_invalidated_unfilled"
    assert event["realized_eligible"] is False
    assert event["observed_next_open_price"] == pytest.approx(stop_price - 1.0)
    for label in (
        "entry_price",
        "future_up",
        "future_bps",
        "candidate_gross_bps",
        "candidate_net_bps_1bp",
        "candidate_net_win_1bp",
        "candidate_outcome_r",
        "delayed_candidate_gross_bps",
    ):
        assert pd.isna(event[label]), label


def _clock_edge_failed_retest(
    start_minute: int, *, length: int = 10, second_episode: bool = False
) -> pd.DataFrame:
    session_date = pd.Timestamp("2025-01-02").date()
    session_open = pd.Timestamp("2025-01-02 09:30", tz=ET)
    rows = []
    for offset in range(min(length, 391 - start_minute)):
        minute = start_minute + offset
        close_ts = session_open + pd.Timedelta(minutes=minute)
        open_, high, low, close = 99.8, 99.85, 99.65, 99.75
        if offset == 2:
            open_, high, low, close = 99.95, 100.05, 99.85, 99.90
        elif offset == 3:
            open_, high, low, close = 99.90, 100.00, 99.70, 99.75
        elif second_episode and offset == 7:
            open_, high, low, close = 99.96, 100.06, 99.86, 99.91
        elif second_episode and offset == 8:
            open_, high, low, close = 99.90, 100.00, 99.71, 99.76
        rows.append(
            {
                "symbol": "SPY",
                "session": session_date,
                "ts": close_ts,
                "bar_open_ts": close_ts - pd.Timedelta(minutes=1),
                "bar_close_ts": close_ts,
                "minute_index": minute,
                "research_eligible": True,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000.0,
                "rvol3": 1.0,
                "atr5": 0.5,
                "atr5_bps": 50.0,
                "rsi3": 50.0,
                "prior_high": 100.0,
                "prior_low": 98.0,
                "premarket_high": np.nan,
                "premarket_low": np.nan,
                "or_high": np.nan,
                "or_low": np.nan,
                "ema9": 99.8,
                "ema20": 99.8,
                "vwap": 99.8,
                "ema_gap_bps": 0.0,
                "ema9_slope_bps": 0.0,
                "ema20_slope_bps": 0.0,
                "close_vwap_bps": 0.0,
                "ret15_bps": 0.0,
                "structure3": 0.0,
                "day_return_bps": 0.0,
                "session_high": 100.05,
                "session_low": 98.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("start_minute", [1, 381, 383, 384, 385, 386])
def test_open_and_close_candidates_are_not_deleted_by_clock(start_minute: int):
    events = generate_candidate_events(_clock_edge_failed_retest(start_minute))

    assert not events.empty
    assert "failed_level_retest" in events["family"].tolist()


def test_later_same_family_episode_is_not_deleted_by_session_global_dedup():
    events = generate_candidate_events(
        _clock_edge_failed_retest(1, length=20, second_episode=True)
    )

    failed = events.loc[events["family"].eq("failed_level_retest")]
    assert len(failed) >= 2
    assert failed["decision_ts"].nunique() >= 2


def test_eod_continuation_is_evaluated_between_old_ten_minute_checkpoints():
    session_date = pd.Timestamp("2025-01-02").date()
    session_open = pd.Timestamp("2025-01-02 09:30", tz=ET)
    rows = []
    for minute in range(330, 371):
        close_ts = session_open + pd.Timedelta(minutes=minute)
        base = 100.0 if minute == 330 else 100.20
        open_, high, low, close = base, base + 0.01, base - 0.01, base
        if minute == 362:
            open_, high, low, close = 100.20, 100.26, 100.10, 100.25
        rows.append(
            {
                "symbol": "SPY",
                "session": session_date,
                "ts": close_ts,
                "bar_open_ts": close_ts - pd.Timedelta(minutes=1),
                "bar_close_ts": close_ts,
                "minute_index": minute,
                "research_eligible": True,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 2_000.0,
                "rvol3": 1.5,
                "atr5": 0.5,
                "atr5_bps": 50.0,
                "rsi3": 60.0,
                "prior_high": 105.0,
                "prior_low": 95.0,
                "premarket_high": np.nan,
                "premarket_low": np.nan,
                "or_high": np.nan,
                "or_low": np.nan,
                "ema9": 100.1,
                "ema20": 100.05,
                "vwap": 100.0,
                "ema_gap_bps": 5.0,
                "ema9_slope_bps": 1.0,
                "ema20_slope_bps": 0.5,
                "close_vwap_bps": 20.0,
                "ret15_bps": 10.0,
                "structure3": 1.0,
                "day_return_bps": 80.0,
                "session_high": 100.26,
                "session_low": 99.5,
            }
        )

    events = generate_candidate_events(pd.DataFrame(rows))
    continuation = events.loc[events["family"].eq("eod_continuation")]

    assert not continuation.empty
    assert 362 in {
        int((timestamp - session_open).total_seconds() // 60)
        for timestamp in continuation["decision_ts"]
    }


def test_eod_reversal_rearms_after_the_signal_clears():
    session_date = pd.Timestamp("2025-01-02").date()
    session_open = pd.Timestamp("2025-01-02 09:30", tz=ET)
    rows = []
    for minute in range(330, 371):
        close_ts = session_open + pd.Timedelta(minutes=minute)
        open_, high, low, close, structure = 100.90, 100.92, 100.88, 100.90, 0.0
        if minute == 330:
            open_, high, low, close = 100.00, 100.02, 99.98, 100.00
        elif minute == 361:
            open_, high, low, close = 100.90, 101.10, 100.80, 100.90
        elif minute == 362:
            open_, high, low, close, structure = 100.85, 100.90, 100.65, 100.70, -1.0
        elif minute == 363:
            open_, high, low, close, structure = 100.70, 100.80, 100.68, 100.75, 0.0
        elif minute == 364:
            open_, high, low, close, structure = 100.74, 100.76, 100.55, 100.60, -1.0
        elif minute > 364:
            open_, high, low, close = 100.62, 100.66, 100.58, 100.63
        rows.append(
            {
                "symbol": "SPY",
                "session": session_date,
                "ts": close_ts,
                "bar_open_ts": close_ts - pd.Timedelta(minutes=1),
                "bar_close_ts": close_ts,
                "minute_index": minute,
                "research_eligible": True,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 2_000.0,
                "rvol3": 1.5,
                "atr5": 0.5,
                "atr5_bps": 50.0,
                "rsi3": 60.0,
                "prior_high": 101.0,
                "prior_low": 95.0,
                "premarket_high": np.nan,
                "premarket_low": np.nan,
                "or_high": np.nan,
                "or_low": np.nan,
                "ema9": 100.8,
                "ema20": 100.7,
                "vwap": 100.5,
                "ema_gap_bps": 10.0,
                "ema9_slope_bps": 1.0,
                "ema20_slope_bps": 0.5,
                "close_vwap_bps": 20.0,
                "ret15_bps": 10.0,
                "structure3": structure,
                "day_return_bps": 100.0,
                "session_high": 101.1,
                "session_low": 99.5,
            }
        )

    events = generate_candidate_events(pd.DataFrame(rows))
    reversal = events.loc[events["family"].eq("eod_profit_taking_reversal")]
    event_minutes = {
        int((timestamp - session_open).total_seconds() // 60)
        for timestamp in reversal["decision_ts"]
    }

    assert {362, 364}.issubset(event_minutes)


def _plan_session(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    start = pd.Timestamp("2025-01-02 10:00", tz=ET)
    return pd.DataFrame(
        [
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "bar_close_ts": start + pd.Timedelta(minutes=i + 1),
            }
            for i, (open_, high, low, close) in enumerate(bars)
        ]
    )


def test_frozen_plan_assumes_stop_first_when_target_and_stop_share_a_bar():
    session = _plan_session([(100.0, 102.5, 98.5, 101.0)])
    result = _simulate_frozen_plan(session, 0, 1, side="long", stop_price=99.0)
    assert result is not None
    assert result["exit_reason"] == "stop"
    assert result["gross_bps"] == pytest.approx(-100.0)
    assert result["outcome_r"] == pytest.approx(-1.0)


def test_frozen_plan_scales_half_at_t1_and_half_at_t2():
    session = _plan_session([(100.0, 102.1, 99.5, 102.0)])
    result = _simulate_frozen_plan(session, 0, 1, side="long", stop_price=99.0)
    assert result is not None
    assert result["exit_reason"] == "t2"
    assert result["effective_exit_price"] == pytest.approx(101.75)
    assert result["outcome_r"] == pytest.approx(1.75)


def test_frozen_plan_marks_an_entry_beyond_the_structural_stop_invalid():
    session = _plan_session([(98.5, 99.0, 98.0, 98.8)])
    assert _simulate_frozen_plan(session, 0, 1, side="long", stop_price=99.0) is None


def test_collapse_discards_a_same_timestamp_direction_conflict():
    ts = pd.Timestamp("2025-01-02 10:00", tz=ET)
    events = pd.DataFrame(
        [
            {"symbol": "SPY", "decision_ts": ts, "candidate_side": "long"},
            {"symbol": "SPY", "decision_ts": ts, "candidate_side": "short"},
        ]
    )
    assert collapse_simultaneous_candidates(events).empty
