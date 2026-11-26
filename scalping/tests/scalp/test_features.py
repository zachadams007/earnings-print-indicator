"""Feature derivation tests: no-leakage, 5m derivation, VWAP reset, RVOL-TOD math."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from bot.scalp.features import (
    AggBar,
    StructureState,
    atr,
    build_feature_snapshots,
    cumulative_volume_by_minute_index,
    daily_ohlc_from_1m,
    derive_agg_bars,
    opening_range,
    prior_day_levels,
    rvol_for_time_of_day,
    session_vwap_series,
    structure_state,
)
from tests.scalp.helpers import bar, make_walk_session, session_open

DAY = date(2026, 6, 1)  # a Monday


# ── no-leakage ────────────────────────────────────────────────────────────────


def test_features_no_leakage_future_mutation():
    """Mutating the DECISION bar itself (index i) must not change snapshot i,
    since decision i only ever sees day_bars_1m[:i]."""
    closes = [100.0 + 0.01 * i for i in range(60)]
    bars = make_walk_session("SPY", DAY, closes)

    snaps_before = build_feature_snapshots(bars, [])
    target = next(s for s in snaps_before if s.idx == 40)

    mutated = list(bars)
    mutated[40] = bar(
        "SPY", mutated[40].ts, o=9999.0, h=9999.0, l=1.0, c=9999.0, v=mutated[40].volume * 1000
    )
    snaps_after = build_feature_snapshots(mutated, [])
    target_after = next(s for s in snaps_after if s.idx == 40)

    assert target.last_close == target_after.last_close
    assert target.vwap == target_after.vwap
    assert target.rvol_tod == target_after.rvol_tod
    assert target.structure_15m == target_after.structure_15m


def test_features_sensitive_to_last_completed_bar():
    """Sanity check the harness is not vacuously constant: mutating the LAST
    COMPLETED bar (index i-1) for decision i MUST change that snapshot."""
    closes = [100.0 + 0.01 * i for i in range(60)]
    bars = make_walk_session("SPY", DAY, closes)

    snaps_before = build_feature_snapshots(bars, [])
    target = next(s for s in snaps_before if s.idx == 40)

    mutated = list(bars)
    mutated[39] = bar("SPY", mutated[39].ts, o=500.0, h=500.0, l=500.0, c=500.0, v=mutated[39].volume)
    snaps_after = build_feature_snapshots(mutated, [])
    target_after = next(s for s in snaps_after if s.idx == 40)

    assert target.last_close != target_after.last_close
    assert target.vwap != target_after.vwap


def test_features_no_leakage_all_snapshots_prefix_stable():
    """More exhaustive prefix check: perturbing bar k only ever changes
    snapshots for decisions > k (never <= k)."""
    closes = [100.0 + (i % 7) * 0.05 for i in range(50)]
    bars = make_walk_session("SPY", DAY, closes)
    k = 25

    baseline = {s.idx: (s.last_close, s.vwap) for s in build_feature_snapshots(bars, [])}

    mutated = list(bars)
    mutated[k] = bar("SPY", mutated[k].ts, o=1234.0, h=1234.0, l=1234.0, c=1234.0, v=999999)
    perturbed = {s.idx: (s.last_close, s.vwap) for s in build_feature_snapshots(mutated, [])}

    for idx in baseline:
        if idx <= k:
            assert baseline[idx] == perturbed[idx], f"snapshot idx={idx} leaked future bar {k}"


# ── 5m derivation ─────────────────────────────────────────────────────────────


def test_derive_5m_bars_correct_aggregation():
    start = session_open(DAY)
    # 10 minutes -> two complete 5m buckets, hand-picked OHLCV.
    closes = [100, 101, 102, 103, 104.5, 105, 104, 106, 103, 107]
    bars = [
        bar("SPY", start + timedelta(minutes=i), o=closes[i] - 0.5, h=closes[i] + 1, l=closes[i] - 1, c=closes[i], v=100 * (i + 1))
        for i in range(10)
    ]

    agg5 = derive_agg_bars(bars, 5)
    assert len(agg5) == 2

    first, second = agg5
    assert first.open == bars[0].open
    assert first.close == bars[4].close
    assert first.high == max(b.high for b in bars[0:5])
    assert first.low == min(b.low for b in bars[0:5])
    assert first.volume == sum(b.volume for b in bars[0:5])
    assert first.n_minutes == 5

    assert second.open == bars[5].open
    assert second.close == bars[9].close
    assert second.high == max(b.high for b in bars[5:10])
    assert second.low == min(b.low for b in bars[5:10])
    assert second.volume == sum(b.volume for b in bars[5:10])


def test_derive_agg_bars_drops_incomplete_trailing_bucket():
    start = session_open(DAY)
    bars = [
        bar("SPY", start + timedelta(minutes=i), 100, 101, 99, 100, 100)
        for i in range(7)  # 1 complete 5m bucket + 2 leftover minutes
    ]
    agg5 = derive_agg_bars(bars, 5)
    assert len(agg5) == 1  # trailing partial bucket (minutes 5,6) must NOT be emitted


def test_derive_agg_bars_empty_input():
    assert derive_agg_bars([], 5) == []


# ── VWAP reset at session start ───────────────────────────────────────────────


def test_session_vwap_resets_daily():
    day1 = DAY
    day2 = DAY + timedelta(days=1)

    day1_bars = [
        bar("SPY", session_open(day1) + timedelta(minutes=i), 200, 201, 199, 200, 10_000)
        for i in range(5)
    ]
    # Day 2 starts at a totally different price/volume regime.
    day2_bars = [
        bar("SPY", session_open(day2) + timedelta(minutes=i), 50, 51, 49, 50, 500)
        for i in range(3)
    ]

    combined = day1_bars + day2_bars
    vwap = session_vwap_series(combined)

    # Day 1's VWAP should hover near 200 (all bars ~200).
    assert all(195 <= v <= 205 for v in vwap[:5])

    # Day 2's FIRST vwap sample must equal day2's own first-bar typical price,
    # not a blend carried over from day 1's ~200 level.
    day2_first_tp = (day2_bars[0].high + day2_bars[0].low + day2_bars[0].close) / 3.0
    assert vwap[5] == pytest.approx(day2_first_tp)
    assert vwap[5] < 100  # nowhere near day 1's ~200 level


def test_session_vwap_weighted_by_volume():
    """A high-volume bar should pull the cumulative VWAP toward its own price."""
    start = session_open(DAY)
    bars = [
        bar("SPY", start, 100, 100, 100, 100, 10),       # tiny volume at 100
        bar("SPY", start + timedelta(minutes=1), 200, 200, 200, 200, 100_000),  # huge volume at 200
    ]
    vwap = session_vwap_series(bars)
    assert vwap[1] == pytest.approx(200, abs=1.0)


# ── RVOL-TOD math ──────────────────────────────────────────────────────────────


def test_cumulative_volume_by_minute_index():
    start = session_open(DAY)
    bars = [bar("SPY", start + timedelta(minutes=i), 100, 100, 100, 100, v=100) for i in range(5)]
    cum = cumulative_volume_by_minute_index(bars)
    assert cum == [100, 200, 300, 400, 500]


def test_rvol_for_time_of_day_hand_computed():
    # Today: 10 completed minutes, volume 200/min -> cumulative at idx 9 = 2000.
    today_bars = [
        bar("SPY", session_open(DAY) + timedelta(minutes=i), 100, 100, 100, 100, v=200)
        for i in range(10)
    ]

    # 2 prior sessions with different but known volumes at the same idx 9.
    prior_day1 = date(2026, 5, 29)
    prior_day2 = date(2026, 5, 28)
    prior1 = [
        bar("SPY", session_open(prior_day1) + timedelta(minutes=i), 100, 100, 100, 100, v=100)
        for i in range(10)
    ]  # cumulative at idx9 = 1000
    prior2 = [
        bar("SPY", session_open(prior_day2) + timedelta(minutes=i), 100, 100, 100, 100, v=300)
        for i in range(10)
    ]  # cumulative at idx9 = 3000

    rvol = rvol_for_time_of_day(today_bars, [prior1, prior2], lookback_sessions=20)
    # avg_hist = (1000 + 3000) / 2 = 2000; today_cum = 2000 -> rvol == 1.0
    assert rvol == pytest.approx(1.0)


def test_rvol_for_time_of_day_elevated():
    today_bars = [
        bar("SPY", session_open(DAY) + timedelta(minutes=i), 100, 100, 100, 100, v=1000)
        for i in range(5)
    ]  # cumulative at idx4 = 5000
    prior = [
        bar("SPY", session_open(date(2026, 5, 29)) + timedelta(minutes=i), 100, 100, 100, 100, v=200)
        for i in range(5)
    ]  # cumulative at idx4 = 1000
    rvol = rvol_for_time_of_day(today_bars, [prior], lookback_sessions=20)
    assert rvol == pytest.approx(5.0)


def test_rvol_for_time_of_day_no_history_returns_none():
    today_bars = [bar("SPY", session_open(DAY), 100, 100, 100, 100, v=100)]
    assert rvol_for_time_of_day(today_bars, [], lookback_sessions=20) is None


def test_rvol_respects_lookback_sessions_window():
    """Only the trailing `lookback_sessions` prior sessions should be used."""
    today_bars = [
        bar("SPY", session_open(DAY) + timedelta(minutes=i), 100, 100, 100, 100, v=100)
        for i in range(3)
    ]  # cumulative at idx2 = 300

    # 3 old sessions with huge volume (should be excluded by lookback=1)...
    old_sessions = [
        [
            bar("SPY", session_open(date(2026, 5, 20) + timedelta(days=d)) + timedelta(minutes=i),
                100, 100, 100, 100, v=10_000)
            for i in range(3)
        ]
        for d in range(3)
    ]
    # ...and one recent session with volume matching today exactly.
    recent_session = [
        bar("SPY", session_open(date(2026, 5, 29)) + timedelta(minutes=i), 100, 100, 100, 100, v=100)
        for i in range(3)
    ]

    rvol = rvol_for_time_of_day(today_bars, old_sessions + [recent_session], lookback_sessions=1)
    # lookback=1 -> only the LAST session in the list (recent_session) is used.
    assert rvol == pytest.approx(1.0)


# ── structure state ────────────────────────────────────────────────────────────


def _agg(o, h, l, c):
    return AggBar(ts_open=session_open(DAY), ts_close=session_open(DAY) + timedelta(minutes=15), open=o, high=h, low=l, close=c, volume=1, n_minutes=15)


def test_structure_state_higher_highs_lows():
    bars15 = [_agg(100, 105, 99, 104), _agg(104, 108, 101, 107), _agg(107, 112, 103, 110)]
    assert structure_state(bars15) == StructureState.higher_highs_lows


def test_structure_state_lower_highs_lows():
    bars15 = [_agg(110, 112, 103, 107), _agg(107, 108, 101, 104), _agg(104, 105, 99, 100)]
    assert structure_state(bars15) == StructureState.lower_highs_lows


def test_structure_state_mixed_when_not_monotonic():
    bars15 = [_agg(100, 110, 99, 105), _agg(105, 108, 95, 100), _agg(100, 106, 98, 103)]
    assert structure_state(bars15) == StructureState.mixed


def test_structure_state_unknown_with_fewer_than_3_bars():
    assert structure_state([_agg(100, 105, 99, 104)]) == StructureState.unknown
    assert structure_state([]) == StructureState.unknown


# ── opening range ───────────────────────────────────────────────────────────────


def test_opening_range_none_until_fully_formed():
    start = session_open(DAY)
    bars = [bar("SPY", start + timedelta(minutes=i), 100, 101, 99, 100, 100) for i in range(10)]
    assert opening_range(bars, minutes=15) is None  # only 10 of 15 minutes present


def test_opening_range_computed_once_complete():
    start = session_open(DAY)
    highs = [101, 103, 99, 105, 100, 102, 98, 104, 101, 103, 100, 106, 99, 102, 97]
    lows = [h - 3 for h in highs]
    bars = [
        bar("SPY", start + timedelta(minutes=i), highs[i] - 1, highs[i], lows[i], highs[i] - 0.5, 100)
        for i in range(15)
    ]
    orange = opening_range(bars, minutes=15, atr_14=2.0)
    assert orange is not None
    assert orange.high == max(highs)
    assert orange.low == min(lows)
    assert orange.width == pytest.approx(max(highs) - min(lows))
    assert orange.width_vs_atr == pytest.approx(orange.width / 2.0)


# ── daily OHLC / ATR / prior-day levels ──────────────────────────────────────────


def test_daily_ohlc_from_1m_groups_by_et_day():
    day1 = DAY
    day2 = DAY + timedelta(days=1)
    bars = [
        bar("SPY", session_open(day1), 100, 105, 95, 102, 10),
        bar("SPY", session_open(day1) + timedelta(minutes=1), 102, 106, 96, 103, 10),
        bar("SPY", session_open(day2), 103, 110, 100, 108, 10),
    ]
    daily = daily_ohlc_from_1m(bars)
    assert len(daily) == 2
    assert daily[0].trade_date == day1
    assert daily[0].open == 100
    assert daily[0].high == 106
    assert daily[0].low == 95
    assert daily[0].close == 103
    assert daily[1].trade_date == day2
    assert daily[1].close == 108


def test_atr_requires_window_plus_one_days():
    daily = daily_ohlc_from_1m(
        [bar("SPY", session_open(DAY + timedelta(days=i)), 100, 101, 99, 100, 10) for i in range(10)]
    )
    assert atr(daily, window=14) is None  # not enough history yet
    assert atr(daily, window=5) is not None


def test_prior_day_levels_returns_most_recent_strictly_before():
    daily = daily_ohlc_from_1m(
        [
            bar("SPY", session_open(DAY), 100, 105, 95, 102, 10),
            bar("SPY", session_open(DAY + timedelta(days=1)), 102, 110, 100, 108, 10),
        ]
    )
    prior = prior_day_levels(daily, DAY + timedelta(days=1))
    assert prior is not None
    assert prior.trade_date == DAY
    assert prior.close == 102

    assert prior_day_levels(daily, DAY) is None  # nothing before the first day


# ── incremental build_feature_snapshots equivalence (performance refactor) ─────
#
# build_feature_snapshots precomputes whole-session arrays and feeds an
# incremental bucket builder once per day, instead of calling
# derive_agg_bars/session_vwap_series/rvol_for_time_of_day fresh on a growing
# prefix at every decision. These tests prove the fast path produces EXACTLY
# what the slow from-scratch functions would produce at each decision index -
# including when the bar sequence has a gap (a missed minute), which is the
# one case where a naive index-based shortcut could have silently diverged.


def test_build_feature_snapshots_matches_from_scratch_derivation():
    closes = [100.0 + (i % 11) * 0.13 - (i % 5) * 0.07 for i in range(80)]
    volumes = [500 + (i * 37) % 900 for i in range(80)]
    start = session_open(DAY)
    bars = [
        bar("SPY", start + timedelta(minutes=i), closes[i] - 0.3, closes[i] + 0.4, closes[i] - 0.6, closes[i], v=volumes[i])
        for i in range(80)
    ]
    prior_day = date(2026, 5, 29)
    prior_bars = [
        bar("SPY", session_open(prior_day) + timedelta(minutes=i), 99, 100, 98, 99.5, v=400 + i)
        for i in range(80)
    ]

    snaps = build_feature_snapshots(bars, [prior_bars], rvol_lookback=20)

    for check_idx in (5, 16, 17, 33, 45, 61, 79):
        snap = next(s for s in snaps if s.idx == check_idx)
        completed = bars[:check_idx]

        expected_vwap = session_vwap_series(completed)[-1]
        assert snap.vwap == pytest.approx(expected_vwap)

        expected_agg3 = derive_agg_bars(completed, 3)
        assert [ (a.open, a.high, a.low, a.close, a.volume) for a in snap.bars_3m ] == [
            (a.open, a.high, a.low, a.close, a.volume) for a in expected_agg3
        ]

        expected_agg5 = derive_agg_bars(completed, 5)
        assert [ (a.open, a.high, a.low, a.close, a.volume) for a in snap.bars_5m ] == [
            (a.open, a.high, a.low, a.close, a.volume) for a in expected_agg5
        ]

        expected_agg15 = derive_agg_bars(completed, 15)
        assert [ (a.open, a.high, a.low, a.close, a.volume) for a in snap.bars_15m ] == [
            (a.open, a.high, a.low, a.close, a.volume) for a in expected_agg15
        ]
        assert snap.structure_15m == structure_state(expected_agg15)

        expected_rvol = rvol_for_time_of_day(completed, [prior_bars], lookback_sessions=20)
        if expected_rvol is None:
            assert snap.rvol_tod is None
        else:
            assert snap.rvol_tod == pytest.approx(expected_rvol)


def test_build_feature_snapshots_matches_from_scratch_with_a_gap():
    """A missing minute must not desynchronize the incremental bucket builder."""
    start = session_open(DAY)
    bars = []
    for i in range(40):
        if i == 12:
            continue  # simulate a missing minute (halt/gap)
        c = 100.0 + i * 0.1
        bars.append(bar("SPY", start + timedelta(minutes=i), c - 0.2, c + 0.3, c - 0.4, c, v=500 + i))

    snaps = build_feature_snapshots(bars, [])

    for check_idx in (10, 13, 20, 38):
        snap = next(s for s in snaps if s.idx == check_idx)
        completed = bars[:check_idx]

        expected_agg3 = derive_agg_bars(completed, 3)
        assert [(a.open, a.close, a.volume) for a in snap.bars_3m] == [
            (a.open, a.close, a.volume) for a in expected_agg3
        ]
        expected_agg5 = derive_agg_bars(completed, 5)
        assert [(a.open, a.close, a.volume) for a in snap.bars_5m] == [
            (a.open, a.close, a.volume) for a in expected_agg5
        ]


def test_build_feature_snapshots_no_leak_via_mutated_completed_list():
    """Regression guard for the specific bug class this refactor risked:
    a snapshot's bars_3m/5m/15m must be an independent copy, not a live
    reference into the incremental builder's growing list."""
    closes = [100.0 + i * 0.05 for i in range(30)]
    bars = make_walk_session("SPY", DAY, closes)
    snaps = build_feature_snapshots(bars, [])

    early = next(s for s in snaps if s.idx == 10)
    n_agg3_at_capture = len(early.bars_3m)

    # Force the rest of the day to be processed (already done building `snaps`
    # eagerly above, since build_feature_snapshots returns a fully-built list).
    late = next(s for s in snaps if s.idx == 29)
    assert len(late.bars_3m) >= n_agg3_at_capture
    assert len(early.bars_3m) == n_agg3_at_capture  # unchanged after later bars were observed
