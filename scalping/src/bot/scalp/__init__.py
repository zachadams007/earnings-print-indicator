"""Stage A no-leakage underlying-direction backtest harness.

Lane A of the SPY/QQQ options scalping copilot super-project. Tests whether
the Long / Short / Wait confluence rubric predicts SPY/QQQ direction over
5/10/15/30/60 minutes, on underlying 1-minute bars only (no options data).

See docs/SCALPING-COPILOT-DATA-FEASIBILITY-2026-07-08.md section 5 ("Stage A")
for the spec this package implements, and
docs/SCALPING-COPILOT-CURSOR-RESEARCH-2026-07-08.md section 3 for the
3-of-5 confirmation rubric shape.

This package is UNRELATED to the crypto funding-harvester lane (project rules
project identity). Do not cite crypto backtests as evidence here, and do not
import from bot.strategies.crypto / bot.live.

Modules
-------
features.py       Point-in-time feature derivation (VWAP, RVOL-TOD, structure, ORB).
rules.py          Parameterized Long/Short/Wait confluence decision function (v1 placeholder).
rules_v01.py      v0.1 core-gate decision model with toggleable trigger families.
rules_v02.py      v0.2 families (band reversal, retest entries) for sweep v2.
exits.py          R-multiple exit engine (structure/ATR stops, targets, time-stops).
evaluate.py       Forward-return evaluation, hit rate, expectancy, MFE/MAE, folds.
evaluate_v2.py    R-multiple evaluation for sweep v2.
controls.py       Negative controls (shuffled bars, delayed signal, random labels).
synthetic.py      Deterministic synthetic 1m bar fixtures (tests + no-data fallback).
noise_boundary.py Independent Zarattini/Aziz/Barbon "Beat the Market" replication;
                  deliberately standalone - does not import sibling modules (see its
                  module docstring for the audit trail and ts-convention rationale).
"""
