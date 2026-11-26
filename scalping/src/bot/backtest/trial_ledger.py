"""Append-only trial ledger for multiple-testing audit trail.

Records each distinct strategy config tried so the Deflated Sharpe Ratio (DSR)
penalty can be applied with the correct N.  Only the headline OOS gate metric
is penalised; per-fold metrics remain at N=1 because folds are the temporal
validation unit, not independent strategy guesses.

File format: one JSON object per line (JSONL).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Default ledger location.  Tests override via the ledger_path parameter.
DEFAULT_LEDGER_PATH: Path = Path("data/trial_ledger.jsonl")


def record_trial(
    config_id: str,
    *,
    window: Optional[str] = None,
    metrics: Optional[dict] = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> None:
    """Append one trial record to the ledger.

    Args:
        config_id: Stable identifier for the strategy variant being evaluated
            (e.g. ``"step2_llm_taxonomy"``).  Must be caller-defined and
            consistent across runs so duplicates can be detected.
        window: Optional human-readable date window string (e.g. ``"2023-01-01/2024-01-01"``).
        metrics: Optional dict of metric values to snapshot alongside the record.
        ledger_path: Path to the JSONL ledger file.  Parent dirs are created if missing.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "config_id": config_id,
        "window": window,
        "metrics": metrics,
    }
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def count_trials(ledger_path: Path = DEFAULT_LEDGER_PATH) -> int:
    """Return total number of records appended (including duplicates).

    Returns 0 if the file does not exist.
    """
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return 0
    count = 0
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                count += 1
    return count


def count_distinct_configs(ledger_path: Path = DEFAULT_LEDGER_PATH) -> int:
    """Return the number of distinct config_id values in the ledger.

    Returns 0 if the file does not exist.
    """
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return 0
    seen: set[str] = set()
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("config_id")
                if cid is not None:
                    seen.add(cid)
            except json.JSONDecodeError:
                pass  # skip malformed lines rather than crashing a gate run
    return len(seen)
