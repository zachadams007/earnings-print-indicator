"""SQLite persistence layer: migrations + CRUD for every schema."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Generator, TypeVar

from .models import (
    Approval,
    Bar,
    Candidate,
    CashReservation,
    DailyJournal,
    DurabilityJudgment,
    ForwardEvent,
    NewsItem,
    Order,
    TradeRecord,
)

T = TypeVar("T")

# ── migrations ──────────────────────────────────────────────────────────────
# Each tuple: (version: int, statements: list[str])
# Versions are applied in order; already-applied versions are skipped.

MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, [
        """CREATE TABLE IF NOT EXISTS news_items (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            symbol TEXT NOT NULL,
            related_symbols TEXT NOT NULL DEFAULT '[]',
            headline TEXT NOT NULL,
            body TEXT,
            published_ts TEXT,
            receipt_ts TEXT NOT NULL,
            source_url TEXT,
            category TEXT,
            raw_payload_hash TEXT NOT NULL,
            ingested_ts TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_items(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_news_receipt_ts ON news_items(receipt_ts)",
    ]),
    (2, [
        """CREATE TABLE IF NOT EXISTS bars (
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            vwap REAL,
            trade_count INTEGER,
            source TEXT NOT NULL,
            ingested_ts TEXT NOT NULL,
            PRIMARY KEY (symbol, ts, timeframe)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON bars(symbol, ts)",
    ]),
    (3, [
        """CREATE TABLE IF NOT EXISTS forward_events (
            event_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            scheduled_ts TEXT NOT NULL,
            known_since_ts TEXT NOT NULL,
            source TEXT NOT NULL,
            expected_move_pct REAL,
            consensus_context TEXT,
            direction_prior TEXT,
            confidence_prior REAL,
            raw_payload_hash TEXT NOT NULL,
            updated_ts TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_fwd_symbol ON forward_events(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_fwd_scheduled ON forward_events(scheduled_ts)",
    ]),
    (4, [
        """CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            underlying_symbol TEXT NOT NULL,
            instrument_symbol TEXT,
            direction TEXT NOT NULL,
            catalyst_ref_id TEXT NOT NULL,
            catalyst_type TEXT NOT NULL,
            created_ts TEXT NOT NULL,
            entry_trigger TEXT NOT NULL,
            entry_score REAL NOT NULL,
            calibrated_prob REAL NOT NULL,
            expected_hold TEXT NOT NULL,
            stop_pct REAL NOT NULL,
            status TEXT NOT NULL,
            rejection_reason TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_cand_status ON candidates(status)",
        "CREATE INDEX IF NOT EXISTS idx_cand_created ON candidates(created_ts)",
        "CREATE INDEX IF NOT EXISTS idx_cand_symbol ON candidates(underlying_symbol)",
    ]),
    (5, [
        """CREATE TABLE IF NOT EXISTS durability_judgments (
            judgment_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            mode TEXT NOT NULL,
            direction TEXT NOT NULL,
            catalyst_type TEXT NOT NULL,
            durability_horizon TEXT NOT NULL,
            novelty_score REAL NOT NULL,
            magnitude_score REAL NOT NULL,
            specificity_score REAL NOT NULL,
            credibility_score REAL NOT NULL,
            trap_risk_score REAL NOT NULL,
            trap_flags TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL,
            calibrated_prob REAL,
            reason_short TEXT NOT NULL,
            raw_response_hash TEXT NOT NULL,
            created_ts TEXT NOT NULL,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_judg_candidate ON durability_judgments(candidate_id)",
        "CREATE INDEX IF NOT EXISTS idx_judg_catalyst_type ON durability_judgments(catalyst_type)",
    ]),
    (6, [
        """CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            broker_order_id TEXT,
            candidate_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            order_type TEXT NOT NULL,
            limit_price REAL,
            stop_price REAL,
            time_in_force TEXT NOT NULL,
            intended_notional REAL NOT NULL,
            reserved_cash REAL NOT NULL,
            max_loss REAL NOT NULL,
            status TEXT NOT NULL,
            submitted_ts TEXT,
            updated_ts TEXT NOT NULL,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_orders_candidate ON orders(candidate_id)",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_idem ON orders(idempotency_key)",
    ]),
    (7, [
        """CREATE TABLE IF NOT EXISTS trade_records (
            trade_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            entry_order_id TEXT NOT NULL,
            exit_order_id TEXT,
            mode TEXT NOT NULL,
            underlying_symbol TEXT NOT NULL,
            instrument_symbol TEXT NOT NULL,
            leverage_multiplier REAL NOT NULL,
            entry_ts TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_ts TEXT,
            exit_price REAL,
            qty REAL NOT NULL,
            gross_pnl REAL,
            net_pnl REAL,
            max_adverse_excursion REAL,
            max_favorable_excursion REAL,
            hold_minutes INTEGER,
            exit_reason TEXT,
            settlement_date TEXT NOT NULL,
            created_ts TEXT NOT NULL,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id),
            FOREIGN KEY (entry_order_id) REFERENCES orders(order_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_trades_candidate ON trade_records(candidate_id)",
        "CREATE INDEX IF NOT EXISTS idx_trades_mode ON trade_records(mode)",
        "CREATE INDEX IF NOT EXISTS idx_trades_entry_ts ON trade_records(entry_ts)",
    ]),
    (8, [
        """CREATE TABLE IF NOT EXISTS cash_reservations (
            reservation_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            amount REAL NOT NULL,
            reserved_at TEXT NOT NULL,
            released_at TEXT,
            status TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_cash_order ON cash_reservations(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_cash_status ON cash_reservations(status)",
    ]),
    (9, [
        """CREATE TABLE IF NOT EXISTS approvals (
            approval_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            status TEXT NOT NULL,
            decided_at TEXT,
            expires_at TEXT NOT NULL,
            operator_message TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_approvals_order ON approvals(order_id)",
    ]),
    (10, [
        """CREATE TABLE IF NOT EXISTS daily_journals (
            journal_id TEXT PRIMARY KEY,
            trading_day TEXT NOT NULL UNIQUE,
            total_candidates INTEGER NOT NULL DEFAULT 0,
            candidates_eligible INTEGER NOT NULL DEFAULT 0,
            candidates_rejected INTEGER NOT NULL DEFAULT 0,
            orders_submitted INTEGER NOT NULL DEFAULT 0,
            orders_filled INTEGER NOT NULL DEFAULT 0,
            trades_closed INTEGER NOT NULL DEFAULT 0,
            gross_pnl REAL NOT NULL DEFAULT 0.0,
            net_pnl REAL NOT NULL DEFAULT 0.0,
            safety_guard_blocks INTEGER NOT NULL DEFAULT 0,
            summary_text TEXT NOT NULL DEFAULT '',
            created_ts TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_journals_day ON daily_journals(trading_day)",
    ]),
    (11, [
        # Additive columns for backtest diagnosis: gap sign/magnitude and trade direction.
        # ALTER TABLE ADD COLUMN is safe on SQLite; new columns default to NULL for all
        # existing live-trade rows.  INSERT OR REPLACE in upsert_trade handles them
        # transparently - live trades simply omit them (None → NULL).
        "ALTER TABLE trade_records ADD COLUMN gap_pct REAL",
        "ALTER TABLE trade_records ADD COLUMN direction TEXT",
    ]),
    (12, [
        # Crypto daily OHLCV bars.  Separate table so the equities 'bars' table
        # schema and its existing indices/tests remain untouched.
        # PRIMARY KEY is (symbol, ts, timeframe, exchange) - allows storing the
        # same candle from multiple exchanges for cross-venue comparison later.
        """CREATE TABLE IF NOT EXISTS crypto_bars (
            symbol      TEXT NOT NULL,
            ts          TEXT NOT NULL,
            timeframe   TEXT NOT NULL,
            open        REAL NOT NULL,
            high        REAL NOT NULL,
            low         REAL NOT NULL,
            close       REAL NOT NULL,
            volume      REAL NOT NULL,
            exchange    TEXT NOT NULL,
            ingested_ts TEXT NOT NULL,
            PRIMARY KEY (symbol, ts, timeframe, exchange)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_crypto_bars_symbol_ts ON crypto_bars(symbol, ts)",
    ]),
    (13, [
        # Perpetual-swap funding-rate history for cash-and-carry analysis.
        # Raw 8-hour records stored as-is; daily aggregation is left to the
        # strategy layer.  PRIMARY KEY is (symbol, ts, exchange) - one row per
        # funding payment per exchange (allows multi-exchange comparison later).
        """CREATE TABLE IF NOT EXISTS funding_rates (
            symbol      TEXT NOT NULL,
            ts          TEXT NOT NULL,
            funding_rate REAL NOT NULL,
            exchange    TEXT NOT NULL,
            ingested_ts TEXT NOT NULL,
            PRIMARY KEY (symbol, ts, exchange)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_funding_rates_symbol_ts ON funding_rates(symbol, ts)",
    ]),
]


# ── serialization helpers ───────────────────────────────────────────────────


def _prep(data: dict) -> dict:
    """Convert Python types to SQLite-compatible scalars."""
    result: dict = {}
    for key, val in data.items():
        if val is None:
            result[key] = None
        elif isinstance(val, datetime):
            result[key] = val.isoformat()
        elif isinstance(val, date):
            result[key] = val.isoformat()
        elif isinstance(val, (list, dict)):
            result[key] = json.dumps(val)
        else:
            result[key] = val
    return result


def _parse(data: dict, json_fields: frozenset[str]) -> dict:
    """Decode JSON list/dict fields so Pydantic gets proper Python types."""
    for field in json_fields:
        if field in data and data[field] is not None:
            data[field] = json.loads(data[field])
    return data


def _upsert(conn: sqlite3.Connection, table: str, data: dict) -> None:
    cols = ", ".join(data.keys())
    placeholders = ", ".join(f":{k}" for k in data.keys())
    conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})", data)


# ── store ───────────────────────────────────────────────────────────────────


class Store:
    """SQLite persistence layer. One connection per operation; WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def run_migrations(self) -> None:
        """Apply all pending schema migrations (idempotent)."""
        with self._db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)

        with self._db() as conn:
            applied = {
                row[0] for row in conn.execute("SELECT version FROM schema_migrations")
            }
            for version, stmts in MIGRATIONS:
                if version not in applied:
                    for stmt in stmts:
                        conn.execute(stmt)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                        (version, datetime.now(UTC).isoformat()),
                    )

    # ── NewsItem ────────────────────────────────────────────────────────────

    _NEWS_JSON = frozenset({"related_symbols"})

    def upsert_news_item(self, item: NewsItem) -> None:
        with self._db() as conn:
            _upsert(conn, "news_items", _prep(item.model_dump()))

    def get_news_item(self, id: str) -> NewsItem | None:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM news_items WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        return NewsItem.model_validate(_parse(dict(row), self._NEWS_JSON))

    # ── Bar ─────────────────────────────────────────────────────────────────

    def upsert_bar(self, bar: Bar) -> None:
        with self._db() as conn:
            _upsert(conn, "bars", _prep(bar.model_dump()))

    def bulk_upsert_bars(self, bars: list[Bar]) -> int:
        """Upsert many bars in ONE connection + transaction (executemany).

        ``upsert_bar`` opens a fresh connection and commits (fsync) per row - fine
        for a handful of bars, ~200 rows/s for a backfill. This bulk path commits
        once for the whole batch (100–1000x faster) and is used for OOS ingest.
        Returns the number of rows written.
        """
        if not bars:
            return 0
        rows = [_prep(b.model_dump()) for b in bars]
        cols = list(rows[0].keys())
        col_sql = ", ".join(cols)
        placeholders = ", ".join(f":{c}" for c in cols)
        sql = f"INSERT OR REPLACE INTO bars ({col_sql}) VALUES ({placeholders})"
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executemany(sql, rows)
            conn.commit()
        finally:
            conn.close()
        return len(rows)

    def get_bar(self, symbol: str, ts: datetime, timeframe: str) -> Bar | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM bars WHERE symbol = ? AND ts = ? AND timeframe = ?",
                (symbol, ts.isoformat(), timeframe),
            ).fetchone()
        if row is None:
            return None
        return Bar.model_validate(dict(row))

    def get_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1m"
    ) -> list[Bar]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM bars WHERE symbol = ? AND timeframe = ? AND ts >= ? AND ts <= ? ORDER BY ts",
                (symbol, timeframe, start.isoformat(), end.isoformat()),
            ).fetchall()
        return [Bar.model_validate(dict(r)) for r in rows]

    # ── ForwardEvent ────────────────────────────────────────────────────────

    def upsert_forward_event(self, event: ForwardEvent) -> None:
        with self._db() as conn:
            _upsert(conn, "forward_events", _prep(event.model_dump()))

    def get_forward_event(self, event_id: str) -> ForwardEvent | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM forward_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        return ForwardEvent.model_validate(dict(row))

    # ── Candidate ───────────────────────────────────────────────────────────

    def upsert_candidate(self, candidate: Candidate) -> None:
        with self._db() as conn:
            _upsert(conn, "candidates", _prep(candidate.model_dump()))

    def get_candidate(self, candidate_id: str) -> Candidate | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            return None
        return Candidate.model_validate(dict(row))

    # ── DurabilityJudgment ──────────────────────────────────────────────────

    _JUDG_JSON = frozenset({"trap_flags"})

    def upsert_judgment(self, judgment: DurabilityJudgment) -> None:
        with self._db() as conn:
            _upsert(conn, "durability_judgments", _prep(judgment.model_dump()))

    def get_judgment(self, judgment_id: str) -> DurabilityJudgment | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM durability_judgments WHERE judgment_id = ?", (judgment_id,)
            ).fetchone()
        if row is None:
            return None
        return DurabilityJudgment.model_validate(_parse(dict(row), self._JUDG_JSON))

    # ── Order ───────────────────────────────────────────────────────────────

    def upsert_order(self, order: Order) -> None:
        with self._db() as conn:
            _upsert(conn, "orders", _prep(order.model_dump()))

    def get_order(self, order_id: str) -> Order | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
        if row is None:
            return None
        return Order.model_validate(dict(row))

    def get_order_by_idempotency_key(self, key: str) -> Order | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE idempotency_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return Order.model_validate(dict(row))

    # ── TradeRecord ─────────────────────────────────────────────────────────

    def upsert_trade(self, trade: TradeRecord) -> None:
        with self._db() as conn:
            _upsert(conn, "trade_records", _prep(trade.model_dump()))

    def get_trade(self, trade_id: str) -> TradeRecord | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM trade_records WHERE trade_id = ?", (trade_id,)
            ).fetchone()
        if row is None:
            return None
        return TradeRecord.model_validate(dict(row))

    # ── CashReservation ─────────────────────────────────────────────────────

    def upsert_cash_reservation(self, res: CashReservation) -> None:
        with self._db() as conn:
            _upsert(conn, "cash_reservations", _prep(res.model_dump()))

    def get_cash_reservation(self, reservation_id: str) -> CashReservation | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM cash_reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
        if row is None:
            return None
        return CashReservation.model_validate(dict(row))

    def get_active_reservations(self) -> list[CashReservation]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM cash_reservations WHERE status = 'active'"
            ).fetchall()
        return [CashReservation.model_validate(dict(r)) for r in rows]

    # ── Approval ────────────────────────────────────────────────────────────

    def upsert_approval(self, approval: Approval) -> None:
        with self._db() as conn:
            _upsert(conn, "approvals", _prep(approval.model_dump()))

    def get_approval(self, approval_id: str) -> Approval | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        if row is None:
            return None
        return Approval.model_validate(dict(row))

    # ── DailyJournal ────────────────────────────────────────────────────────

    def upsert_journal(self, journal: DailyJournal) -> None:
        with self._db() as conn:
            _upsert(conn, "daily_journals", _prep(journal.model_dump()))

    def get_journal(self, trading_day: date) -> DailyJournal | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM daily_journals WHERE trading_day = ?",
                (trading_day.isoformat(),),
            ).fetchone()
        if row is None:
            return None
        return DailyJournal.model_validate(dict(row))

    # ── CryptoBar ────────────────────────────────────────────────────────────

    def upsert_crypto_bar(self, bar: dict) -> None:
        """Insert or replace one crypto bar dict (keys must match crypto_bars schema)."""
        with self._db() as conn:
            _upsert(conn, "crypto_bars", bar)

    def bulk_upsert_crypto_bars(self, bars: list[dict]) -> int:
        """Upsert many crypto bars in one transaction. Returns row count.

        Mirrors bulk_upsert_bars: one connection, one commit, ~100-1000x faster
        than calling upsert_crypto_bar() in a loop.
        """
        if not bars:
            return 0
        cols = list(bars[0].keys())
        col_sql = ", ".join(cols)
        placeholders = ", ".join(f":{c}" for c in cols)
        sql = f"INSERT OR REPLACE INTO crypto_bars ({col_sql}) VALUES ({placeholders})"
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executemany(sql, bars)
            conn.commit()
        finally:
            conn.close()
        return len(bars)

    def get_crypto_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
        exchange: str | None = None,
    ) -> list[dict]:
        """Return crypto bar dicts for a symbol in [start, end], ordered by ts.

        Args:
            symbol: e.g. "BTC/USD"
            start: inclusive lower bound (UTC datetime)
            end: inclusive upper bound (UTC datetime)
            timeframe: e.g. "1d"
            exchange: optional filter; None returns all exchanges

        Returns:
            List of row dicts matching the crypto_bars table schema.
        """
        params: list = [symbol, timeframe, start.isoformat(), end.isoformat()]
        exchange_clause = ""
        if exchange is not None:
            exchange_clause = "AND exchange = ?"
            params.append(exchange)

        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT * FROM crypto_bars
                   WHERE symbol = ? AND timeframe = ?
                     AND ts >= ? AND ts <= ?
                     {exchange_clause}
                   ORDER BY ts""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_crypto_bar_count(self, symbol: str, timeframe: str = "1d") -> int:
        """Return the number of crypto bars stored for a symbol."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM crypto_bars WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            ).fetchone()
        return int(row[0])

    def get_crypto_bar_date_range(
        self, symbol: str, timeframe: str = "1d"
    ) -> tuple[str | None, str | None]:
        """Return (min_ts, max_ts) ISO strings for a symbol, or (None, None) if empty."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM crypto_bars WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    # ── FundingRate ──────────────────────────────────────────────────────────

    def upsert_funding_rate(self, record: dict) -> None:
        """Insert or replace one funding-rate record dict.

        Required keys: symbol, ts, funding_rate, exchange, ingested_ts.
        """
        with self._db() as conn:
            _upsert(conn, "funding_rates", record)

    def bulk_upsert_funding_rates(self, records: list[dict]) -> int:
        """Upsert many funding-rate records in one transaction. Returns row count.

        Mirrors bulk_upsert_crypto_bars: one connection, one commit.
        """
        if not records:
            return 0
        cols = list(records[0].keys())
        col_sql = ", ".join(cols)
        placeholders = ", ".join(f":{c}" for c in cols)
        sql = f"INSERT OR REPLACE INTO funding_rates ({col_sql}) VALUES ({placeholders})"
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executemany(sql, records)
            conn.commit()
        finally:
            conn.close()
        return len(records)

    def get_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        exchange: str | None = None,
    ) -> list[dict]:
        """Return funding-rate records for a symbol in [start, end], ordered by ts.

        Args:
            symbol:   e.g. "BTC/USDT:USDT"
            start:    inclusive lower bound (UTC datetime)
            end:      inclusive upper bound (UTC datetime)
            exchange: optional filter; None returns all exchanges

        Returns:
            List of row dicts matching the funding_rates table schema.
        """
        params: list = [symbol, start.isoformat(), end.isoformat()]
        exchange_clause = ""
        if exchange is not None:
            exchange_clause = "AND exchange = ?"
            params.append(exchange)

        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT * FROM funding_rates
                   WHERE symbol = ?
                     AND ts >= ? AND ts <= ?
                     {exchange_clause}
                   ORDER BY ts""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_funding_rate_count(self, symbol: str, exchange: str | None = None) -> int:
        """Return the number of funding-rate records stored for a symbol."""
        params: list = [symbol]
        exchange_clause = ""
        if exchange is not None:
            exchange_clause = "AND exchange = ?"
            params.append(exchange)
        with self._db() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM funding_rates WHERE symbol = ? {exchange_clause}",
                params,
            ).fetchone()
        return int(row[0])

    def get_funding_rate_date_range(
        self, symbol: str, exchange: str | None = None
    ) -> tuple[str | None, str | None]:
        """Return (min_ts, max_ts) ISO strings for a symbol, or (None, None) if empty."""
        params: list = [symbol]
        exchange_clause = ""
        if exchange is not None:
            exchange_clause = "AND exchange = ?"
            params.append(exchange)
        with self._db() as conn:
            row = conn.execute(
                f"SELECT MIN(ts), MAX(ts) FROM funding_rates WHERE symbol = ? {exchange_clause}",
                params,
            ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def applied_migrations(self) -> list[int]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        return [r[0] for r in rows]

    # ── Backtest query helpers ───────────────────────────────────────────────

    def get_news_visible_at(
        self,
        ts: datetime,
        symbol: str,
        lookback_hours: float = 18.0,
    ) -> list[NewsItem]:
        """Return news items visible (receipt_ts <= ts) for a symbol, within lookback window.

        PIT contract: receipt_ts must be <= ts for a news item to be "known" at bar time ts.
        """
        from datetime import timedelta
        cutoff = (ts - timedelta(hours=lookback_hours)).isoformat()
        ts_str = ts.isoformat()
        with self._db() as conn:
            rows = conn.execute(
                """SELECT * FROM news_items
                   WHERE receipt_ts <= ? AND receipt_ts >= ?
                     AND (symbol = ? OR related_symbols LIKE ?)
                   ORDER BY receipt_ts""",
                (ts_str, cutoff, symbol, f'%"{symbol}"%'),
            ).fetchall()
        return [NewsItem.model_validate(_parse(dict(r), self._NEWS_JSON)) for r in rows]

    def get_bars_for_day(
        self, symbol: str, trade_date: "date", timeframe: str = "1m"
    ) -> list[Bar]:
        """Return all bars for a symbol on a given calendar date (NY timezone)."""
        day_str = trade_date.isoformat()
        with self._db() as conn:
            rows = conn.execute(
                """SELECT * FROM bars
                   WHERE symbol = ? AND timeframe = ?
                     AND substr(ts, 1, 10) = ?
                   ORDER BY ts""",
                (symbol, timeframe, day_str),
            ).fetchall()
        return [Bar.model_validate(dict(r)) for r in rows]

    def get_bars_for_symbols_days(
        self,
        symbols: list[str],
        start_date: "date",
        end_date: "date",
        timeframe: str = "1m",
    ) -> dict[tuple[str, "date"], list[Bar]]:
        """Return bars grouped by (symbol, calendar date) for an inclusive date range."""
        if not symbols or start_date > end_date:
            return {}

        placeholders = ",".join("?" for _ in symbols)
        start_str = start_date.isoformat()
        end_exclusive = (end_date + timedelta(days=1)).isoformat()

        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT * FROM bars
                   WHERE symbol IN ({placeholders}) AND timeframe = ?
                     AND ts >= ? AND ts < ?
                   ORDER BY symbol, ts""",
                (*symbols, timeframe, start_str, end_exclusive),
            ).fetchall()

        grouped: dict[tuple[str, date], list[Bar]] = {}
        for row in rows:
            raw = dict(row)
            day = date.fromisoformat(raw["ts"][:10])
            bar = Bar.model_validate(raw)
            grouped.setdefault((bar.symbol, day), []).append(bar)
        return grouped

    def get_prior_close(
        self, symbol: str, trade_date: "date", timeframe: str = "1m"
    ) -> float | None:
        """Return the last close price before trade_date for a symbol."""
        day_str = trade_date.isoformat()
        with self._db() as conn:
            row = conn.execute(
                """SELECT close FROM bars
                   WHERE symbol = ? AND timeframe = ? AND substr(ts, 1, 10) < ?
                   ORDER BY ts DESC LIMIT 1""",
                (symbol, timeframe, day_str),
            ).fetchone()
        return float(row[0]) if row else None

    def get_distinct_symbols(self, timeframe: str = "1m") -> list[str]:
        """Return all distinct symbols in the bars table."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM bars WHERE timeframe = ? ORDER BY symbol",
                (timeframe,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_trading_dates(
        self, symbol: str, start: "date", end: "date", timeframe: str = "1m"
    ) -> list["date"]:
        """Return sorted list of calendar dates that have bars for a symbol."""
        from datetime import date as date_type
        with self._db() as conn:
            rows = conn.execute(
                """SELECT DISTINCT substr(ts, 1, 10) as d FROM bars
                   WHERE symbol = ? AND timeframe = ?
                     AND d >= ? AND d <= ?
                   ORDER BY d""",
                (symbol, timeframe, start.isoformat(), end.isoformat()),
            ).fetchall()
        return [date_type.fromisoformat(r[0]) for r in rows]

    def get_bars_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> list[Bar]:
        """Return bars in (start, end] inclusive range ordered by ts."""
        return self.get_bars(symbol, start, end, timeframe)

    def get_all_bars_for_range(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> list[Bar]:
        """Return bars for multiple symbols in a time range, sorted by ts then symbol."""
        if not symbols:
            return []
        placeholders = ",".join("?" for _ in symbols)
        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT * FROM bars
                   WHERE symbol IN ({placeholders}) AND timeframe = ?
                     AND ts >= ? AND ts <= ?
                   ORDER BY ts, symbol""",
                (*symbols, timeframe, start.isoformat(), end.isoformat()),
            ).fetchall()
        return [Bar.model_validate(dict(r)) for r in rows]

    def get_all_news_for_range(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[NewsItem]:
        """Return all news items for symbols in [start, end], sorted by receipt_ts."""
        if not symbols:
            return []
        placeholders = ",".join("?" for _ in symbols)
        with self._db() as conn:
            rows = conn.execute(
                f"""SELECT * FROM news_items
                   WHERE symbol IN ({placeholders})
                     AND receipt_ts >= ? AND receipt_ts <= ?
                   ORDER BY receipt_ts""",
                (*symbols, start.isoformat(), end.isoformat()),
            ).fetchall()
        return [NewsItem.model_validate(_parse(dict(r), self._NEWS_JSON)) for r in rows]

    def get_trade_records(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        mode: str | None = None,
    ) -> list[TradeRecord]:
        """Return closed trade records, optionally filtered by date range and mode."""
        conditions: list[str] = ["exit_ts IS NOT NULL"]
        params: list = []
        if start:
            conditions.append("entry_ts >= ?")
            params.append(start.isoformat())
        if end:
            conditions.append("entry_ts <= ?")
            params.append(end.isoformat())
        if mode:
            conditions.append("mode = ?")
            params.append(mode)
        where = " AND ".join(conditions)
        with self._db() as conn:
            rows = conn.execute(
                f"SELECT * FROM trade_records WHERE {where} ORDER BY entry_ts",
                params,
            ).fetchall()
        return [TradeRecord.model_validate(dict(r)) for r in rows]
