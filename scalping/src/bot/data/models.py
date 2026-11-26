"""Pydantic schemas for every persisted record type (build-plan §5)."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── enums ──────────────────────────────────────────────────────────────────


class Mode(str, Enum):
    reactive = "reactive"
    anticipatory = "anticipatory"


class Direction(str, Enum):
    positive = "positive"
    negative = "negative"
    unclear = "unclear"


class DurabilityHorizon(str, Enum):
    minutes = "minutes"
    hours = "hours"
    one_day = "one_day"
    two_days = "two_days"
    avoid = "avoid"


class CandidateStatus(str, Enum):
    seen = "seen"
    eligible = "eligible"
    rejected = "rejected"
    ordered = "ordered"
    traded = "traded"
    expired = "expired"


class OrderSide(str, Enum):
    buy = "buy"
    sell = "sell"


class OrderStatus(str, Enum):
    pending = "pending"
    submitted = "submitted"
    filled = "filled"
    canceled = "canceled"
    rejected = "rejected"


class ReservationStatus(str, Enum):
    active = "active"
    released = "released"
    expired = "expired"


class ApprovalStatus(str, Enum):
    approved = "approved"
    rejected = "rejected"
    expired = "expired"
    operator_error = "operator_error"


# ── data ingestion schemas ──────────────────────────────────────────────────


class NewsItem(BaseModel):
    """Raw Benzinga (or other provider) news item with PIT receipt timestamp."""

    id: str
    provider: str
    symbol: str
    related_symbols: list[str] = Field(default_factory=list)
    headline: str
    body: Optional[str] = None
    published_ts: Optional[datetime] = None
    receipt_ts: datetime
    source_url: Optional[str] = None
    category: Optional[str] = None
    raw_payload_hash: str
    ingested_ts: datetime


class Bar(BaseModel):
    """OHLCV bar from Alpaca IEX or SIP feed."""

    symbol: str
    ts: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    source: str
    ingested_ts: datetime


class ForwardEvent(BaseModel):
    """A scheduled forward catalyst (earnings, FDA/PDUFA, econ, etc.)."""

    event_id: str
    symbol: str
    event_type: str
    scheduled_ts: datetime
    known_since_ts: datetime
    source: str
    expected_move_pct: Optional[float] = None
    consensus_context: Optional[str] = None
    direction_prior: Optional[str] = None
    confidence_prior: Optional[float] = None
    raw_payload_hash: str
    updated_ts: datetime


# ── signal / evaluation schemas ─────────────────────────────────────────────


class Candidate(BaseModel):
    """A candidate trade opportunity, from news/events through entry gate."""

    candidate_id: str
    mode: Mode
    underlying_symbol: str
    instrument_symbol: Optional[str] = None
    direction: Direction
    catalyst_ref_id: str
    catalyst_type: str
    created_ts: datetime
    entry_trigger: str
    entry_score: float
    calibrated_prob: float
    expected_hold: DurabilityHorizon
    stop_pct: float
    status: CandidateStatus
    rejection_reason: Optional[str] = None


class DurabilityJudgment(BaseModel):
    """Structured LLM output for catalyst durability, validated and stored for calibration."""

    judgment_id: str
    candidate_id: str
    model: str
    prompt_version: str
    mode: Mode
    direction: Direction
    catalyst_type: str
    durability_horizon: DurabilityHorizon
    novelty_score: float
    magnitude_score: float
    specificity_score: float
    credibility_score: float
    trap_risk_score: float
    trap_flags: list[str] = Field(default_factory=list)
    confidence: float
    calibrated_prob: Optional[float] = None
    reason_short: str
    raw_response_hash: str
    created_ts: datetime


# ── execution schemas ───────────────────────────────────────────────────────


class Order(BaseModel):
    """An order intent record - created before any broker call, never modified."""

    order_id: str
    idempotency_key: str
    broker_order_id: Optional[str] = None
    candidate_id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str
    intended_notional: float
    reserved_cash: float
    max_loss: float
    status: OrderStatus
    submitted_ts: Optional[datetime] = None
    updated_ts: datetime


class TradeRecord(BaseModel):
    """Completed (or open) trade in the outcome ledger."""

    trade_id: str
    candidate_id: str
    entry_order_id: str
    exit_order_id: Optional[str] = None
    mode: Mode
    underlying_symbol: str
    instrument_symbol: str
    leverage_multiplier: float
    entry_ts: datetime
    entry_price: float
    exit_ts: Optional[datetime] = None
    exit_price: Optional[float] = None
    qty: float
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    hold_minutes: Optional[int] = None
    exit_reason: Optional[str] = None
    settlement_date: date
    created_ts: datetime
    # Backtest annotation fields (nullable; NULL for live trades)
    gap_pct: Optional[float] = None   # overnight gap that triggered this entry (signed %)
    direction: Optional[str] = None   # "positive" | "negative" inferred from gap sign


# ── safety / settlement schemas ─────────────────────────────────────────────


class CashReservation(BaseModel):
    """Settled-cash reservation held against a pending order (§8)."""

    reservation_id: str
    order_id: str
    amount: float
    reserved_at: datetime
    released_at: Optional[datetime] = None
    status: ReservationStatus


class Approval(BaseModel):
    """Telegram human-approval record (§4 - Approve Gate)."""

    approval_id: str
    order_id: str
    candidate_id: str
    status: ApprovalStatus
    decided_at: Optional[datetime] = None
    expires_at: datetime
    operator_message: Optional[str] = None


# ── journal schema ──────────────────────────────────────────────────────────


class DailyJournal(BaseModel):
    """Machine-readable daily summary row (§4 - Daily Journal)."""

    journal_id: str
    trading_day: date
    total_candidates: int = 0
    candidates_eligible: int = 0
    candidates_rejected: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    trades_closed: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    safety_guard_blocks: int = 0
    summary_text: str = ""
    created_ts: datetime
