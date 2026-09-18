"""SQLAlchemy models and engine for the learn-anything backend."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.orm import Session as DBSession

# --- Engine (PostgreSQL) ---

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Expected e.g. postgresql+psycopg://user:pass@host:5432/db"
    )

engine = create_engine(DATABASE_URL)


class Base(DeclarativeBase):
    pass


SessionFactory = sessionmaker(bind=engine)


# --- Enum ---


class Phase(str, Enum):
    CLARIFYING = "clarifying"
    PROBING = "probing"
    PLANNING = "planning"
    REVIEWING = "reviewing"
    GENERATING = "generating"
    EXECUTING = "executing"
    COMPLETE = "complete"
    ERROR = "error"


# --- Models ---


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    phase: Mapped[str] = mapped_column(String, default=Phase.CLARIFYING.value)
    goal: Mapped[str] = mapped_column(Text)
    # Human-readable language name detected from the learner's text (e.g.
    # "Spanish"). Drives the language of every user-facing reply and the
    # generated materials. Nullable for legacy rows; readers fall back to
    # language.DEFAULT_LANGUAGE.
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    narrowed_goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    boundary_map: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Timezone-aware (UTC) so values read back carry an explicit offset and
    # serialize as RFC 3339 date-times. A bare wall-clock string is ambiguous
    # and clients parse it as local time.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Relationships
    probe_questions: Mapped[list[ProbeQuestion]] = relationship(
        back_populates="session", cascade="all, delete"
    )
    plan: Mapped[Plan | None] = relationship(
        back_populates="session", uselist=False, cascade="all, delete"
    )
    materials: Mapped[list[StepMaterial]] = relationship(
        back_populates="session", cascade="all, delete"
    )
    step_progress: Mapped[list[StepProgress]] = relationship(
        back_populates="session", cascade="all, delete"
    )
    slide_contents: Mapped[list[SlideContent]] = relationship(
        back_populates="session", cascade="all, delete"
    )


class ProbeQuestion(Base):
    __tablename__ = "probe_questions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True
    )
    question_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    text: Mapped[str] = mapped_column(Text)
    options: Mapped[list[str]] = mapped_column(JSON)
    correct_index: Mapped[int] = mapped_column(Integer)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    strand: Mapped[str] = mapped_column(String)
    difficulty: Mapped[int] = mapped_column(Integer)
    # Answer (null until answered)
    selected_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    answered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    session: Mapped[Session] = relationship(back_populates="probe_questions")


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), unique=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    prose_summary: Mapped[str] = mapped_column(Text)
    dependency_dag: Mapped[str] = mapped_column(Text)
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    adjustments: Mapped[list[str]] = mapped_column(JSON, default=list)

    session: Mapped[Session] = relationship(back_populates="plan")


class StepMaterial(Base):
    __tablename__ = "step_materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True
    )
    step_id: Mapped[str] = mapped_column(String)
    slides: Mapped[list[str]] = mapped_column(JSON)
    questions: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    # A row is provisional while its step is still generating (empty
    # questions/summary, partial slides) and is flipped to True when the step
    # completes. The driver skips only complete rows, so a provisional row is
    # re-run (cleaned up) on a fresh generation, not skipped.
    is_complete: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    session: Mapped[Session] = relationship(back_populates="materials")


class SlideContent(Base):
    __tablename__ = "slide_contents"

    # Globally unique sole PK: "{session_id}_{step_id}_slide_{n}" — the session
    # prefix makes it safe across sessions (step IDs repeat per session).
    # Served to the sandbox service via GET /slides/{slide_id} (internal).
    slide_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True
    )
    step_id: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    # True for the placeholder slide generated when a slide exhausts all
    # attempts. Placeholders are always generated and persisted, but the
    # endpoints only return them in dev mode (DEV_MODE=1).
    is_placeholder: Mapped[bool] = mapped_column(Boolean, default=False)

    session: Mapped[Session] = relationship(back_populates="slide_contents")


class FailedSlide(Base):
    """Per-attempt debug record for a slide that failed to compile.

    One row is written per failed attempt, so a slide that fails more than
    once has multiple rows (same ``slide_index``, distinct ``prompt``/``jsx``/
    ``error``). Captures the three things needed to debug a failure:
    ``prompt`` (what we sent the LLM), ``jsx`` (what the LLM returned) and
    ``error`` (the sandbox compile error).
    """

    __tablename__ = "failed_slides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True
    )
    step_id: Mapped[str] = mapped_column(String)
    slide_index: Mapped[int] = mapped_column(Integer)
    # The exact prompt sent to the LLM for this attempt: a JSON list of
    # messages ({"role": "system"|"human", "content": ...}) as sent.
    prompt: Mapped[str] = mapped_column(Text)
    # The JSX returned by the LLM for this attempt.
    jsx: Mapped[str] = mapped_column(Text)
    # The error returned by the sandbox compile for this attempt.
    error: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class GraphStageTiming(Base):
    """Per-stage wall-clock timing for a graph execution.

    One row per node/stage execution of ANY graph. Currently written by
    the material graph (plan_slide_contents, each write_slide /
    compile_slide attempt, write_questions, summarize_step); the ``graph``
    column identifies the producing graph and ``context`` carries
    stage-specific detail (material rows:
    ``{"step_id": ..., "slide_index": ... | null, "attempt": ... | null}``).
    Append-only — a regenerated step appends a fresh set of rows; prior
    attempt rows are kept as history (same policy as failed_slides).
    """

    __tablename__ = "graph_stage_timings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True
    )
    # Which graph produced this row: "material" for now;
    # "plan" / "probe" / "clarify" possible later.
    graph: Mapped[str] = mapped_column(String)
    # Node name within the graph (e.g. "write_slide", "design_plan").
    stage: Mapped[str] = mapped_column(String)
    # Stage-specific detail; NULL for stages without extra context.
    context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class StepProgress(Base):
    __tablename__ = "step_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True
    )
    step_id: Mapped[str] = mapped_column(String)
    slides_done: Mapped[bool] = mapped_column(Boolean, default=False)
    answers: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    complete: Mapped[bool] = mapped_column(Boolean, default=False)

    session: Mapped[Session] = relationship(back_populates="step_progress")


# --- FastAPI dependency ---


def get_db() -> Iterator[DBSession]:
    """Yield a DB session per request."""
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()
