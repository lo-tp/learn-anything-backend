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
    narrowed_goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    boundary_map: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
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
        ForeignKey("sessions.session_id"), index=True
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
    answered_at: Mapped[datetime | None] = mapped_column(nullable=True)

    session: Mapped[Session] = relationship(back_populates="probe_questions")


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id"), unique=True
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
        ForeignKey("sessions.session_id"), index=True
    )
    step_id: Mapped[str] = mapped_column(String)
    slides: Mapped[list[str]] = mapped_column(JSON)
    questions: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON)

    session: Mapped[Session] = relationship(back_populates="materials")


class SlideContent(Base):
    __tablename__ = "slide_contents"

    # Globally unique sole PK: "{session_id}_{step_id}_slide_{n}" — the session
    # prefix makes it safe across sessions (step IDs repeat per session).
    # Served to the sandbox service via GET /slides/{slide_id} (internal).
    slide_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id"), index=True
    )
    step_id: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)

    session: Mapped[Session] = relationship(back_populates="slide_contents")


class StepProgress(Base):
    __tablename__ = "step_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id"), index=True
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
