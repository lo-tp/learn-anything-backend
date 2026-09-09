"""SQLAlchemy models and engine for the learn-anything backend."""

from __future__ import annotations

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
from sqlalchemy.pool import StaticPool

# --- Engine (in-memory, single connection) ---

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # required for in-memory SQLite with multiple connections
)


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


# --- Models ---


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    phase: Mapped[str] = mapped_column(String, default=Phase.CLARIFYING.value)
    goal: Mapped[str] = mapped_column(Text)
    narrowed_goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    boundary_map: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
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


class ProbeQuestion(Base):
    __tablename__ = "probe_questions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id"), index=True
    )
    question_id: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    options: Mapped[list[str]] = mapped_column(JSON)
    correct_index: Mapped[int] = mapped_column(Integer)
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


# --- Init ---

Base.metadata.create_all(engine)


# --- FastAPI dependency ---


def get_db() -> Iterator[DBSession]:
    """Yield a DB session per request."""
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()
