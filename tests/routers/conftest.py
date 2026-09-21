"""Shared fixtures for router unit tests.

Strategy
--------
* The domain DB is backed by a fresh in-memory SQLite database per test
  (one shared connection via ``StaticPool``), so the routers' real ORM logic
  (``db.get`` / ``db.query`` / ``db.add`` / ``db.commit``) is exercised for
  real.
* The LangGraph graphs and the LLM/language helpers are mocked at the router
  module boundary (``patch("routers.<mod>.<name>")``) so no network/LLM calls
  happen.

``DATABASE_URL`` is set to a throwaway SQLite URL *before* ``db`` is imported,
because ``db.models`` creates its engine at import time.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

# Must happen before any ``db`` import in this test package.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
# Pinned (not setdefault) before any module import: a developer's local
# ``.env`` (loaded via ``load_dotenv()`` in ``llm.py`` during test imports)
# may carry a real ``JWT_SECRET``, which must not change auth test behavior.
os.environ["JWT_SECRET"] = "test-secret"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db import Base, Phase, Session, get_db


@pytest.fixture()
def db_engine():
    """A fresh in-memory SQLite engine with all tables created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db(db_engine):
    """A raw DB session for seeding rows and asserting on persisted state."""
    session = sessionmaker(bind=db_engine)()
    yield session
    session.close()


@pytest.fixture()
def client(db_engine):
    """A FastAPI TestClient with ``get_db`` pointed at the test engine."""
    from routers import auth, clarify, plan, probe, sessions, slides

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(sessions.router)
    app.include_router(clarify.router)
    app.include_router(probe.router)
    app.include_router(plan.router)
    app.include_router(slides.router)

    TestingSession = sessionmaker(bind=db_engine)

    def _get_db() -> Iterator:
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _get_db

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def make_session(db):
    """Factory that inserts and returns a ``Session`` row."""

    def _make(
        session_id: str | None = None,
        phase: Phase = Phase.CLARIFYING,
        goal: str = "Learn calculus",
        narrowed_goal: str | None = None,
        language: str = "English",
        boundary_map: dict | None = None,
        **extra,
    ) -> Session:
        session = Session(
            session_id=session_id or uuid.uuid4().hex,
            phase=phase.value,
            goal=goal,
            narrowed_goal=narrowed_goal,
            language=language,
            boundary_map=boundary_map,
            **extra,
        )
        db.add(session)
        db.commit()
        return session

    return _make
