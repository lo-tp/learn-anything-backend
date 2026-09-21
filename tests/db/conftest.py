"""Shared fixtures for db model tests.

Mirrors the in-memory SQLite strategy in ``tests/routers/conftest.py``:
``DATABASE_URL`` is set to a throwaway URL *before* ``db`` is imported
(because ``db.models`` creates its engine at import time), and tables are
built from ``Base.metadata`` so the ORM models — not Alembic — are what the
tests exercise.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db import Base


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
