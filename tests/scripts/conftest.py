"""Shared fixtures for script tests.

Mirrors ``tests/db/conftest.py``: ``DATABASE_URL`` is set to a throwaway URL
*before* ``db`` is imported (``db.models`` builds its engine at import time), so
importing a script that reads the models never reaches for a real Postgres. The
seeded in-memory engine lets a test drive a script's real DB read path.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "test-secret")

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
def db_session_factory(db_engine) -> sessionmaker:
    """A session factory over the seeded engine, for a script to read from."""
    return sessionmaker(bind=db_engine)
