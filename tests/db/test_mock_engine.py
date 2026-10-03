"""Tests for the in-memory engine in MOCK_LLM mode (#118).

Acceptance criteria:
- MOCK_LLM=1 alone starts a fully in-memory app (no .env, no Postgres).
"""

from __future__ import annotations

import threading
import time

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker


class TestMakeEngineMockMode:
    def test_ram_engine_with_all_tables(self, monkeypatch):
        from db.models import Session, make_engine

        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        engine = make_engine()
        try:
            # In-memory SQLite, with every table created at startup.
            assert engine.dialect.name == "sqlite"
            assert "sessions" in inspect(engine).get_table_names()
            assert "plans" in inspect(engine).get_table_names()

            # Usable: a row round-trips on the shared in-memory connection.
            db = sessionmaker(bind=engine)()
            db.add(Session(session_id="m1", phase="clarifying", goal="g"))
            db.commit()
            assert db.get(Session, "m1") is not None
            db.close()
        finally:
            engine.dispose()

    def test_database_url_ignored_in_mock_mode(self, monkeypatch):
        """In mock mode DATABASE_URL is ignored (RAM DB wins)."""
        from db.models import make_engine

        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://nobody@nowhere/db")
        engine = make_engine()
        try:
            assert engine.dialect.name == "sqlite"
        finally:
            engine.dispose()

    def test_missing_database_url_fails_outside_mock_mode(self, monkeypatch):
        monkeypatch.delenv("MOCK_LLM", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from db.models import make_engine

        with pytest.raises(RuntimeError):
            make_engine()


class TestMockEngineConcurrentCheckouts:
    def test_second_concurrent_checkout_sees_the_same_database(self, monkeypatch):
        """Regression (#118): an overflow checkout opened a *second, empty* RAM DB.

        ``QueuePool(pool_size=1)`` leaves SQLAlchemy's default ``max_overflow=10``,
        so a checkout taken while another holds the pooled connection opened a
        brand-new ``sqlite://`` database — one with no tables at all:
        ``OperationalError: no such table: sessions`` (seen on GET /sessions
        while a concurrent request held the real connection).
        """
        from db.models import Session, make_engine

        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        engine = make_engine()
        try:
            held = engine.connect()  # "request 1" holds the pooled connection
            outcome: dict[str, object] = {}

            def second_request() -> None:
                try:
                    other = engine.connect()
                    other.execute(select(Session)).fetchall()
                    other.close()
                    outcome["ok"] = True
                except OperationalError as exc:
                    outcome["error"] = exc

            thread = threading.Thread(target=second_request)
            thread.start()
            time.sleep(0.2)  # let it try to check out while the pool is busy
            held.close()  # "request 1" finishes; the waiter may proceed
            thread.join(10)

            assert not thread.is_alive(), "second checkout never completed"
            assert "ok" in outcome, f"second checkout failed: {outcome.get('error')}"
        finally:
            engine.dispose()


class TestSeedMockUser:
    def test_seeds_fixed_user_with_valid_password(self, db, db_engine, monkeypatch):
        monkeypatch.setenv("MOCK_LLM", "1")
        from core import security
        from db import User, seed_mock_user

        seed_mock_user(engine=db_engine)

        user = db.query(User).filter(User.email == "mock@example.com").first()
        assert user is not None
        assert user.display_name == "Mock User"
        assert security.verify_password(user.password_hash, "12345678")

    def test_seed_is_idempotent(self, db, db_engine, monkeypatch):
        monkeypatch.setenv("MOCK_LLM", "1")
        from db import User, seed_mock_user

        seed_mock_user(engine=db_engine)
        seed_mock_user(engine=db_engine)

        assert (
            db.query(User).filter(User.email == "mock@example.com").count() == 1
        )
