"""Tests for the in-memory engine in MOCK_LLM mode (#118).

Acceptance criteria:
- MOCK_LLM=1 alone starts a fully in-memory app (no .env, no Postgres).
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect
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
