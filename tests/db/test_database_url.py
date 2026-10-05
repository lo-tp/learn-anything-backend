"""Tests for DATABASE_URL reading / driver normalisation.

Render (and most other hosts) hand out a driver-less ``postgresql://``
connection string. This project depends on psycopg3 only, so SQLAlchemy
must be pointed at that driver explicitly — without anyone editing the
host's value.
"""

from __future__ import annotations

import pytest

from db.models import database_url_from_env, psycopg_url


class TestPsycopgUrl:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            (
                "postgresql://user:pw@db.onrender.com:5432/learn_anything",
                "postgresql+psycopg://user:pw@db.onrender.com:5432/learn_anything",
            ),
            (
                "postgres://user:pw@localhost:5432/learn_anything",
                "postgresql+psycopg://user:pw@localhost:5432/learn_anything",
            ),
            # Already names a driver → left exactly as given.
            (
                "postgresql+psycopg://user:pw@localhost:5432/learn_anything",
                "postgresql+psycopg://user:pw@localhost:5432/learn_anything",
            ),
            # Non-Postgres URLs are not touched (tests and mock mode use these).
            ("sqlite:///:memory:", "sqlite:///:memory:"),
            ("sqlite://", "sqlite://"),
        ],
    )
    def test_only_the_driver_part_of_a_pg_scheme_is_rewritten(
        self, given: str, expected: str
    ) -> None:
        assert psycopg_url(given) == expected

    def test_query_params_survive(self) -> None:
        got = psycopg_url("postgresql://u:p@h:5432/d?sslmode=require")
        assert got == "postgresql+psycopg://u:p@h:5432/d?sslmode=require"


class TestDatabaseUrlFromEnv:
    def test_normalises_the_host_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql://u:p@db.internal:5432/learn_anything"
        )
        assert database_url_from_env() == (
            "postgresql+psycopg://u:p@db.internal:5432/learn_anything"
        )

    def test_fails_fast_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
            database_url_from_env()
