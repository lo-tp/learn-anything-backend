"""Which origins this API answers.

The seam is the list itself (:func:`main.cors_allow_origins`), because the two
failures a wrong list produces are invisible: a browser refuses a request with a
message that points at the page, and a list that allows one origin answers a
request from anywhere else with no header at all. A developer walking the app
through on a phone is the case that gets both at once, so these tests assert
what the list says and — once — that the middleware actually obeys it.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

SERVED = "https://learn.lotp.xyz"
LAN = "http://192.168.31.178:3000"
OTHER = "http://10.0.0.9:3000"


def _origins(
    monkeypatch, *, served: str | None = SERVED, extra: str | None = None
) -> list[str]:
    """The list this process would build, with both names stated.

    ``served=None`` is the name left unset — the default stands, as it always did.
    ``served=""`` is the name set to nothing, which ``core/settings.py`` treats as
    unset too, and which is how a deployment says "answer this list and nothing
    else".
    """
    from main import cors_allow_origins

    if served is None:
        monkeypatch.delenv("FRONTEND_DOMAIN", raising=False)
    else:
        monkeypatch.setenv("FRONTEND_DOMAIN", served)
    if extra is None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CORS_ORIGINS", extra)
    return cors_allow_origins()


class TestTheListOfOrigins:
    def test_the_served_origin_alone_is_the_policy_when_nothing_is_added(
        self, monkeypatch
    ):
        """The behaviour before ``CORS_ORIGINS`` existed, stated as a contract."""
        assert _origins(monkeypatch) == [SERVED]

    def test_an_origin_for_another_device_is_added_after_it(self, monkeypatch):
        assert _origins(monkeypatch, extra=LAN) == [SERVED, LAN]

    def test_several_are_added_in_the_order_written(self, monkeypatch):
        assert _origins(monkeypatch, extra=f"{LAN},https://preview.lotp.xyz") == [
            SERVED,
            LAN,
            "https://preview.lotp.xyz",
        ]

    def test_naming_the_served_origin_again_changes_nothing(self, monkeypatch):
        """The intuitive thing to do when a request from it is failing."""
        assert _origins(monkeypatch, extra=f"{SERVED},{LAN}") == [SERVED, LAN]

    def test_a_name_that_is_not_set_keeps_the_default_it_always_had(self, monkeypatch):
        """No ``FRONTEND_DOMAIN`` is the fresh-clone case, and it has always meant
        the local development origin."""
        assert _origins(monkeypatch, served=None) == ["http://localhost:3000"]

    def test_a_deployment_that_names_no_served_origin_answers_only_its_list(
        self, monkeypatch
    ):
        """``FRONTEND_DOMAIN=`` with a list is a deployment saying "these, nothing else"."""
        assert _origins(monkeypatch, served="", extra=LAN) == [LAN]

    def test_a_process_that_names_nothing_at_all_answers_no_origin(self, monkeypatch):
        """Not a wildcard, and not an accident: an empty list is what it says."""
        assert _origins(monkeypatch, served="", extra=None) == []

    def test_the_two_shapes_that_could_only_produce_a_cors_error_add_nothing(
        self, monkeypatch, caplog
    ):
        """The served origin stands; the two mistakes are reported and dropped."""
        assert _origins(monkeypatch, extra="192.168.31.178:3000,*,") == [SERVED]
        assert "no scheme" in caplog.text
        assert "wildcard" in caplog.text


class TestTheMiddlewareAnswersWhatTheListSays:
    """One pass through the real middleware, because a list nobody obeyed would
    pass every test above."""

    @staticmethod
    def _client(origins: list[str]) -> TestClient:
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/probe")
        def probe() -> dict[str, str]:
            return {"ok": "yes"}

        return TestClient(app)

    @pytest.fixture()
    def client(self, monkeypatch) -> TestClient:
        return self._client(_origins(monkeypatch, extra=LAN))

    def test_a_request_from_the_other_device_is_answered_with_that_origin(self, client):
        preflight = client.options(
            "/probe",
            headers={
                "Origin": LAN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.headers["access-control-allow-origin"] == LAN
        assert preflight.headers["access-control-allow-credentials"] == "true"

    def test_a_signed_request_carries_the_origin_rather_than_a_wildcard(self, client):
        """The sign-in cookie is the reason a wildcard could never be the answer."""
        response = client.get("/probe", headers={"Origin": LAN, "Cookie": "access_token=x"})
        assert response.headers["access-control-allow-origin"] == LAN

    def test_the_origin_the_deployment_is_served_from_still_works(self, client):
        response = client.get("/probe", headers={"Origin": SERVED, "Cookie": "access_token=x"})
        assert response.headers["access-control-allow-origin"] == SERVED

    def test_an_origin_that_was_never_named_is_answered_with_nothing(self, client):
        response = client.get("/probe", headers={"Origin": OTHER, "Cookie": "access_token=x"})
        assert "access-control-allow-origin" not in response.headers
