"""Unit tests for routers/explore.py — the Explore feed (#144, #178).

Acceptance criteria (issue #144, uncapped by issue #178):
- An unauthenticated request for the public list returns the newest-first
  list of every Session that reached materials — no cap (#178).
- Only Sessions that reached materials appear; Sessions still in intake
  (clarifying, probing, planning, reviewing) are excluded.
- The payload carries no owner identity: no email, no display name, no user id.
- The list shape matches the existing Session list item, so the generated
  client types stay authoritative.
- Verified with a request that sends no cookie at all.

The feed deliberately stays its own endpoint rather than being folded into
``GET /sessions?phase=…``: the two surfaces are expected to diverge (Explore
is a curated public feed, History is a User's own list), so their contracts are
tested separately here and in ``test_sessions.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.security import hash_password
from db import Phase, User

# Every field the public list item may carry. Anything else would be a change
# to the shape the generated client types are built from — and an addition
# such as an email or a display name would be owner identity.
LIST_ITEM_FIELDS = {"session_id", "phase", "goal", "narrowed_goal", "created_at"}

MATERIAL_PHASES = [Phase.GENERATING, Phase.EXECUTING, Phase.COMPLETE]
INTAKE_PHASES = [Phase.CLARIFYING, Phase.PROBING, Phase.PLANNING, Phase.REVIEWING]

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _goal_list(payload: dict) -> list[str]:
    """The goals of a successful Explore response, in the returned order."""
    return [s["goal"] for s in payload["sessions"]]


def _response_ref(spec: dict, path: str) -> str:
    """The OpenAPI content-schema $ref of a path's 200 response."""
    response = spec["paths"][path]["get"]["responses"]["200"]
    return response["content"]["application/json"]["schema"]["$ref"]


class TestUnauthenticatedRequest:
    """A Visitor with no cookie at all gets real content, not an error."""

    def test_no_cookie_returns_200_with_content(
        self, client, db, make_session, monkeypatch
    ):
        # The test app opens its sign-in gate by default; delete DEV_MODE so
        # this request is made exactly as production serves it — no gate
        # bypass, no cookie, no header.
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(
            session_id="public-1",
            phase=Phase.EXECUTING,
            goal="Learn reservoir computing",
            created_at=BASE,
        )

        resp = client.get("/explore/sessions")

        assert resp.status_code == 200
        assert _goal_list(resp.json()) == ["Learn reservoir computing"]

    def test_empty_explore_is_a_successful_empty_list(self, client, db):
        resp = client.get("/explore/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_signing_in_changes_nothing(
        self, client, db, make_session, monkeypatch
    ):
        # Explore is one surface for everyone: no owner scoping, and no
        # filter by who is asking.
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(
            phase=Phase.COMPLETE,
            goal="Learn reservoir computing",
            created_at=BASE,
        )
        anonymous = client.get("/explore/sessions").json()
        signed_in = client.get(
            "/explore/sessions", cookies={"access_token": "not-a-valid-token"}
        ).json()
        assert signed_in == anonymous


class TestPhaseFilter:
    """Only Sessions that reached materials are on Explore."""

    def test_material_phases_appear_intake_phases_do_not(
        self, client, db, make_session
    ):
        for i, phase in enumerate(INTAKE_PHASES):
            make_session(
                session_id=f"intake-{i}",
                phase=phase,
                goal=f"intake {phase.value}",
                created_at=BASE + timedelta(days=i),
            )
        for i, phase in enumerate(MATERIAL_PHASES):
            make_session(
                session_id=f"material-{i}",
                phase=phase,
                goal=f"material {phase.value}",
                created_at=BASE + timedelta(days=10 + i),
            )

        resp = client.get("/explore/sessions")
        assert [s["phase"] for s in resp.json()["sessions"]] == [
            "complete",
            "executing",
            "generating",
        ]

    def test_error_phase_is_excluded(self, client, db, make_session):
        make_session(
            session_id="errored",
            phase=Phase.ERROR,
            goal="gone wrong",
            created_at=BASE,
        )
        assert client.get("/explore/sessions").json() == {"sessions": []}


class TestOrderAndNoCap:
    """Newest first, and every eligible Session is in the answer (#178)."""

    def test_newest_first(self, client, db, make_session):
        for i in range(3):
            make_session(
                session_id=f"a{i}",
                phase=Phase.EXECUTING,
                goal=f"goal {i}",
                created_at=BASE + timedelta(days=i),
            )

        resp = client.get("/explore/sessions")
        assert _goal_list(resp.json()) == ["goal 2", "goal 1", "goal 0"]

    def test_the_feed_is_not_capped(self, client, db, make_session):
        # The Explore surface has no pager (#143), so a cap is not a paused
        # feed but a lost one: #178 removes it. 23 eligible Sessions — well
        # past the twenty this route used to answer — plus 5 newer intake
        # Sessions that must neither appear nor consume a slot.
        for i in range(23):
            make_session(
                session_id=f"material-{i:02d}",
                phase=Phase.EXECUTING,
                goal=f"material {i:02d}",
                created_at=BASE + timedelta(days=i),
            )
        for i in range(5):
            make_session(
                session_id=f"intake-{i:02d}",
                phase=Phase.CLARIFYING,
                goal=f"intake {i:02d}",
                created_at=BASE + timedelta(days=100 + i),
            )

        resp = client.get("/explore/sessions")
        goals = _goal_list(resp.json())
        assert len(goals) == 23
        assert goals == [f"material {i:02d}" for i in range(22, -1, -1)]


class TestNoOwnerIdentity:
    def test_payload_carries_no_owner_fields(self, client, db, make_session):
        db.add(
            User(
                email="ada@example.com",
                display_name="Ada Lovelace",
                password_hash=hash_password("supersecret"),
            )
        )
        db.commit()
        make_session(
            session_id="owned-1",
            phase=Phase.EXECUTING,
            goal="Learn calculus",
            narrowed_goal="Learn integrals by substitution",
            created_at=BASE,
        )

        resp = client.get("/explore/sessions")
        items = resp.json()["sessions"]
        assert items
        for item in items:
            assert set(item) == LIST_ITEM_FIELDS
        # The words that could name an owner appear nowhere in the response.
        assert "ada@example.com" not in resp.text
        assert "Ada Lovelace" not in resp.text


class TestSchemaIsTheSessionListSchema:
    """The generated client types stay authoritative: one schema, one name."""

    def test_openapi_reuses_the_session_list_schema(self, client):
        spec = client.get("/openapi.json").json()
        session_list_ref = "#/components/schemas/SessionList"

        # The same response schema as History, not a look-alike copy of it.
        assert _response_ref(spec, "/explore/sessions") == session_list_ref
        assert _response_ref(spec, "/sessions") == session_list_ref

        sessions_prop = spec["components"]["schemas"]["SessionList"]["properties"][
            "sessions"
        ]
        assert sessions_prop["items"] == {
            "$ref": "#/components/schemas/SessionListItem"
        }
