"""Unit tests for routers/sessions.py — the public Session reads (#178).

The list and the deck read are public: the phase filter is a query parameter
and not a server-side policy, nothing is capped, and the payload carries no
owner identity. The Explore feed (``test_explore.py``) renders the same list
item and stays a separate surface with its own phase rule (#144).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from core.security import hash_password
from db import Phase, Plan, StepProgress, User

# RFC 3339 date-time with an explicit UTC designator (pydantic emits "Z"
# for UTC; "+00:00" is also acceptable).
UTC_OFFSET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|\+00:00)$")


def _signed_in(auth_cookie: str) -> dict[str, str]:
    """A sign-in cookie, for the tests that ask as a signed-in User.

    Reads no longer require one (#178); a Visitor's answer is asserted in
    ``TestTheListIsAPublicRead`` and in ``test_auth_gate``. The token does not
    need a matching ``User`` row — the list is not scoped per User — so the
    plain ``auth_cookie`` fixture is enough.
    """
    return {"access_token": auth_cookie}


class TestListSessions:
    """The Session list, asked by a signed-in User (a Visitor's is below)."""

    def test_empty(self, client, db, auth_cookie):
        resp = client.get("/sessions", cookies=_signed_in(auth_cookie))
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_returns_all_newest_first(self, client, db, make_session, auth_cookie):
        make_session(
            session_id="a",
            phase=Phase.CLARIFYING,
            goal="goal A",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        make_session(
            session_id="b",
            phase=Phase.PROBING,
            goal="goal B",
            created_at=datetime(2024, 2, 1, tzinfo=UTC),
        )

        resp = client.get("/sessions", cookies=_signed_in(auth_cookie))
        assert resp.status_code == 200
        ids = [s["session_id"] for s in resp.json()["sessions"]]
        assert ids == ["b", "a"]
        # Spot-check a serialized field.
        first = resp.json()["sessions"][0]
        assert first["goal"] == "goal B"
        assert first["phase"] == "probing"
        # created_at must carry an explicit UTC offset (RFC 3339), otherwise
        # clients parse it as local time.
        assert UTC_OFFSET_RE.match(first["created_at"]), first["created_at"]

    def test_naive_legacy_created_at_serialized_as_utc(
        self, client, db, make_session, auth_cookie
    ):
        # Rows written before the timestamptz migration (or via a driver that
        # strips the offset) come back naive; the response must still carry
        # an explicit UTC offset so the instant is unambiguous.
        make_session(
            session_id="a",
            # Naive UTC wall-clock: what legacy rows look like when read back.
            created_at=datetime(2026, 9, 17, 1, 2, 49, tzinfo=UTC).replace(
                tzinfo=None
            ),
        )
        resp = client.get("/sessions", cookies=_signed_in(auth_cookie))
        created_at = resp.json()["sessions"][0]["created_at"]
        assert UTC_OFFSET_RE.match(created_at), created_at
        assert created_at.startswith("2026-09-17T01:02:49")

    def test_filter_by_single_phase(self, client, db, make_session, auth_cookie):
        make_session(session_id="a", phase=Phase.CLARIFYING)
        make_session(session_id="b", phase=Phase.PROBING)
        make_session(session_id="c", phase=Phase.PROBING)

        resp = client.get(
            "/sessions",
            params={"phase": ["probing"]},
            cookies=_signed_in(auth_cookie),
        )
        ids = sorted(s["session_id"] for s in resp.json()["sessions"])
        assert ids == ["b", "c"]

    def test_filter_by_multiple_phases(self, client, db, make_session, auth_cookie):
        make_session(session_id="a", phase=Phase.CLARIFYING)
        make_session(session_id="b", phase=Phase.PROBING)
        make_session(session_id="c", phase=Phase.PLANNING)

        resp = client.get(
            "/sessions",
            params={"phase": ["clarifying", "planning"]},
            cookies=_signed_in(auth_cookie),
        )
        ids = sorted(s["session_id"] for s in resp.json()["sessions"])
        assert ids == ["a", "c"]

    def test_filter_matches_nothing(self, client, db, make_session, auth_cookie):
        make_session(session_id="a", phase=Phase.CLARIFYING)

        resp = client.get(
            "/sessions",
            params={"phase": ["complete"]},
            cookies=_signed_in(auth_cookie),
        )
        assert resp.json() == {"sessions": []}


# Every field a list item may carry. Anything else would be a change to the
# shape the generated client types are built from — and an addition such as an
# email or a display name would be owner identity (#144).
LIST_ITEM_FIELDS = {"session_id", "phase", "goal", "narrowed_goal", "created_at"}

# The phases the app lists — the same set the frontend passes as
# ``CONFIRMING_PHASES``: a Session that reached materials has a deck to show.
MATERIAL_PHASES = [Phase.GENERATING, Phase.EXECUTING, Phase.COMPLETE]

BASE = datetime(2024, 1, 1, tzinfo=UTC)


class TestTheListIsAPublicRead:
    """#178: one list, asked the same way by a Visitor and by a User.

    These tests delete ``DEV_MODE`` because that is how production is
    configured (``.env.prod`` sets ``DEV_MODE=0``) — the configuration in which
    a Visitor used to get a 401 here, and in which the Explore feed used to be
    the only list they could see.
    """

    def test_no_cookie_returns_the_list_with_content(
        self, client, db, make_session, monkeypatch
    ):
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(
            session_id="public-1",
            phase=Phase.EXECUTING,
            goal="Learn reservoir computing",
            created_at=BASE,
        )

        resp = client.get("/sessions", params={"phase": [p.value for p in MATERIAL_PHASES]})
        assert resp.status_code == 200
        assert [s["goal"] for s in resp.json()["sessions"]] == [
            "Learn reservoir computing"
        ]

    def test_an_empty_list_is_a_successful_empty_list(self, client, db, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_signing_in_changes_nothing(self, client, db, make_session, monkeypatch):
        # One surface for everyone: no owner scoping, and no filter by who asks.
        monkeypatch.delenv("DEV_MODE", raising=False)
        make_session(
            phase=Phase.COMPLETE,
            goal="Learn reservoir computing",
            created_at=BASE,
        )
        anonymous = client.get("/sessions").json()
        signed_in = client.get(
            "/sessions", cookies={"access_token": "not-a-valid-token"}
        ).json()
        assert signed_in == anonymous

    def test_the_phase_filter_is_a_query_not_a_policy(self, client, db, make_session, monkeypatch):
        # The list applies whatever phases the caller asks for and nothing else:
        # intake phases stay askable (History of a User's own in-progress work),
        # and an uncalled filter lists every row.
        monkeypatch.delenv("DEV_MODE", raising=False)
        for i, phase in enumerate([Phase.CLARIFYING, *MATERIAL_PHASES]):
            make_session(
                session_id=f"s-{phase.value}",
                phase=phase,
                goal=f"{phase.value} goal",
                created_at=BASE + timedelta(days=i),
            )

        filtered = client.get(
            "/sessions", params={"phase": [p.value for p in MATERIAL_PHASES]}
        ).json()
        assert [s["phase"] for s in filtered["sessions"]] == [
            "complete",
            "executing",
            "generating",
        ]
        unfiltered = client.get("/sessions").json()
        assert [s["phase"] for s in unfiltered["sessions"]] == [
            "complete",
            "executing",
            "generating",
            "clarifying",
        ]

    def test_the_list_is_not_capped(self, client, db, make_session, monkeypatch):
        # There is no pager (#143), so nothing may be cut off the end of it:
        # every Session that reached materials is reachable from the first page.
        monkeypatch.delenv("DEV_MODE", raising=False)
        for i in range(23):
            make_session(
                session_id=f"material-{i:02d}",
                phase=Phase.EXECUTING,
                goal=f"material {i:02d}",
                created_at=BASE + timedelta(days=i),
            )

        resp = client.get(
            "/sessions", params={"phase": [p.value for p in MATERIAL_PHASES]}
        )
        goals = [s["goal"] for s in resp.json()["sessions"]]
        assert len(goals) == 23
        assert goals == [f"material {i:02d}" for i in range(22, -1, -1)]

    def test_the_payload_carries_no_owner_fields(self, client, db, make_session, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
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

        resp = client.get("/sessions")
        items = resp.json()["sessions"]
        assert items
        for item in items:
            assert set(item) == LIST_ITEM_FIELDS
        # The words that could name an owner appear nowhere in the response.
        assert "ada@example.com" not in resp.text
        assert "Ada Lovelace" not in resp.text


class TestGetSession:
    def test_not_found(self, client, db):
        resp = client.get("/sessions/nope")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Session not found"

    def test_no_plan(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.CLARIFYING).session_id
        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "a"
        assert body["phase"] == "clarifying"
        assert body["narrowed_goal"] is None
        assert body["progress"] == {
            "current_step_id": None,
            "completed_steps": [],
            "total_steps": 0,
            "step_scores": {},
        }

    def test_with_plan_and_progress(self, client, db, make_session):
        sid = make_session(
            session_id="a", phase=Phase.EXECUTING, narrowed_goal="ng"
        ).session_id
        db.add(
            Plan(
                session_id=sid,
                version=1,
                prose_summary="ps",
                dependency_dag="dag",
                steps=[
                    {"id": "s1", "title": "T1", "description": "d",
                     "depends_on": [], "depth": 0},
                    {"id": "s2", "title": "T2", "description": "d",
                     "depends_on": ["s1"], "depth": 1},
                    {"id": "s3", "title": "T3", "description": "d",
                     "depends_on": ["s2"], "depth": 2},
                ],
                adjustments=[],
            )
        )
        db.add(StepProgress(session_id=sid, step_id="s1", complete=True, score=0.9))
        db.add(StepProgress(session_id=sid, step_id="s2", complete=True, score=1.0))
        db.add(StepProgress(session_id=sid, step_id="s3", complete=False, score=None))
        db.commit()

        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["narrowed_goal"] == "ng"
        assert body["progress"] == {
            "current_step_id": "s3",
            "completed_steps": ["s1", "s2"],
            "total_steps": 3,
            "step_scores": {"s1": 0.9, "s2": 1.0},
        }

    def test_all_steps_complete(self, client, db, make_session):
        sid = make_session(session_id="a", phase=Phase.COMPLETE).session_id
        db.add(
            Plan(
                session_id=sid,
                version=1,
                prose_summary="ps",
                dependency_dag="dag",
                steps=[{"id": "s1", "title": "T", "description": "d",
                        "depends_on": [], "depth": 0}],
                adjustments=[],
            )
        )
        db.add(StepProgress(session_id=sid, step_id="s1", complete=True, score=1.0))
        db.commit()

        resp = client.get(f"/sessions/{sid}")
        body = resp.json()
        assert body["progress"]["current_step_id"] is None
        assert body["progress"]["completed_steps"] == ["s1"]
