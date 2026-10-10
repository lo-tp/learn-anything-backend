"""The Explore feed: the public list of Session records (#144, #178).

Explore is the surface a Visitor (a person with no account) browses. Unlike
``routers.sessions``, this router carries **no sign-in dependency**: an
unauthenticated request here is a normal, successful request — not a 401 and
not an empty list. What it lists is restricted instead: only Sessions that
reached material generation have anything to show, and nothing in the payload
says who owns them.

It is a separate route from ``GET /sessions``, not an alias of it. Today the
two differ only in which phases they list, and neither is capped (#178: the
Explore surface has no pager, so a cap is a lost Session rather than a paused
one). They stay separate because the two surfaces are expected to diverge —
Explore is a curated public feed, History is a User's own list — and one shared
endpoint would have to be split apart again when they do.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession

from db import Phase, Session, get_db
from routers.sessions import SessionList, session_list_item

router = APIRouter(tags=["explore"])

# A Session reaches Explore once material generation has started. The intake
# phases (clarifying, probing, planning, reviewing) have no materials to show,
# and an errored Session never reached them either. This set is kept in step
# with the phases the frontend asks History for (``CONFIRMING_PHASES``) until
# Explore's own curation rules differ from the app's.
MATERIAL_PHASES: tuple[Phase, ...] = (Phase.GENERATING, Phase.EXECUTING, Phase.COMPLETE)


@router.get("/explore/sessions", response_model=SessionList)
def list_explore_sessions(db: DBSession = Depends(get_db)) -> SessionList:
    """List the public Explore feed: Sessions that reached materials.

    Newest first, and every one of them: there is no pager here, so a cap
    would hide Sessions outright rather than defer them (#178 removes the
    twenty-row cap this route used to apply). Reuses the ``SessionList`` /
    ``SessionListItem`` schema from ``/sessions`` and carries no owner
    identity — no email, display name, or user id — so a Visitor sees what
    people are learning without being told who they are.
    """
    rows = (
        db.query(Session)
        .filter(Session.phase.in_([p.value for p in MATERIAL_PHASES]))
        .order_by(Session.created_at.desc())
        .all()
    )
    return SessionList(sessions=[session_list_item(s) for s in rows])
