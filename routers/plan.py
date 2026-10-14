"""Plan phase endpoints: generate, adjust, approve, and material polling.

The writes here are gated per route (#178); ``GET /sessions/{session_id}/materials``
— the generated deck — is a public read and declares no gate. Ownership begins
at the write, not the read.

Approving the plan schedules ``generate_materials`` as a FastAPI background
task: a per-step driver that commits one ``StepMaterial`` row (plus one
``SlideContent`` row per slide and ``FailedSlide`` rows for failed attempts)
per step. The per-slide retry loop lives inside the material graph
(``graphs/material/``); the driver simply invokes the graph once per step
and persists the results. The domain DB is the resume point — a step with an
existing ``StepMaterial`` row is skipped; a failure lands the session in the
``error`` phase (terminal, per design).

One slide, on its own, is the other half of that storage. Each ``SlideContent``
row keeps the content spec it was written from, so
``POST /sessions/{session_id}/slides/{slide_id}/regenerate`` can schedule
``regenerate_slide_material``: a background task that rebuilds everything
``write_slide`` needs from this database and runs the graph's own nodes over one
spec — no graph, no checkpointer, no interrupt (#197).

A rewrite marks the row it is running (``is_regenerating``): the mark is what
tells a client a rewrite is in flight, what makes a second request for the same
slide a refusal, and what app startup clears when a previous process left one
behind. ``docs/api-and-graphs.md`` states the contract.
"""

import logging
import threading
import time
from collections import deque
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from core.language import (
    DEFAULT_LANGUAGE,
    detect_language,
    has_meaningful_signal,
    localize_status,
)
from core.llm import llm
from core.mock_llm import is_mock_mode
from core.security import is_dev_mode, require_auth
from db import (
    FailedSlide,
    GraphStageTiming,
    Phase,
    Plan,
    Session,
    SessionFactory,
    SlideContent,
    StepMaterial,
    get_db,
)
from graphs import (
    checkpointer,
    graph_config,
    material_graph,
    plan_graph,
    pre_material_llm,
)
from graphs.material.regenerate import regenerate_slide
from services.stage_timings import (
    empty_stage_timings,
    record_stage_timings,
    stage_timing_rows,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["plan"])

# The plan writes keep the DEV_MODE-bounded gate (#89): a local run can walk
# the phases without a cookie, production asks for one. It is declared per route
# rather than on the router because this router also serves the deck read —
# `GET /sessions/{session_id}/materials` — which is a public read (#178).
GATE_THE_WRITES = [Depends(require_auth)]

# One lock per session: a double-approve must not race check-then-insert on
# the shared in-memory SQLite connection. The guard protects lock creation.
_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


# --- Schemas ---


class AdjustIn(BaseModel):
    adjustment: str


class StepOut(BaseModel):
    id: str
    letter: str = ""  # display index (A, B, C, ...); empty for legacy plans
    title: str
    description: str
    depends_on: list[str]
    depth: int


class PlanBody(BaseModel):
    prose_summary: str
    dependency_dag: str
    steps: list[StepOut]


class PlanOut(BaseModel):
    phase: Phase
    plan: PlanBody


class ApproveOut(BaseModel):
    phase: Phase
    message: str


class RegenerateIn(BaseModel):
    # What the learner wants different about this slide. Optional, and it shapes
    # how the slide is rendered — not what the deck's plan asks it to cover.
    feedback: str | None = None


class RegenerateOut(BaseModel):
    slide_id: str
    message: str


class SummaryOut(BaseModel):
    step_id: str
    title: str
    key_points: list[str]


class SlideItem(BaseModel):
    type: Literal["slide"]
    slide_id: str
    # "A rewrite of this slide is running." A rewritten slide keeps its ID, so
    # this is the only thing in the deck a client can compare its own request
    # against (#197). No default: a slide item that does not say is one that is
    # lying.
    regenerating: bool


class QuestionItem(BaseModel):
    type: Literal["question"]
    id: str
    text: str
    options: list[str]
    correct_index: int
    explanation: str


MaterialItem = Annotated[
    SlideItem | QuestionItem,
    Field(discriminator="type"),
]


class MaterialOut(BaseModel):
    step_id: str
    summary: SummaryOut
    items: list[MaterialItem]


class MaterialsOut(BaseModel):
    phase: Phase
    generated_steps: list[MaterialOut]


def _session_lock(session_id: str) -> threading.Lock:
    """Get (or create) the per-session generation lock."""
    with _locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


# --- Background material generation driver ---


def generate_materials(session_id: str) -> None:
    """Generate material for every plan step, one committed row per step.

    Plain ``def`` so FastAPI runs it in the thread pool after the approve
    response is sent. Opens its OWN db session (the request's is closed by
    then). The DB is the resume point: a ``StepMaterial`` row for a step means
    it is done — skip it (but keep its summary in the accumulated context).

    Defensive guard (#118): mock mode never schedules this — a no-op even if
    it is somehow invoked.

    The per-slide retry loop lives inside the material graph; the driver
    streams the graph per step and persists each slide to the DB immediately
    after it compiles, so a crash mid-step loses at most the in-flight
    slide. Orphan SlideContent/FailedSlide rows from a previous partial run
    are cleaned up before re-invoking.
    """
    if is_mock_mode():
        return
    with _session_lock(session_id):
        started = time.monotonic()
        db = SessionFactory()
        try:
            session = db.get(Session, session_id)
            if session is None:
                logger.warning(
                    "Material generation: session %s not found, aborting",
                    session_id,
                )
                return
            steps = session.plan.steps if session.plan is not None else None
            boundary_map = session.boundary_map
            if not steps or boundary_map is None:
                logger.warning(
                    "Material generation: session %s has no plan or "
                    "boundary map, aborting",
                    session_id,
                )
                return

            # All materials are generated in the learner's language.
            language = session.language or DEFAULT_LANGUAGE
            logger.info(
                "Material generation started: session=%s, steps=%d",
                session_id,
                len(steps),
            )
            summaries: list[dict] = []
            for step in steps:  # already in dependency order
                existing = (
                    db.query(StepMaterial)
                    .filter_by(session_id=session_id, step_id=step["id"])
                    .first()
                )
                if existing is not None and existing.is_complete:
                    # Already generated — resume past it, but keep its summary.
                    logger.debug(
                        "Material generation: step %s already generated, skipping",
                        step["id"],
                    )
                    summaries.append(existing.summary)
                    continue

                step_started = time.monotonic()
                logger.debug(
                    "Material generation: generating step %s "
                    "(established concepts: %d)",
                    step["id"],
                    len(summaries),
                )
                # Clean up from a previous partial run of this step: a
                # provisional StepMaterial row (crashed before completing)
                # plus its orphan slide/failed rows. The step is re-run from
                # scratch.
                if existing is not None:
                    db.delete(existing)
                db.query(SlideContent).filter_by(
                    session_id=session_id,
                    step_id=step["id"],
                ).delete()
                db.query(FailedSlide).filter_by(
                    session_id=session_id,
                    step_id=step["id"],
                ).delete()
                db.commit()

                # Create a provisional StepMaterial row up front so the FE can
                # see this step appear and its slides land one by one. It is
                # filled in as slides compile and finalized (questions, summary,
                # is_complete=True) when the step finishes.
                provisional = StepMaterial(
                    session_id=session_id,
                    step_id=step["id"],
                    slides=[],
                    questions=[],
                    summary={
                        "step_id": step["id"],
                        "title": step.get("title", ""),
                        "key_points": [],
                    },
                    is_complete=False,
                )
                db.add(provisional)
                db.commit()

                # Run the graph with interrupt/resume: the graph pauses after
                # each successful compile (pause_after_compile); the driver
                # saves the slide to the DB and resumes. Each compiled slide is
                # also appended to the provisional row so the FE sees it.
                # Stage timings from state are persisted the same way: each
                # not-yet-saved entry becomes a GraphStageTiming row in the
                # same transaction as the slide commit (append-only — prior
                # attempt rows from a re-run are kept).
                config = graph_config(session_id, f"material:{step['id']}")
                state: dict = material_graph.invoke(
                    {
                        "step": step,
                        "established_concepts": summaries,
                        "learner_context": boundary_map,
                        "language": language,
                    },
                    config,
                )
                slide_ids: list[str] = []
                timings_persisted = 0
                while state.get("__interrupt__"):
                    slides = state.get("slides") or []
                    n = len(slides)
                    slide_id = f"{session_id}_{step['id']}_slide_{n}"
                    slide_ids.append(slide_id)
                    specs = state.get("slide_contents") or []
                    # The spec this slide was written from (#197). The graph's
                    # own cursor is the one that says which spec produced the
                    # JSX just compiled: the two disagree when a slide was
                    # skipped entirely, because the skipped one added nothing to
                    # `slides` and every later row shifts down by one.
                    spec_index = state.get("slide_index", 0) - 1
                    spec = specs[spec_index] if 0 <= spec_index < len(specs) else None
                    db.add(
                        SlideContent(
                            slide_id=slide_id,
                            session_id=session_id,
                            step_id=step["id"],
                            content=slides[-1],
                            content_spec=spec,
                            is_placeholder=state.get(
                                "current_slide_is_placeholder", False
                            ),
                        )
                    )
                    # Persist any stage timings that have accumulated since the
                    # last commit (one row per stage execution, append-only).
                    new_rows = stage_timing_rows(
                        session_id, "material", (state.get("stage_timings") or [])[timings_persisted:]
                    )
                    db.add_all(new_rows)
                    timings_persisted += len(new_rows)
                    # Publish the new slide on the provisional row (reassign a
                    # fresh list so SQLAlchemy's JSON column detects the change).
                    provisional.slides = provisional.slides + [slide_id]
                    db.commit()
                    state = material_graph.invoke(Command(resume=True), config)
                result = state

                # Finalize the provisional row and persist failed attempts and
                # any remaining stage timings in one transaction (the slides
                # are already committed above).
                failed = result.get("failed_attempts") or []
                for fa in failed:
                    db.add(
                        FailedSlide(
                            session_id=session_id,
                            step_id=step["id"],
                            slide_index=fa["index"] + 1,  # 1-based for DB
                            prompt=fa.get("prompt", ""),
                            jsx=fa["jsx"],
                            error=fa["error"],
                        )
                    )
                db.add_all(
                    stage_timing_rows(
                        session_id,
                        "material",
                        (result.get("stage_timings") or [])[timings_persisted:],
                    )
                )
                provisional.slides = slide_ids
                provisional.questions = result["questions"]
                provisional.summary = result["summary"]
                provisional.is_complete = True
                # Total wall time to generate this step's material: from the
                # first graph invoke to finalization, including the slide
                # commits between interrupts. Persisted as a generate_step
                # timing row alongside the per-stage rows (same append-only
                # history policy — a re-run appends a fresh row).
                step_total = time.monotonic() - step_started
                db.add(
                    GraphStageTiming(
                        session_id=session_id,
                        graph="material",
                        stage="generate_step",
                        context={
                            "step_id": step["id"],
                            "slide_index": None,
                            "attempt": None,
                        },
                        duration_seconds=step_total,
                    )
                )
                db.commit()
                summaries.append(result["summary"])
                logger.info(
                    "Material generation: step %s done (%d slides, %d failed "
                    "attempts, %d questions, %.1fs)",
                    step["id"],
                    len(slide_ids),
                    len(failed),
                    len(result["questions"]),
                    step_total,
                )

            session.phase = Phase.EXECUTING.value
            db.commit()
            logger.info(
                "Material generation complete: session=%s, %.1fs total",
                session_id,
                time.monotonic() - started,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "Material generation failed for session %s after %.1fs",
                session_id,
                time.monotonic() - started,
            )
            try:
                # Re-fetch: the rollback expired the objects loaded above.
                errored = db.get(Session, session_id)
                if errored is not None:
                    errored.phase = Phase.ERROR.value
                    db.commit()
            except Exception:
                logger.exception(
                    "Failed to record error state for session %s", session_id
                )
        finally:
            db.close()


# --- Single-slide regeneration driver (#197) ---


def _plan_position(session: Session, step_id: str) -> int | None:
    """Where this step sits in the plan, or ``None`` if the plan no longer has it.

    A plan can be adjusted after its deck was generated, and a slide whose step
    is gone has no title, description or position to be rewritten from.
    """
    if session.plan is None:
        return None
    for position, step in enumerate(session.plan.steps):
        if step.get("id") == step_id:
            return position
    return None


def _deck_position(slide: SlideContent) -> int:
    """Which slot of its step's deck this row is, 1-based.

    The slide's own name carries it: the deck driver assigns
    ``{session_id}_{step_id}_slide_{n}`` where ``n`` is the slide's position in
    the deck. That position — not the one-slide run's counting, where every
    slide is slide 1 — is what a regeneration's ``failed_slides`` and
    ``graph_stage_timings`` rows must key on, because those tables are keyed on
    ``(session, step, slide_index)`` and read as one set by
    ``scripts/slide_analysis.py`` and ``scripts/classify_slide_failures.py``.

    A generated run keys its rows on the position in the *spec* list; the two
    differ only when a slide was skipped entirely (its attempts exhausted *and*
    the placeholder could not compile), which shifts every later deck slot down
    by one. A stored row cannot tell which slide was skipped — no row was
    written for it — so a regeneration keys on the deck slot it can see.

    Falls back to 1 for a slide_id that does not carry the suffix.
    """
    _, separator, suffix = slide.slide_id.rpartition("_slide_")
    if separator and suffix.isdigit():
        return int(suffix)
    return 1


def _regeneration_inputs(
    db: DBSession, session: Session, slide: SlideContent
) -> tuple[dict, list[dict]] | None:
    """The plan step this slide belongs to, and the concepts it was built on.

    The step is the one the slide was generated for; the concepts are the
    summaries of every earlier plan step, in plan order — the values the
    material generation driver had accumulated by the time it reached this step.
    A prior step whose row is still provisional is skipped for the same reason
    the driver skips it: a step that never finished established nothing, and its
    placeholder summary would be offered to the model as something the learner
    has already been taught.

    Returns ``None`` when the plan no longer has the step.
    """
    target = _plan_position(session, slide.step_id)
    if target is None or session.plan is None:
        return None

    steps = session.plan.steps
    order = {step["id"]: rank for rank, step in enumerate(steps)}
    earlier = [step["id"] for step in steps[:target]]
    summaries: list[dict] = []
    if earlier:
        rows = (
            db.query(StepMaterial)
            .filter(
                StepMaterial.session_id == session.session_id,
                StepMaterial.step_id.in_(earlier),
                StepMaterial.is_complete.is_(True),
            )
            .all()
        )
        rows.sort(key=lambda row: (order[row.step_id], row.id))
        summaries = [row.summary for row in rows]
    return steps[target], summaries


def _regeneration_timings(timings: list[dict], deck_position: int) -> list[dict]:
    """A regeneration's stage rows, attributed to the deck slot they changed.

    The nodes record a stage as the graph's own run sees it — the cursor inside
    a one-slide run is always slide 1 — and the rows are read afterwards
    alongside a generated deck's rows. Re-keying on the deck slot is what makes
    a regenerated slide's costs land on the slide they belong to;
    ``regenerated`` says which of the two runs wrote the row, so "what does
    generating a deck cost" and "what does asking for one slide back cost" can
    still be answered separately (#194).
    """
    return [
        {
            **timing,
            "context": {
                **(timing.get("context") or {}),
                "slide_index": deck_position,
                "regenerated": True,
            },
        }
        for timing in timings
    ]


def _clear_regeneration_mark(db: DBSession, slide_id: str) -> None:
    """Clear the in-flight mark on a row that still carries it.

    Idempotent: once the task has made its own commit the row is already clear,
    so this writes nothing. It exists for the exits that never reached that
    commit — a refusal under the lock, or a task that raised.
    """
    slide = db.get(SlideContent, slide_id)
    if slide is not None and slide.is_regenerating:
        slide.is_regenerating = False
        db.commit()


def clear_abandoned_regenerations() -> None:
    """Clear every in-flight mark the previous process left behind.

    A mark describes a rewrite this process is running — the rewrite *is* a
    FastAPI background task, so it cannot outlive the process that started it.
    A run killed by a deploy, a reload or an OOM leaves its mark in a database
    nothing can finish, and that mark is worse than no mark: it tells every
    reader of the deck that a slide is being replaced, and it refuses the next
    request for that slide, which is the learner's only route back to it.

    Called from app startup, where no rewrite is running by definition, so every
    mark found is stale. That premise is the deployment's, not the code's:
    production runs one backend replica (infra's
    ``manifests/overlays/prod/replica-floor.yaml``), so a starting process is
    never clearing a mark some other live process owns. The one window that does
    overlap is a rolling replace — a pod still finishing a rewrite inside its 5s
    graceful shutdown while its replacement boots. What that costs is a deck that
    briefly reports "no rewrite running" while one still is, and a learner's
    repeat request accepted and paid for; the alternative — a mark that outlives
    its work and refuses that slide forever — is the worse of the two.

    Best-effort: a slow or unreachable database must not become a reason the app
    cannot serve a request — the rule ``/health`` follows for the same reason.
    """
    db = SessionFactory()
    try:
        cleared = db.query(SlideContent).filter(
            SlideContent.is_regenerating.is_(True)
        ).update({"is_regenerating": False}, synchronize_session=False)
        db.commit()
        if cleared:
            logger.warning(
                "Cleared %d regeneration mark(s) left by a previous process", cleared
            )
    except Exception:  # startup does not depend on the database — same rule as /health
        db.rollback()
        logger.exception("Could not clear stale regeneration marks")
    finally:
        db.close()


def regenerate_slide_material(
    session_id: str, slide_id: str, feedback: str | None
) -> None:
    """Rewrite one slide from the content spec its own row stores.

    Plain ``def`` so FastAPI runs it in the thread pool after the 202 is sent,
    with its OWN db session. It opens with the one thing that used to be lost —
    ``SlideContent.content_spec`` — rebuilds the rest of what ``write_slide``
    needs from the plan, the prior steps' summaries, the session's boundary map
    and its language, and hands the result to :func:`graphs.material.regenerate`
    (which runs the material graph's own nodes over that one spec).

    The row is only replaced by a slide that actually compiled: a regeneration
    that exhausted its attempts ends with the placeholder stand-in, and
    exchanging a bad slide for a placeholder is not an improvement the learner
    asked for. The failed attempts and the stage timings are kept either way —
    append-only history, the same policy the deck driver follows.

    Takes the per-session generation lock: a slide is not rewritten while the
    deck it belongs to is being written. Two requests for the same slide never
    reach that queue at all — the second is refused by the endpoint, on the mark
    the first left.

    The request's mark is cleared on every outcome that reached one, in the
    task's own commit where there is one and in a final guard where there is
    not. Mock mode returns before any of this, and cannot have a mark to clear:
    the endpoint refuses a slide rewrite there outright. An abandoned mark — the
    work died with the process — is what
    :func:`clear_abandoned_regenerations` answers at startup.
    """
    if is_mock_mode():
        return
    with _session_lock(session_id):
        started = time.monotonic()
        db = SessionFactory()
        try:
            slide = db.get(SlideContent, slide_id)
            if slide is None or slide.session_id != session_id:
                # The deck changed under the request (a dev regeneration wipes
                # slides); there is nothing left to rewrite.
                logger.warning(
                    "Slide regeneration: slide %s is not part of session %s",
                    slide_id, session_id,
                )
                return
            session = db.get(Session, session_id)
            if session is None:
                logger.warning(
                    "Slide regeneration: session %s not found", session_id
                )
                return
            inputs = _regeneration_inputs(db, session, slide)
            if inputs is None:
                logger.warning(
                    "Slide regeneration: session %s has no plan step %s",
                    session_id, slide.step_id,
                )
                return
            step, established_concepts = inputs
            if slide.content_spec is None:
                # A row from before the spec was stored (#197): the only route
                # back for this slide is regenerating the whole deck.
                logger.warning(
                    "Slide regeneration: slide %s has no stored content spec",
                    slide_id,
                )
                return
            # Every row this task writes describes this deck slot, not the
            # one-slide run that produced it (see _deck_position, #197).
            position = _deck_position(slide)

            result = regenerate_slide(
                llm,
                step=step,
                content_spec=slide.content_spec,
                established_concepts=established_concepts,
                learner_context=session.boundary_map or {},
                language=session.language,
                feedback=feedback,
            )

            for attempt in result["failed_attempts"]:
                db.add(
                    FailedSlide(
                        session_id=session_id,
                        step_id=slide.step_id,
                        slide_index=position,
                        prompt=attempt.get("prompt", ""),
                        jsx=attempt["jsx"],
                        error=attempt["error"],
                    )
                )
            db.add_all(
                stage_timing_rows(
                    session_id,
                    "material",
                    _regeneration_timings(result["stage_timings"], position),
                )
            )

            jsx = result["jsx"]
            if jsx is not None and not result["is_placeholder"]:
                slide.content = jsx
                # A placeholder row that regenerated into a real slide is now a
                # real slide, and the deck shows it again.
                slide.is_placeholder = False
                logger.info(
                    "Slide regenerated: session=%s slide=%s attempts=%d "
                    "(%d chars, %.1fs)",
                    session_id, slide_id, len(result["failed_attempts"]) + 1,
                    len(jsx), time.monotonic() - started,
                )
            else:
                logger.warning(
                    "Slide regeneration produced nothing usable for session=%s "
                    "slide=%s after %d attempt(s); keeping the slide the learner "
                    "has",
                    session_id, slide_id, len(result["failed_attempts"]),
                )
            # The rewrite is over either way: the learner got the slide they
            # asked for, or the deck kept the one they already have. Clearing
            # the mark in this same commit is what makes "the new slide is
            # here" and "no rewrite is running" one fact a reader cannot catch
            # out of step.
            slide.is_regenerating = False
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "Slide regeneration failed for session %s slide %s after %.1fs",
                session_id, slide_id, time.monotonic() - started,
            )
        finally:
            # An exit that never reached that commit — a refusal under the lock,
            # or a task that raised — still has to end the signal, or the deck
            # goes on reporting a rewrite that is not running.
            _clear_regeneration_mark(db, slide_id)
            db.close()


# --- Helpers ---


def _get_session_or_404(db: DBSession, session_id: str) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _ordered_materials(session: Session) -> list[StepMaterial]:
    """StepMaterial rows in plan step order (deterministic).

    Rows whose step_id is not in the plan sort after the plan-ordered rows
    (by row id); if the session has no plan, rows keep insertion (row-id)
    order. Ordering is applied here, endpoint-side — the relationship has
    no order_by.
    """
    materials = list(session.materials)
    plan_steps = session.plan.steps if session.plan is not None else None
    if plan_steps is None:
        return sorted(materials, key=lambda m: m.id)
    order = {s["id"]: i for i, s in enumerate(plan_steps)}
    return sorted(
        materials,
        key=lambda m: (
            order[m.step_id] if m.step_id in order else len(plan_steps),
            m.id,
        ),
    )


def _cleanup_thread(session_id: str) -> None:
    """Best-effort: clear the plan checkpoint thread so the client can retry."""
    try:
        checkpointer.delete_thread(f"{session_id}:plan")
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to clean up plan thread")


def validate_plan(plan: dict) -> None:
    """Validate a rendered plan: non-empty steps, valid dep IDs, acyclic."""
    steps = plan.get("steps", [])
    if not steps:
        raise HTTPException(status_code=500, detail="Generated plan failed validation")

    ids = {s["id"] for s in steps}
    for s in steps:
        for dep in s.get("depends_on", []):
            if dep not in ids:
                raise HTTPException(
                    status_code=500, detail="Generated plan failed validation"
                )

    # Kahn's algorithm for cycle detection
    in_degree: dict[str, int] = {s["id"]: 0 for s in steps}
    adj: dict[str, list[str]] = {s["id"]: [] for s in steps}
    for s in steps:
        for dep in s.get("depends_on", []):
            adj[dep].append(s["id"])
            in_degree[s["id"]] += 1

    queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(ids):
        raise HTTPException(status_code=500, detail="Generated plan failed validation")


def _persist_plan(
    db: DBSession,
    session: Session,
    plan: dict,
    version: int,
    adjustments: list[str] | None = None,
) -> None:
    """Insert or update the Plan row for this session."""
    existing = session.plan
    if existing is None:
        row = Plan(
            session_id=session.session_id,
            version=version,
            prose_summary=plan["prose_summary"],
            dependency_dag=plan["dependency_dag"],
            steps=plan["steps"],
            adjustments=adjustments or [],
        )
        db.add(row)
    else:
        existing.version = version
        existing.prose_summary = plan["prose_summary"]
        existing.dependency_dag = plan["dependency_dag"]
        existing.steps = plan["steps"]
        if adjustments is not None:
            existing.adjustments = adjustments
    db.commit()


# --- Routes ---


@router.post(
    "/sessions/{session_id}/plan/generate",
    response_model=PlanOut,
    dependencies=GATE_THE_WRITES,
)
def generate_plan(session_id: str, db: DBSession = Depends(get_db)) -> PlanOut:
    """Trigger plan generation."""
    session = _get_session_or_404(db, session_id)
    if session.phase == Phase.REVIEWING.value:
        raise HTTPException(
            status_code=409,
            detail="Session is already reviewing — use /plan/adjust instead",
        )
    if session.phase != Phase.PLANNING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'planning'",
        )

    config = graph_config(session_id, "plan")

    try:
        result = plan_graph.invoke(
            {
                "goal": session.narrowed_goal or session.goal,
                "language": session.language or DEFAULT_LANGUAGE,
                "boundary_map": session.boundary_map or {},
                "research": None,
                "current_plan": None,
                "adjustment": None,
                "pass_count": 0,
                "stage_timings": empty_stage_timings(),
            },
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan generation failed")

    record_stage_timings(db, session_id, "plan", result)

    if "__interrupt__" not in result:
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan generation did not pause as expected"
        )

    plan = result.get("current_plan")
    if plan is None:
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan generation did not produce a plan"
        )

    try:
        validate_plan(plan)
    except HTTPException:
        _cleanup_thread(session_id)
        raise

    _persist_plan(db, session, plan, version=1)
    session.phase = Phase.REVIEWING.value
    db.commit()

    return PlanOut(
        phase=Phase.REVIEWING,
        plan=PlanBody(
            prose_summary=plan["prose_summary"],
            dependency_dag=plan["dependency_dag"],
            steps=[StepOut(**s) for s in plan["steps"]],
        ),
    )


@router.post(
    "/sessions/{session_id}/plan/adjust",
    response_model=PlanOut,
    dependencies=GATE_THE_WRITES,
)
def adjust_plan(
    session_id: str, body: AdjustIn, db: DBSession = Depends(get_db)
) -> PlanOut:
    """Submit a free-text adjustment and get the regenerated plan."""
    session = _get_session_or_404(db, session_id)
    if session.phase != Phase.REVIEWING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'reviewing'",
        )

    config = graph_config(session_id, "plan")

    # Follow the learner's language if the adjustment carries enough signal.
    if has_meaningful_signal(body.adjustment):
        session.language = detect_language(body.adjustment, pre_material_llm)
        db.commit()
    try:
        result = plan_graph.invoke(
            Command(
                resume={"action": "adjust", "text": body.adjustment},
                update={
                    "language": session.language or DEFAULT_LANGUAGE,
                    "stage_timings": empty_stage_timings(),
                },
            ),
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan adjustment failed")

    record_stage_timings(db, session_id, "plan", result)

    if "__interrupt__" not in result:
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan adjustment did not pause as expected"
        )

    plan = result.get("current_plan")
    if plan is None:
        _cleanup_thread(session_id)
        raise HTTPException(
            status_code=500, detail="Plan adjustment did not produce a plan"
        )

    try:
        validate_plan(plan)
    except HTTPException:
        _cleanup_thread(session_id)
        raise

    existing = session.plan
    new_version = existing.version + 1 if existing else 1
    new_adjustments = (
        list(existing.adjustments) + [body.adjustment]
        if existing
        else [body.adjustment]
    )
    _persist_plan(db, session, plan, version=new_version, adjustments=new_adjustments)
    db.commit()

    return PlanOut(
        phase=Phase.REVIEWING,
        plan=PlanBody(
            prose_summary=plan["prose_summary"],
            dependency_dag=plan["dependency_dag"],
            steps=[StepOut(**s) for s in plan["steps"]],
        ),
    )


@router.post(
    "/sessions/{session_id}/plan/approve",
    response_model=ApproveOut,
    status_code=202,
    dependencies=GATE_THE_WRITES,
)
def approve_plan(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
) -> ApproveOut:
    """Approve the plan, then start material generation in the background."""
    session = _get_session_or_404(db, session_id)
    if session.phase != Phase.REVIEWING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Session is in phase '{session.phase}', not 'reviewing'",
        )

    config = graph_config(session_id, "plan")

    try:
        result = plan_graph.invoke(
            Command(
                resume={"action": "approve"},
                update={"stage_timings": empty_stage_timings()},
            ),
            config,
        )
    except Exception:  # noqa: BLE001 — catch all to clean up thread
        _cleanup_thread(session_id)
        raise HTTPException(status_code=500, detail="Plan approval failed")

    record_stage_timings(db, session_id, "plan", result)

    plan = result.get("current_plan")
    if plan is None:
        raise HTTPException(
            status_code=500, detail="Plan approval did not produce a plan"
        )

    # Re-persist the final plan (idempotent)
    existing = session.plan
    if existing is not None:
        _persist_plan(
            db,
            session,
            plan,
            version=existing.version,
            adjustments=existing.adjustments,
        )

    if is_mock_mode():
        # #118: mock mode never advances into slide generation — the session
        # stays in REVIEWING and generate_materials is not scheduled.
        return ApproveOut(
            phase=Phase.REVIEWING,
            message="Plan approved. Slide generation is disabled in mock mode.",
        )

    session.phase = Phase.GENERATING.value
    db.commit()

    # Runs in the thread pool after the 202 is sent; opens its own db session.
    background_tasks.add_task(generate_materials, session_id)

    return ApproveOut(
        phase=Phase.GENERATING,
        # The confirmation is a user-facing reply, so it is in the learner's
        # language (falls back to the English text if translation fails).
        message=localize_status(
            pre_material_llm,
            session.language or DEFAULT_LANGUAGE,
            "Plan approved. Material generation started in the background.",
        ),
    )


@router.post(
    "/dev/sessions/{session_id}/regenerate",
    response_model=ApproveOut,
    status_code=202,
    dependencies=GATE_THE_WRITES,
)
def dev_regenerate(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
) -> ApproveOut:
    """Dev-only: wipe existing materials and re-run generation.

    Requires the session to already have a plan and boundary_map
    (i.e. it has passed the planning phase). Skips all upstream phases.
    """
    # #118: dev regeneration is refused in mock mode (slide generation is
    # disabled there).
    if is_mock_mode():
        raise HTTPException(
            status_code=400, detail="Dev regeneration is disabled in mock mode."
        )
    if not is_dev_mode():
        raise HTTPException(status_code=404, detail="Not found")

    session = _get_session_or_404(db, session_id)
    if session.plan is None or session.boundary_map is None:
        raise HTTPException(
            status_code=409,
            detail="Session has no plan or boundary_map — cannot regenerate",
        )

    # Wipe existing materials so generation starts fresh.
    for m in list(session.materials):
        db.delete(m)
    for s in list(session.slide_contents):
        db.delete(s)

    # Delete material checkpoint threads so stale in-memory state
    # (e.g. old slide_contents/slides) does not resume into the fresh run.
    steps = session.plan.steps
    for step in steps:
        try:
            checkpointer.delete_thread(f"{session_id}:material:{step['id']}")
        except Exception:  # noqa: BLE001 — best-effort cleanup
            logger.warning(
                "Failed to delete material checkpoint thread for step %s",
                step["id"],
            )

    session.phase = Phase.GENERATING.value
    db.commit()

    background_tasks.add_task(generate_materials, session_id)
    return ApproveOut(
        phase=Phase.GENERATING,
        message="Material generation restarted (dev).",
    )


@router.post(
    "/sessions/{session_id}/slides/{slide_id}/regenerate",
    response_model=RegenerateOut,
    status_code=202,
    dependencies=GATE_THE_WRITES,
)
def request_slide_regeneration(
    session_id: str,
    slide_id: str,
    background_tasks: BackgroundTasks,
    body: RegenerateIn | None = None,
    db: DBSession = Depends(get_db),
) -> RegenerateOut:
    """Regenerate one slide of a deck, leaving every other slide alone.

    A gated write (#178): it changes the deck, and the request that asks for it
    belongs to whoever is working on that Session.

    Everything needed to write the slide again is already in this backend: the
    content spec on the slide's own row (#197), the step in the plan, the prior
    steps' summaries, the boundary map, the language. That is what makes a
    single slide a self-contained operation instead of a reason to wipe the deck
    and re-run generation (``POST /dev/sessions/{id}/regenerate``).

    Answers ``202`` because writing a slide is several model calls: the work runs
    in the background and the row is replaced when a new slide compiles. A
    regeneration that exhausts its attempts changes nothing — the slide the
    learner has stays.

    The request also marks the slide (``is_regenerating``), in the commit that
    answers it, and the deck read surfaces the mark as ``regenerating`` on the
    slide item. A ``202`` on its own tells a client only that something started,
    and a rewritten slide keeps its ID, so the mark is the only thing it has to
    watch; a second request for a marked slide is refused, which is the rule the
    whole-deck route already holds for a deck. Clearing the mark is the task's
    job, on every outcome.
    """
    # #118: mock mode never generates slides, so there is no slide to rewrite.
    if is_mock_mode():
        raise HTTPException(
            status_code=400, detail="Slide generation is disabled in mock mode."
        )

    session = _get_session_or_404(db, session_id)
    slide = db.get(SlideContent, slide_id)
    if slide is None or slide.session_id != session_id:
        raise HTTPException(status_code=404, detail="Slide not found")
    # A placeholder slide is invisible outside dev mode here as everywhere else
    # in this API (GET /slides/{id}, the deck read): in production it was never
    # in the deck, so there is nothing on screen to ask about.
    if slide.is_placeholder and not is_dev_mode():
        raise HTTPException(status_code=404, detail="Slide not found")
    if session.phase == Phase.GENERATING.value:
        raise HTTPException(
            status_code=409,
            detail="Material generation is still running for this session",
        )
    # One rewrite per slide at a time, which is the rule the whole-deck route
    # already holds for a deck: a second request is refused instead of queueing
    # behind the first and paying for the same slide twice. Two requests that
    # arrive together, before either has marked the row, are still both accepted
    # and serialised by the task's per-session lock — the mark is what makes the
    # repeated request visible to ask about, not a second lock.
    if slide.is_regenerating:
        raise HTTPException(
            status_code=409,
            detail="This slide is already being regenerated",
        )
    if _plan_position(session, slide.step_id) is None:
        raise HTTPException(
            status_code=409,
            detail="Session has no plan step for this slide — cannot regenerate",
        )
    if slide.content_spec is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This slide has no stored content spec — it was generated "
                "before specs were stored. Regenerate the whole deck instead."
            ),
        )

    # The mark is written here, in the request that is answered with the 202,
    # for the same reason `approve_plan` writes `phase = generating` before it
    # schedules a deck run: the work has to be visible in the very next read,
    # not whenever the thread pool gets to it. Clearing it is the task's job, on
    # every outcome.
    slide.is_regenerating = True
    db.commit()

    feedback = ((body.feedback or "") if body is not None else "").strip()
    background_tasks.add_task(
        regenerate_slide_material, session_id, slide_id, feedback or None
    )
    # The acknowledgement is a user-facing reply, so it is in the learner's
    # language (falls back to the English text if translation fails) — the same
    # rule the plan approval follows.
    return RegenerateOut(
        slide_id=slide_id,
        message=localize_status(
            pre_material_llm,
            session.language or DEFAULT_LANGUAGE,
            "Regeneration started. The slide is replaced when a new one compiles.",
        ),
    )


@router.get(
    "/sessions/{session_id}/materials",
    response_model=MaterialsOut,
)
def get_materials(session_id: str, db: DBSession = Depends(get_db)) -> MaterialsOut:
    """Poll material generation progress with full content.

    A public read (#178): this is the deck itself, and a Visitor opens it by
    address (ADR-0004). No sign-in dependency is declared on this route.
    """
    session = _get_session_or_404(db, session_id)

    try:
        phase = Phase(session.phase)
    except ValueError:
        logger.error("Session %s has unknown phase %r", session_id, session.phase)
        raise HTTPException(
            status_code=500,
            detail=f"Session {session_id} has unknown phase {session.phase!r}",
        )

    # Placeholder slides (a failed slide's stand-in) are always generated
    # and persisted, but are only returned here in dev mode — in production
    # the slot is omitted from the deck. ``is_dev_mode`` is the only reading of
    # that flag this endpoint may use, so it agrees with the write gate and with
    # GET /slides/{slide_id} about which slides exist.
    #
    # One pass over the deck's rows: which slots are placeholders, and which are
    # being rewritten right now. A slide the deck is rewriting is never hidden by
    # the placeholder filter: that slot is about to hold a real slide, and the
    # rewrite is what a reader of that slot is waiting for.
    deck_rows = {s.slide_id: s for s in session.slide_contents}
    if is_dev_mode():
        placeholder_ids: set[str] = set()
    else:
        placeholder_ids = {
            sid for sid, s in deck_rows.items() if s.is_placeholder
        }
    regenerating_ids = {
        sid for sid, s in deck_rows.items() if s.is_regenerating
    }

    generated_steps = [
        MaterialOut(
            step_id=m.step_id,
            summary=SummaryOut(**m.summary),
            items=[
                *[
                    SlideItem(
                        type="slide",
                        slide_id=sid,
                        regenerating=sid in regenerating_ids,
                    )
                    for sid in m.slides
                    if sid not in placeholder_ids or sid in regenerating_ids
                ],
                *[QuestionItem(type="question", **q) for q in m.questions],
            ],
        )
        for m in _ordered_materials(session)
    ]

    return MaterialsOut(phase=phase, generated_steps=generated_steps)
